"""Local Gemma-3-12B-IT provider using HuggingFace Transformers.

Key differences from API providers:
  • Runs on local GPU via device_map="auto"
  • Returns per-token top-k logprobs from call_with_logprobs()
  • Gemma does not support a system role — system instruction is prepended
    as a prefix to the user message (same approach as pilot GemmaProvider)
  • Optional 4-bit quantization via load_in_4bit=True

Model IDs:
  "google/gemma-3-12b-it"  (default)
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from .base import LLMProvider
from ..utils import SafetyBlockError

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "google/gemma-3-12b-it"


class GemmaProvider(LLMProvider):
    """Gemma-3 instruct model running locally via transformers."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device_map: str = "auto",
        load_in_4bit: bool = False,
        torch_dtype: str = "bfloat16",
        hf_token: Optional[str] = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        except ImportError as e:
            raise ImportError(
                "transformers and torch are required for GemmaProvider. "
                "Install with: pip install transformers torch"
            ) from e

        import os
        token = hf_token or os.environ.get("HF_TOKEN")

        dtype = getattr(torch, torch_dtype)

        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            logger.info("GemmaProvider — 4-bit quantization enabled")

        logger.info("Loading tokenizer: %s ...", model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=token,
        )

        logger.info("Loading model: %s (device_map=%s) ...", model_id, device_map)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=dtype,
            quantization_config=quantization_config,
            token=token,
        )
        self._model.eval()
        self._model_id = model_id
        self._torch = torch
        logger.info("GemmaProvider ready — model=%s", model_id)

    @property
    def provider_name(self) -> str:
        return "gemma"

    @property
    def model_name(self) -> str:
        return self._model_id

    @property
    def supports_logprobs(self) -> bool:
        return True

    def _build_input_ids(self, system_instruction: str, user_message: str):
        """Gemma has no system role — combine into single user turn."""
        combined = f"{system_instruction.strip()}\n\n{user_message.strip()}"
        messages = [{"role": "user", "content": combined}]
        result = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        # Newer transformers returns BatchEncoding; extract the tensor.
        if hasattr(result, "input_ids"):
            result = result.input_ids
        return result.to(self._model.device)

    def _decode_output(self, input_ids, output_ids) -> str:
        """Decode only the newly generated tokens (skip the prompt)."""
        new_ids = output_ids[0, input_ids.shape[1]:]
        return self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    def call(
        self,
        system_instruction: str,
        user_message: str,
        temperature: float = 0.0,
        max_tokens: int = 4000,
        expect_json: bool = False,
    ) -> str:
        import torch
        input_ids = self._build_input_ids(system_instruction, user_message)
        do_sample = temperature > 0.0

        with torch.inference_mode():
            output = self._model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature if do_sample else None,
                do_sample=do_sample,
                top_p=0.95 if do_sample else None,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        return self._decode_output(input_ids, output)

    def call_with_logprobs(
        self,
        system_instruction: str,
        user_message: str,
        temperature: float = 0.0,
        max_tokens: int = 4000,
        top_k_logprobs: int = 20,
    ) -> tuple[str, list[dict[str, float]]]:
        """Generate text and return per-token top-k logprobs.

        Returns:
            (text, logprobs) where logprobs is a list of dicts:
            [{token_str: log_prob}, ...], one dict per generated token.
        """
        import torch
        input_ids = self._build_input_ids(system_instruction, user_message)
        do_sample = temperature > 0.0

        with torch.inference_mode():
            output = self._model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature if do_sample else None,
                do_sample=do_sample,
                top_p=0.95 if do_sample else None,
                pad_token_id=self._tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        text = self._decode_output(input_ids, output.sequences)

        # Extract top-k logprobs per generated token
        logprobs: list[dict[str, float]] = []
        for step_scores in output.scores:
            # step_scores shape: (batch=1, vocab_size)
            step_lp = torch.log_softmax(step_scores[0], dim=-1)
            top = step_lp.topk(min(top_k_logprobs, step_lp.size(-1)))
            token_lp = {}
            for idx, lp in zip(top.indices.tolist(), top.values.tolist()):
                token_str = self._tokenizer.decode([idx])
                token_lp[token_str] = float(lp)
            logprobs.append(token_lp)

        return text, logprobs
