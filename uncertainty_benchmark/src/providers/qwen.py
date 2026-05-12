"""Local Qwen3-4B provider using HuggingFace Transformers.

Key differences from Gemma/Llama:
  • Thinking mode: always disabled via enable_thinking=False in apply_chat_template
    (validated in Phase 0.0 pre-flight: 100% native JSON compliance without thinking)
  • Supports system role
  • strip_thinking() fallback in case thinking tokens slip through

Model IDs:
  "Qwen/Qwen3-4B"  (default — validated in pre-flight)
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from .base import LLMProvider

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Qwen/Qwen3-4B"

# Fallback: strip any <think>...</think> blocks that slip through
_THINKING_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_thinking(text: str) -> str:
    return _THINKING_RE.sub("", text).strip()


class QwenProvider(LLMProvider):
    """Qwen3-4B running locally via transformers with thinking mode disabled."""

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
                "transformers and torch are required for QwenProvider. "
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
            logger.info("QwenProvider — 4-bit quantization enabled")

        logger.info("Loading tokenizer: %s ...", model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=token,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

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
        logger.info("QwenProvider ready — model=%s (thinking=OFF)", model_id)

    @property
    def provider_name(self) -> str:
        return "qwen"

    @property
    def model_name(self) -> str:
        return self._model_id

    @property
    def supports_logprobs(self) -> bool:
        return True

    def _build_input_ids(self, system_instruction: str, user_message: str):
        """Qwen3 supports system role. enable_thinking=False suppresses <think> blocks."""
        messages = [
            {"role": "system", "content": system_instruction.strip()},
            {"role": "user",   "content": user_message.strip()},
        ]
        try:
            result = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt",
            )
        except TypeError:
            # Older tokenizer versions may not have enable_thinking
            logger.warning("enable_thinking kwarg not supported — using standard template")
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
        new_ids = output_ids[0, input_ids.shape[1]:]
        text = self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        return strip_thinking(text)

    def call(
        self,
        system_instruction: str,
        user_message: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
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
        max_tokens: int = 512,
        top_k_logprobs: int = 20,
    ) -> tuple[str, list[dict[str, float]]]:
        """Generate with thinking=OFF and return per-token top-k logprobs.

        Validated in Phase 0.0 pre-flight: vocab_size=151,936, 46 steps typical.
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

        logprobs: list[dict[str, float]] = []
        input_len = input_ids.shape[1]
        for step_idx, step_scores in enumerate(output.scores):
            step_lp = torch.log_softmax(step_scores[0], dim=-1)
            top = step_lp.topk(min(top_k_logprobs, step_lp.size(-1)))
            token_lp: dict[str, float] = {}
            for idx, lp in zip(top.indices.tolist(), top.values.tolist()):
                token_str = self._tokenizer.decode([idx])
                token_lp[token_str] = float(lp)
            # Store the log-prob of the actually generated token for LNPE
            gen_id = output.sequences[0, input_len + step_idx].item()
            token_lp["__gen_lp__"] = float(step_lp[gen_id])
            logprobs.append(token_lp)

        return text, logprobs
