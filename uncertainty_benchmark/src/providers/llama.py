"""Local large-70B provider — DeepSeek-R1-Distill-Llama-70B (default).

Architecture: Llama-3 base, distilled from DeepSeek-R1 (reasoning model).
This model emits <think>...</think> blocks which are stripped automatically.

Key properties:
  • Supports system role (Llama-3 chat template)
  • 70B — device_map="auto" spans all available GPUs
  • 4-bit NF4 quantization by default (~35-40 GB VRAM, fits A100-80GB)
  • Thinking blocks stripped from output before returning text

Model IDs:
  "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"  (default, ungated)
  "meta-llama/Llama-3.3-70B-Instruct"           (gated, requires HF approval)
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from .base import LLMProvider

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"

# Strip <think>...</think> reasoning blocks emitted by DeepSeek distill models
_THINKING_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove DeepSeek-style <think>...</think> blocks from output."""
    return _THINKING_RE.sub("", text).strip()


class LlamaProvider(LLMProvider):
    """DeepSeek-R1-Distill-Llama-70B (or any Llama-3-based model) running locally."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device_map: str = "auto",
        load_in_4bit: bool = True,
        torch_dtype: str = "bfloat16",
        hf_token: Optional[str] = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        except ImportError as e:
            raise ImportError(
                "transformers and torch are required for LlamaProvider. "
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
            logger.info("LlamaProvider — 4-bit quantization enabled")

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
        logger.info("LlamaProvider ready — model=%s", model_id)

    @property
    def provider_name(self) -> str:
        # Use "deepseek" when running the DeepSeek distill; "llama" otherwise.
        if "deepseek" in self._model_id.lower():
            return "deepseek"
        return "llama"

    @property
    def model_name(self) -> str:
        return self._model_id

    @property
    def supports_logprobs(self) -> bool:
        return True

    def _build_input_ids(self, system_instruction: str, user_message: str):
        """Llama-3 chat template — supports system role natively.

        For DeepSeek-R1-Distill models: pre-fills an empty <think></think>
        block so the model skips extended chain-of-thought reasoning and
        outputs the JSON answer directly. Without this, the model exhausts
        max_tokens on thinking before ever producing the JSON.
        """
        messages = [
            {"role": "system", "content": system_instruction.strip()},
            {"role": "user",   "content": user_message.strip()},
        ]

        is_deepseek = "deepseek" in self._model_id.lower()

        if is_deepseek:
            # Pre-fill empty thinking block: forces model past reasoning phase.
            # Use continue_final_message=True to extend this partial turn.
            messages.append({"role": "assistant", "content": "<think>\n\n</think>\n\n"})
            result = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=False,
                continue_final_message=True,
                return_tensors="pt",
                tokenize=True,
            )
        else:
            result = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                tokenize=True,
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
        for step_scores in output.scores:
            step_lp = torch.log_softmax(step_scores[0], dim=-1)
            top = step_lp.topk(min(top_k_logprobs, step_lp.size(-1)))
            token_lp = {}
            for idx, lp in zip(top.indices.tolist(), top.values.tolist()):
                token_str = self._tokenizer.decode([idx])
                token_lp[token_str] = float(lp)
            logprobs.append(token_lp)

        return text, logprobs
