"""Abstract base class for all LLM providers."""

from __future__ import annotations

import abc
from typing import Optional


class LLMProvider(abc.ABC):
    """Strategy interface for all LLM backends."""

    @abc.abstractmethod
    def call(
        self,
        system_instruction: str,
        user_message: str,
        temperature: float = 0.0,
        max_tokens: int = 4000,
        expect_json: bool = False,
    ) -> str:
        """Make a single (one-shot) call and return the response text."""
        ...

    def call_with_logprobs(
        self,
        system_instruction: str,
        user_message: str,
        temperature: float = 0.0,
        max_tokens: int = 4000,
        top_k_logprobs: int = 20,
    ) -> tuple[str, Optional[list[dict[str, float]]]]:
        """Call and return (text, per_token_logprobs).

        per_token_logprobs is a list of dicts mapping token_str → log_prob
        (one dict per generated token, top-k only).

        Default implementation falls back to call() with no logprobs.
        Local providers (Gemma, Llama, Qwen) override this.
        """
        text = self.call(
            system_instruction=system_instruction,
            user_message=user_message,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return text, None

    def call_multiturn(
        self,
        system_instruction: str,
        contents: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 4000,
        expect_json: bool = False,
    ) -> str:
        """Multi-turn call with role-alternating conversation history.

        ``contents`` is a list of {"role": "user"|"model", "text": str}.
        Must end with a user turn. Returns the model's reply text.

        Default raises NotImplementedError; API providers override this.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement call_multiturn"
        )

    @property
    def supports_logprobs(self) -> bool:
        """True if this provider returns per-token logprobs from call_with_logprobs."""
        return False

    @property
    @abc.abstractmethod
    def provider_name(self) -> str: ...

    @property
    @abc.abstractmethod
    def model_name(self) -> str: ...
