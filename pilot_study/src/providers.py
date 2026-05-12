"""LLM provider abstraction: LLMProvider ABC + GeminiProvider + GemmaProvider.

To add a new provider (e.g. a local Ollama model), subclass LLMProvider
and implement `call`, `provider_name`, and `model_name`.

GemmaProvider wraps Gemma models served through the Gemini API.
Gemma does not support system_instruction or JSON mode, so this provider:
  - Prepends the system instruction as a prefix to the first user turn
  - Returns raw text (JSON parsing is handled downstream by parse_json_response)
"""

from __future__ import annotations

import abc
import logging
import os
from typing import Optional, Union

import tenacity
from google import genai
from google.genai import types

from .utils import SafetyBlockError, extract_non_thinking_text

logger = logging.getLogger(__name__)


# ── Abstract base ──────────────────────────────────────────────────────────

class LLMProvider(abc.ABC):
    """Strategy interface for all LLM backends."""

    @abc.abstractmethod
    def call(
        self,
        system_instruction: str,
        user_message: str,
        temperature: float = 0.0,
        max_tokens: int = 4000,
        expect_json: Union[bool, types.Schema] = False,
    ) -> str:
        """Make a single (one-shot) call and return the response text."""
        ...

    def call_multiturn(
        self,
        system_instruction: str,
        contents: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 4000,
        expect_json: Union[bool, types.Schema] = False,
    ) -> str:
        """Multi-turn call with proper conversation history.

        ``contents`` is a list of ``{"role": "user"|"model", "text": str}``
        dicts representing the accumulated conversation so far (must end with
        a user turn).  The provider appends the model reply and returns it.

        Default implementation raises NotImplementedError — concrete providers
        must override this to participate in the multi-turn pipeline.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement call_multiturn")

    @property
    @abc.abstractmethod
    def provider_name(self) -> str: ...

    @property
    @abc.abstractmethod
    def model_name(self) -> str: ...


# ── Gemini ─────────────────────────────────────────────────────────────────

class GeminiProvider(LLMProvider):
    """Google Gemini via the google-genai SDK. Reads VERTEX_API_KEY from env."""

    def __init__(
        self,
        model_id: str = "gemini-2.5-flash",
        default_temperature: float = 0.0,
        api_version: str = "v1beta",
    ) -> None:
        api_key = os.environ.get("VERTEX_API_KEY")
        if not api_key:
            raise EnvironmentError("VERTEX_API_KEY not set. Add it to your .env file.")
        self._api_key = api_key
        self._api_version = api_version
        self._model_id = model_id
        self._default_temperature = default_temperature
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(api_version=api_version),
        )
        logger.info("GeminiProvider ready — model=%s api_version=%s", model_id, api_version)

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._model_id

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=2, min=4, max=60),
        stop=tenacity.stop_after_attempt(6),
        retry=tenacity.retry_if_not_exception_type(SafetyBlockError),
        before_sleep=lambda rs: logger.warning(
            "Gemini retry — sleeping %.0fs (attempt %d)",
            rs.next_action.sleep if rs.next_action else 0,
            rs.attempt_number,
        ),
        reraise=True,
    )
    def call(
        self,
        system_instruction: str,
        user_message: str,
        temperature: float = 0.0,
        max_tokens: int = 4000,
        expect_json: Union[bool, types.Schema] = False,
    ) -> str:
        full_prompt = f"{system_instruction.strip()}\n\n{user_message.strip()}"

        # Try primary model first; fall back to v1beta only if a different version
        # was requested. No model substitution — tenacity handles transient retries.
        unique: list[tuple[str, str]] = [(self._model_id, self._api_version)]
        if self._api_version != "v1beta":
            unique.append((self._model_id, "v1beta"))

        last_error: Optional[Exception] = None
        for model_id, api_version in unique:
            try:
                client = self._client
                if api_version != self._api_version:
                    client = genai.Client(
                        api_key=self._api_key,
                        http_options=types.HttpOptions(api_version=api_version),
                    )
                config_kwargs: dict = {
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                    "top_p": 0.95,
                }
                if expect_json is not False:
                    config_kwargs["response_mime_type"] = "application/json"
                    if isinstance(expect_json, types.Schema):
                        config_kwargs["response_schema"] = expect_json

                response = client.models.generate_content(
                    model=model_id,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(**config_kwargs),
                )

                if model_id != self._model_id:
                    logger.warning("Fallback model used: requested=%s actual=%s", self._model_id, model_id)
                if api_version != self._api_version:
                    logger.warning("Fallback api_version used: requested=%s actual=%s", self._api_version, api_version)

                model_response = extract_non_thinking_text(response)
                if model_response.was_blocked:
                    raise SafetyBlockError(f"Response blocked: finish_reason={model_response.finish_reason}")
                return model_response.text

            except SafetyBlockError:
                raise
            except Exception as exc:
                last_error = exc
                logger.warning("Gemini call failed (model=%s api_version=%s): %s", model_id, api_version, exc)

        raise RuntimeError(f"All Gemini call attempts failed. Last error: {last_error}") from last_error

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=2, min=4, max=60),
        stop=tenacity.stop_after_attempt(6),
        retry=tenacity.retry_if_not_exception_type(SafetyBlockError),
        before_sleep=lambda rs: logger.warning(
            "Gemini multi-turn retry — sleeping %.0fs (attempt %d)",
            rs.next_action.sleep if rs.next_action else 0,
            rs.attempt_number,
        ),
        reraise=True,
    )
    def call_multiturn(
        self,
        system_instruction: str,
        contents: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 4000,
        expect_json: Union[bool, types.Schema] = False,
    ) -> str:
        """Multi-turn call using the Gemini ``contents`` API.

        ``contents`` is a list of ``{"role": "user"|"model", "text": str}``
        dicts.  The list must end with a user turn; the model's reply is
        returned as a string (NOT appended to the list — callers manage state).
        ``system_instruction`` is injected via ``GenerateContentConfig`` so it
        is never exposed as a user message in the conversation history.
        """
        content_objects = [
            types.Content(
                role=c["role"],
                parts=[types.Part(text=c["text"])],
            )
            for c in contents
        ]

        config_kwargs: dict = {
            "system_instruction": system_instruction,
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "top_p": 0.95,
        }
        if expect_json is not False:
            config_kwargs["response_mime_type"] = "application/json"
            if isinstance(expect_json, types.Schema):
                config_kwargs["response_schema"] = expect_json

        response = self._client.models.generate_content(
            model=self._model_id,
            contents=content_objects,
            config=types.GenerateContentConfig(**config_kwargs),
        )

        model_response = extract_non_thinking_text(response)
        if model_response.was_blocked:
            raise SafetyBlockError(f"Response blocked: finish_reason={model_response.finish_reason}")
        logger.debug("call_multiturn — %d turns in context", len(contents))
        return model_response.text


# ── Gemma (via Gemini API) ─────────────────────────────────────────────────

class GemmaProvider(GeminiProvider):
    """Gemma models accessed through the Google Gemini API.

    Limitations vs GeminiProvider:
      - system_instruction is NOT supported — it is prepended to the first user
        turn as a plain text prefix instead.
      - JSON mode (response_mime_type / response_schema) is NOT supported —
        the model is instructed via prompt to return JSON, and the raw text is
        returned for downstream parsing by parse_json_response().
    """

    @property
    def provider_name(self) -> str:
        return "gemma"

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=2, min=4, max=60),
        stop=tenacity.stop_after_attempt(6),
        retry=tenacity.retry_if_not_exception_type(SafetyBlockError),
        before_sleep=lambda rs: logger.warning(
            "Gemma retry — sleeping %.0fs (attempt %d)",
            rs.next_action.sleep if rs.next_action else 0,
            rs.attempt_number,
        ),
        reraise=True,
    )
    def call(
        self,
        system_instruction: str,
        user_message: str,
        temperature: float = 0.0,
        max_tokens: int = 4000,
        expect_json: Union[bool, types.Schema] = False,
    ) -> str:
        """Single-turn call. system_instruction is prepended to user_message."""
        combined = f"{system_instruction.strip()}\n\n{user_message.strip()}"
        response = self._client.models.generate_content(
            model=self._model_id,
            contents=combined,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                top_p=0.95,
            ),
        )
        model_response = extract_non_thinking_text(response)
        if model_response.was_blocked:
            raise SafetyBlockError(f"Response blocked: finish_reason={model_response.finish_reason}")
        return model_response.text

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=2, min=4, max=60),
        stop=tenacity.stop_after_attempt(6),
        retry=tenacity.retry_if_not_exception_type(SafetyBlockError),
        before_sleep=lambda rs: logger.warning(
            "Gemma multi-turn retry — sleeping %.0fs (attempt %d)",
            rs.next_action.sleep if rs.next_action else 0,
            rs.attempt_number,
        ),
        reraise=True,
    )
    def call_multiturn(
        self,
        system_instruction: str,
        contents: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 4000,
        expect_json: Union[bool, types.Schema] = False,
    ) -> str:
        """Multi-turn call. system_instruction is prepended to the LAST user turn.

        Gemma does not support a system turn, so the instruction is injected
        as a prefix to the final user message in the conversation (the one that
        requests the model's next response).
        """
        if not contents:
            raise ValueError("contents must not be empty")

        # Build Content objects; prepend instruction to the last user turn
        content_objects: list[types.Content] = []
        for idx, c in enumerate(contents):
            text = c["text"]
            if idx == len(contents) - 1 and c["role"] == "user" and system_instruction:
                text = f"{system_instruction.strip()}\n\n{text}"
            content_objects.append(
                types.Content(role=c["role"], parts=[types.Part(text=text)])
            )

        response = self._client.models.generate_content(
            model=self._model_id,
            contents=content_objects,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                top_p=0.95,
            ),
        )
        model_response = extract_non_thinking_text(response)
        if model_response.was_blocked:
            raise SafetyBlockError(f"Response blocked: finish_reason={model_response.finish_reason}")
        logger.debug("Gemma call_multiturn — %d turns in context", len(contents))
        return model_response.text
