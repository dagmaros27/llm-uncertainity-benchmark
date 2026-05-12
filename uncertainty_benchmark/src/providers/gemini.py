"""Gemini provider — Google Gemini API via the google-genai SDK.

Prefers VERTEX_API_KEY (paid tier, used by the pilot). Falls back to
GEMINI_API_KEY (AI Studio free tier) only if Vertex isn't set.

Supports:
  • call()           — single-turn with optional JSON schema
  • call_multiturn() — role-alternating conversation history
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Union

import tenacity
from google import genai
from google.genai import types

from .base import LLMProvider
from ..utils import SafetyBlockError, extract_non_thinking_text

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):

    def __init__(
        self,
        model_id: str = "gemini-2.5-flash",
        default_temperature: float = 0.0,
        api_version: str = "v1beta",
    ) -> None:
        api_key = os.environ.get("VERTEX_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "VERTEX_API_KEY (or GEMINI_API_KEY) not set. Add it to your .env file."
            )
        key_source = "VERTEX_API_KEY" if os.environ.get("VERTEX_API_KEY") else "GEMINI_API_KEY"
        self._api_key = api_key
        self._api_version = api_version
        self._model_id = model_id
        self._default_temperature = default_temperature
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(api_version=api_version),
        )
        logger.info(
            "GeminiProvider ready — model=%s api_version=%s key=%s",
            model_id, api_version, key_source,
        )

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._model_id

    @property
    def supports_logprobs(self) -> bool:
        return False

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
        expect_json: Union[bool, "types.Schema"] = False,
    ) -> str:
        full_prompt = f"{system_instruction.strip()}\n\n{user_message.strip()}"
        config_kwargs: dict = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "top_p": 0.95,
        }
        if expect_json is not False:
            config_kwargs["response_mime_type"] = "application/json"
            if not isinstance(expect_json, bool):
                config_kwargs["response_schema"] = expect_json

        try:
            response = self._client.models.generate_content(
                model=self._model_id,
                contents=full_prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            model_response = extract_non_thinking_text(response)
            if model_response.was_blocked:
                raise SafetyBlockError(
                    f"Response blocked: finish_reason={model_response.finish_reason}"
                )
            return model_response.text
        except SafetyBlockError:
            raise
        except Exception as exc:
            logger.warning("Gemini call failed (model=%s): %s", self._model_id, exc)
            raise

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
        expect_json: Union[bool, "types.Schema"] = False,
    ) -> str:
        """Multi-turn call using the Gemini contents API.

        ``contents`` must end with a user turn. System instruction is injected via
        GenerateContentConfig (never placed inside the contents list).
        """
        content_objects = [
            types.Content(role=c["role"], parts=[types.Part(text=c["text"])])
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
            if not isinstance(expect_json, bool):
                config_kwargs["response_schema"] = expect_json

        try:
            response = self._client.models.generate_content(
                model=self._model_id,
                contents=content_objects,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            model_response = extract_non_thinking_text(response)
            if model_response.was_blocked:
                raise SafetyBlockError(
                    f"Response blocked: finish_reason={model_response.finish_reason}"
                )
            logger.debug("call_multiturn — %d turns in context", len(contents))
            return model_response.text
        except SafetyBlockError:
            raise
        except Exception as exc:
            logger.warning(
                "Gemini multi-turn call failed (model=%s): %s", self._model_id, exc
            )
            raise
