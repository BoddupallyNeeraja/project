"""
Translation client supporting DeepL and Google Cloud Translation APIs.

The implementation keeps external dependencies minimal by relying on the
``requests`` package and environment variables for authentication.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import requests


class TranslationProviderNotConfigured(RuntimeError):
    """Raised when neither DeepL nor Google Cloud credentials are available."""


@dataclass
class TranslationConfig:
    """Structured configuration for the translator."""

    provider: str
    api_key: str
    base_url: Optional[str] = None


class TranslationClient:
    """
    High-level translation helper that chooses the active provider automatically.

    Supported providers:
      - DeepL: requires ``DEEPL_API_KEY`` set in the environment.
      - Google Cloud Translation: requires ``GOOGLE_TRANSLATE_API_KEY`` set.
    """

    def __init__(self) -> None:
        config = self._load_config()
        self.provider = config.provider
        self.api_key = config.api_key
        self.base_url = config.base_url

    def translate_text(
        self,
        text: str,
        target_language: str,
        source_language: Optional[str] = None,
    ) -> str:
        """
        Translate a single ``text`` string into ``target_language``.
        """

        if not text.strip():
            return ""

        if self.provider == "deepl":
            return self._translate_with_deepl(
                text=text,
                target_language=target_language,
                source_language=source_language,
            )

        if self.provider == "google":
            return self._translate_with_google(
                text=text,
                target_language=target_language,
                source_language=source_language,
            )

        raise TranslationProviderNotConfigured(
            "No active translation provider configured."
        )

    def translate_batch(
        self,
        text: str,
        languages: Iterable[str],
        source_language: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Translate ``text`` into multiple ``languages``.
        """

        results: Dict[str, str] = {}
        for lang in languages:
            normalized = lang.strip()
            if not normalized:
                continue
            translated = self.translate_text(
                text=text,
                target_language=normalized,
                source_language=source_language,
            )
            results[normalized] = translated
        return results

    # Provider configuration -------------------------------------------------

    @staticmethod
    def _load_config() -> TranslationConfig:
        deepl_key = os.getenv("DEEPL_API_KEY")
        google_key = os.getenv("GOOGLE_TRANSLATE_API_KEY")

        if deepl_key:
            base_url = os.getenv(
                "DEEPL_API_URL",
                "https://api-free.deepl.com/v2/translate",
            )
            return TranslationConfig(provider="deepl", api_key=deepl_key, base_url=base_url)

        if google_key:
            base_url = os.getenv(
                "GOOGLE_TRANSLATE_API_URL",
                "https://translation.googleapis.com/language/translate/v2",
            )
            return TranslationConfig(provider="google", api_key=google_key, base_url=base_url)

        raise TranslationProviderNotConfigured(
            "Set either DEEPL_API_KEY or GOOGLE_TRANSLATE_API_KEY to enable translation."
        )

    # Provider implementations -----------------------------------------------

    def _translate_with_deepl(
        self,
        text: str,
        target_language: str,
        source_language: Optional[str] = None,
    ) -> str:
        data = {
            "text": text,
            "target_lang": target_language.upper(),
        }
        if source_language:
            data["source_lang"] = source_language.upper()

        headers = {
            "Authorization": f"DeepL-Auth-Key {self.api_key}",
        }

        response = requests.post(
            self.base_url,
            data=data,
            headers=headers,
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        try:
            return payload["translations"][0]["text"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Unexpected DeepL response format: {payload}") from exc

    def _translate_with_google(
        self,
        text: str,
        target_language: str,
        source_language: Optional[str] = None,
    ) -> str:
        params = {
            "q": text,
            "target": target_language,
            "format": "text",
            "key": self.api_key,
        }
        if source_language:
            params["source"] = source_language

        response = requests.post(
            self.base_url,
            params=params,
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        try:
            return payload["data"]["translations"][0]["translatedText"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Unexpected Google Translate response: {payload}") from exc
