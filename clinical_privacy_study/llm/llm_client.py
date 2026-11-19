"""OpenAI-compatible chat transport. Credentials are read only at request time."""

import json
import logging
import os
from dataclasses import dataclass
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from config import Llm


@dataclass(frozen=True)
class Generation:
    text: str
    returned_model: str
    fingerprint: str | None = None


class LlmClient(Protocol):
    def generate(self, prompt: str, model: str) -> Generation:
        """Return completion text and provider-reported model identity."""
        ...


class NoRedirect(HTTPRedirectHandler):
    """Never forward requests or authorization headers across redirects."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class ChatClient:
    def __init__(self, settings: Llm, logger: logging.Logger) -> None:
        self.settings = settings
        self.logger = logger
        default = ("http://127.0.0.1:8000/v1/chat/completions" if settings.local
                   else "https://api.openai.com/v1/chat/completions")
        self.endpoint = os.environ.get("CLINICAL_LLM_ENDPOINT", default)
        url = urlsplit(self.endpoint)
        if (not url.hostname or url.username or url.password or url.query or url.fragment
                or url.scheme not in ("http", "https")):
            raise ValueError("Use a chat endpoint without credentials or query parameters.")
        if settings.local:
            if url.hostname not in ("localhost", "127.0.0.1", "::1"):
                raise ValueError("Local mode requires a loopback endpoint.")
        elif url.scheme != "https":
            raise ValueError("Remote chat endpoints require HTTPS.")

    def generate(self, prompt: str, model: str) -> Generation:
        headers = {"Content-Type": "application/json"}
        if not self.settings.local:
            key = os.environ.get("CLINICAL_LLM_API_KEY")
            if not key:
                raise ValueError("Set CLINICAL_LLM_API_KEY in this process environment.")
            headers["Authorization"] = f"Bearer {key}"
        payload = {"model": model, "temperature": self.settings.temperature,
                   "max_tokens": self.settings.max_tokens,
                   "messages": [{"role": "user", "content": prompt}]}
        request = Request(self.endpoint, json.dumps(payload).encode(), headers=headers)
        self.logger.info("LLM request: model=%s host=%s", model, urlsplit(self.endpoint).hostname)
        try:
            with build_opener(NoRedirect()).open(request, timeout=60) as response:
                result = json.load(response)
            choice = result["choices"][0]
            if choice.get("finish_reason") not in (None, "stop"):
                raise ValueError("Incomplete or filtered LLM generation.")
            text = choice["message"]["content"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Empty LLM completion.")
            return Generation(text, result["model"], result.get("system_fingerprint"))
        except HTTPError as exc:
            raise ValueError(f"Chat provider returned HTTP {exc.code}.") from None
        except (URLError, TimeoutError, KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"Chat request failed: {type(exc).__name__}.") from None
