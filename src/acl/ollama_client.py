from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    """Tiny dependency-free client for Ollama's local chat API."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        temperature: float,
        num_ctx: int,
    ) -> dict[str, Any]:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,
            "format": schema,
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx,
            },
        }

        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OllamaError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise OllamaError(
                f"Could not reach Ollama at {self.base_url}. "
                "Make sure Ollama is running."
            ) from exc
        except TimeoutError as exc:
            raise OllamaError("Ollama request timed out.") from exc

        try:
            content = body["message"]["content"]
            return json.loads(content)
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise OllamaError(
                f"Ollama returned an unexpected response: {body!r}"
            ) from exc
