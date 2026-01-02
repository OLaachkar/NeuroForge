"""
Minimal Ollama HTTP client for local LLM chat.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, Iterable, List, Optional


class OllamaClient:
    def __init__(self, host: str = "http://localhost:11434") -> None:
        self.host = host.rstrip("/")

    def _request(self, path: str, payload: Dict[str, Any], stream: bool) -> Iterable[Dict[str, Any]]:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.host}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        response = urllib.request.urlopen(request, timeout=60)
        if not stream:
            body = response.read().decode("utf-8")
            if body:
                yield json.loads(body)
            return
        for raw_line in response:
            line = raw_line.strip()
            if not line:
                continue
            yield json.loads(line.decode("utf-8"))

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        options: Optional[Dict[str, Any]] = None,
        stream: bool = True,
    ) -> Iterable[Dict[str, Any]]:
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        if options:
            payload["options"] = options
        return self._request("/api/chat", payload, stream)

    def tags(self) -> Dict[str, Any]:
        try:
            request = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(request, timeout=5) as response:
                return json.load(response)
        except urllib.error.URLError:
            return {}
