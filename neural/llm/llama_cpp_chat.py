"""
llama.cpp bridge with brain-driven logits processing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .brain_logits import BrainLogitsProcessor

try:
    from llama_cpp import Llama
except ImportError:  # pragma: no cover - optional dependency
    Llama = None


@dataclass
class ChatConfig:
    n_ctx: int = 4096
    n_gpu_layers: int = 35
    n_batch: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    max_tokens: int = 256


class BrainLlamaChat:
    """Thin wrapper around llama.cpp with a brain-driven logits processor."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = 35,
        n_batch: int = 512,
        seed: int = 0,
    ) -> None:
        if Llama is None:
            raise RuntimeError("llama-cpp-python is not installed.")

        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            seed=seed,
        )
        self.processor = BrainLogitsProcessor(self.llm)

    def build_prompt(self, history: List[Tuple[str, str]], user_input: str, memory: str = "") -> str:
        lines: List[str] = []
        if memory:
            lines.append("Context:")
            lines.append(memory.strip())
            lines.append("")
        for role, content in history:
            prefix = "User" if role == "user" else "Assistant"
            lines.append(f"{prefix}: {content}")
        lines.append(f"User: {user_input}")
        lines.append("Assistant:")
        return "\n".join(lines)

    def generate(
        self,
        prompt: str,
        brain_state: Dict,
        temperature: float,
        top_p: float,
        repeat_penalty: float,
        max_tokens: int,
    ) -> str:
        self.processor.set_state(brain_state)
        output = self.llm.create_completion(
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            max_tokens=max_tokens,
            stop=["\nUser:"],
            logits_processor=[self.processor],
        )
        return output["choices"][0]["text"].strip()
