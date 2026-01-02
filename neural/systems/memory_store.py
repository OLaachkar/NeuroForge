"""
Lightweight local memory store for conversation snippets.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Dict, List, Optional


_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def tokenize_text(text: str) -> List[str]:
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    return [t for t in tokens if len(t) > 1]


class MemoryStore:
    """Simple on-disk store with keyword overlap retrieval."""

    def __init__(self, path: str, max_entries: int = 200) -> None:
        self.path = path
        self.max_entries = max_entries
        self.entries: List[Dict] = []
        self.next_id = 1
        self.load()

    def load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            self.entries = list(data.get("entries", []))
            self.next_id = int(data.get("next_id", len(self.entries) + 1))
        except (json.JSONDecodeError, OSError, ValueError):
            self.entries = []
            self.next_id = 1

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        payload = {"next_id": self.next_id, "entries": self.entries[-self.max_entries :]}
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def add(self, text: str, tags: Optional[List[str]] = None, affect: Optional[Dict[str, float]] = None) -> Dict:
        tokens = tokenize_text(text)
        affect_payload = None
        if affect:
            affect_payload = {
                "valence": _clamp(float(affect.get("valence", 0.0)), -1.0, 1.0),
                "arousal": _clamp(float(affect.get("arousal", 0.0)), 0.0, 1.0),
                "tension": _clamp(float(affect.get("tension", 0.0)), 0.0, 1.0),
            }
        entry = {
            "id": self.next_id,
            "text": text,
            "tags": tags or [],
            "tokens": tokens,
            "affect": affect_payload,
            "timestamp": time.time(),
        }
        self.entries.append(entry)
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries :]
        self.next_id += 1
        return entry

    def search(self, query: str, limit: int = 5, affect: Optional[Dict[str, float]] = None) -> List[Dict]:
        query_tokens = set(tokenize_text(query))
        if not query_tokens:
            return []
        scored = []
        for entry in self.entries:
            entry_tokens = set(entry.get("tokens", []))
            if not entry_tokens:
                continue
            overlap = query_tokens.intersection(entry_tokens)
            if not overlap:
                continue
            score = len(overlap) / len(query_tokens.union(entry_tokens))
            if affect and entry.get("affect"):
                score += 0.15 * _affect_similarity(affect, entry["affect"])
            scored.append((score, entry))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [entry for _, entry in scored[:limit]]


def _affect_similarity(query_affect: Dict[str, float], entry_affect: Dict[str, float]) -> float:
    valence_q = float(query_affect.get("valence", 0.0))
    arousal_q = float(query_affect.get("arousal", 0.0))
    tension_q = float(query_affect.get("tension", 0.0))

    valence_e = float(entry_affect.get("valence", 0.0))
    arousal_e = float(entry_affect.get("arousal", 0.0))
    tension_e = float(entry_affect.get("tension", 0.0))

    valence_sim = 1.0 - min(1.0, abs(valence_q - valence_e) / 2.0)
    arousal_sim = 1.0 - min(1.0, abs(arousal_q - arousal_e))
    tension_sim = 1.0 - min(1.0, abs(tension_q - tension_e))
    return 0.5 * valence_sim + 0.3 * arousal_sim + 0.2 * tension_sim
