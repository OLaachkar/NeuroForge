"""
Lightweight affect inference with optional embedding support.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Dict, List, Optional

import numpy as np

try:  # Optional: richer embeddings if available.
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None


_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
_POSITIVE = {
    "good", "great", "thanks", "thank", "love", "nice", "happy", "excited",
    "wonderful", "amazing", "awesome", "cool", "sweet", "pleased",
}
_NEGATIVE = {
    "bad", "sad", "angry", "hate", "upset", "terrible", "awful", "worried",
    "afraid", "scared", "panic", "annoyed", "hurt", "tired",
}
_AROUSAL = {
    "urgent", "fast", "quick", "now", "alert", "focus", "intense", "energy",
    "panic", "rush", "danger", "fight", "run",
}
_CALM = {
    "calm", "steady", "slow", "relaxed", "soft", "quiet", "safe", "gentle",
}
_STRESS = {
    "stress", "overwhelmed", "anxious", "panic", "danger", "threat", "fear",
    "worry", "tense", "pressure",
}


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _hash_embed(tokens: List[str], dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    if not tokens:
        return vec
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little", signed=False)
        idx = value % dim
        sign = 1.0 if (value >> 63) & 1 == 0 else -1.0
        vec[idx] += sign
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class AffectInferencer:
    """Infer affect signals from text with optional embeddings."""

    def __init__(
        self,
        dim: int = 256,
        use_sentence_transformers: Optional[bool] = None,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        if use_sentence_transformers is None:
            use_sentence_transformers = os.getenv("NEURAL_USE_SENTENCE_TRANSFORMERS", "") == "1"
        self.use_sentence_transformers = bool(use_sentence_transformers and SentenceTransformer)
        self.model = SentenceTransformer(model_name) if self.use_sentence_transformers else None
        self.dim = dim

        self.anchor_pos = self._embed_text("good happy safe calm reward success")
        self.anchor_neg = self._embed_text("bad sad angry threat danger failure")
        self.anchor_arousal = self._embed_text("urgent danger fast alert energy")
        self.anchor_calm = self._embed_text("calm slow safe steady relaxed")

    def _embed_text(self, text: str) -> np.ndarray:
        if self.model is not None:
            vec = np.asarray(self.model.encode(text, normalize_embeddings=True), dtype=np.float32)
            return vec
        return _hash_embed(_tokenize(text), self.dim)

    def infer(self, text: str, recent_texts: Optional[List[str]] = None) -> Dict[str, float]:
        tokens = _tokenize(text)
        embedding = self._embed_text(text)

        pos_hits = sum(1 for t in tokens if t in _POSITIVE)
        neg_hits = sum(1 for t in tokens if t in _NEGATIVE)
        arousal_hits = sum(1 for t in tokens if t in _AROUSAL)
        calm_hits = sum(1 for t in tokens if t in _CALM)
        stress_hits = sum(1 for t in tokens if t in _STRESS)

        lex_valence = 0.0
        if pos_hits or neg_hits:
            lex_valence = (pos_hits - neg_hits) / float(pos_hits + neg_hits)

        valence = 0.6 * (self._similarity(embedding, self.anchor_pos) - self._similarity(embedding, self.anchor_neg))
        valence += 0.4 * lex_valence
        valence = float(np.clip(valence, -1.0, 1.0))

        arousal = 0.5 + 0.5 * (
            self._similarity(embedding, self.anchor_arousal) - self._similarity(embedding, self.anchor_calm)
        )
        if arousal_hits or calm_hits:
            arousal += 0.2 * ((arousal_hits - calm_hits) / float(arousal_hits + calm_hits))
        punct_boost = min(0.3, 0.05 * text.count("!") + 0.03 * text.count("?"))
        arousal = float(np.clip(arousal + punct_boost, 0.0, 1.0))

        stress = 0.0
        if tokens:
            stress = (stress_hits + max(0, neg_hits)) / float(len(tokens))
        stress = float(np.clip(0.5 * stress + 0.3 * max(0.0, -valence), 0.0, 1.0))

        novelty = 0.4
        if recent_texts:
            sims = []
            for prior in recent_texts:
                prior_vec = self._embed_text(prior)
                sims.append(self._similarity(embedding, prior_vec))
            if sims:
                novelty = 1.0 - max(sims)
        novelty = float(np.clip(novelty, 0.0, 1.0))

        return {
            "valence": valence,
            "arousal": arousal,
            "stress": stress,
            "novelty": novelty,
        }

    def _similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        if self.model is not None:
            return float(np.dot(a, b))
        return _cosine(a, b)
