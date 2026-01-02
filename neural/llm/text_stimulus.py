"""
Text-to-stimulus mapping for brain input.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional

import numpy as np

from .affect_inference import AffectInferencer


def text_to_stimulus(text: str, size: int, strength: float = 0.3) -> np.ndarray:
    """Deterministically map text to a sensory input vector."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, strength, size).astype(float)


_DEFAULT_INFERENCER = AffectInferencer()


def text_to_affect(text: str, recent_texts: Optional[List[str]] = None) -> Dict[str, float]:
    """Estimate affect signals from text (embedding-backed when available)."""
    return _DEFAULT_INFERENCER.infer(text, recent_texts)
