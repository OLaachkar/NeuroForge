"""
Brain-driven logits processor for llama.cpp.
Directly biases token scores based on brain state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List

import numpy as np


def _tokenize_variants(llm, words: Iterable[str]) -> List[int]:
    token_ids: List[int] = []
    for word in words:
        for prefix in ("", " "):
            ids = llm.tokenize((prefix + word).encode("utf-8"))
            if ids:
                token_ids.append(ids[0])
    return list(set(token_ids))


@dataclass
class TokenSets:
    positive: List[int] = field(default_factory=list)
    calm: List[int] = field(default_factory=list)
    excited: List[int] = field(default_factory=list)
    uncertain: List[int] = field(default_factory=list)
    negation: List[int] = field(default_factory=list)
    exclaim: List[int] = field(default_factory=list)


def build_token_sets(llm) -> TokenSets:
    positive = [
        "yes", "yeah", "good", "great", "nice", "sure", "okay", "thanks",
        "glad", "happy", "love", "cool",
    ]
    calm = [
        "calm", "steady", "slow", "soft", "gentle", "quiet", "relaxed",
        "balanced", "clear",
    ]
    excited = [
        "wow", "amazing", "excited", "fast", "urgent", "quick", "energy",
        "intense", "bold",
    ]
    uncertain = [
        "maybe", "perhaps", "unsure", "guess", "might", "possibly",
    ]
    negation = ["not", "no", "never", "cannot", "can't", "don't"]
    exclaim = ["!"]

    return TokenSets(
        positive=_tokenize_variants(llm, positive),
        calm=_tokenize_variants(llm, calm),
        excited=_tokenize_variants(llm, excited),
        uncertain=_tokenize_variants(llm, uncertain),
        negation=_tokenize_variants(llm, negation),
        exclaim=_tokenize_variants(llm, exclaim),
    )


class BrainLogitsProcessor:
    """Adjusts logits based on a cached brain state."""

    def __init__(self, llm) -> None:
        self.token_sets = build_token_sets(llm)
        self.state: Dict[str, Dict[str, float]] = {}
        self.eos_token = llm.token_eos()

    def set_state(self, state: Dict) -> None:
        self.state = state

    def __call__(self, input_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
        neuro = self.state.get("neurotransmitter_levels", {})
        affect = self.state.get("affect_state", {})
        dopamine = neuro.get("dopamine", 0.5)
        serotonin = neuro.get("serotonin", 0.5)
        norepi = neuro.get("norepinephrine", 0.5)
        ach = neuro.get("acetylcholine", 0.5)
        gaba = neuro.get("gaba", 0.5)
        valence = affect.get("valence", 0.0)
        arousal = affect.get("arousal", 0.0)
        tension = affect.get("tension", 0.0)

        pos_bias = 0.6 * (dopamine - 0.5)
        calm_bias = 0.5 * (serotonin - 0.5)
        excite_bias = 0.6 * (norepi - 0.5)
        uncertain_bias = 0.4 * (0.5 - ach)
        negation_bias = 0.4 * (gaba - 0.5)
        exclaim_bias = 0.3 * (norepi - 0.5)

        pos_bias += 0.25 * valence
        excite_bias += 0.3 * (arousal - 0.3)
        calm_bias += 0.2 * (0.3 - tension)
        uncertain_bias += 0.25 * tension
        negation_bias += 0.2 * max(0.0, -valence)

        for token_id in self.token_sets.positive:
            scores[token_id] += pos_bias
        for token_id in self.token_sets.calm:
            scores[token_id] += calm_bias
        for token_id in self.token_sets.excited:
            scores[token_id] += excite_bias
        for token_id in self.token_sets.uncertain:
            scores[token_id] += uncertain_bias
        for token_id in self.token_sets.negation:
            scores[token_id] += negation_bias
        for token_id in self.token_sets.exclaim:
            scores[token_id] += exclaim_bias

        scores[self.eos_token] -= max(0.0, norepi - 0.5) * 0.2
        return scores
