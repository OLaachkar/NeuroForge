"""
Self-model subsystem tracking internal state for phenomenology reports.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


class SelfModel:
    """Tracks internal state variables for introspective reporting."""

    def __init__(self, reward_history_tau_ms: float = 5000.0) -> None:
        self.energy = 0.0
        self.prediction_error = 0.0
        self.reward_history = 0.0
        self.attention_focus = 0.0
        self.affect = {"valence": 0.0, "arousal": 0.2, "tension": 0.2}
        self.reward_history_tau_ms = reward_history_tau_ms

    def update(self, brain, dt: float, experience: Optional[Dict[str, float]] = None) -> None:
        rates = [region.get_spike_rate() for region in brain.regions.values()]
        mean_rate = float(np.mean(rates)) if rates else 0.0
        self.energy = float(1.0 - np.exp(-mean_rate / 10.0))

        reward = float((experience or {}).get("reward", 0.0))
        if self.reward_history_tau_ms > 0.0:
            alpha = 1.0 - np.exp(-dt / self.reward_history_tau_ms)
            self.reward_history += alpha * (reward - self.reward_history)

        self.prediction_error = float(getattr(brain.neurotransmitters, "last_rpe", 0.0))

        neuro = brain.neurotransmitters.get_state()
        ach = neuro.get("acetylcholine", 0.5)
        norepi = neuro.get("norepinephrine", 0.5)
        self.attention_focus = float(np.clip(0.5 * ach + 0.5 * norepi, 0.0, 1.0))

        self.affect = brain.neurotransmitters.get_affect_state()

    def get_state(self) -> Dict[str, float]:
        return {
            "energy": float(self.energy),
            "prediction_error": float(self.prediction_error),
            "reward_history": float(self.reward_history),
            "attention_focus": float(self.attention_focus),
            "valence": float(self.affect.get("valence", 0.0)),
            "arousal": float(self.affect.get("arousal", 0.0)),
            "tension": float(self.affect.get("tension", 0.0)),
        }

    def generate_report(self) -> str:
        valence = float(self.affect.get("valence", 0.0))
        arousal = float(self.affect.get("arousal", 0.0))
        tension = float(self.affect.get("tension", 0.0))

        mood = "neutral"
        if valence > 0.3:
            mood = "positive"
        elif valence < -0.3:
            mood = "negative"

        activation = "low"
        if arousal > 0.7:
            activation = "high"
        elif arousal > 0.4:
            activation = "moderate"

        strain = "calm"
        if tension > 0.7:
            strain = "strained"
        elif tension > 0.4:
            strain = "tense"

        return (
            "Internally generated state report (simulation): "
            f"mood={mood}, activation={activation}, strain={strain}, "
            f"energy={self.energy:.2f}, attention={self.attention_focus:.2f}, "
            f"prediction_error={self.prediction_error:.2f}."
        )
