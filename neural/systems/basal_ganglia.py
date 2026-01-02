"""
Basal ganglia-inspired action selection loop.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
    if logits.size == 0:
        return np.array([])
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


class BasalGangliaSystem:
    """Simplified basal ganglia loop for action selection and reward learning."""

    def __init__(
        self,
        num_actions: int,
        motor_output_size: int,
        feature_dim: int = 16,
        learning_rate: float = 0.05,
        decay: float = 0.001,
        temperature: float = 1.0,
        gating_strength: float = 0.2,
        seed: Optional[int] = None,
    ) -> None:
        self.num_actions = max(1, int(num_actions))
        self.motor_output_size = max(1, int(motor_output_size))
        self.feature_dim = max(1, int(feature_dim))
        self.learning_rate = float(learning_rate)
        self.decay = float(decay)
        self.temperature = float(temperature)
        self.gating_strength = float(np.clip(gating_strength, 0.0, 1.0))
        self.rng = np.random.default_rng(seed)

        self.weights = self.rng.normal(0.0, 0.1, size=(self.num_actions, self.feature_dim))
        self.bias = np.zeros(self.num_actions, dtype=float)
        self.projection: Optional[np.ndarray] = None

        self.action_groups = np.array_split(
            np.arange(self.motor_output_size, dtype=int),
            self.num_actions,
        )
        self.last_action: Optional[int] = None
        self.last_features: Optional[np.ndarray] = None
        self.last_logits: Optional[np.ndarray] = None
        self.last_probs: Optional[np.ndarray] = None

    def _ensure_projection(self, input_dim: int) -> None:
        if self.projection is None or self.projection.shape[1] != input_dim:
            scale = 1.0 / max(1.0, np.sqrt(input_dim))
            self.projection = self.rng.normal(0.0, scale, size=(self.feature_dim, input_dim))

    def _features(self, cortex_output: np.ndarray) -> np.ndarray:
        cortex_output = np.asarray(cortex_output, dtype=float)
        self._ensure_projection(cortex_output.size)
        assert self.projection is not None
        features = self.projection @ cortex_output
        return np.tanh(features)

    def _modulated_temperature(self, neuro: Dict[str, float]) -> float:
        dopamine = neuro.get("dopamine", 0.5)
        norepi = neuro.get("norepinephrine", 0.5)
        ach = neuro.get("acetylcholine", 0.5)

        temp = self.temperature
        temp *= 1.0 + 1.5 * (norepi - 0.5)
        temp *= 1.0 - 0.5 * (dopamine - 0.5)
        temp *= 1.0 - 0.3 * (ach - 0.5)
        return float(np.clip(temp, 0.1, 2.0))

    def select_action(
        self,
        cortex_output: np.ndarray,
        neuromodulators: Dict[str, float],
    ) -> Tuple[int, np.ndarray]:
        features = self._features(cortex_output)
        logits = self.weights @ features + self.bias
        dopamine = neuromodulators.get("dopamine", 0.5)
        dopamine_gain = float(np.clip(1.0 + 1.0 * (dopamine - 0.5), 0.2, 1.8))
        logits = logits * dopamine_gain
        temp = self._modulated_temperature(neuromodulators)
        probs = _softmax(logits / temp)
        action = int(self.rng.choice(self.num_actions, p=probs))

        self.last_action = action
        self.last_features = features
        self.last_logits = logits
        self.last_probs = probs
        return action, probs

    def update(self, reward: float, rpe: float, dt: float) -> None:
        if self.last_action is None or self.last_features is None:
            return
        dt_s = max(0.0, dt) / 1000.0
        lr = self.learning_rate * dt_s
        decay = self.decay * dt_s
        target = float(rpe if rpe is not None else reward)

        if decay > 0.0:
            self.weights *= (1.0 - decay)
            self.bias *= (1.0 - decay)

        self.weights[self.last_action] += lr * target * self.last_features
        self.bias[self.last_action] += lr * target

    def apply_gate(
        self,
        motor_output: np.ndarray,
        neuromodulators: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        motor_output = np.asarray(motor_output, dtype=float)
        if self.last_action is None or not self.action_groups:
            return motor_output
        dopamine = 0.5
        if neuromodulators:
            dopamine = neuromodulators.get("dopamine", 0.5)
        gate_strength = self.gating_strength * (1.0 + 0.8 * (dopamine - 0.5))
        gate_strength = float(np.clip(gate_strength, 0.0, 1.0))
        gate = np.full(motor_output.shape, gate_strength, dtype=float)
        indices = self.action_groups[self.last_action]
        gate[indices] = 1.0
        return motor_output * gate

    def get_state(self) -> Dict[str, object]:
        return {
            "last_action": self.last_action,
            "action_probs": None if self.last_probs is None else self.last_probs.tolist(),
            "action_logits": None if self.last_logits is None else self.last_logits.tolist(),
        }
