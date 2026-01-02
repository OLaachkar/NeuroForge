"""
Neurotransmitter System
Models global neurotransmitter systems:
- Dopamine (reward, motivation)
- Serotonin (mood, regulation)
- Acetylcholine (attention, learning)
- GABA (inhibition)
- Glutamate (excitation)
"""

import numpy as np
from typing import Dict, List, Any, Optional
from ..regions.brain_region import BrainRegion


class NeurotransmitterSystem:
    """
    Models global neuromodulatory systems that affect brain-wide activity.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}

        def _merge(base: Dict[str, float], overrides: Optional[Dict[str, float]]) -> Dict[str, float]:
            merged = dict(base)
            if overrides:
                merged.update(overrides)
            return merged

        base_levels = {
            'dopamine': 0.5,      # Reward, motivation, learning
            'serotonin': 0.5,     # Mood, regulation
            'acetylcholine': 0.5,  # Attention, learning
            'gaba': 0.5,          # Global inhibition
            'glutamate': 0.5,      # Global excitation
            'norepinephrine': 0.5, # Arousal, attention
        }
        baseline_levels = _merge(base_levels, config.get("baseline_levels"))
        initial_levels = _merge(baseline_levels, config.get("initial_levels"))
        self.levels = initial_levels
        
        self.target_regions: Dict[str, List[BrainRegion]] = {}
        
        self.basal_levels = baseline_levels

        base_affect = {
            'valence': 0.0,
            'arousal': 0.2,
            'tension': 0.2,
        }
        affect_baseline = _merge(base_affect, config.get("affect_baseline"))
        self.affect_state = _merge(affect_baseline, config.get("affect_initial"))
        self.affect_baseline = affect_baseline
        self.affect_decay_rates = _merge(
            {
                'valence': 0.002,
                'arousal': 0.004,
                'tension': 0.004,
            },
            config.get("affect_decay_rates"),
        )

        self.reward_expectation = float(config.get("reward_expectation", 0.0))
        self.reward_tau_ms = float(config.get("reward_tau_ms", 2000.0))
        self.rpe_gain = float(config.get("rpe_gain", 1.0))
        self.last_rpe = 0.0

        self.experience_gains = _merge(
            {
                'valence': 1.0,
                'arousal': 1.0,
                'stress': 1.0,
                'novelty': 1.0,
                'reward': 1.0,
            },
            config.get("experience_gains"),
        )
        self.affect_gains = _merge(
            {
                'valence': 1.2,
                'arousal': 1.5,
                'tension': 1.2,
            },
            config.get("affect_gains"),
        )
        self.neuro_gains = _merge(
            {
                'dopamine': 1.0,
                'serotonin': 1.0,
                'norepinephrine': 1.0,
                'acetylcholine': 1.0,
                'gaba': 1.0,
                'glutamate': 1.0,
            },
            config.get("neuro_gains"),
        )
        
        self.decay_rates = _merge(
            {
                'dopamine': 0.01,
                'serotonin': 0.005,
                'acetylcholine': 0.02,
                'gaba': 0.01,
                'glutamate': 0.01,
                'norepinephrine': 0.015,
            },
            config.get("decay_rates"),
        )
    
    def set_level(self, neurotransmitter: str, level: float):
        """Set neurotransmitter level"""
        if neurotransmitter in self.levels:
            self.levels[neurotransmitter] = np.clip(level, 0.0, 1.0)
    
    def release(self, neurotransmitter: str, amount: float):
        """Release neurotransmitter"""
        if neurotransmitter in self.levels:
            self.levels[neurotransmitter] = np.clip(
                self.levels[neurotransmitter] + amount, 0.0, 1.0
            )
    
    def update(self, dt: float, experience: Dict[str, float] | None = None):
        """Update neurotransmitter levels (decay toward basal + experience)."""
        for key in self.affect_state:
            diff = self.affect_baseline[key] - self.affect_state[key]
            decay = self.affect_decay_rates[key] * dt
            self.affect_state[key] += diff * decay

        for nt in self.levels:
            diff = self.basal_levels[nt] - self.levels[nt]
            decay = self.decay_rates[nt] * dt
            self.levels[nt] += diff * decay

        if experience:
            self.apply_experience(experience, dt)

        self._clamp_states()

    def apply_experience(self, experience: Dict[str, float], dt: float) -> None:
        """Apply experience signals to affect and neurotransmitters."""
        dt_s = max(0.0, dt) / 1000.0
        valence = float(experience.get("valence", 0.0)) * self.experience_gains["valence"]
        arousal = float(experience.get("arousal", 0.0)) * self.experience_gains["arousal"]
        stress = float(experience.get("stress", 0.0)) * self.experience_gains["stress"]
        novelty = float(experience.get("novelty", 0.0)) * self.experience_gains["novelty"]
        reward = float(experience.get("reward", 0.0)) * self.experience_gains["reward"]

        valence = float(np.clip(valence, -1.0, 1.0))
        arousal = float(np.clip(arousal, 0.0, 1.0))
        stress = float(np.clip(stress, 0.0, 1.0))
        novelty = float(np.clip(novelty, 0.0, 1.0))
        reward = float(np.clip(reward, -1.0, 1.0))

        rpe = reward - self.reward_expectation
        self.last_rpe = float(rpe)
        if self.reward_tau_ms > 0:
            self.reward_expectation += (reward - self.reward_expectation) * (dt / self.reward_tau_ms)

        self.affect_state["valence"] += (
            (valence + 0.5 * rpe - 0.4 * stress) * self.affect_gains["valence"] * dt_s
        )
        self.affect_state["arousal"] += (
            (arousal + 0.6 * novelty + 0.6 * stress) * self.affect_gains["arousal"] * dt_s
        )
        self.affect_state["tension"] += (
            (stress + max(0.0, -valence)) * self.affect_gains["tension"] * dt_s
        )

        self._shift_level(
            "dopamine",
            self.neuro_gains["dopamine"] * (self.rpe_gain * rpe + 0.2 * valence) * dt_s,
        )
        self._shift_level(
            "serotonin",
            self.neuro_gains["serotonin"] * (0.3 * valence - 0.5 * stress) * dt_s,
        )
        self._shift_level(
            "norepinephrine",
            self.neuro_gains["norepinephrine"] * (0.4 * arousal + 0.4 * novelty + 0.6 * stress) * dt_s,
        )
        self._shift_level(
            "acetylcholine",
            self.neuro_gains["acetylcholine"] * (0.4 * novelty + 0.2 * arousal) * dt_s,
        )
        self._shift_level(
            "gaba",
            self.neuro_gains["gaba"] * (0.3 * self.affect_state["tension"]) * dt_s,
        )
        self._shift_level(
            "glutamate",
            self.neuro_gains["glutamate"] * (0.3 * self.affect_state["arousal"]) * dt_s,
        )
    
    def modulate_region(self, region: BrainRegion):
        """Apply neuromodulatory effects to a brain region"""
        dopamine_factor = 1.0 + (self.levels['dopamine'] - 0.5) * 0.5
        
        ach_factor = 1.0 + (self.levels['acetylcholine'] - 0.5) * 0.3
        
        serotonin_factor = 1.0 + (self.levels['serotonin'] - 0.5) * 0.2
        
        gaba_factor = 1.0 - (self.levels['gaba'] - 0.5) * 0.4
        
        glutamate_factor = 1.0 + (self.levels['glutamate'] - 0.5) * 0.4
        
        for neuron in region.neurons:
            modulation_factor = (
                dopamine_factor * ach_factor * serotonin_factor *
                gaba_factor * glutamate_factor
            )
            modulation_factor = max(0.1, modulation_factor)
            neuron.threshold = neuron.base_threshold * modulation_factor + neuron.homeostatic_offset
            
            for synapse in neuron.output_synapses:
                synapse.a_ltp_eff = synapse.a_ltp_base * dopamine_factor * ach_factor
                synapse.a_ltd_eff = synapse.a_ltd_base * serotonin_factor * (1.0 / max(0.1, dopamine_factor))
    
    def reward_signal(self, reward_value: float, dt: float = 1.0):
        """Send reward signal (dopamine release)."""
        self.apply_experience({"reward": reward_value}, dt)
    
    def stress_signal(self, stress_level: float, dt: float = 1.0):
        """Send stress signal (norepinephrine, cortisol-like)."""
        self.apply_experience({"stress": stress_level}, dt)

    def _shift_level(self, neurotransmitter: str, delta: float) -> None:
        if neurotransmitter in self.levels:
            self.levels[neurotransmitter] = np.clip(
                self.levels[neurotransmitter] + delta, 0.0, 1.0
            )

    def _clamp_states(self) -> None:
        self.affect_state["valence"] = float(np.clip(self.affect_state["valence"], -1.0, 1.0))
        self.affect_state["arousal"] = float(np.clip(self.affect_state["arousal"], 0.0, 1.0))
        self.affect_state["tension"] = float(np.clip(self.affect_state["tension"], 0.0, 1.0))
    
    def get_state(self) -> Dict[str, float]:
        """Get current neurotransmitter levels."""
        return self.levels.copy()

    def get_affect_state(self) -> Dict[str, float]:
        """Get current affect state."""
        return self.affect_state.copy()

