"""
Biological Synapse Model
Models synaptic connections with:
- Short-term plasticity (depression/facilitation)
- Long-term plasticity (LTP/LTD)
- Neurotransmitter dynamics
- Different synapse types
"""

import numpy as np
from dataclasses import dataclass
from .neuron import BiologicalNeuron


@dataclass
class SynapseState:
    """Current state of a synapse"""
    weight: float = 1.0  # Synaptic strength
    neurotransmitter_level: float = 0.0
    vesicle_pool: float = 1.0  # Available neurotransmitter vesicles
    last_release_time: float = -np.inf
    last_pre_spike_time: float = -np.inf
    last_post_spike_time: float = -np.inf


class Synapse:
    """
    Biologically realistic synapse with plasticity mechanisms.
    Implements STDP (Spike-Timing Dependent Plasticity) and short-term plasticity.
    """
    
    def __init__(
        self,
        pre_neuron: BiologicalNeuron,
        post_neuron: BiologicalNeuron,
        initial_weight: float = 0.5,
        max_weight: float = 2.0,
        min_weight: float = 0.0,
        neurotransmitter_type: str = "glutamate",  # glutamate, GABA, dopamine, etc.
        tau_syn: float = 5.0,  # Synaptic time constant (ms)
        tau_ltp: float = 20.0,  # LTP time constant
        tau_ltd: float = 20.0,  # LTD time constant
        a_ltp: float = 0.01,  # LTP learning rate
        a_ltd: float = 0.01,  # LTD learning rate
        use_probability: float = 0.3,  # Release probability
        tau_recovery: float = 800.0,  # Vesicle recovery time
        facilitation_factor: float = 1.0,
    ):
        self.pre_neuron = pre_neuron
        self.post_neuron = post_neuron
        self.initial_weight = initial_weight
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.neurotransmitter_type = neurotransmitter_type
        self.tau_syn = tau_syn
        
        self.tau_ltp = tau_ltp
        self.tau_ltd = tau_ltd
        self.a_ltp_base = a_ltp
        self.a_ltd_base = a_ltd
        self.a_ltp_eff = a_ltp
        self.a_ltd_eff = a_ltd
        
        self.base_use_probability = use_probability
        self.use_probability = use_probability
        self.tau_recovery = tau_recovery
        self.facilitation_factor = facilitation_factor
        
        self.state = SynapseState()
        self.state.weight = initial_weight
        
    def update(self, dt: float, t_ms: float):
        """Update synapse state"""
        self.state.neurotransmitter_level *= np.exp(-dt / self.tau_syn)
        
        recovery_rate = (1.0 - self.state.vesicle_pool) / self.tau_recovery
        self.state.vesicle_pool = min(1.0, self.state.vesicle_pool + recovery_rate * dt)
        
        if self.pre_neuron.state.just_spiked:
            self._handle_pre_spike(t_ms)
        
        if self.post_neuron.state.just_spiked:
            self._handle_post_spike(t_ms)

        self.use_probability += (
            self.base_use_probability - self.use_probability
        ) * min(1.0, dt / self.tau_recovery)
    
    def _handle_pre_spike(self, t_ms: float):
        """Handle pre-synaptic spike"""
        if np.random.random() < self.use_probability * self.state.vesicle_pool:
            release_amount = self.state.vesicle_pool * self.facilitation_factor
            self.state.neurotransmitter_level += release_amount * self.state.weight
            self.state.vesicle_pool *= (1.0 - release_amount)
            self.state.last_release_time = t_ms
        
        self.use_probability = min(1.0, self.use_probability * 1.1)

        if np.isfinite(self.state.last_post_spike_time):
            delta_t = t_ms - self.state.last_post_spike_time
            if delta_t > 0.0:
                weight_change = -self.a_ltd_eff * np.exp(-delta_t / self.tau_ltd)
                self._apply_weight_change(weight_change)

        self.state.last_pre_spike_time = t_ms
    
    def _handle_post_spike(self, t_ms: float):
        """Handle post-synaptic spike"""
        if np.isfinite(self.state.last_pre_spike_time):
            delta_t = t_ms - self.state.last_pre_spike_time
            if delta_t > 0.0:
                weight_change = self.a_ltp_eff * np.exp(-delta_t / self.tau_ltp)
                self._apply_weight_change(weight_change)

        self.state.last_post_spike_time = t_ms
    
    def get_current(self) -> float:
        """Get current synaptic current"""
        return self.state.neurotransmitter_level
    
    def is_active(self) -> bool:
        """Check if synapse is currently active"""
        return self.state.neurotransmitter_level > 0.01
    
    def set_weight(self, weight: float):
        """Manually set synaptic weight"""
        self.state.weight = np.clip(weight, self.min_weight, self.max_weight)
    
    def get_weight(self) -> float:
        """Get current synaptic weight"""
        return self.state.weight

    def _apply_weight_change(self, delta: float) -> None:
        """Apply and clamp a weight update."""
        self.state.weight = np.clip(self.state.weight + delta, self.min_weight, self.max_weight)

