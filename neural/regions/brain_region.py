"""
Brain Region Model
Models different brain regions (cortex, hippocampus, etc.) with:
- Specific neuron populations
- Regional connectivity patterns
- Specialized functions
"""

import numpy as np
from typing import List, Dict, Optional
from ..core.neuron import BiologicalNeuron, NeuronType
from ..core.synapse import Synapse


class BrainRegion:
    """
    Represents a brain region with specific structure and function.
    """
    
    def __init__(
        self,
        region_name: str,
        num_neurons: int,
        excitatory_ratio: float = 0.8,  # 80% excitatory, 20% inhibitory
        connectivity_density: float = 0.1,  # 10% of possible connections
        region_type: str = "cortex",  # cortex, hippocampus, thalamus, etc.
        neuron_model: str = "lif",
        izhikevich_params: Optional[Dict[str, float]] = None,
        tonic_current: float = 0.0,
        noise_std: float = 0.0,
        enable_homeostasis: bool = False,
        homeostasis_target_hz: float = 5.0,
        homeostasis_tau_ms: float = 1000.0,
        homeostasis_gain: float = 0.001,
        homeostasis_offset_limit: float = 15.0,
    ):
        self.region_name = region_name
        self.region_type = region_type
        self.num_neurons = num_neurons
        self.excitatory_ratio = excitatory_ratio
        self.neuron_model = neuron_model
        self.izhikevich_params = izhikevich_params or {}
        self.tonic_current = tonic_current
        self.noise_std = noise_std
        self.enable_homeostasis = enable_homeostasis
        self.homeostasis_target_hz = homeostasis_target_hz
        self.homeostasis_tau_ms = homeostasis_tau_ms
        self.homeostasis_gain = homeostasis_gain
        self.homeostasis_offset_limit = homeostasis_offset_limit
        
        self.neurons: List[BiologicalNeuron] = []
        self.synapses: List[Synapse] = []
        
        self._create_neurons()
        self._create_connections(connectivity_density)
        
        self.activity_level = 0.0
        self.oscillation_frequency = 0.0  # For rhythmic activity
        self.simulation_time_ms = 0.0
        
    def _create_neurons(self):
        """Create neuron population for this region"""
        num_excitatory = int(self.num_neurons * self.excitatory_ratio)
        
        for i in range(self.num_neurons):
            if i < num_excitatory:
                neuron_type = NeuronType.EXCITATORY
            else:
                neuron_type = NeuronType.INHIBITORY
            
            neuron = BiologicalNeuron(
                neuron_id=len(self.neurons),
                neuron_type=neuron_type,
                neuron_model=self.neuron_model,
                izhikevich_params=self.izhikevich_params,
            )
            self.neurons.append(neuron)
    
    def _create_connections(self, density: float):
        """Create internal connections within region"""
        num_connections = int(self.num_neurons * self.num_neurons * density)
        
        for _ in range(num_connections):
            pre_idx = np.random.randint(0, len(self.neurons))
            post_idx = np.random.randint(0, len(self.neurons))
            
            if pre_idx == post_idx:
                continue  # No self-connections for now
            
            pre_neuron = self.neurons[pre_idx]
            post_neuron = self.neurons[post_idx]
            
            if pre_neuron.neuron_type == NeuronType.EXCITATORY:
                initial_weight = np.random.uniform(0.3, 0.7)
            else:
                initial_weight = np.random.uniform(0.5, 1.0)
            
            synapse = Synapse(pre_neuron, post_neuron, initial_weight=initial_weight)
            self.synapses.append(synapse)
            pre_neuron.connect_to(post_neuron, synapse)
    
    def connect_to_region(self, target_region: 'BrainRegion', 
                         connection_density: float = 0.05,
                         weight_range: tuple = (0.3, 0.7)):
        """Connect this region to another region"""
        num_connections = int(len(self.neurons) * len(target_region.neurons) * connection_density)
        
        for _ in range(num_connections):
            pre_idx = np.random.randint(0, len(self.neurons))
            post_idx = np.random.randint(0, len(target_region.neurons))
            
            pre_neuron = self.neurons[pre_idx]
            post_neuron = target_region.neurons[post_idx]
            
            initial_weight = np.random.uniform(weight_range[0], weight_range[1])
            synapse = Synapse(pre_neuron, post_neuron, initial_weight=initial_weight)
            self.synapses.append(synapse)
            pre_neuron.connect_to(post_neuron, synapse)
    
    def update_neurons(self, dt: float, external_input: Optional[np.ndarray] = None, t_ms: Optional[float] = None):
        """Update neurons only for this region"""
        if t_ms is None:
            self.simulation_time_ms += dt
            t_ms = self.simulation_time_ms
        else:
            self.simulation_time_ms = t_ms

        if external_input is not None:
            for i, neuron in enumerate(self.neurons):
                current = self.tonic_current
                if i < len(external_input):
                    current += external_input[i]
                if self.noise_std > 0.0:
                    current += np.random.normal(0.0, self.noise_std)
                neuron.update(dt, current, t_ms)
                self._update_homeostasis(neuron, dt)
        else:
            for neuron in self.neurons:
                current = self.tonic_current
                if self.noise_std > 0.0:
                    current += np.random.normal(0.0, self.noise_std)
                neuron.update(dt, current, t_ms)
                self._update_homeostasis(neuron, dt)
        
        self.activity_level = np.mean([n.state.membrane_potential for n in self.neurons])

    def update_synapses(self, dt: float, t_ms: Optional[float] = None):
        """Update synapses only for this region"""
        if t_ms is None:
            t_ms = self.simulation_time_ms
        for synapse in self.synapses:
            synapse.update(dt, t_ms)

    def update(self, dt: float, external_input: Optional[np.ndarray] = None,
               t_ms: Optional[float] = None, update_synapses: bool = True):
        """Update all neurons (and optionally synapses) in this region"""
        self.update_neurons(dt, external_input, t_ms)
        if update_synapses:
            self.update_synapses(dt, t_ms)
    
    def get_output(self) -> np.ndarray:
        """Get output from this region (neuron activities)"""
        return np.array([n.get_output() for n in self.neurons])
    
    def get_activity(self) -> float:
        """Get overall activity level"""
        return self.activity_level
    
    def get_spike_rate(self) -> float:
        """Get average spike rate (Hz)"""
        if not self.neurons:
            return 0.0
        return float(np.mean([n.state.firing_rate_hz for n in self.neurons]))

    def _update_homeostasis(self, neuron: BiologicalNeuron, dt: float) -> None:
        """Update firing rate estimates and optional homeostasis."""
        if dt <= 0.0:
            return
        inst_rate = 1000.0 / dt if neuron.state.just_spiked else 0.0
        alpha = 1.0 - np.exp(-dt / self.homeostasis_tau_ms)
        neuron.state.firing_rate_hz += alpha * (inst_rate - neuron.state.firing_rate_hz)

        if self.enable_homeostasis:
            error = neuron.state.firing_rate_hz - self.homeostasis_target_hz
            neuron.homeostatic_offset += self.homeostasis_gain * error * dt
            neuron.homeostatic_offset = float(np.clip(
                neuron.homeostatic_offset,
                -self.homeostasis_offset_limit,
                self.homeostasis_offset_limit,
            ))

