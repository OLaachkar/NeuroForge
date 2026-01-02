"""
Biological Neuron Model
Models a single neuron with realistic biological properties including:
- Membrane potential dynamics
- Action potential generation
- Refractory periods
- Different neuron types (excitatory, inhibitory)
"""

import numpy as np
from typing import List, Optional, TYPE_CHECKING, Dict
from dataclasses import dataclass
from enum import Enum

if TYPE_CHECKING:
    from .synapse import Synapse


class NeuronType(Enum):
    EXCITATORY = "excitatory"
    INHIBITORY = "inhibitory"
    MODULATORY = "modulatory"


@dataclass
class NeuronState:
    """Current state of a neuron"""
    membrane_potential: float = -70.0  # mV (resting potential)
    spike_count: int = 0
    last_spike_time: float = -np.inf
    is_refractory: bool = False
    refractory_remaining_ms: float = 0.0
    just_spiked: bool = False
    firing_rate_hz: float = 0.0
    calcium_level: float = 0.0  # For plasticity
    neurotransmitter_release: float = 0.0
    recovery_variable: float = 0.0  # Izhikevich recovery variable


class BiologicalNeuron:
    """
    A biologically realistic neuron model using integrate-and-fire dynamics
    with additional biological properties.
    """
    
    def __init__(
        self,
        neuron_id: int,
        neuron_type: NeuronType = NeuronType.EXCITATORY,
        neuron_model: str = "lif",
        izhikevich_params: Optional[Dict[str, float]] = None,
        resting_potential: float = -70.0,
        threshold: float = -55.0,
        reset_potential: float = -70.0,
        membrane_resistance: float = 100.0,  # MΩ
        membrane_capacitance: float = 0.1,  # nF
        refractory_period: float = 2.0,  # ms
        tau_m: float = 20.0,  # Membrane time constant (ms)
    ):
        self.neuron_id = neuron_id
        self.neuron_type = neuron_type
        self.neuron_model = neuron_model.lower()
        self.resting_potential = resting_potential
        self.threshold = threshold
        self.base_threshold = threshold
        self.reset_potential = reset_potential
        self.membrane_resistance = membrane_resistance
        self.membrane_capacitance = membrane_capacitance
        self.refractory_period = refractory_period
        self.tau_m = tau_m
        
        self.state = NeuronState()
        self.state.membrane_potential = resting_potential

        self.homeostatic_offset = 0.0

        self._internal_time_ms = 0.0
        
        self.synaptic_inputs: List = []
        
        self.output_synapses: List = []

        self.izh_a = 0.02
        self.izh_b = 0.2
        self.izh_c = -65.0
        self.izh_d = 8.0
        self.izh_v_peak = 30.0
        self.izh_input_scale = membrane_resistance

        if self.neuron_model == "izhikevich":
            self._init_izhikevich(izhikevich_params or {})
        
    def update(self, dt: float, external_current: float = 0.0, t_ms: Optional[float] = None) -> bool:
        """
        Update neuron state for one time step.
        Returns True if neuron fired an action potential.
        """
        if t_ms is None:
            self._internal_time_ms += dt
            t_ms = self._internal_time_ms

        if self.neuron_model == "izhikevich":
            return self._update_izhikevich(dt, external_current, t_ms)

        self.state.just_spiked = False
        self.state.refractory_remaining_ms = max(0.0, self.state.refractory_remaining_ms - dt)
        self.state.is_refractory = self.state.refractory_remaining_ms > 0.0

        if self.state.is_refractory:
            self.state.membrane_potential += (
                self.reset_potential - self.state.membrane_potential
            ) * min(1.0, dt / self.tau_m)
            self._decay_intracellular(dt)
            return False
        
        total_input = external_current
        for synapse in self.synaptic_inputs:
            if synapse.is_active():
                if synapse.pre_neuron.neuron_type == NeuronType.INHIBITORY:
                    total_input -= synapse.get_current()
                else:
                    total_input += synapse.get_current()
        
        dV = ((self.resting_potential - self.state.membrane_potential) + 
              self.membrane_resistance * total_input) / self.tau_m
        
        self.state.membrane_potential += dV * dt
        
        fired = False
        if self.state.membrane_potential >= self.threshold:
            fired = True
            self.state.just_spiked = True
            self.state.membrane_potential = self.reset_potential
            self.state.spike_count += 1
            self.state.last_spike_time = t_ms
            self.state.refractory_remaining_ms = self.refractory_period
            self.state.is_refractory = True
            self.state.calcium_level += 1.0  # Calcium influx for plasticity
            
            self.state.neurotransmitter_release = 1.0
        
        self._decay_intracellular(dt)
        
        return fired

    def _init_izhikevich(self, params: Dict[str, float]) -> None:
        """Initialize Izhikevich parameters."""
        if self.neuron_type == NeuronType.INHIBITORY:
            defaults = {"a": 0.1, "b": 0.2, "c": -65.0, "d": 2.0}
        else:
            defaults = {"a": 0.02, "b": 0.2, "c": -65.0, "d": 8.0}

        self.izh_a = float(params.get("a", defaults["a"]))
        self.izh_b = float(params.get("b", defaults["b"]))
        self.izh_c = float(params.get("c", defaults["c"]))
        self.izh_d = float(params.get("d", defaults["d"]))
        self.izh_v_peak = float(params.get("v_peak", 30.0))
        self.izh_input_scale = float(params.get("input_scale", self.membrane_resistance))

        v_init = float(params.get("v_init", self.resting_potential))
        u_init = float(params.get("u_init", self.izh_b * v_init))
        self.state.membrane_potential = v_init
        self.state.recovery_variable = u_init

    def _update_izhikevich(self, dt: float, external_current: float, t_ms: float) -> bool:
        """Update neuron using Izhikevich dynamics."""
        self.state.just_spiked = False
        self.state.is_refractory = False
        self.state.refractory_remaining_ms = 0.0

        total_input = external_current
        for synapse in self.synaptic_inputs:
            if synapse.is_active():
                if synapse.pre_neuron.neuron_type == NeuronType.INHIBITORY:
                    total_input -= synapse.get_current()
                else:
                    total_input += synapse.get_current()

        v = self.state.membrane_potential
        u = self.state.recovery_variable
        current = total_input * self.izh_input_scale

        dv = (0.04 * v * v + 5.0 * v + 140.0 - u + current) * dt
        du = (self.izh_a * (self.izh_b * v - u)) * dt
        v += dv
        u += du

        fired = False
        if v >= self.izh_v_peak:
            fired = True
            self.state.just_spiked = True
            self.state.spike_count += 1
            self.state.last_spike_time = t_ms
            v = self.izh_c
            u += self.izh_d
            self.state.calcium_level += 1.0
            self.state.neurotransmitter_release = 1.0

        self.state.membrane_potential = v
        self.state.recovery_variable = u
        self._decay_intracellular(dt)
        return fired
    
    def reset_refractory(self):
        """Reset refractory period (called by brain after refractory time)"""
        self.state.is_refractory = False
        self.state.refractory_remaining_ms = 0.0
    
    def get_output(self) -> float:
        """Get current output (spike or membrane potential)"""
        return 1.0 if self.state.just_spiked else 0.0

    def get_voltage(self) -> float:
        """Get current membrane potential"""
        return self.state.membrane_potential
    
    def add_synapse(self, synapse):
        """Add an input synapse"""
        self.synaptic_inputs.append(synapse)
    
    def connect_to(self, post_neuron: 'BiologicalNeuron', synapse):
        """Connect this neuron to another via a synapse"""
        self.output_synapses.append(synapse)
        post_neuron.add_synapse(synapse)

    def _decay_intracellular(self, dt: float) -> None:
        """Decay intracellular signals each step."""
        self.state.calcium_level *= np.exp(-dt / 50.0)
        self.state.neurotransmitter_release *= np.exp(-dt / 5.0)

