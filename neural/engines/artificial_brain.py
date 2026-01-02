"""
Artificial Biological Brain
Main orchestrator that combines all systems to create a functioning brain.
"""

import numpy as np
import time
from collections import deque
from typing import Any, Dict, List, Optional, Callable
from ..regions.brain_region import BrainRegion
from ..systems.memory_system import MemorySystem
from ..systems.neurotransmitter_system import NeurotransmitterSystem
from ..systems.self_model import SelfModel
from ..systems.basal_ganglia import BasalGangliaSystem


class ArtificialBrain:
    """
    A complete artificial biological brain that attempts to replicate
    human brain behavior through biologically-inspired mechanisms.
    """
    
    def __init__(
        self,
        num_cortical_neurons: int = 1000,
        num_hippocampal_neurons: int = 500,
        num_thalamic_neurons: int = 300,
        simulation_dt: float = 0.1,  # ms
        seed: Optional[int] = None,
        tonic_current: float = 0.2,
        noise_std: float = 0.02,
        enable_homeostasis: bool = True,
        homeostasis_target_hz: float = 5.0,
        homeostasis_tau_ms: float = 1000.0,
        homeostasis_gain: float = 0.001,
        homeostasis_offset_limit: float = 15.0,
        region_params: Optional[Dict[str, Dict[str, Any]]] = None,
        neuromodulation_config: Optional[Dict[str, Any]] = None,
        basal_ganglia_config: Optional[Dict[str, Any]] = None,
        memory_config: Optional[Dict[str, Any]] = None,
    ):
        self.simulation_dt = simulation_dt
        self.simulation_time = 0.0
        self.seed = seed
        self.region_params = region_params or {}

        if seed is not None:
            np.random.seed(seed)
        
        print("Creating artificial brain...")
        
        print("  Creating brain regions...")
        self.regions: Dict[str, BrainRegion] = {}
        
        cortex_params = self._merge_region_params(
            'cortex',
            {
                'region_name': 'cortex',
                'num_neurons': num_cortical_neurons,
                'excitatory_ratio': 0.8,
                'connectivity_density': 0.1,
                'region_type': 'cortex',
                'tonic_current': tonic_current,
                'noise_std': noise_std,
                'enable_homeostasis': enable_homeostasis,
                'homeostasis_target_hz': homeostasis_target_hz,
                'homeostasis_tau_ms': homeostasis_tau_ms,
                'homeostasis_gain': homeostasis_gain,
                'homeostasis_offset_limit': homeostasis_offset_limit,
            },
        )
        use_column = bool(cortex_params.pop("use_column", False))
        if use_column:
            from ..regions.cortical_column import CorticalColumn

            self.regions["cortex"] = CorticalColumn(**cortex_params)
        else:
            cortex_params.pop("layer_specs", None)
            cortex_params.pop("layer_connectivity", None)
            cortex_params.pop("layer_connectivity_scale", None)
            self.regions["cortex"] = BrainRegion(**cortex_params)
        
        self.regions['hippocampus'] = BrainRegion(**self._merge_region_params(
            'hippocampus',
            {
                'region_name': 'hippocampus',
                'num_neurons': num_hippocampal_neurons,
                'excitatory_ratio': 0.9,
                'connectivity_density': 0.15,
                'region_type': 'hippocampus',
                'tonic_current': tonic_current,
                'noise_std': noise_std,
                'enable_homeostasis': enable_homeostasis,
                'homeostasis_target_hz': homeostasis_target_hz,
                'homeostasis_tau_ms': homeostasis_tau_ms,
                'homeostasis_gain': homeostasis_gain,
                'homeostasis_offset_limit': homeostasis_offset_limit,
            },
        ))
        
        self.regions['thalamus'] = BrainRegion(**self._merge_region_params(
            'thalamus',
            {
                'region_name': 'thalamus',
                'num_neurons': num_thalamic_neurons,
                'excitatory_ratio': 0.7,
                'connectivity_density': 0.2,
                'region_type': 'thalamus',
                'tonic_current': tonic_current,
                'noise_std': noise_std,
                'enable_homeostasis': enable_homeostasis,
                'homeostasis_target_hz': homeostasis_target_hz,
                'homeostasis_tau_ms': homeostasis_tau_ms,
                'homeostasis_gain': homeostasis_gain,
                'homeostasis_offset_limit': homeostasis_offset_limit,
            },
        ))
        
        print("  Connecting brain regions...")
        self.regions['thalamus'].connect_to_region(
            self.regions['cortex'],
            connection_density=0.1
        )
        self.regions['cortex'].connect_to_region(
            self.regions['hippocampus'],
            connection_density=0.08
        )
        self.regions['hippocampus'].connect_to_region(
            self.regions['cortex'],
            connection_density=0.08
        )
        
        print("  Initializing memory system...")
        self.memory = MemorySystem(
            self.regions['hippocampus'],
            self.regions['cortex'],
            config=memory_config,
            seed=seed,
        )
        
        print("  Initializing neurotransmitter systems...")
        self.neurotransmitters = NeurotransmitterSystem(neuromodulation_config)
        self.self_model = SelfModel()
        
        self.sensory_inputs: Dict[str, np.ndarray] = {}
        
        self.motor_outputs: np.ndarray = np.zeros(100)  # 100 motor neurons
        
        motor_cortex_size = min(100, num_cortical_neurons)
        self.motor_neurons = self.regions['cortex'].neurons[:motor_cortex_size]

        bg_cfg = basal_ganglia_config or {}
        bg_enabled = bool(bg_cfg.get("enabled", True))
        if bg_enabled:
            num_actions = int(bg_cfg.get("num_actions", min(8, motor_cortex_size)))
            num_actions = max(1, min(num_actions, motor_cortex_size))
            self.basal_ganglia = BasalGangliaSystem(
                num_actions=num_actions,
                motor_output_size=len(self.motor_neurons),
                feature_dim=int(bg_cfg.get("feature_dim", 16)),
                learning_rate=float(bg_cfg.get("learning_rate", 0.05)),
                decay=float(bg_cfg.get("decay", 0.001)),
                temperature=float(bg_cfg.get("temperature", 1.0)),
                gating_strength=float(bg_cfg.get("gating_strength", 0.2)),
                seed=seed,
            )
        else:
            self.basal_ganglia = None
        
        self.stats = {
            'total_spikes': 0,
            'simulation_time': 0.0,
            'region_activities': {},
            'region_spikes': {},
            'rpe_history': deque(maxlen=200),
            'reward_history': deque(maxlen=200),
            'action_history': deque(maxlen=200),
        }
        
        print("Artificial brain created successfully!")
        print(f"  Total neurons: {self.get_total_neurons()}")
        print(f"  Total synapses: {self.get_total_synapses()}")
    
    def get_total_neurons(self) -> int:
        """Get total number of neurons"""
        return sum(len(region.neurons) for region in self.regions.values())
    
    def get_total_synapses(self) -> int:
        """Get total number of synapses"""
        return sum(len(region.synapses) for region in self.regions.values())

    def _merge_region_params(self, name: str, defaults: Dict[str, Any]) -> Dict[str, Any]:
        """Merge per-region overrides while filtering unknown keys."""
        overrides = self.region_params.get(name, {})
        merged = defaults.copy()
        merged.update(overrides)
        allowed = {
            'region_name',
            'num_neurons',
            'excitatory_ratio',
            'connectivity_density',
            'region_type',
            'neuron_model',
            'izhikevich_params',
            'tonic_current',
            'noise_std',
            'enable_homeostasis',
            'homeostasis_target_hz',
            'homeostasis_tau_ms',
            'homeostasis_gain',
            'homeostasis_offset_limit',
            'use_column',
            'layer_specs',
            'layer_connectivity',
            'layer_connectivity_scale',
        }
        return {k: v for k, v in merged.items() if k in allowed}

    def iter_synapses(self):
        """Iterate over all synapses exactly once."""
        for region in self.regions.values():
            for synapse in region.synapses:
                yield synapse
    
    def add_sensory_input(self, modality: str, input_data: np.ndarray):
        """Add sensory input (vision, hearing, touch, etc.)"""
        self.sensory_inputs[modality] = input_data
    
    def process_sensory_input(self) -> np.ndarray:
        """Process all sensory inputs and route to thalamus"""
        if not self.sensory_inputs:
            return np.zeros(len(self.regions['thalamus'].neurons))
        
        total_input_size = len(self.regions['thalamus'].neurons)
        combined_input = np.zeros(total_input_size)
        
        input_idx = 0
        for modality, data in self.sensory_inputs.items():
            for value in data.flatten():
                if input_idx < total_input_size:
                    combined_input[input_idx] = value * 0.1
                    input_idx += 1
        
        return combined_input
    
    def get_motor_output(self) -> np.ndarray:
        """Get motor output from motor neurons"""
        motor_output = np.zeros(len(self.motor_neurons))
        for i, neuron in enumerate(self.motor_neurons):
            motor_output[i] = neuron.get_output()
        return motor_output
    
    def step(self, external_reward: float = 0.0, experience: Optional[Dict[str, float]] = None):
        """
        Run one simulation step.
        This is the core brain update loop.
        """
        dt = self.simulation_dt
        self.simulation_time += dt
        
        sensory_input = self.process_sensory_input()
        
        experience_input = dict(experience or {})
        if external_reward != 0.0:
            experience_input["reward"] = experience_input.get("reward", 0.0) + external_reward
        self.neurotransmitters.update(dt, experience=experience_input if experience_input else None)
        self.stats['rpe_history'].append(self.neurotransmitters.last_rpe)
        self.stats['reward_history'].append(float(experience_input.get("reward", 0.0)))
        
        for region_name, region in self.regions.items():
            self.neurotransmitters.modulate_region(region)
            
            if region_name == 'thalamus':
                region_input = sensory_input
            else:
                region_input = None
            
            region.update_neurons(dt, region_input, t_ms=self.simulation_time)

            step_spikes = sum(1 for n in region.neurons if n.state.just_spiked)
            self.stats['total_spikes'] += step_spikes
            self.stats['region_spikes'][region_name] = (
                self.stats['region_spikes'].get(region_name, 0) + step_spikes
            )
            self.stats['region_activities'][region_name] = region.get_activity()

        for region in self.regions.values():
            region.update_synapses(dt, t_ms=self.simulation_time)
        
        self.memory.update(dt)

        self.self_model.update(self, dt, experience_input if experience_input else None)
        
        motor_output = self.get_motor_output()
        if self.basal_ganglia is not None:
            reward_value = experience_input.get("reward", 0.0) if experience_input else 0.0
            self.basal_ganglia.update(reward_value, self.neurotransmitters.last_rpe, dt)
            cortex_output = self.regions['cortex'].get_output()
            self.basal_ganglia.select_action(cortex_output, self.neurotransmitters.get_state())
            motor_output = self.basal_ganglia.apply_gate(motor_output, self.neurotransmitters.get_state())
            self.stats['action_history'].append(self.basal_ganglia.last_action)
        self.motor_outputs = motor_output
        
        self.stats['simulation_time'] = self.simulation_time
    
    def learn_pattern(self, pattern: np.ndarray) -> int:
        """Learn and store a pattern"""
        return self.memory.encode(pattern, self.regions['hippocampus'])
    
    def recall_pattern(self, partial_pattern: np.ndarray) -> Optional[np.ndarray]:
        """Recall a pattern from partial cue"""
        return self.memory.recall(partial_pattern, self.regions['cortex'])
    
    def run(
        self,
        duration_ms: float,
        callback: Optional[Callable] = None,
        experience: Optional[Dict[str, float]] = None,
    ):
        """
        Run simulation for specified duration.
        callback is called each step with (brain, dt) arguments.
        """
        num_steps = int(duration_ms / self.simulation_dt)
        
        for step in range(num_steps):
            self.step(experience=experience)
            
            if callback:
                callback(self, self.simulation_dt)
    
    def get_brain_state(self) -> Dict:
        """Get comprehensive brain state"""
        return {
            'simulation_time': self.simulation_time,
            'region_activities': {
                name: region.get_activity() 
                for name, region in self.regions.items()
            },
            'spike_rates': {
                name: region.get_spike_rate() 
                for name, region in self.regions.items()
            },
            'neurotransmitter_levels': self.neurotransmitters.get_state(),
            'affect_state': self.neurotransmitters.get_affect_state(),
            'reward_expectation': self.neurotransmitters.reward_expectation,
            'last_rpe': self.neurotransmitters.last_rpe,
            'rpe_history': list(self.stats['rpe_history']),
            'reward_history': list(self.stats['reward_history']),
            'action_history': list(self.stats['action_history']),
            'self_model_state': self.self_model.get_state(),
            'memory_stats': self.memory.get_memory_stats(),
            'basal_ganglia_state': self.basal_ganglia.get_state() if self.basal_ganglia else None,
            'total_neurons': self.get_total_neurons(),
            'total_synapses': self.get_total_synapses(),
            'total_spikes': self.stats['total_spikes'],
        }

    def get_phenomenology_report(self) -> str:
        """Return a labeled, internally generated state report."""
        return self.self_model.generate_report()
    
    def print_state(self):
        """Print current brain state"""
        state = self.get_brain_state()
        print(f"\n=== Brain State at {state['simulation_time']:.1f} ms ===")
        print(f"Total Neurons: {state['total_neurons']}")
        print(f"Total Synapses: {state['total_synapses']}")
        print("\nRegion Activities:")
        for name, activity in state['region_activities'].items():
            print(f"  {name}: {activity:.2f} mV")
        print("\nSpike Rates:")
        for name, rate in state['spike_rates'].items():
            print(f"  {name}: {rate:.2f} Hz")
        print("\nNeurotransmitters:")
        for nt, level in state['neurotransmitter_levels'].items():
            print(f"  {nt}: {level:.3f}")
        print(f"  last_rpe: {state.get('last_rpe', 0.0):.3f}")
        if state.get("basal_ganglia_state"):
            bg_state = state["basal_ganglia_state"]
            if bg_state and bg_state.get("last_action") is not None:
                print(f"\nBasal Ganglia: last_action={bg_state.get('last_action')}")
        print("\nMemory:")
        mem_stats = state['memory_stats']
        print(f"  Working Memory: {mem_stats['working_memory_count']} patterns")
        print(f"  Long-term Memory: {mem_stats['long_term_memory_count']} patterns")

