"""
Scaled Artificial Brain
Optimized version for handling millions of neurons using:
- GPU acceleration (optional)
- Sparse connectivity matrices
- Vectorized operations
- Event-driven updates
"""

import numpy as np
from scipy import sparse
from typing import Dict, List, Optional, Tuple
import time

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None


class VectorizedNeuronLayer:
    """
    Vectorized neuron layer - processes many neurons in parallel.
    Much faster than individual neuron objects.
    """
    
    def __init__(
        self,
        num_neurons: int,
        excitatory_ratio: float = 0.8,
        use_gpu: bool = False,
    ):
        self.num_neurons = num_neurons
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np
        
        num_excitatory = int(num_neurons * excitatory_ratio)
        self.neuron_types = self.xp.zeros(num_neurons, dtype=self.xp.int32)
        self.neuron_types[:num_excitatory] = 1  # 1 = excitatory, 0 = inhibitory
        
        self.membrane_potentials = self.xp.full(num_neurons, -70.0, dtype=self.xp.float32)
        
        self.thresholds = self.xp.full(num_neurons, -55.0, dtype=self.xp.float32)
        self.resting_potentials = self.xp.full(num_neurons, -70.0, dtype=self.xp.float32)
        self.reset_potentials = self.xp.full(num_neurons, -70.0, dtype=self.xp.float32)
        
        self.refractory_times = self.xp.zeros(num_neurons, dtype=self.xp.float32)
        self.is_refractory = self.xp.zeros(num_neurons, dtype=self.xp.bool_)
        
        self.spike_counts = self.xp.zeros(num_neurons, dtype=self.xp.int32)
        self.spiked = self.xp.zeros(num_neurons, dtype=self.xp.bool_)
        
        self.tau_m = 20.0  # ms
        self.membrane_resistance = 100.0  # MΩ
        self.refractory_period = 2.0  # ms
        
    def update(self, dt: float, synaptic_inputs: np.ndarray, external_current: Optional[np.ndarray] = None):
        """
        Update all neurons in parallel.
        
        Args:
            dt: Time step (ms)
            synaptic_inputs: Input current for each neuron (nA)
            external_current: External current injection (optional)
        """
        if self.use_gpu and isinstance(synaptic_inputs, np.ndarray):
            synaptic_inputs = cp.asarray(synaptic_inputs)
        
        if external_current is not None:
            if self.use_gpu and isinstance(external_current, np.ndarray):
                external_current = cp.asarray(external_current)
            total_input = synaptic_inputs + external_current
        else:
            total_input = synaptic_inputs
        
        self.refractory_times -= dt
        self.is_refractory = self.refractory_times > 0
        
        self.membrane_potentials = self.xp.where(
            self.is_refractory,
            self.reset_potentials,
            self.membrane_potentials
        )
        
        dV = ((self.resting_potentials - self.membrane_potentials) + 
              self.membrane_resistance * total_input) / self.tau_m
        
        self.membrane_potentials = self.xp.where(
            ~self.is_refractory,
            self.membrane_potentials + dV * dt,
            self.membrane_potentials
        )
        
        self.spiked = (self.membrane_potentials >= self.thresholds) & (~self.is_refractory)
        
        self.spike_counts += self.spiked.astype(self.xp.int32)
        
        self.membrane_potentials = self.xp.where(
            self.spiked,
            30.0,  # Spike peak
            self.membrane_potentials
        )
        
        self.refractory_times = self.xp.where(
            self.spiked,
            self.refractory_period,
            self.refractory_times
        )
    
    def get_spikes(self) -> np.ndarray:
        """Get current spike outputs"""
        if self.use_gpu:
            return cp.asnumpy(self.spiked.astype(self.xp.float32))
        return self.spiked.astype(np.float32)
    
    def get_membrane_potentials(self) -> np.ndarray:
        """Get current membrane potentials"""
        if self.use_gpu:
            return cp.asnumpy(self.membrane_potentials)
        return self.membrane_potentials
    
    def get_activity(self) -> float:
        """Get average activity"""
        if self.use_gpu:
            return float(cp.mean(self.membrane_potentials))
        return float(np.mean(self.membrane_potentials))


class SparseSynapticMatrix:
    """
    Sparse connectivity matrix for efficient synapse storage and computation.
    Uses scipy.sparse for memory efficiency.
    """
    
    def __init__(
        self,
        num_pre: int,
        num_post: int,
        connectivity_density: float = 0.1,
        initial_weight_range: Tuple[float, float] = (0.3, 0.7),
        use_gpu: bool = False,
    ):
        self.num_pre = num_pre
        self.num_post = num_post
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np
        
        num_connections = int(num_pre * num_post * connectivity_density)
        
        pre_indices = np.random.randint(0, num_pre, num_connections)
        post_indices = np.random.randint(0, num_post, num_connections)
        
        weights = np.random.uniform(
            initial_weight_range[0],
            initial_weight_range[1],
            num_connections
        )
        
        self.weight_matrix = sparse.csr_matrix(
            (weights, (pre_indices, post_indices)),
            shape=(num_pre, num_post),
            dtype=np.float32
        )
        
        self.neurotransmitter_levels = self.xp.zeros(num_post, dtype=self.xp.float32)
        self.tau_syn = 5.0  # ms
        
        self.learning_rate_ltp = 0.01
        self.learning_rate_ltd = 0.01
        self.max_weight = 2.0
        self.min_weight = 0.0
        
        self.ltp_traces = self.xp.zeros(num_pre, dtype=self.xp.float32)
        self.ltd_traces = self.xp.zeros(num_post, dtype=self.xp.float32)
        self.tau_ltp = 20.0
        self.tau_ltd = 20.0
        
    def propagate_spikes(self, pre_spikes: np.ndarray, dt: float):
        """
        Propagate spikes through synapses.
        
        Args:
            pre_spikes: Binary spike array from pre-synaptic neurons
            dt: Time step
        """
        if self.use_gpu:
            if isinstance(pre_spikes, np.ndarray):
                pre_spikes = cp.asarray(pre_spikes)
            post_input = self.weight_matrix.T.dot(cp.asnumpy(pre_spikes))
            post_input = cp.asarray(post_input)
        else:
            post_input = self.weight_matrix.T.dot(pre_spikes)
        
        self.neurotransmitter_levels += post_input * 0.5
        
        self.neurotransmitter_levels *= self.xp.exp(-dt / self.tau_syn)
        
        self.ltp_traces = self.xp.where(
            pre_spikes > 0.5,
            self.ltp_traces + self.learning_rate_ltp,
            self.ltp_traces * self.xp.exp(-dt / self.tau_ltp)
        )
    
    def apply_stdp(self, post_spikes: np.ndarray, dt: float):
        """Apply Spike-Timing Dependent Plasticity"""
        if self.use_gpu:
            if isinstance(post_spikes, np.ndarray):
                post_spikes = cp.asarray(post_spikes)
        else:
            post_spikes = np.asarray(post_spikes)
        
        self.ltd_traces = self.xp.where(
            post_spikes > 0.5,
            self.ltd_traces + self.learning_rate_ltd,
            self.ltd_traces * self.xp.exp(-dt / self.tau_ltd)
        )
        
        if not self.use_gpu:
            ltp_np = self.ltp_traces if isinstance(self.ltp_traces, np.ndarray) else cp.asnumpy(self.ltp_traces)
            ltd_np = self.ltd_traces if isinstance(self.ltd_traces, np.ndarray) else cp.asnumpy(self.ltd_traces)
            
            rows, cols = self.weight_matrix.nonzero()
            
            if len(rows) > 0:
                pre_activity = ltp_np[rows]
                post_activity = ltd_np[cols]
                
                updates = pre_activity * post_activity * 0.0001
                
                self.weight_matrix.data = np.clip(
                    self.weight_matrix.data + updates,
                    self.min_weight,
                    self.max_weight
                )
    
    def get_output(self) -> np.ndarray:
        """Get synaptic output currents"""
        if self.use_gpu:
            return cp.asnumpy(self.neurotransmitter_levels)
        return self.neurotransmitter_levels


class ScaledBrainRegion:
    """
    Scaled brain region using vectorized operations.
    Can handle millions of neurons efficiently.
    """
    
    def __init__(
        self,
        region_name: str,
        num_neurons: int,
        excitatory_ratio: float = 0.8,
        connectivity_density: float = 0.1,
        use_gpu: bool = False,
    ):
        self.region_name = region_name
        self.num_neurons = num_neurons
        self.use_gpu = use_gpu
        
        self.neurons = VectorizedNeuronLayer(
            num_neurons,
            excitatory_ratio,
            use_gpu
        )
        
        self.synapses = SparseSynapticMatrix(
            num_neurons,
            num_neurons,
            connectivity_density,
            use_gpu=use_gpu
        )
        
        self.activity_level = 0.0
        
    def update(self, dt: float, external_input: Optional[np.ndarray] = None):
        """Update region"""
        pre_spikes = self.neurons.get_spikes()
        self.synapses.propagate_spikes(pre_spikes, dt)
        
        synaptic_input = self.synapses.get_output()
        
        self.neurons.update(dt, synaptic_input, external_input)
        
        post_spikes = self.neurons.get_spikes()
        self.synapses.apply_stdp(post_spikes, dt)
        
        self.activity_level = self.neurons.get_activity()
    
    def connect_to(self, target_region: 'ScaledBrainRegion', 
                   connection_density: float = 0.05):
        """Connect this region to another"""
        inter_synapses = SparseSynapticMatrix(
            self.num_neurons,
            target_region.num_neurons,
            connection_density,
            use_gpu=self.use_gpu
        )
        
        if not hasattr(self, 'output_synapses'):
            self.output_synapses = []
        self.output_synapses.append((target_region, inter_synapses))
        
        if not hasattr(target_region, 'input_synapses'):
            target_region.input_synapses = []
        target_region.input_synapses.append((self, inter_synapses))
    
    def get_output(self) -> np.ndarray:
        """Get output spikes"""
        return self.neurons.get_spikes()
    
    def get_activity(self) -> float:
        """Get activity level"""
        return self.activity_level
    
    def get_spike_rate(self) -> float:
        """Get average spike rate"""
        if self.use_gpu:
            spikes = cp.asnumpy(self.neurons.spike_counts)
        else:
            spikes = self.neurons.spike_counts
        return float(np.mean(spikes)) if len(spikes) > 0 else 0.0


class ScaledArtificialBrain:
    """
    Scaled version of artificial brain that can handle millions of neurons.
    Uses vectorized operations and sparse matrices for efficiency.
    """
    
    def __init__(
        self,
        num_cortical_neurons: int = 10000,
        num_hippocampal_neurons: int = 5000,
        num_thalamic_neurons: int = 3000,
        simulation_dt: float = 0.1,
        use_gpu: bool = False,
    ):
        self.simulation_dt = simulation_dt
        self.simulation_time = 0.0
        self.use_gpu = use_gpu and GPU_AVAILABLE
        
        print(f"Creating scaled artificial brain (GPU: {self.use_gpu})...")
        print(f"  Cortex: {num_cortical_neurons:,} neurons")
        print(f"  Hippocampus: {num_hippocampal_neurons:,} neurons")
        print(f"  Thalamus: {num_thalamic_neurons:,} neurons")
        
        self.regions = {}
        
        self.regions['cortex'] = ScaledBrainRegion(
            'cortex',
            num_cortical_neurons,
            excitatory_ratio=0.8,
            connectivity_density=0.1,
            use_gpu=self.use_gpu
        )
        
        self.regions['hippocampus'] = ScaledBrainRegion(
            'hippocampus',
            num_hippocampal_neurons,
            excitatory_ratio=0.9,
            connectivity_density=0.15,
            use_gpu=self.use_gpu
        )
        
        self.regions['thalamus'] = ScaledBrainRegion(
            'thalamus',
            num_thalamic_neurons,
            excitatory_ratio=0.7,
            connectivity_density=0.2,
            use_gpu=self.use_gpu
        )
        
        print("  Connecting regions...")
        self.regions['thalamus'].connect_to(self.regions['cortex'], 0.1)
        self.regions['cortex'].connect_to(self.regions['hippocampus'], 0.08)
        self.regions['hippocampus'].connect_to(self.regions['cortex'], 0.08)
        
        self.sensory_inputs = {}
        
        print("Scaled brain created!")
        print(f"  Total neurons: {self.get_total_neurons():,}")
        print(f"  Estimated synapses: {self.get_estimated_synapses():,}")
    
    def get_total_neurons(self) -> int:
        """Get total number of neurons"""
        return sum(region.num_neurons for region in self.regions.values())
    
    def get_estimated_synapses(self) -> int:
        """Estimate total synapses (sparse matrices)"""
        total = 0
        for region in self.regions.values():
            total += region.synapses.weight_matrix.nnz
            if hasattr(region, 'output_synapses'):
                for _, synapses in region.output_synapses:
                    total += synapses.weight_matrix.nnz
        return total
    
    def add_sensory_input(self, modality: str, input_data: np.ndarray):
        """Add sensory input"""
        self.sensory_inputs[modality] = input_data
    
    def step(self):
        """Run one simulation step"""
        dt = self.simulation_dt
        self.simulation_time += dt
        
        if self.sensory_inputs and 'thalamus' in self.regions:
            thalamus = self.regions['thalamus']
            sensory_input = np.zeros(thalamus.num_neurons)
            
            input_idx = 0
            for modality, data in self.sensory_inputs.items():
                for value in data.flatten():
                    if input_idx < len(sensory_input):
                        sensory_input[input_idx] = value * 0.1
                        input_idx += 1
            
            thalamus.update(dt, sensory_input)
        else:
            self.regions['thalamus'].update(dt)
        
        for name, region in self.regions.items():
            if name != 'thalamus':
                if hasattr(region, 'input_synapses'):
                    total_input = None
                    for source_region, synapses in region.input_synapses:
                        source_spikes = source_region.get_output()
                        synapses.propagate_spikes(source_spikes, dt)
                        syn_output = synapses.get_output()
                        
                        if total_input is None:
                            total_input = syn_output.copy()
                        else:
                            total_input += syn_output
                    
                    region.update(dt, total_input)
                else:
                    region.update(dt)
    
    def run(self, duration_ms: float):
        """Run simulation"""
        num_steps = int(duration_ms / self.simulation_dt)
        start_time = time.time()
        
        for step in range(num_steps):
            self.step()
            
            if step % 1000 == 0 and step > 0:
                elapsed = time.time() - start_time
                rate = step / elapsed
                print(f"  Step {step}/{num_steps} ({rate:.1f} steps/sec)")
        
        elapsed = time.time() - start_time
        print(f"Simulation complete: {elapsed:.2f}s for {duration_ms}ms simulation")
        print(f"  Speed: {duration_ms/elapsed:.1f}x real-time")
    
    def print_state(self):
        """Print brain state"""
        print(f"\n=== Scaled Brain State at {self.simulation_time:.1f} ms ===")
        print(f"Total Neurons: {self.get_total_neurons():,}")
        print(f"Estimated Synapses: {self.get_estimated_synapses():,}")
        print("\nRegion Activities:")
        for name, region in self.regions.items():
            print(f"  {name}: {region.get_activity():.2f} mV (spike rate: {region.get_spike_rate():.2f} Hz)")


if __name__ == "__main__":
    print("=" * 60)
    print("SCALED BRAIN TEST")
    print("=" * 60)
    
    print("\n1. Small brain (10K neurons):")
    brain1 = ScaledArtificialBrain(
        num_cortical_neurons=5000,
        num_hippocampal_neurons=3000,
        num_thalamic_neurons=2000,
        use_gpu=False
    )
    brain1.run(10.0)  # 10ms
    brain1.print_state()
    
    print("\n2. Medium brain (100K neurons):")
    brain2 = ScaledArtificialBrain(
        num_cortical_neurons=50000,
        num_hippocampal_neurons=30000,
        num_thalamic_neurons=20000,
        use_gpu=False
    )
    brain2.run(10.0)
    brain2.print_state()
    
    print("\n3. Large brain (1M neurons) - may take a while:")
    try:
        brain3 = ScaledArtificialBrain(
            num_cortical_neurons=500000,
            num_hippocampal_neurons=300000,
            num_thalamic_neurons=200000,
            use_gpu=False
        )
        brain3.run(1.0)  # Just 1ms for testing
        brain3.print_state()
    except MemoryError:
        print("  Not enough memory for 1M neurons on this system")

