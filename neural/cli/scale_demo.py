"""
Simple demonstration of scaling the artificial brain.
Shows how to go from small to large networks.
"""

import numpy as np
import time
from neural.engines.scaled_brain import ScaledArtificialBrain, GPU_AVAILABLE


def demo_scaling():
    """Demonstrate scaling from small to large networks"""
    
    print("=" * 70)
    print("ARTIFICIAL BRAIN SCALING DEMONSTRATION")
    print("=" * 70)
    
    sizes = [
        ("Small", 1000, 500, 300),
        ("Medium", 5000, 2000, 1000),
        ("Large", 20000, 10000, 5000),
    ]
    
    for size_name, cortex, hippo, thalamus in sizes:
        print(f"\n{'='*70}")
        print(f"Testing {size_name} Brain")
        print(f"  Neurons: {cortex + hippo + thalamus:,}")
        print(f"{'='*70}")
        
        try:
            start = time.time()
            use_gpu = GPU_AVAILABLE
            brain = ScaledArtificialBrain(
                num_cortical_neurons=cortex,
                num_hippocampal_neurons=hippo,
                num_thalamic_neurons=thalamus,
                use_gpu=use_gpu
            )
            if use_gpu:
                print(f"  Using GPU acceleration")
            creation_time = time.time() - start
            
            print(f"\nCreation: {creation_time:.2f}s")
            print(f"Total neurons: {brain.get_total_neurons():,}")
            print(f"Estimated synapses: {brain.get_estimated_synapses():,}")
            
            visual_input = np.random.rand(50) * 0.5
            brain.add_sensory_input('vision', visual_input)
            
            print("\nRunning 10ms simulation...")
            sim_start = time.time()
            brain.run(10.0)
            sim_time = time.time() - sim_start
            
            print(f"\nSimulation: {sim_time:.2f}s")
            print(f"Speed: {10.0/sim_time:.1f}x real-time")
            
            brain.print_state()
            
        except MemoryError:
            print(f"\n[ERROR] Not enough memory for {size_name} brain on this system")
        except Exception as e:
            print(f"\n[ERROR] {e}")
    
    print("\n" + "=" * 70)
    print("SCALING SUMMARY")
    print("=" * 70)
    print("""
The scaled version uses:
  * Vectorized operations (all neurons updated in parallel)
  * Sparse matrices (only stores active connections)
  * Efficient memory usage
  
To scale further:
  1. Use GPU acceleration (set use_gpu=True)
  2. Increase available RAM
  3. Use distributed computing for very large networks
  
Current limits:
  - CPU: ~1M neurons (depends on RAM)
  - GPU: ~10M neurons (depends on GPU RAM)
  - Distributed: 100M+ neurons (requires cluster)
    """)


if __name__ == "__main__":
    demo_scaling()

