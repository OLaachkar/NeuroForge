"""
Benchmark script to test scaling performance.
Compares original vs scaled implementations.
"""

import numpy as np
import time
from neural.engines.artificial_brain import ArtificialBrain
from neural.engines.scaled_brain import ScaledArtificialBrain


def benchmark_original(size_name: str, num_cortical: int, num_hippo: int, num_thalamus: int):
    """Benchmark original implementation"""
    print(f"\n{'='*60}")
    print(f"ORIGINAL IMPLEMENTATION - {size_name}")
    print(f"{'='*60}")
    
    try:
        start = time.time()
        brain = ArtificialBrain(
            num_cortical_neurons=num_cortical,
            num_hippocampal_neurons=num_hippo,
            num_thalamic_neurons=num_thalamus
        )
        creation_time = time.time() - start
        
        print(f"Creation time: {creation_time:.2f}s")
        print(f"Neurons: {brain.get_total_neurons():,}")
        print(f"Synapses: {brain.get_total_synapses():,}")
        
        sim_start = time.time()
        brain.run(10.0)  # 10ms simulation
        sim_time = time.time() - sim_start
        
        print(f"Simulation time: {sim_time:.2f}s")
        print(f"Speed: {10.0/sim_time:.2f}x real-time")
        
        return creation_time, sim_time
        
    except MemoryError:
        print("  Memory error - too large for original implementation")
        return None, None


def benchmark_scaled(size_name: str, num_cortical: int, num_hippo: int, num_thalamus: int, use_gpu: bool = False):
    """Benchmark scaled implementation"""
    print(f"\n{'='*60}")
    print(f"SCALED IMPLEMENTATION - {size_name} (GPU: {use_gpu})")
    print(f"{'='*60}")
    
    try:
        start = time.time()
        brain = ScaledArtificialBrain(
            num_cortical_neurons=num_cortical,
            num_hippocampal_neurons=num_hippo,
            num_thalamic_neurons=num_thalamus,
            use_gpu=use_gpu
        )
        creation_time = time.time() - start
        
        print(f"Creation time: {creation_time:.2f}s")
        print(f"Neurons: {brain.get_total_neurons():,}")
        print(f"Estimated synapses: {brain.get_estimated_synapses():,}")
        
        sim_start = time.time()
        brain.run(10.0)  # 10ms simulation
        sim_time = time.time() - sim_start
        
        print(f"Simulation time: {sim_time:.2f}s")
        print(f"Speed: {10.0/sim_time:.2f}x real-time")
        
        return creation_time, sim_time
        
    except MemoryError:
        print("  Memory error - too large for scaled implementation")
        return None, None
    except Exception as e:
        print(f"  Error: {e}")
        return None, None


def main():
    """Run scaling benchmarks"""
    print("=" * 60)
    print("SCALING BENCHMARK")
    print("=" * 60)
    print("\nComparing original vs scaled implementations")
    print("Testing different brain sizes...")
    
    test_sizes = [
        ("Small", 500, 200, 100),
        ("Medium", 2000, 1000, 500),
        ("Large", 5000, 2000, 1000),
        ("Very Large", 10000, 5000, 3000),
    ]
    
    results = []
    
    for size_name, cortex, hippo, thalamus in test_sizes:
        print(f"\n{'='*60}")
        print(f"TESTING: {size_name} Brain")
        print(f"  Cortex: {cortex:,} neurons")
        print(f"  Hippocampus: {hippo:,} neurons")
        print(f"  Thalamus: {thalamus:,} neurons")
        print(f"{'='*60}")
        
        orig_create, orig_sim = benchmark_original(size_name, cortex, hippo, thalamus)
        
        scaled_create, scaled_sim = benchmark_scaled(size_name, cortex, hippo, thalamus, use_gpu=False)
        
        gpu_create, gpu_sim = None, None
        try:
            import cupy
            gpu_create, gpu_sim = benchmark_scaled(size_name, cortex, hippo, thalamus, use_gpu=True)
        except ImportError:
            print("\nGPU version skipped (CuPy not available)")
        
        results.append({
            'size': size_name,
            'neurons': cortex + hippo + thalamus,
            'original': (orig_create, orig_sim),
            'scaled_cpu': (scaled_create, scaled_sim),
            'scaled_gpu': (gpu_create, gpu_sim),
        })
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Size':<15} {'Neurons':<12} {'Original':<20} {'Scaled CPU':<20} {'Scaled GPU':<20}")
    print("-" * 60)
    
    for result in results:
        orig_str = f"{result['original'][1]:.2f}s" if result['original'][1] else "N/A"
        scaled_str = f"{result['scaled_cpu'][1]:.2f}s" if result['scaled_cpu'][1] else "N/A"
        gpu_str = f"{result['scaled_gpu'][1]:.2f}s" if result['scaled_gpu'][1] else "N/A"
        
        print(f"{result['size']:<15} {result['neurons']:<12,} {orig_str:<20} {scaled_str:<20} {gpu_str:<20}")
    
    print("\n" + "=" * 60)
    print("SPEEDUP ANALYSIS")
    print("=" * 60)
    
    for result in results:
        if result['original'][1] and result['scaled_cpu'][1]:
            speedup = result['original'][1] / result['scaled_cpu'][1]
            print(f"{result['size']}: {speedup:.2f}x faster with scaled CPU version")
        
        if result['scaled_cpu'][1] and result['scaled_gpu'][1]:
            gpu_speedup = result['scaled_cpu'][1] / result['scaled_gpu'][1]
            print(f"{result['size']}: {gpu_speedup:.2f}x faster with GPU")


if __name__ == "__main__":
    main()

