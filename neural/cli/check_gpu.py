"""Check GPU availability"""
try:
    import cupy as cp
    print(f"CuPy version: {cp.__version__}")
    print(f"CUDA available: {cp.cuda.is_available()}")
    if cp.cuda.is_available():
        device = cp.cuda.Device()
        print(f"GPU device: {device}")
        print(f"Compute capability: {device.compute_capability}")
        print(f"GPU memory: {cp.get_default_memory_pool().get_limit() / 1e9:.2f} GB")
    else:
        print("CUDA not available - check CUDA installation")
except ImportError:
    print("CuPy not installed")
    print("\nTo install CuPy for RTX 3060:")
    print("1. First install CUDA Toolkit (11.x or 12.x)")
    print("2. Then install CuPy:")
    print("   pip install cupy-cuda11x  # For CUDA 11.x")
    print("   # or")
    print("   pip install cupy-cuda12x  # For CUDA 12.x")

