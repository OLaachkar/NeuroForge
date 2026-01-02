"""
Setup script for GPU acceleration on RTX 3060
"""

import subprocess
import sys

def check_cuda():
    """Check if CUDA is available"""
    try:
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("[OK] CUDA Toolkit found")
            print(result.stdout.split('\n')[3])  # Version line
            return True
    except:
        pass
    
    try:
        result = subprocess.run(['nvidia-smi'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("[OK] NVIDIA GPU detected")
            for line in result.stdout.split('\n'):
                if 'Driver Version' in line:
                    print(f"  {line.strip()}")
            return True
    except:
        pass
    
    print("[WARNING] CUDA not found in PATH")
    print("  But GPU may still work - try installing CuPy anyway")
    return False

def install_cupy():
    """Install CuPy - try both CUDA 11 and 12"""
    print("\n" + "="*60)
    print("Installing CuPy for GPU acceleration...")
    print("="*60)
    
    print("\nTrying CUDA 12.x...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
                              'cupy-cuda12x', '--upgrade'])
        print("[OK] CuPy (CUDA 12.x) installed successfully!")
        return True
    except:
        print("  CUDA 12.x not available, trying CUDA 11.x...")
    
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
                              'cupy-cuda11x', '--upgrade'])
        print("[OK] CuPy (CUDA 11.x) installed successfully!")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to install CuPy: {e}")
        print("\nManual installation:")
        print("  1. Install CUDA Toolkit from NVIDIA")
        print("  2. Then: pip install cupy-cuda11x  (or cupy-cuda12x)")
        return False

def test_gpu():
    """Test GPU after installation"""
    print("\n" + "="*60)
    print("Testing GPU...")
    print("="*60)
    
    try:
        import cupy as cp
        print(f"[OK] CuPy version: {cp.__version__}")
        
        if cp.cuda.is_available():
            device = cp.cuda.Device()
            print(f"[OK] CUDA available")
            print(f"  Device: {device}")
            print(f"  Compute capability: {device.compute_capability}")
            
            a = cp.array([1, 2, 3])
            b = cp.array([4, 5, 6])
            c = a + b
            print(f"[OK] GPU computation test: {cp.asnumpy(c)}")
            
            mempool = cp.get_default_memory_pool()
            mempool.set_limit(size=8 * 1024**3)  # 8GB limit
            print(f"[OK] GPU memory pool configured")
            
            return True
        else:
            print("[ERROR] CUDA not available - check CUDA installation")
            return False
    except ImportError:
        print("[ERROR] CuPy not installed")
        return False
    except Exception as e:
        print(f"[ERROR] GPU test failed: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("GPU Setup for RTX 3060")
    print("="*60)
    
    cuda_available = check_cuda()
    
    if install_cupy():
        if test_gpu():
            print("\n" + "="*60)
            print("[SUCCESS] GPU setup complete! You can now use GPU acceleration.")
            print("="*60)
            print("\nUsage:")
            print("  from neural.engines.scaled_brain import ScaledArtificialBrain")
            print("  brain = ScaledArtificialBrain(..., use_gpu=True)")
        else:
            print("\n[WARNING] GPU installed but not working - check CUDA installation")
    else:
        print("\n[WARNING] Please install CuPy manually")

