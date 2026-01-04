
import os
import sys
import time
import shutil
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.propagation.sionna_wrapper import SionnaModel

def test_caching():
    print("--- Testing Sionna Caching ---")
    
    config_path = 'config/sionna_parameters.yaml'
    
    # 1. Setup Model
    try:
        model = SionnaModel(config_path)
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return

    # Define test parameters
    map_shape = (10, 10) # Small grid
    scale = 10.0
    sensors = np.array([[5, 5], [2, 2]])
    
    # Clear existing cache for this test (to be safe)
    # The cache dir is data/cache/sionna
    cache_dir = Path("data/cache/sionna")
    if cache_dir.exists():
        print(f"Cleaning cache directory: {cache_dir}")
        # Ideally we only delete files related to this test, but hash is unknown.
        # For this test, we accept clearing it or just rely on the print output.
        # Let's clean it to ensure "Run 1" is actually a computation.
        for f in cache_dir.glob("*.npy"):
            f.unlink()
            
    # Run 1: Computation
    print("\n--- Run 1 (Computation) ---")
    start_time = time.time()
    A1 = model.compute_propagation_matrix(sensors, map_shape, scale=scale, verbose=True)
    run1_duration = time.time() - start_time
    print(f"Run 1 completed in {run1_duration:.4f} seconds")
    
    # Run 2: Cache Hit
    print("\n--- Run 2 (Cache Retrieval) ---")
    start_time = time.time()
    A2 = model.compute_propagation_matrix(sensors, map_shape, scale=scale, verbose=True)
    run2_duration = time.time() - start_time
    print(f"Run 2 completed in {run2_duration:.4f} seconds")
    
    # Verification
    print("\n--- Verification ---")
    
    # Check timings
    if run2_duration < run1_duration * 0.5: # Expecting massive speedup, e.g. < 10%
        print(f"✓ Speedup verified: {run1_duration:.4f}s -> {run2_duration:.4f}s")
    else:
        print(f"❌ Speedup NOT observed. Caching might not be working.")
        
    # Check consistency
    if np.allclose(A1, A2):
        print("✓ Results match exactly.")
    else:
        print("❌ Results mismatch!")
        
    # Check cache file existence
    npy_files = list(cache_dir.glob("*.npy"))
    if len(npy_files) > 0:
        print(f"✓ Cache file created: {npy_files[0].name}")
    else:
        print("❌ No cache file found.")

if __name__ == "__main__":
    test_caching()
