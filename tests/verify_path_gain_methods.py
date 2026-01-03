
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.propagation.sionna_wrapper import SionnaModel

def verify_methods():
    print("Initializing SionnaModel...")
    config_path = 'config/sionna_parameters.yaml'
    
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return

    try:
        model = SionnaModel(config_path)
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return

    # Define a small test scenario
    # 2 sensors
    sensor_locations = np.array([
        [10, 10],
        [40, 40]
    ])
    
    # 50x50 grid
    map_shape = (50, 50)
    scale = 10.0 # 10m per pixel
    
    print("\n--- Running Legacy Raytracing Method (Point-to-Point) ---")
    start_time = time.time()
    # We might need to reduce config batch size or num_samples for speed in test
    # But usually compute_propagation_matrix loops over grid points. 
    # 50x50 = 2500 grid points. Might take a while if not careful.
    # Let's reduce grid size for the slow method check
    
    small_map_shape = (2, 2)
    N_small = small_map_shape[0] * small_map_shape[1]
    
    print(f"Computing for {N_small} grid points...")
    
    # Reduce num_samples for test speed
    original_samples = model.num_samples
    model.num_samples = 100000
    print(f"Reducing num_samples to {model.num_samples} for verification")
    
    A_legacy = model.compute_propagation_matrix(
        sensor_locations, 
        small_map_shape, 
        scale=scale, 
        verbose=True,
        method='raytracing'
    )
    time_legacy = time.time() - start_time
    print(f"Legacy method took {time_legacy:.2f} seconds")
    
    print("\n--- Running New Coverage Map Method ---")
    start_time = time.time()
    A_fast = model.compute_propagation_matrix(
        sensor_locations, 
        small_map_shape, 
        scale=scale, 
        verbose=True,
        method='coverage_map'
    )
    time_fast = time.time() - start_time
    print(f"Fast method took {time_fast:.2f} seconds")
    
    # Compare results (Linear)
    # Convert to dB for easy viewing
    A_legacy_db = 10 * np.log10(A_legacy + 1e-30)
    A_fast_db = 10 * np.log10(A_fast + 1e-30)
    
    print("\n--- Comparison ---")
    print(f"Legacy - Mean: {np.mean(A_legacy_db[A_legacy_db > -199]):.2f} dB")
    print(f"Fast   - Mean: {np.mean(A_fast_db[A_fast_db > -199]):.2f} dB")
    
    # Check correlation
    # Filter out invalid values
    valid_mask = (A_legacy_db > -150) & (A_fast_db > -150)
    if np.sum(valid_mask) > 0:
        corr = np.corrcoef(A_legacy[valid_mask], A_fast[valid_mask])[0, 1]
        print(f"Correlation coefficient on valid paths (Linear): {corr:.4f}")
        
        diff = np.abs(A_legacy_db[valid_mask] - A_fast_db[valid_mask])
        print(f"Mean absolute difference: {np.mean(diff):.2f} dB")
    else:
        print("No common valid paths found (all masked/blocked?)")
        
    print("\nSpeedup factor: {:.2f}x".format(time_legacy / time_fast))

if __name__ == "__main__":
    verify_methods()
