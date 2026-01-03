
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.propagation.sionna_wrapper import SionnaModel
from src.propagation.tirem_wrapper import TiremModel

def compare_models():
    print("--- TIREM vs Raytracing Comparison ---")
    
    # 1. Setup Parameters
    config_sionna = 'config/sionna_parameters.yaml'
    config_tirem = 'config/tirem_parameters.yaml'
    
    if not os.path.exists(config_sionna) or not os.path.exists(config_tirem):
        print("Configuration files not found.")
        return

    # Load map metadata to determine scenarios
    map_file = "SLCmap_5May2022.mat"
    if not os.path.exists(map_file):
        print(f"Map file {map_file} not found in root.")
        # Try to find it
        if os.path.exists(os.path.join("..", map_file)):
            map_file = os.path.join("..", map_file)
        else:
            print("Could not locate map file.")
            return
            
    try:
        mat = sio.loadmat(map_file)
        slc = mat['SLC'][0][0]
        # Get dimensions
        # cellsize usually at index corresponding to 'cellsize' field
        # We can just rely on the shape of 'dem'
        dtype_names = slc.dtype.names
        dem_idx = dtype_names.index('dem')
        dem = slc[dem_idx]
        native_height, native_width = dem.shape
        
        cellsize_idx = dtype_names.index('cellsize')
        native_cellsize = float(slc[cellsize_idx][0][0])
        
        print(f"Map Native Size: {native_height}x{native_width}")
        print(f"Native Resolution: {native_cellsize} m")
        
    except Exception as e:
        print(f"Error loading map metadata: {e}")
        return

    # Define test resolution
    # Let's downsample by factor of 150 for speed (requested by user)
    downsample = 150
    scale = native_cellsize * downsample
    map_shape = (native_height // downsample, native_width // downsample)
    
    print(f"Test Configuration:")
    print(f"  Scale: {scale} m/pixel")
    print(f"  Map Shape: {map_shape} ({map_shape[0]*map_shape[1]} points)")
    
    # Define sensors (receivers)
    # Pick a few locations that are likely valid (middle of map)
    # Coordinates in pixel space (col, row)
    sensors = np.array([
        [map_shape[1] // 2, map_shape[0] // 2],       # Center
        [map_shape[1] // 4, map_shape[0] // 4],       # Top-left quadrant
        [3 * map_shape[1] // 4, 3 * map_shape[0] // 4] # Bottom-right quadrant
    ])
    
    print(f"  Sensors: {len(sensors)}")

    # 2. Run TIREM
    print("\n--- Running TIREM ---")
    try:
        tirem_model = TiremModel(config_tirem)
        A_tirem = tirem_model.compute_propagation_matrix(
            sensors, map_shape, scale=scale, n_jobs=-1, verbose=True
        )
    except Exception as e:
        print(f"TIREM failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Running Raytracing (Vectorized Reciprocity Cloud)
    print("\n--- Running Raytracing (Fast Reciprocity Cloud) ---")
    try:
        sionna_model = SionnaModel(config_sionna)
        A_rt = sionna_model.compute_propagation_matrix(
            sensors, map_shape, scale=scale, verbose=True, method='coverage_map'
        )
    except Exception as e:
        print(f"Raytracing failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Analysis & Visualization
    print("\n--- Comparison Analysis ---")
    
    # Filter valid values (> -150 dB usually implies valid signal)
    # Or just compare everything
    
    for i in range(len(sensors)):
        sensor_loc = sensors[i]
        grid_tirem = A_tirem[i].reshape(map_shape) # Linear
        grid_rt = A_rt[i].reshape(map_shape)       # Linear (Updated Standard)
        
        # Convert TIREM to dB
        grid_tirem_db = 10 * np.log10(grid_tirem + 1e-30)
        
        # Convert RT to dB
        grid_rt_db = 10 * np.log10(grid_rt + 1e-30)
        
        # Mask out very low values for statistics (noise floor)
        # Use a reasonable floor like -160 dB
        mask = (grid_tirem_db > -160) & (grid_rt_db > -160)
        
        valid_tirem = grid_tirem_db[mask]
        valid_rt = grid_rt_db[mask]
        
        diff = valid_rt - valid_tirem
        
        print(f"\nSensor {i} at {sensor_loc}:")
        if len(diff) > 0:
            mae = np.mean(np.abs(diff))
            me = np.mean(diff)
            corr = np.corrcoef(valid_tirem, valid_rt)[0, 1]
            print(f"  Points with > -160dB: {len(diff)}/{mask.size} ({100*len(diff)/mask.size:.1f}%)")
            print(f"  Correlation: {corr:.4f}")
            print(f"  Mean Error (RT - TIREM): {me:.2f} dB")
            print(f"  Mean Abs Error: {mae:.2f} dB")
            print(f"  RMSE: {np.sqrt(np.mean(diff**2)):.2f} dB")
        else:
            print("  No common valid signal points found.")
            
        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Using fixed reasonable range for path gain
        vmin, vmax = -160, -60
        
        im0 = axes[0].imshow(grid_tirem_db, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0].set_title(f'TIREM (dB) (Sensor {i})')
        axes[0].scatter(sensor_loc[0], sensor_loc[1], c='red', marker='x')
        plt.colorbar(im0, ax=axes[0], label='Path Gain (dB)')
        
        im1 = axes[1].imshow(grid_rt_db, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        axes[1].set_title(f'Raytracing (dB) (Sensor {i})')
        axes[1].scatter(sensor_loc[0], sensor_loc[1], c='red', marker='x')
        plt.colorbar(im1, ax=axes[1], label='Path Gain (dB)')
        
        # Difference Heatmap (RT - TIREM in dB)
        # We need full grid difference, masking invalid
        diff_grid = grid_rt_db - grid_tirem_db
        # Hide invalid
        diff_grid[~mask] = np.nan
        
        im2 = axes[2].imshow(diff_grid, origin='lower', cmap='coolwarm', vmin=-20, vmax=20)
        axes[2].set_title(f'Difference (RT - TIREM)')
        plt.colorbar(im2, ax=axes[2], label='Delta (dB)')
        
        plt.tight_layout()
        output_file = f"comparison_sensor_{i}.png"
        plt.savefig(output_file)
        print(f"  Saved plot to {output_file}")
        plt.close(fig)

        # Plot Histograms
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(valid_tirem, bins=50, alpha=0.5, label='TIREM (dB)', density=True)
        ax.hist(valid_rt, bins=50, alpha=0.5, label='Raytracing (dB)', density=True)
        ax.set_title(f"Path Gain Distribution (Sensor {i})")
        ax.set_xlabel("Path Gain (dB)")
        ax.legend()
        plt.savefig(f"comparison_hist_{i}.png")
        plt.close(fig)

if __name__ == "__main__":
    compare_models()
