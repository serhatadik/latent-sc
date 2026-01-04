
import sys
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add OpenGERT path for utils
# opengert_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../OpenGERT'))
# sys.path.insert(0, opengert_path)

try:
    import sionna
    from sionna.rt import load_scene
except ImportError:
    print("[ERROR] Sionna not found. Please install checks.")
    sys.exit(1)

# Import the helper from OpenGERT styled path if available, or define it inline 
# if we want to be self-contained (safest option given environment complexities).
# The user pointed to OpenGERT/opengert/RT/utils/sionna_utils.py
# Let's try to import it dynamically or copy the function since libraries might be messy.

# To be robust, I will use the function logic provided by the user directly in this script.
# This avoids path hacking issues.

import drjit as dr
import mitsuba as mi

def find_highest_z_at_xy(scene, query_x, query_y, include_ground=True):
    """
    Finds the surface z coordinate of the object closest to a given (x, y) pair in the scene.
    Modified to default include_ground=True for Digital Surface Model generation.
    """
    closest_object_key = None
    minimal_distance = None

    # First pass: find the object whose vertex is closest to the query point (2D)
    # This is a heuristic optimization to avoid checking every triangle in every object?
    # Actually, Sionna scenes are usually BVH accelerated, but this helper iterates objects.
    # For many buildings this might be slow, but for checking mismatch it's fine.
    
    # NOTE: The original helper used vertex distance. This assumes vertices are dense enough.
    # For large terrain triangles, the vertices might be far, but the face covers the point.
    # So this 'closest vertex' heuristic is flawed for terrain.
    # However, for finding "which building is here", it works okay if buildings are small.
    
    # BETTER APPROACH for "One-to-one comparison":
    # Use Sionna's built-in ray tracer (scene.compute_paths equivalent logic) via Mitsuba?
    # Or just use the provided helper as requested. 
    # The user said: "You can refer to the utility/helper functions for querying the height... at a particular x,y"
    
    # If the helper relies on 'vertex' proximity, it might be inaccurate for large faces.
    # Let's stick to the user's suggestion but be aware of limitations.
    # Wait, the provided helper `find_highest_z_at_xy` does:
    # 1. Finds object with closest vertex.
    # 2. WITHIN that object, finds highest Z of vertices near query.
    
    # This effectively finds the "height of the nearest vertex". 
    # It does NOT do raycasting/interpolation on the face.
    # So it will look like a "Voronoi" height map or point cloud projection.
    # This might be sufficient for "let us see the error", but not exact surface.
    
    # Let's try to implement a robust RAYCAST using Mitsuba directly if possible, seeing as Sionna uses Mitsuba.
    # mi.Scene.ray_intersect
    
    # Construct a ray from high up pointing down
    # origin = [x, y, 1000], direction = [0, 0, -1]
    
    ray = mi.Ray3f(o=[query_x, query_y, 2000.0], d=[0, 0, -1])
    si = scene._scene.ray_intersect(ray)
    
    if np.array(si.is_valid())[0]:
        # Hit something
        # position is si.p
        hit_pos = np.array(si.p)
        return hit_pos[2] # Z value
    else:
        return np.nan


def load_slcmap(slcmap_path, downsample_factor=1):
    print(f"[INFO] Loading SLCMap from: {slcmap_path}")
    mat_data = sio.loadmat(str(slcmap_path))
    map_struct = mat_data['SLC']
    SLC = map_struct[0][0]
    
    column_map = {name: i for i, name in enumerate(SLC.dtype.names)}
    
    dem = SLC[column_map["dem"]].astype(np.float64)
    hybrid_bldg = SLC[column_map["hybrid_bldg"]].astype(np.float64)
    buildings = 0.3048 * hybrid_bldg
    
    if "cellsize" in column_map:
        cellsize = float(SLC[column_map["cellsize"]][0][0])
    else:
        cellsize = 1.0

    axis = SLC[column_map["axis"]].flatten()
    
    if downsample_factor > 1:
        dem = dem[::downsample_factor, ::downsample_factor]
        buildings = buildings[::downsample_factor, ::downsample_factor]
        cellsize *= downsample_factor
    
    combined = dem + buildings
    
    return {
        'dem': dem,
        'buildings': buildings,
        'combined': combined,
        'cellsize': cellsize,
        'axis': axis
    }

def main():
    # Parameters
    # Using 10x downsample to align with standard scene generation
    DOWNSAMPLE = 10
    
    slcmap_path = Path("SLCmap_5May2022.mat")
    scene_xml_path = Path("data/sionna_scenes/scene.xml")
    
    if not slcmap_path.exists():
        print(f"[ERROR] SLCMap not found: {slcmap_path}")
        return
        
    if not scene_xml_path.exists():
        print(f"[ERROR] Scene XML not found: {scene_xml_path}")
        print("Please run 'python scripts/run_conversion.py' first.")
        return

    # 1. Load TIREM Geometry
    tirem_data = load_slcmap(slcmap_path, downsample_factor=DOWNSAMPLE)
    tirem_combined = tirem_data['combined']
    cellsize = tirem_data['cellsize']
    axis = tirem_data['axis']
    
    # Calculate center for local coordinate conversion
    # Sionna scene is centered at (0,0) which corresponds to the center of the map bounds
    center_x = (axis[0] + axis[1]) / 2
    center_y = (axis[2] + axis[3]) / 2
    
    rows, cols = tirem_combined.shape
    print(f"[INFO] Grid shape: {rows}x{cols} (Scale: {cellsize}m)")
    
    # 2. Load Sionna Scene
    print(f"[INFO] Loading Sionna scene from {scene_xml_path}...")
    scene = load_scene(str(scene_xml_path))
    
    # 3. Compute Difference Map using Raycasting
    # We will iterate over the grid and raycast for each pixel center
    print("[INFO] Computing height map via raycasting (this may take a moment)...")
    
    sionna_z = np.zeros_like(tirem_combined)
    
    # Generating grid coordinates
    # TIREM grid logic:
    # row 0 is at axis[2] (min_y) ? Or max_y?
    # standard image: row 0 is top. 
    # SLCMapToScene logic: y = axis[2] + row * cellsize (implies row 0 is bottom/min_y)
    
    # Let's vectorize the raycasting if possible.
    # mitsuba.Scene.ray_intersect allows wavefront (batch) raycasting!
    
    r_idx = np.arange(rows)
    c_idx = np.arange(cols)
    C, R = np.meshgrid(c_idx, r_idx) # Grid of indices
    
    # Flatten
    c_flat = C.flatten()
    r_flat = R.flatten()
    
    # Convert to Local Coordinates
    # Matches SLCMapToScene transform
    x_local = axis[0] + c_flat * cellsize - center_x
    y_local = axis[2] + r_flat * cellsize - center_y
    z_start = np.full_like(x_local, 2000.0)
    
    n_rays = len(x_local)
    print(f"[INFO] Batch raycasting {n_rays} rays...")
    
    # Create Mitsuba Rays
    # We need to construct them in Dr.Jit arrays
    
    # o_x = dr.ravel(dr.float32(x_local)) # This might fail if x_local is numpy
    # Use direct mitsuba/drjit conversion
    
    ray_o = mi.Point3f(x_local, y_local, z_start)
    ray_d = mi.Vector3f(0, 0, -1)
    
    ray = mi.Ray3f(ray_o, ray_d)
    
    # Raycast
    si = scene._scene.ray_intersect(ray)
    
    # Extract Z hits
    # si.p is the intersection point
    # si.is_valid() boolean mask
    
    hits = si.p.z.numpy()
    valid = si.is_valid().numpy()
    
    # Reshape back to grid
    sionna_z_flat = hits
    
    # Handle misses (set to NaN or min)
    # If invalid, it means ray didn't hit anything (hole in mesh?)
    # or below 2000m (unlikely)
    
    sionna_z_flat[~valid] = np.nan
    sionna_z = sionna_z_flat.reshape((rows, cols))
    
    # 4. Compare
    # TIREM Combined vs Sionna Z
    
    diff = sionna_z - tirem_combined
    valid_mask = ~np.isnan(diff)
    
    if np.sum(valid_mask) == 0:
        print("[ERROR] No valid intersections found! Check coordinate alignment.")
        # Debug shift?
        print(f"Sample Ray: {x_local[0]}, {y_local[0]}")
        print(f"Scene bounds: {scene._scene.bbox()}")
    
    else:
        print("\nSTATISTICS:")
        print(f"  Valid Pixels: {np.sum(valid_mask)}/{diff.size}")
        print(f"  Mean Error: {np.mean(diff[valid_mask]):.4f} m")
        print(f"  Max Error:  {np.max(np.abs(diff[valid_mask])):.4f} m")
        print(f"  RMSE:       {np.sqrt(np.mean(diff[valid_mask]**2)):.4f} m")

    # 5. Visualize
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # TIREM
    im0 = axes[0,0].imshow(tirem_combined, origin='lower', cmap='terrain')
    axes[0,0].set_title("TIREM Geometry (Combined)", fontsize=12)
    plt.colorbar(im0, ax=axes[0,0], label='Elevation (m)')
    
    # Sionna
    im1 = axes[0,1].imshow(sionna_z, origin='lower', cmap='terrain', vmin=np.nanmin(tirem_combined), vmax=np.nanmax(tirem_combined))
    axes[0,1].set_title("Sionna Scene (Raycast)", fontsize=12)
    plt.colorbar(im1, ax=axes[0,1], label='Elevation (m)')
    
    # Difference
    max_diff = 10.0
    im2 = axes[1,0].imshow(diff, origin='lower', cmap='seismic', vmin=-max_diff, vmax=max_diff)
    axes[1,0].set_title("Height Difference (Sionna - TIREM)\nBlue = Sionna Lower, Red = Sionna Higher", fontsize=12)
    plt.colorbar(im2, ax=axes[1,0], label='Delta (m)')
    
    # Histogram
    if np.any(valid_mask):
        axes[1,1].hist(diff[valid_mask].flatten(), bins=100, range=(-5, 5), color='purple', alpha=0.7)
        axes[1,1].set_title("Error Distribution (Center Zoom +/- 5m)", fontsize=12)
        axes[1,1].set_xlabel("Error (m)")
        axes[1,1].set_ylabel("Pixel Count")
        
        # Add stats text
        stats_text = (f"RMSE: {np.sqrt(np.mean(diff[valid_mask]**2)):.2f}m\n"
                      f"Mean: {np.mean(diff[valid_mask]):.2f}m")
        axes[1,1].text(0.95, 0.95, stats_text, transform=axes[1,1].transAxes, ha='right', va='top', bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    output_path = "results/geometry_alignment_test.png"
    plt.savefig(output_path, dpi=150)
    print(f"\n[INFO] Saved comparison figure to: {output_path}")

if __name__ == "__main__":
    main()
