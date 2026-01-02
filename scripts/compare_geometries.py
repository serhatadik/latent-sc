"""
Geometry Comparison Script: TIREM vs Sionna Ray-Tracing

This script visualizes and compares the geometry representations used by:
1. TIREM: Raster-based DEM + building heights from SLCMap
2. Sionna RT: Triangulated mesh scene generated from SLCMap

Run with: python scripts/compare_geometries.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
from pathlib import Path

# Try to import trimesh for 3D mesh visualization
try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False
    print("[WARN] trimesh not available, skipping 3D mesh visualization")


def load_slcmap(slcmap_path, downsample_factor=1):
    """Load SLCMap data for TIREM visualization."""
    print(f"[INFO] Loading SLCMap from: {slcmap_path}")
    
    mat_data = sio.loadmat(str(slcmap_path))
    map_struct = mat_data['SLC']
    SLC = map_struct[0][0]
    
    column_map = {name: i for i, name in enumerate(SLC.dtype.names)}
    
    dem = SLC[column_map["dem"]].astype(np.float64)
    hybrid_bldg = SLC[column_map["hybrid_bldg"]].astype(np.float64)
    buildings = 0.3048 * hybrid_bldg  # Convert feet to meters
    
    cellsize = float(SLC[column_map["cellsize"]][0][0])
    axis = SLC[column_map["axis"]].flatten()
    
    if downsample_factor > 1:
        dem = dem[::downsample_factor, ::downsample_factor]
        buildings = buildings[::downsample_factor, ::downsample_factor]
        cellsize *= downsample_factor
    
    combined = dem + buildings
    
    print(f"  DEM shape: {dem.shape}")
    print(f"  Cell size: {cellsize} m")
    print(f"  Elevation range: [{dem.min():.1f}, {dem.max():.1f}] m")
    print(f"  Building height range: [0, {buildings.max():.1f}] m")
    print(f"  Building cells: {np.sum(buildings > 1.0)}")
    
    return {
        'dem': dem,
        'buildings': buildings,
        'combined': combined,
        'cellsize': cellsize,
        'axis': axis
    }


def load_sionna_meshes(scene_dir):
    """Load Sionna scene meshes for visualization."""
    if not HAS_TRIMESH:
        return None
        
    meshes_dir = Path(scene_dir) / "meshes"
    if not meshes_dir.exists():
        print(f"[WARN] Meshes directory not found: {meshes_dir}")
        return None
    
    print(f"[INFO] Loading meshes from: {meshes_dir}")
    
    all_meshes = []
    mesh_files = list(meshes_dir.glob("*.ply"))
    
    terrain_mesh = None
    building_meshes = []
    
    for mesh_file in mesh_files:
        mesh = trimesh.load(str(mesh_file))
        if 'terrain' in mesh_file.name.lower():
            terrain_mesh = mesh
        else:
            building_meshes.append(mesh)
    
    print(f"  Terrain mesh: {'loaded' if terrain_mesh else 'not found'}")
    print(f"  Building meshes: {len(building_meshes)}")
    
    return {
        'terrain': terrain_mesh,
        'buildings': building_meshes
    }


def plot_tirem_geometry(data, ax1, ax2):
    """Plot TIREM raster geometry."""
    dem = data['dem']
    buildings = data['buildings']
    combined = data['combined']
    
    # Plot 1: DEM with buildings overlay
    ls = LightSource(azdeg=315, altdeg=45)
    dem_shaded = ls.shade(dem, cmap=plt.cm.terrain, blend_mode='soft')
    
    ax1.imshow(dem_shaded, origin='lower', aspect='equal')
    
    # Overlay buildings as semi-transparent
    building_mask = buildings > 1.0
    building_overlay = np.ma.masked_where(~building_mask, buildings)
    im = ax1.imshow(building_overlay, origin='lower', aspect='equal', 
                    cmap='Reds', alpha=0.7, vmin=0, vmax=buildings.max())
    
    ax1.set_title('TIREM: DEM + Buildings (Raster)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Column (pixels)')
    ax1.set_ylabel('Row (pixels)')
    
    # Colorbar for buildings
    cbar = plt.colorbar(im, ax=ax1, shrink=0.6, label='Building Height (m)')
    
    # Plot 2: Combined elevation (DEM + buildings)
    combined_shaded = ls.shade(combined, cmap=plt.cm.terrain, blend_mode='soft')
    ax2.imshow(combined_shaded, origin='lower', aspect='equal')
    ax2.set_title('TIREM: Combined Elevation (DEM + Buildings)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Column (pixels)')
    ax2.set_ylabel('Row (pixels)')


def plot_sionna_geometry(meshes, data, ax1, ax2):
    """Plot Sionna mesh geometry (2D projection)."""
    if meshes is None:
        ax1.text(0.5, 0.5, 'Sionna meshes not available\n(trimesh required)', 
                 ha='center', va='center', transform=ax1.transAxes, fontsize=12)
        ax2.text(0.5, 0.5, 'Sionna meshes not available\n(trimesh required)', 
                 ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        return
    
    terrain = meshes['terrain']
    buildings = meshes['buildings']
    
    # Get bounds
    cellsize = data['cellsize']
    axis = data['axis']
    center_x = (axis[0] + axis[1]) / 2
    center_y = (axis[2] + axis[3]) / 2
    
    # Plot 1: Top-down view of terrain vertices
    if terrain is not None:
        verts = terrain.vertices
        ax1.scatter(verts[:, 0], verts[:, 1], c=verts[:, 2], 
                    cmap='terrain', s=0.1, alpha=0.5)
    
    # Overlay building footprints (Actual Mesh Vertices)
    all_bldg_verts = []
    for bldg in buildings:
        all_bldg_verts.append(bldg.vertices)
        
    if all_bldg_verts:
        # Combine all building vertices for efficient plotting
        combined_bldg_verts = np.vstack(all_bldg_verts)
        
        # Plot top-down view of building vertices
        # This accurately shows the irregular raster-based shapes rather than bounding boxes
        ax1.scatter(combined_bldg_verts[:, 0], combined_bldg_verts[:, 1], 
                    c='red', s=0.5, alpha=0.5, label='Sionna Mesh Vertices')
        
    # Standardize view
    ax1.set_xlim(data['dem'].shape[1] * cellsize / -2, data['dem'].shape[1] * cellsize / 2)
    ax1.set_ylim(data['dem'].shape[0] * cellsize / -2, data['dem'].shape[0] * cellsize / 2)
    ax1.set_title('Sionna RT: Mesh Footprints (Top-Down)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X (local meters)')
    ax1.set_ylabel('Y (local meters)')
    ax1.set_aspect('equal')
    
    # Plot 2: Building heights histogram comparison
    tirem_heights = data['buildings'][data['buildings'] > 1.0].flatten()
    
    sionna_heights = []
    for bldg in buildings:
        verts = bldg.vertices
        z_vals = np.unique(np.round(verts[:, 2], 1))
        if len(z_vals) >= 2:
            sionna_heights.append(z_vals[-1] - z_vals[0])
    
    ax2.hist(tirem_heights, bins=30, alpha=0.6, label=f'TIREM ({len(tirem_heights)} cells)', color='blue')
    if sionna_heights:
        ax2.hist(sionna_heights, bins=30, alpha=0.6, label=f'Sionna ({len(sionna_heights)} buildings)', color='red')
    
    ax2.set_title('Building Height Distribution Comparison', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Building Height (m)')
    ax2.set_ylabel('Count')
    ax2.legend()


def plot_3d_comparison(data, meshes, downsample=5):
    """Create 3D visualization comparing both representations."""
    fig = plt.figure(figsize=(16, 7))
    
    # TIREM 3D surface
    ax1 = fig.add_subplot(121, projection='3d')
    
    dem = data['dem'][::downsample, ::downsample]
    combined = data['combined'][::downsample, ::downsample]
    buildings = data['buildings'][::downsample, ::downsample]
    
    height, width = dem.shape
    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)
    
    # Plot terrain
    ax1.plot_surface(X, Y, dem, cmap='terrain', alpha=0.7, 
                     linewidth=0, antialiased=False)
    
    # Plot buildings as vertical bars at each cell with buildings
    bldg_mask = buildings > 1.0
    for i in range(height):
        for j in range(width):
            if bldg_mask[i, j]:
                z_base = dem[i, j]
                z_top = combined[i, j]
                ax1.bar3d(j, i, z_base, 0.8, 0.8, buildings[i, j], 
                          color='red', alpha=0.7)
    
    ax1.set_title('TIREM: Raster Grid Representation', fontsize=11, fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Elevation (m)')
    
    # Sionna 3D mesh
    ax2 = fig.add_subplot(122, projection='3d')
    
    if meshes is not None and meshes['terrain'] is not None:
        terrain = meshes['terrain']
        verts = terrain.vertices
        
        # Sample terrain vertices for faster plotting
        sample_idx = np.random.choice(len(verts), min(10000, len(verts)), replace=False)
        sampled = verts[sample_idx]
        
        ax2.scatter(sampled[:, 0], sampled[:, 1], sampled[:, 2], 
                    c=sampled[:, 2], cmap='terrain', s=0.5, alpha=0.3, label='Terrain')
        
        # Plot buildings - Gather ALL vertices
        all_bldg_verts = []
        for bldg in meshes['buildings']:
            all_bldg_verts.append(bldg.vertices)
            
        if all_bldg_verts:
            bldg_verts = np.vstack(all_bldg_verts)
            
            # Downsample if too many points for matplotlib
            if len(bldg_verts) > 50000:
                idx = np.random.choice(len(bldg_verts), 50000, replace=False)
                bldg_verts = bldg_verts[idx]
                
            ax2.scatter(bldg_verts[:, 0], bldg_verts[:, 1], bldg_verts[:, 2], 
                        c='red', s=0.5, alpha=0.5, label='Buildings')
        
        ax2.set_title('Sionna RT: Triangulated Mesh (Point Cloud)', fontsize=11, fontweight='bold')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_zlabel('Z (m)')
        # Force same z limits as TIREM for comparison
        ax2.set_zlim(ax1.get_zlim())
    else:
        ax2.text2D(0.5, 0.5, 'Sionna meshes not available', 
                   ha='center', va='center', transform=ax2.transAxes)
    
    plt.tight_layout()
    return fig


def print_statistics(data, meshes):
    """Print statistical comparison."""
    print("\n" + "="*60)
    print("GEOMETRY STATISTICS COMPARISON")
    print("="*60)
    
    dem = data['dem']
    buildings = data['buildings']
    
    print(f"\n{'TIREM (Raster):':<30}")
    print(f"  Grid size: {dem.shape[0]} x {dem.shape[1]}")
    print(f"  Total cells: {dem.size:,}")
    print(f"  Building cells: {np.sum(buildings > 1.0):,}")
    print(f"  Terrain elevation: [{dem.min():.1f}, {dem.max():.1f}] m")
    print(f"  Building heights: [0, {buildings.max():.1f}] m")
    print(f"  Mean building height: {buildings[buildings > 1.0].mean():.1f} m")
    
    if meshes is not None:
        terrain = meshes['terrain']
        bldgs = meshes['buildings']
        
        print(f"\n{'Sionna RT (Mesh):':<30}")
        if terrain:
            print(f"  Terrain vertices: {len(terrain.vertices):,}")
            print(f"  Terrain faces: {len(terrain.faces):,}")
            print(f"  Terrain Z range: [{terrain.vertices[:, 2].min():.1f}, {terrain.vertices[:, 2].max():.1f}] m")
        
        print(f"  Building meshes: {len(bldgs)}")
        total_bldg_verts = sum(len(b.vertices) for b in bldgs)
        total_bldg_faces = sum(len(b.faces) for b in bldgs)
        print(f"  Total building vertices: {total_bldg_verts:,}")
        print(f"  Total building faces: {total_bldg_faces:,}")
    
    print("\n" + "="*60)


def main():
    # Paths
    slcmap_path = Path("SLCmap_5May2022.mat")
    sionna_scene_dir = Path("data/sionna_scenes")
    output_dir = Path("results/geometry_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not slcmap_path.exists():
        print(f"[ERROR] SLCMap not found: {slcmap_path}")
        return
    
    # Use same downsampling as scene generation
    downsample_factor = 10
    
    # Load data
    print("\n" + "="*60)
    print("LOADING GEOMETRY DATA")
    print("="*60 + "\n")
    
    data = load_slcmap(slcmap_path, downsample_factor=downsample_factor)
    meshes = load_sionna_meshes(sionna_scene_dir)
    
    # Print statistics
    print_statistics(data, meshes)
    
    # Create 2D comparison plots
    print("\n[INFO] Creating 2D comparison plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    plot_tirem_geometry(data, axes[0, 0], axes[0, 1])
    plot_sionna_geometry(meshes, data, axes[1, 0], axes[1, 1])
    
    plt.tight_layout()
    
    output_path = output_dir / "geometry_comparison_2d.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Saved 2D comparison to: {output_path}")
    plt.show()
    
    # Create 3D comparison
    if HAS_TRIMESH:
        print("\n[INFO] Creating 3D comparison plot...")
        fig_3d = plot_3d_comparison(data, meshes, downsample=2)
        
        output_path_3d = output_dir / "geometry_comparison_3d.png"
        plt.savefig(output_path_3d, dpi=150, bbox_inches='tight')
        print(f"[INFO] Saved 3D comparison to: {output_path_3d}")
        plt.show()
    
    print("\n[INFO] Comparison complete!")


if __name__ == "__main__":
    main()
