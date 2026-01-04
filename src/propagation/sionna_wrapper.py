"""
Sionna RT propagation model wrapper.

This module provides a Sionna ray-tracing based propagation model that
implements the same interface as TiremModel for use in joint_sparse_reconstruction.
"""

import os
import numpy as np
import yaml
from pathlib import Path
from typing import Optional, Tuple
from .base import PropagationModel

# Delay Sionna imports to allow the module to be loaded even without Sionna
_SIONNA_AVAILABLE = None


def _check_sionna():
    """Check if Sionna is available and import it."""
    global _SIONNA_AVAILABLE
    if _SIONNA_AVAILABLE is None:
        try:
            import sionna
            _SIONNA_AVAILABLE = True
        except ImportError:
            _SIONNA_AVAILABLE = False
    return _SIONNA_AVAILABLE


class SionnaModel(PropagationModel):
    """
    Sionna RT propagation model wrapper.
    
    Uses Sionna ray-tracing to compute path gains between transmitter
    locations and receiver (sensor) locations.
    
    Parameters
    ----------
    config_path : str or Path
        Path to sionna_parameters.yaml configuration file
        
    Attributes
    ----------
    scene : sionna.rt.Scene
        The loaded Sionna scene
    freq : float
        Frequency in Hz
    tx_height : float
        Transmitter height in meters
    rx_height : float
        Receiver height in meters
    """
    
    def __init__(self, config_path: str):
        """
        Initialize Sionna model from config file.
        
        Parameters
        ----------
        config_path : str or Path
            Path to sionna_parameters.yaml
        """
        if not _check_sionna():
            raise ImportError(
                "Sionna is required for ray-tracing propagation model. "
                "Install with: pip install sionna"
            )
            
        # Import Sionna modules
        from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray
        self._load_scene = load_scene
        self._Transmitter = Transmitter
        self._Receiver = Receiver
        self._PlanarArray = PlanarArray
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.freq = float(self.config.get('frequency_hz', 3534e6))
        self.tx_height = self.config.get('transmitter_height_m', 8.8)
        self.rx_height = self.config.get('receiver_height_m', 2.0)
        self.max_depth = self.config.get('max_depth', 5)
        self.diffraction = self.config.get('diffraction', True)
        self.num_samples = self.config.get('num_samples', int(1e6))
        
        # Scene paths
        self.scene_cache_dir = Path(self.config.get('scene_cache_dir', './data/sionna_scenes'))
        self.scene_xml = self.config.get('scene_xml', 'scene.xml')
        self.map_file = self.config.get('map_file', 'SLCmap_5May2022.mat')
        
        # Scene holder
        self.scene = None
        self._scene_loaded = False
        
        # Center coordinates (set when scene is loaded)
        self.center_x = None
        self.center_y = None
        
        # Resolve paths
        config_dir = Path(config_path).parent
        if not os.path.isabs(self.map_file):
            map_path = config_dir.parent / self.map_file
            if not map_path.exists():
                map_path = Path.cwd() / self.map_file
            self.map_path = str(map_path.resolve())
        else:
            self.map_path = self.map_file
            
        self._load_elevation_map()

    def _load_elevation_map(self):
        """Load the SLC map elevation data for ground height lookup."""
        import scipy.io as sio
        
        if not os.path.exists(self.map_path):
             print(f"[WARNING] Map file not found: {self.map_path}. Z-heights will be flat 0.")
             self.elev_map = None
             return
             
        try:
            mat = sio.loadmat(self.map_path)
            SLC = mat['SLC'][0][0]
            column_map = {name: i for i, name in enumerate(SLC.dtype.names)}
            
            # Extract elevation (DEM + Buildings) matching TIREM
            # Check fields
            if 'dem' in column_map and 'hybrid_bldg' in column_map:
                dem = SLC[column_map['dem']].astype(np.float32)
                bldg = SLC[column_map['hybrid_bldg']].astype(np.float32) * 0.3048 # Feet to meters
                self.elev_map = dem + bldg
            elif 'data' in column_map:
                 self.elev_map = SLC[column_map['data']].astype(np.float32)
            else:
                self.elev_map = None
                
            # Get dimensions and resolution
            if 'cellsize' in column_map:
                self.map_cellsize = float(SLC[column_map['cellsize']][0][0])
            else:
                self.map_cellsize = 0.5 # Default
                
            # Bounds
            if 'axis' in column_map:
                 self.map_axis = SLC[column_map['axis']].flatten() # [min_x, max_x, min_y, max_y]
                 self.map_min_x = self.map_axis[0]
                 self.map_max_y = self.map_axis[3] # Top-left origin usage often implies we need top-left?
                 # Actually SLC map arrays are usually (height, width).
                 # TIREM wrapper logic:
                 # row = (max_y - y) / cellsize ?? OR row = (y - min_y) / cellsize?
                 # Need to verify coordinate system.
                 # SLCMapToScene uses:
                 # easting = axis[0] + col * cellsize
                 # northing = axis[2] + row * cellsize (so row 0 is min_y? or max_y?)
                 # axis is [e_min, e_max, n_min, n_max]
                 
                 # In SLCMapToScene:
                 # rows, cols = meshgrid(arange(height), arange(width))
                 # y = axis[2] + rows * cellsize
                 # This implies row 0 is at n_min (bottom).
                 # But usually images are row 0 top.
                 # Let's trust SLCMapToScene logic: row increases Y.
                 pass
                 
        except Exception as e:
            print(f"[ERROR] Failed to load elevation map: {e}")
            self.elev_map = None

    def _get_terrain_elevation(self, x_local, y_local):
        """
        Get terrain elevation at local coordinates (vectorized).
        x_local, y_local: arrays of local coordinates (meters from scene center)
        """
        if self.elev_map is None or self.center_x is None:
            return np.zeros_like(x_local)
            
        # Convert to global UTM
        x_utm = x_local + self.center_x
        y_utm = y_local + self.center_y
        
        # Convert to grid indices
        # From SLCMapToScene: 
        # easting = axis[0] + col * cellsize -> col = (easting - axis[0]) / cellsize
        # northing = axis[2] + row * cellsize -> row = (northing - axis[2]) / cellsize
        
        # We need self.map_axis to be available. It is loaded in _load_elevation_map.
        # But we need to ensure map_axis matches the one used for scene generation.
        # Ideally, we used the same map.
        
        if not hasattr(self, 'map_axis'):
            return np.zeros_like(x_local)
            
        col = (x_utm - self.map_axis[0]) / self.map_cellsize
        row = (y_utm - self.map_axis[2]) / self.map_cellsize
        
        # Round to nearest integer
        col_idx = np.rint(col).astype(np.int32)
        row_idx = np.rint(row).astype(np.int32)
        
        # Clip
        h, w = self.elev_map.shape
        col_idx = np.clip(col_idx, 0, w - 1)
        row_idx = np.clip(row_idx, 0, h - 1)
        
        return self.elev_map[row_idx, col_idx]
            
    def _ensure_scene(self, verbose: bool = True):
        """
        Ensure scene is loaded, generating from SLCMap if needed.
        """
        if self._scene_loaded:
            return
            
        scene_path = self.scene_cache_dir / self.scene_xml
        meta_path = self.scene_cache_dir / "scene_meta.yaml"
        
        # Check if scene and metadata already exist
        should_generate = True # FORCE REGENERATION for debug
        # if scene_path.exists() and meta_path.exists():
        if False: # Disabled cache check

            try:
                with open(meta_path, 'r') as f:
                    meta = yaml.safe_load(f)
                    self.center_x = meta.get('center_x', 0.0)
                    self.center_y = meta.get('center_y', 0.0)
                    self.scene_rows = meta.get('rows')
                    self.scene_cols = meta.get('cols')
                    
                # If rows/cols missing (old meta), force regeneration
                if self.scene_rows is not None and self.scene_cols is not None:
                    if verbose:
                        print(f"[INFO] Loading cached scene from: {scene_path}")
                    self.scene = self._load_scene(str(scene_path))
                    should_generate = False
                else:
                    if verbose:
                        print(f"[INFO] Cached scene metadata incomplete. Regenerating...")
            except Exception as e:
                if verbose:
                    print(f"[WARNING] Failed to load scene metadata: {e}. Regenerating...")
        
        if should_generate:
            # Generate scene from SLCMap
            if verbose:
                print(f"[INFO] Generating scene from SLCMap: {self.map_path}")
                
            from .slcmap_to_scene import SLCMapToScene
            
            # Ensure using combined mesh for RadioMapSolver compatibility
            # We need one "terrain" object that includes everything
            
            converter = SLCMapToScene(
                self.map_path,
                output_dir=str(self.scene_cache_dir),
                downsample_factor=self.config.get('downsample_factor', 1)
            )
            
            # Export with separate_buildings=False to create one combined "terrain" mesh
            scene_path = converter.export_to_mitsuba_xml(self.scene_xml, separate_buildings=False)
            
            # Save center coordinates and dimensions
            self.center_x = converter.center_x
            self.center_y = converter.center_y
            self.scene_rows = converter.dem.shape[0]
            self.scene_cols = converter.dem.shape[1]
            if verbose:
                print(f"[DEBUG] Generated scene rows/cols from dem: {self.scene_rows}x{self.scene_cols}")
            
            meta_path = self.scene_cache_dir / "scene_meta.yaml"
            with open(meta_path, 'w') as f:
                yaml.dump({
                    'center_x': float(self.center_x),
                    'center_y': float(self.center_y),
                    'cellsize': float(converter.cellsize),
                    'map_file': str(self.map_file),
                    'rows': int(self.scene_rows),
                    'cols': int(self.scene_cols)
                }, f)
                
            # Load the generated scene
            self.scene = self._load_scene(scene_path)
            
        # Configure scene
        self.scene.frequency = self.freq
        self.scene.tx_array = self._PlanarArray(
            num_rows=1, num_cols=1,
            vertical_spacing=0.5, horizontal_spacing=0.5,
            pattern="iso", polarization="V"
        )
        self.scene.rx_array = self._PlanarArray(
            num_rows=1, num_cols=1,
            vertical_spacing=0.5, horizontal_spacing=0.5,
            pattern="iso", polarization="V"
        )
        
        self._scene_loaded = True
        if verbose:
            print(f"[INFO] Scene loaded with {len(self.scene._scene_objects)} objects")
            
    def _utm_to_local(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert UTM coordinates to local scene coordinates."""
        return x - self.center_x, y - self.center_y
        
    def _local_to_utm(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert local scene coordinates to UTM."""
        return x + self.center_x, y + self.center_y

    def _get_vertex_mapping_data(self):
        """
        Pre-compute face-to-vertex indices for the regular grid mesh.
        Matches logic in SLCMapToScene.generate_terrain_mesh.
        """
        rows = self.scene_rows
        cols = self.scene_cols
        
        # Number of quads
        h_quads = rows - 1
        w_quads = cols - 1
        n_faces = h_quads * w_quads * 2
        
        # Generate faces array (matches slcmap_to_scene)
        # Quad indices
        r_idx = np.arange(h_quads)
        c_idx = np.arange(w_quads)
        R, C = np.meshgrid(r_idx, c_idx, indexing='ij')
        
        idx00 = R * cols + C
        idx01 = R * cols + (C + 1)
        idx10 = (R + 1) * cols + C
        idx11 = (R + 1) * cols + (C + 1)
        
        # Flatten (assuming row-major flattening matches meshgrid indexing='ij')
        # meshgrid(ij) -> R varies slowly (rows), C varies quickly (cols).
        # This is strictly consistent with nested loops: for r: for c:
        
        idx00 = idx00.ravel()
        idx01 = idx01.ravel()
        idx10 = idx10.ravel()
        idx11 = idx11.ravel()
        
        # Triangle 1: 00, 10, 01
        t1 = np.column_stack([idx00, idx10, idx01])
        # Triangle 2: 10, 11, 01
        t2 = np.column_stack([idx10, idx11, idx01])
        
        # Interleave to match likely generation order: for i, for j: append T1, append T2.
        # This implies: Face 0, Face 1, Face 2, Face 3...
        # So we construct faces array of shape (N, 3)
        # 0::2 is T1, 1::2 is T2
        
        faces = np.empty((n_faces, 3), dtype=np.int32)
        faces[0::2] = t1
        faces[1::2] = t2
        
        # Vertex counts for averaging
        vertex_counts = np.zeros(rows * cols, dtype=np.float32)
        
        # We need counts of how many faces touch each vertex
        # Just compute it once
        # faces.ravel() contains all vertex indices used in all faces
        np.add.at(vertex_counts, faces.ravel(), 1)
        
        return faces, vertex_counts
        
    def compute_propagation_matrix(
        self,
        sensor_locations: np.ndarray,
        map_shape: Tuple[int, int],
        scale: float = 1.0,
        n_jobs: int = -1,
        verbose: bool = True
    ) -> np.ndarray:
        """
        Compute propagation matrix (path gains) using Sionna RadioMapSolver.
        
        Uses Terrain-Aware Vectorized Reciprocity via RadioMapSolver:
        1. Place a Transmitter at the Sensor location (Reciprocity).
        2. Compute coverage on the entire terrain mesh.
        3. Map mesh face values to grid vertices.
        4. Interpolate to requested map geometry.
        
        Parameters
        ----------
        sensor_locations : ndarray of shape (M, 2)
            Sensor coordinates in pixel space (col, row)
        map_shape : tuple of (height, width)
            Shape of the reconstruction grid
        scale : float
            Pixel-to-meter scaling factor
        n_jobs : int
            Number of parallel jobs (currently not used - Sionna uses GPU)
        verbose : bool
            Print progress information
            
        Returns
        -------
        A : ndarray of shape (M, N)
            Path gain matrix (linear power ratio)
        """
        import hashlib
        import json
        from scipy import ndimage
        
        # Define cache directory - use absolute path based on this file's location
        _THIS_DIR = Path(__file__).parent.resolve()
        _PROJECT_ROOT = _THIS_DIR.parent.parent
        CACHE_DIR = _PROJECT_ROOT / "data" / "cache" / "sionna"
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Create a unique cache key based on all parameters that affect the result
        cache_params = {
            'sensor_locations': sensor_locations.tolist() if isinstance(sensor_locations, np.ndarray) else sensor_locations,
            'map_shape': list(map_shape),
            'scale': float(scale),
            'sionna_config': self.config,
            'map_path': str(self.map_path)
        }
        
        cache_string = json.dumps(cache_params, sort_keys=True)
        cache_hash = hashlib.md5(cache_string.encode('utf-8')).hexdigest()
        cache_file = CACHE_DIR / f"sionna_prop_matrix_{cache_hash}.npy"
        
        if cache_file.exists():
            if verbose:
                print(f"[INFO] Loading cached propagation matrix from: {cache_file}")
            try:
                return np.load(cache_file)
            except Exception as e:
                print(f"[WARNING] Failed to load cache: {e}. Recomputing...")

        self._ensure_scene(verbose=verbose)
        
        # Import RadioMapSolver here
        from sionna.rt import RadioMapSolver
        
        M = len(sensor_locations)
        height_query, width_query = map_shape
        N_query = height_query * width_query
        
        if verbose:
            print(f"[INFO] Computing propagation matrix via RadioMapSolver")
            print(f"  Sensors: {M}")
            print(f"  Scene Grid: {self.scene_rows}x{self.scene_cols}")
            print(f"  Target Grid: {height_query}x{width_query}")
            
        # 1. Initialize Solver
        solver = RadioMapSolver()
        
        # Get the terrain mesh object
        # Sionna may rename objects - check multiple possible keys
        terrain_obj = None
        possible_keys = ['terrain', 'merged-shapes']  # Sionna often uses 'merged-shapes'
        
        for key in possible_keys:
            if key in self.scene.objects:
                terrain_obj = self.scene.objects[key]
                if verbose:
                    print(f"[DEBUG] Found terrain as '{key}'")
                break
        
        if terrain_obj is None:
            # Fallback: use first object in scene
            if len(self.scene.objects) > 0:
                key = list(self.scene.objects.keys())[0]
                terrain_obj = self.scene.objects[key]
                if verbose:
                    print(f"[DEBUG] Using first object as terrain: '{key}'")
            else:
                raise ValueError(f"No objects found in scene. Keys: {list(self.scene.objects.keys())}")
        
        # Clone as mesh for proper RadioMapSolver usage
        # This is the official Sionna approach per their tutorials
        terrain_mesh = terrain_obj.clone(as_mesh=True)
        
        # CRITICAL: Elevate the measurement surface above the terrain!
        # If measurement surface is at same position as terrain, rays hit terrain
        # but never "hit" the measurement surface. 
        # Elevate by receiver height to measure signal at receiver level.
        from sionna.rt import transform_mesh
        transform_mesh(terrain_mesh, translation=[0, 0, self.rx_height])
        
        if verbose:
            if hasattr(terrain_mesh, 'face_count'):
                print(f"[DEBUG] Terrain mesh face count: {terrain_mesh.face_count()}")
            print(f"[DEBUG] Measurement surface elevated by {self.rx_height}m")



        # 2. Pre-compute face mapping
        if verbose:
            print("[INFO] Pre-computing mesh face-to-vertex mapping...")
        faces, vertex_counts = self._get_vertex_mapping_data()
        
        # 3. Setup Coordinate Mapping for Interpolation
        # We need to map query pixels (map_shape) to scene pixels (scene_rows, scene_cols)
        
        width_m = width_query * scale
        height_m = height_query * scale
        
        # Query coords in local meters (assuming centered)
        q_cols = np.arange(width_query)
        q_rows = np.arange(height_query)
        q_x_local = (q_cols * scale) - (width_m / 2.0)
        q_y_local = (q_rows * scale) - (height_m / 2.0)
        
        if not hasattr(self, 'map_axis'):
             self._load_elevation_map()
             
        meta_path = self.scene_cache_dir / "scene_meta.yaml"
        with open(meta_path, 'r') as f:
             meta = yaml.safe_load(f)
             scene_cellsize = meta.get('cellsize')
             
        # Coordinate transform function for interpolation
        # map_coordinates expects indices in [0, H-1], [0, W-1]
        
        def coords_to_scene_indices(x_local, y_local):
            x_utm = x_local + self.center_x
            y_utm = y_local + self.center_y
            # Matches generate_terrain_mesh logic:
            # x = axis[0] + col * cellsize - center_x -> col = (x + center_x - axis[0]) / cellsize
            # y = axis[2] + row * cellsize - center_y -> row = (y + center_x - axis[2]) / cellsize
            col = (x_utm - self.map_axis[0]) / scene_cellsize
            row = (y_utm - self.map_axis[2]) / scene_cellsize
            return row, col
            
        # Grid mesh for interpolation
        QX, QY = np.meshgrid(q_x_local, q_y_local) # QX is (H, W) array of x coords
        Q_rows, Q_cols = coords_to_scene_indices(QX, QY)
        
        # Stack for map_coordinates: (2, N_query) -> row_coords, col_coords
        interpolation_coords = np.stack([Q_rows.ravel(), Q_cols.ravel()])
        
        # 4. Compute
        A = np.zeros((M, N_query), dtype=np.float64)
        
        # Add temporary Transmitter
        try:
             self.scene.remove("tx_temp")
        except:
             pass
        tx = self._Transmitter(name="tx_temp", position=[0,0,0])
        self.scene.add(tx)
        
        try:
            for i in range(M):
                if verbose:
                    print(f"  Processing sensor {i+1}/{M}")
                    
                # Setup Transmitter at Sensor Location (Reciprocity)
                sx_pix, sy_pix = sensor_locations[i]
                sx_local = sx_pix * scale - width_m / 2.0
                sy_local = sy_pix * scale - height_m / 2.0
                
                # Use terrain height from our pre-loaded map? Or from scene?
                # Using _get_terrain_elevation (scalar) is consistent.
                sz_ground_arr = self._get_terrain_elevation(np.array([sx_local]), np.array([sy_local]))
                sz_ground = sz_ground_arr[0]
                sz = sz_ground + self.rx_height # Sensor height AGL
                
                tx.position = [float(sx_local), float(sy_local), float(sz)]
                
                # Compute coverage
                count_samples = self.config.get('num_samples', 10_000_000)
                
                # Use unified __call__ API with mesh argument
                # Note: confirmed RadioMapSolver defaults to 1m resolution if cell_size not passed.
                # Reverting to default (no cell_size).
                radio_map = solver(
                    self.scene, 
                    measurement_surface=terrain_mesh, 
                    samples_per_tx=count_samples,
                    diffraction=True, # Enable diffraction for terrain shadowing
                    max_depth=5 # Ensure sufficient bounces/diffraction
                )
                
                # radio_map.path_gain is [num_tx, num_cells].
                path_gain = radio_map.path_gain.numpy().flatten()
                
                if verbose:
                    print(f"[DEBUG] path_gain size: {path_gain.size}")
                    print(f"[DEBUG] path_gain range: [{path_gain.min():.2e}, {path_gain.max():.2e}]")
                    print(f"[DEBUG] path_gain non-zero: {np.sum(path_gain > 0)}")
                    print(f"[DEBUG] path_gain mean (where >0): {np.mean(path_gain[path_gain > 0]):.2e}" if np.any(path_gain > 0) else "[DEBUG] No non-zero path gains!")
                
                # Check for grid structure
                
                r = self.scene_rows
                c = self.scene_cols
                
                if path_gain.size == r * c:
                     # Match Vertices
                     if verbose:
                         print(f"[DEBUG] Matched vertex grid: {r}x{c}")
                     scene_grid = path_gain.reshape(r, c)
                     coord_offset = 0.0
                     
                elif path_gain.size == (r-1) * (c-1):
                     if verbose: 
                         print(f"[DEBUG] Detected Cell-based grid ({r-1}x{c-1}).")
                     # Try Row-Major (Height, Width) assuming standard Numpy order matches Sionna
                     scene_grid = path_gain.reshape(r-1, c-1)
                     coord_offset = -0.5
                     
                # Scenario 2: Half resolution grid (H/2 * W/2) - Common with some mesh settings
                # This check might need adjustment if it's (r-1)//2 ? 
                # Let's check exact match first.
                elif path_gain.size == (r // 2) * (c // 2):
                    if verbose:
                        print(f"[DEBUG] Detected half-resolution output ({r // 2}x{c // 2}). Upsampling.")
                        
                    # Revert flipud
                    low_res_grid = path_gain.reshape(r // 2, c // 2)
                    
                    # Upsample to full resolution
                    scene_grid = ndimage.zoom(low_res_grid, 2.0, order=1)
                    coord_offset = 0.0 # Approximation

                # Scenario 3: Face-based mapping (original plan)
                # Check if it matches face count
                elif path_gain.size == faces.shape[0]:
                     if verbose:
                         print(f"[DEBUG] Using face-based mapping: {path_gain.size} faces -> {r}x{c} vertices")
                     # Map to Vertices
                     vertex_gains = np.zeros(self.scene_rows * self.scene_cols, dtype=np.float64)
                     vals_expanded = np.repeat(path_gain, 3) 
                     indices_expanded = faces.ravel()
                     np.add.at(vertex_gains, indices_expanded, vals_expanded)
                     mask = vertex_counts > 0
                     vertex_gains[mask] /= vertex_counts[mask]
                     scene_grid = vertex_gains.reshape(self.scene_rows, self.scene_cols)
                     coord_offset = 0.0
                     
                     if verbose:
                         print(f"[DEBUG] scene_grid range after mapping: [{scene_grid.min():.2e}, {scene_grid.max():.2e}]")
                     
                else:
                    msg = f"Shape mismatch! path_gain: {path_gain.shape}, Expected Vertices: {r*c}, Cells: {(r-1)*(c-1)}, Half-Cells: {(r//2)*(c//2)}, Faces: {faces.shape[0]}"
                    raise RuntimeError(msg)

                # Interpolate to Query Grid
                # interpolation_coords are in Vertex Indices. 
                # If we have a Cell Grid, we shift coordinates by offset.
                
                interp_coords_shifted = interpolation_coords + coord_offset
                
                query_gains = ndimage.map_coordinates(
                    scene_grid, 
                    interp_coords_shifted, 
                    order=1, 
                    mode='nearest' # Extrapolate edges
                )
                
                A[i, :] = query_gains
                

        finally:
            try:
                self.scene.remove("tx_temp")
            except:
                pass
                
        if verbose:
            print(f"[INFO] Propagation matrix computation complete")
            print(f"  Mean path gain: {np.mean(A[A > 1e-20]):.2e}")
            
        # Save to cache
        if verbose:
            print(f"[INFO] Saving propagation matrix to cache: {cache_file}")
        try:
            np.save(cache_file, A)
        except Exception as e:
            print(f"[WARNING] Failed to save cache: {e}")
            
        return A
