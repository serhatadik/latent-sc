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
from scipy.interpolate import RegularGridInterpolator
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
        if scene_path.exists() and meta_path.exists():
            if verbose:
                print(f"[INFO] Loading cached scene from: {scene_path}")
            self.scene = self._load_scene(str(scene_path))
            
            # Load center coordinates from scene metadata
            with open(meta_path, 'r') as f:
                meta = yaml.safe_load(f)
                self.center_x = meta.get('center_x', 0.0)
                self.center_y = meta.get('center_y', 0.0)
        else:
            # Generate scene from SLCMap
            if verbose:
                print(f"[INFO] Generating scene from SLCMap: {self.map_path}")
                
            from .slcmap_to_scene import SLCMapToScene
            
            converter = SLCMapToScene(
                self.map_path,
                output_dir=str(self.scene_cache_dir),
                downsample_factor=self.config.get('downsample_factor', 1)
            )
            scene_path = converter.export_to_mitsuba_xml(self.scene_xml)
            
            # Save center coordinates
            self.center_x = converter.center_x
            self.center_y = converter.center_y
            
            meta_path = self.scene_cache_dir / "scene_meta.yaml"
            with open(meta_path, 'w') as f:
                yaml.dump({
                    'center_x': float(self.center_x),
                    'center_y': float(self.center_y),
                    'cellsize': float(converter.cellsize),
                    'map_file': str(self.map_file)
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
        
    def compute_propagation_matrix(
        self,
        sensor_locations: np.ndarray,
        map_shape: Tuple[int, int],
        scale: float = 1.0,
        n_jobs: int = -1,
        verbose: bool = True,
        method: str = 'coverage_map'
    ) -> np.ndarray:
        """
        Compute propagation matrix (path gains) using Sionna ray-tracing.
        
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
        method : str, optional
            Computation method: 'coverage_map' (default, fast) or 'raytracing' (slow, point-to-point)
            
        Returns
        -------
        A : ndarray of shape (M, N)
            Path gain matrix (linear power ratio)
            A[m, n] = path gain from grid point n to sensor m
        """
        self._ensure_scene(verbose=verbose)

        if method == 'coverage_map':
            return self._compute_via_coverage_map(sensor_locations, map_shape, scale, verbose)
        
        import tensorflow as tf
        
        M = len(sensor_locations)
        height, width = map_shape
        N = height * width
        
        if verbose:
            print(f"[INFO] Computing propagation matrix: {M} sensors × {N} grid points")
            
        # Initialize output matrix
        A = np.zeros((M, N), dtype=np.float64)  # Initialize with 0 gain
        
        # Convert sensor locations to local coordinates
        # sensor_locations are in pixel space (col, row)
        # We assume the grid is centered on the scene origin (0,0)
        # x_local = (col * scale) - (width_m / 2)
        # y_local = (row * scale) - (height_m / 2)
        
        width_m = width * scale
        height_m = height * scale
        
        sensor_x_local = (sensor_locations[:, 0] * scale) - (width_m / 2.0)
        sensor_y_local = (sensor_locations[:, 1] * scale) - (height_m / 2.0)
        
        # For each grid point, compute path gain to all sensors
        # This is computationally intensive - we'll batch by grid points
        batch_size = self.config.get('batch_size', 100)
        
        for grid_idx in range(0, N, batch_size):
            end_idx = min(grid_idx + batch_size, N)
            
            if verbose and grid_idx % (batch_size * 10) == 0:
                print(f"  Progress: {grid_idx}/{N} grid points ({100*grid_idx/N:.1f}%)")
                
            for tx_idx in range(grid_idx, end_idx):
                # Get grid point coordinates
                row = tx_idx // width
                col = tx_idx % width
                
                # Local coordinates centered
                tx_x_local = (col * scale) - (width_m / 2.0)
                tx_y_local = (row * scale) - (height_m / 2.0)
                
                # Find ground height at TX location
                tx_z_ground = self._get_terrain_elevation(tx_x_local, tx_y_local)
                tx_z = tx_z_ground + self.tx_height
                
                # Add transmitter
                tx = self._Transmitter(
                    name="tx_temp",
                    position=[tx_x_local, tx_y_local, tx_z],
                    orientation=[0, 0, 0]
                )
                self.scene.add(tx)
                
                try:
                    # Add all receivers (Batched)
                    # We compute RX heights once
                    rx_z_ground = self._get_terrain_elevation(sensor_x_local, sensor_y_local) # Vectorized
                    rx_z = rx_z_ground + self.rx_height
                    
                    for rx_idx in range(len(sensor_x_local)):
                        rx = self._Receiver(
                            name=f"rx_{rx_idx}",
                            position=[sensor_x_local[rx_idx], sensor_y_local[rx_idx], rx_z[rx_idx]],
                            orientation=[0, 0, 0]
                        )
                        self.scene.add(rx)
                        
                    # Compute paths
                    paths = self.scene.compute_paths(
                        max_depth=self.max_depth,
                        diffraction=self.diffraction,
                        num_samples=self.num_samples
                    )
                    
                    # Get channel impulse response
                    a, tau = paths.cir()
                    
                    # Compute path gain (sum of squared amplitudes)
                    # a shape: [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time]
                    path_power = tf.reduce_sum(tf.abs(a)**2, axis=(2, 4, 5, 6))  # Sum over antennas, paths, time
                    path_power = path_power.numpy().squeeze()  # [num_rx]
                    
                    # Convert to dB
                    # path_gain_db = 10 * np.log10(path_power + 1e-30)
                    
                    # Store in matrix (Linear)
                    A[:, tx_idx] = path_power
                    
                finally:
                    # Clean up receivers and transmitter
                    for rx_idx in range(M):
                        try:
                            self.scene.remove(f"rx_{rx_idx}")
                        except:
                            pass
                    try:
                        self.scene.remove("tx_temp")
                    except:
                        pass
                        
        if verbose:
            print(f"[INFO] Propagation matrix computation complete")
        if verbose:
            print(f"[INFO] Propagation matrix computation complete")
            print(f"  Mean path gain: {np.mean(A[A > 1e-20]):.2e}")
            print(f"  Range: [{np.min(A):.2e}, {np.max(A):.2e}]")
            
        return A

    def _compute_via_coverage_map(
        self,
        sensor_locations: np.ndarray,
        map_shape: Tuple[int, int],
        scale: float,
        verbose: bool = True
    ) -> np.ndarray:
        """
        Compute propagation matrix using reciprocity and vectorized 'Receiver Cloud'.
        
        This replaces the old 'coverage_map' method which assumes a flat plane.
        Instead, we use reciprocity:
        1. Place a Transmitter at the Sensor location (Reciprocity).
        2. Place a single Receiver object with multiple positions corresponding to the grid points
           adjusted for correct terrain height.
        3. Compute paths in one vectorized batch.
        """
        import tensorflow as tf
        
        M = len(sensor_locations)
        height, width = map_shape
        N = height * width
        
        if verbose:
            print(f"[INFO] Computing propagation matrix via Vectorized Reciprocity: {M} sensors × {N} grid points")
            
        # Initialize output matrix
        A = np.zeros((M, N), dtype=np.float64)
        
        # Convert sensor locations to local coordinates
        width_m = width * scale
        height_m = height * scale
        
        sensor_x_local = (sensor_locations[:, 0] * scale) - (width_m / 2.0)
        sensor_y_local = (sensor_locations[:, 1] * scale) - (height_m / 2.0)
        
        # Grid coordinates
        grid_rows = np.arange(height)
        grid_cols = np.arange(width)
        grid_x_local = (grid_cols * scale) - (width_m / 2.0) # 1D
        grid_y_local = (grid_rows * scale) - (height_m / 2.0) # 1D
        
        # Meshgrid (flattened)
        grid_X_local, grid_Y_local = np.meshgrid(grid_x_local, grid_y_local)
        grid_pts_x = grid_X_local.ravel()
        grid_pts_y = grid_Y_local.ravel()
        
        # Determine terrain height for all grid points
        if verbose:
            print("[INFO] querying terrain elevation for grid points...")
            
        # Batch terrain query to avoid memory issues if N is large?
        # 12k points is small. 1M points is large.
        # _get_terrain_elevation is efficient (numpy indexing).
        grid_pts_z_ground = self._get_terrain_elevation(grid_pts_x, grid_pts_y)
        
        # Grid points act as Transmitters in reality, so use tx_height (AGL)
        grid_pts_z = grid_pts_z_ground + self.tx_height
        
        # Combine into (N, 3) positions
        grid_positions = np.column_stack((grid_pts_x, grid_pts_y, grid_pts_z))
        
        # Determine sensor heights
        sensor_z_ground = self._get_terrain_elevation(sensor_x_local, sensor_y_local)
        # Sensors are Receivers in reality, so use rx_height. In Reciprocity, they are Tx.
        # But we must preserve their physical 3D location.
        sensor_z = sensor_z_ground + self.rx_height 
        
        # Add ONE Transmitter (Reciprocity: Sensor acts as Tx)
        try:
             self.scene.remove("tx_temp")
        except:
             pass
             
        tx = self._Transmitter(
            name="tx_temp",
            position=[0, 0, 0], 
            orientation=[0, 0, 0]
        )
        self.scene.add(tx)
        
        # Sionna cannot handle unlimited number of paths/receivers in one go safely on small GPUs.
        # We should batch strict limit. E.g. 500 grid points.
        GRID_BATCH_SIZE = self.config.get('grid_batch_size', 500)
        
        try:
            for i in range(M):
                if verbose:
                    print(f"  Processing sensor {i+1}/{M} (Reciprocity)")
                
                # Move transmitter to sensor location
                tx.position = [sensor_x_local[i], sensor_y_local[i], sensor_z[i]]
                
                # Create Receiver Object Pool
                # We reuse these objects to avoid scene.add/remove overhead and TF retracing
                rx_pool_names = []
                if verbose:
                    print(f"[INFO] initializing receiver pool of size {GRID_BATCH_SIZE}...")
                
                try:
                    for idx in range(GRID_BATCH_SIZE):
                        rx_name = f"rx_pool_{idx}"
                        rx_pool_names.append(rx_name)
                        try:
                            self.scene.remove(rx_name)
                        except:
                            pass
                        # Initial dummy position
                        rx = self._Receiver(
                            name=rx_name,
                            position=[0,0,-1000],
                            orientation=[0,0,0]
                        )
                        self.scene.add(rx)

                    # Loop over grid batches
                    for batch_start in range(0, N, GRID_BATCH_SIZE):
                        batch_end = min(batch_start + GRID_BATCH_SIZE, N)
                        curr_batch_size = batch_end - batch_start
                        
                        # Update positions for current batch
                        for idx in range(curr_batch_size):
                            # Map to corresponding receiver in pool
                            rx_name = rx_pool_names[idx]
                            # Update position directly
                            # Accessing object from scene is safer?
                            # self.scene.get(rx_name).position = ...
                            # Or just keep reference? 
                            # self.scene.get(rx_name) returns the object.
                            obj = self.scene.get(rx_name)
                            obj.position = grid_positions[batch_start + idx]
                            
                        # Move unused receivers in pool out of the way (if last batch is small)
                        if curr_batch_size < GRID_BATCH_SIZE:
                            for idx in range(curr_batch_size, GRID_BATCH_SIZE):
                                obj = self.scene.get(rx_pool_names[idx])
                                obj.position = [0,0,-1000]

                        # Compute paths
                        paths = self.scene.compute_paths(
                            max_depth=self.max_depth,
                            diffraction=self.diffraction,
                            num_samples=self.num_samples
                        )
                        
                        # Compute power
                        a, tau = paths.cir()
                        # Reduce
                        path_power = tf.reduce_sum(tf.abs(a)**2, axis=(-1, -2)) # Sum time, paths
                        path_power = tf.reduce_sum(path_power, axis=(-1, -2))   # Sum ant
                        path_power = path_power.numpy().flatten()
                        
                        # Store valid part
                        # path_power corresponds to ALL receivers in scene?
                        # No, path_power shape depends on scene receivers.
                        # We have 1 Tx and GRID_BATCH_SIZE Rx.
                        # So path_power should have GRID_BATCH_SIZE elements.
                        
                        if len(path_power) >= curr_batch_size:
                            A[i, batch_start:batch_end] = path_power[:curr_batch_size]
                        else:
                            if verbose:
                                print(f"[WARNING] Batch output mismatch: {len(path_power)} vs {curr_batch_size}")

                finally:
                    # Clean up pool
                    for rx_name in rx_pool_names:
                        try:
                            self.scene.remove(rx_name)
                        except:
                            pass
                        
        finally:
            try:
                self.scene.remove("tx_temp")
            except:
                pass
                
        if verbose:
            print(f"[INFO] Propagation matrix computation complete")
            print(f"  Mean path gain: {np.mean(A[A > 1e-20]):.2e}")
            
        return A
        height, width = map_shape
        N = height * width
        
        if verbose:
            print(f"[INFO] Computing propagation matrix via Coverage Map: {M} sensors × {N} grid points")
            
        # Initialize output matrix
        A = np.zeros((M, N), dtype=np.float64)
        
        # Convert sensor locations to local coordinates
        width_m = width * scale
        height_m = height * scale
        
        sensor_x_local = (sensor_locations[:, 0] * scale) - (width_m / 2.0)
        sensor_y_local = (sensor_locations[:, 1] * scale) - (height_m / 2.0)
        
        # Grid coordinates for interpolation
        # The reconstruction grid targets (where we want path gain values):
        grid_rows = np.arange(height)
        grid_cols = np.arange(width)
        
        # Local coordinates
        grid_x_local = (grid_cols * scale) - (width_m / 2.0)
        grid_y_local = (grid_rows * scale) - (height_m / 2.0)
        
        # Meshgrid for interpolation query points (flattened)
        # We need to query at every (col, row) pair
        # Note: RegularGridInterpolator expects (x, y) coordinates if defined that way
        
        # Let's define the target grid points in local coordinates
        # shape (N, 2)
        grid_X_local, grid_Y_local = np.meshgrid(
             grid_x_local,
             grid_y_local
        )
        target_points = np.column_stack((grid_X_local.ravel(), grid_Y_local.ravel()))
        
        # Add a temporary transmitter to be moved around
        # We'll re-use the same transmitter object
        
        # Ensure cleaning up any previous temp objects
        try:
             self.scene.remove("tx_temp")
        except:
             pass
             
        # Add a placeholder transmitter
        tx = self._Transmitter(
            name="tx_temp",
            position=[0, 0, self.rx_height], # Reciprocity: TX at RX height
            orientation=[0, 0, 0]
        )
        self.scene.add(tx)
        
        try:
            for i in range(M):
                if verbose:
                    print(f"  Processing sensor {i+1}/{M}")
                
                # Move transmitter to sensor location (Reciprocity)
                # Note: modifying position directly is efficient in Sionna
                tx.position = [sensor_x_local[i], sensor_y_local[i], self.rx_height]
                
                # Compute coverage map
                # cm_cell_size should match scale loosely, but doesn't have to be exact match to grid
                # Smaller cell size = better resolution but more memory
                # We'll use scale as cell size
                cm_cell_size = (scale, scale)
                
                cm = self.scene.coverage_map(
                    max_depth=self.max_depth,
                    diffraction=self.diffraction,
                    num_samples=self.num_samples,
                    cm_cell_size=cm_cell_size,
                    check_scene=False
                )
                
                # Coverage map data
                # cm.path_gain shape: [1, height_cm, width_cm]
                pg = cm.path_gain.numpy()[0]
                
                # Coverage map coordinates
                # cm.center is center of the map
                # cm.size is size of the map (meters)
                # cm.cell_centers contains coordinates
                
                cm_centers = cm.cell_centers.numpy() # [height_cm, width_cm, 3]
                # Extract X and Y coordinates (local)
                cm_X = cm_centers[:, :, 0]
                cm_Y = cm_centers[:, :, 1]
                
                # Create interpolator
                # RegularGridInterpolator requires 1D axes
                # Sionna Coverage Map is regular grid
                
                # Unique X and Y coordinates sorted
                # Note: Sionna cm X accounts for rows or cols? 
                # usually cm_centers varies X along cols and Y along rows or vice versa
                
                x_axis = np.unique(cm_X)
                y_axis = np.unique(cm_Y)
                
                # Check dimensions
                if pg.shape != (len(y_axis), len(x_axis)):
                    # Sometimes axes might be swapped or float errors
                    # Let's try to infer from shape
                    pass
                
                # Convert to dB
                # pg_db = 10 * np.log10(pg + 1e-30)
                
                # Interpolator
                # RegularGridInterpolator(points, values)
                # points: tuple of (y, x) or (x, y) depending on data layout
                # Sionna pg usually (y, x) (row, col)
                
                # WARNING: RegularGridInterpolator expects strictly increasing axes
                # Ensure they are sorted
                
                # Construct interpolator
                # We need to map target_points (local x, local y) to pg_db
                # If pg_db is (Y, X), interpolator expects (y, x)
                
                interp = RegularGridInterpolator((y_axis, x_axis), pg, bounds_error=False, fill_value=0.0)
                
                # Query points: (y, x)
                query_points = target_points[:, [1, 0]] # Swap to (y, x)
                
                interpolated_pg = interp(query_points)
                
                # Store in A
                A[i, :] = interpolated_pg
                
        finally:
            try:
                self.scene.remove("tx_temp")
            except:
                pass
                
        if verbose:
            print(f"[INFO] Propagation matrix computation complete")
            print(f"  Mean path gain: {np.mean(A[A > 1e-20]):.2e}")
            
        return A
