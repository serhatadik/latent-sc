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
            
        self.freq = self.config.get('frequency_hz', 3534e6)
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
            
    def _ensure_scene(self, verbose: bool = True):
        """
        Ensure scene is loaded, generating from SLCMap if needed.
        """
        if self._scene_loaded:
            return
            
        scene_path = self.scene_cache_dir / self.scene_xml
        
        # Check if scene already exists
        if scene_path.exists():
            if verbose:
                print(f"[INFO] Loading cached scene from: {scene_path}")
            self.scene = self._load_scene(str(scene_path))
            
            # Load center coordinates from scene metadata
            meta_path = self.scene_cache_dir / "scene_meta.yaml"
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    meta = yaml.safe_load(f)
                    self.center_x = meta.get('center_x', 0)
                    self.center_y = meta.get('center_y', 0)
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
                    'center_x': self.center_x,
                    'center_y': self.center_y,
                    'cellsize': converter.cellsize,
                    'map_file': self.map_file
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
        verbose: bool = True
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
            
        Returns
        -------
        A : ndarray of shape (M, N)
            Path gain matrix in dB (negative values = path loss)
            A[m, n] = path gain from grid point n to sensor m
        """
        self._ensure_scene(verbose=verbose)
        
        import tensorflow as tf
        
        M = len(sensor_locations)
        height, width = map_shape
        N = height * width
        
        if verbose:
            print(f"[INFO] Computing propagation matrix: {M} sensors × {N} grid points")
            
        # Initialize output matrix
        A = np.full((M, N), -200.0)  # Initialize with very low gain (no signal)
        
        # Convert sensor locations to local coordinates
        # sensor_locations are in pixel space (col, row)
        # Need to convert to meters then to local coordinates
        sensor_x_utm = sensor_locations[:, 0] * scale  # col → x (easting)
        sensor_y_utm = sensor_locations[:, 1] * scale  # row → y (northing)
        sensor_x_local, sensor_y_local = self._utm_to_local(sensor_x_utm, sensor_y_utm)
        
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
                
                tx_x_utm = col * scale
                tx_y_utm = row * scale
                tx_x_local, tx_y_local = self._utm_to_local(tx_x_utm, tx_y_utm)
                
                # Find ground height at TX location (approximate from scene)
                tx_z = self.tx_height  # Use configured height above ground
                
                # Add transmitter
                tx = self._Transmitter(
                    name="tx_temp",
                    position=[tx_x_local, tx_y_local, tx_z],
                    orientation=[0, 0, 0]
                )
                self.scene.add(tx)
                
                try:
                    # Add all receivers
                    for rx_idx, (rx_x, rx_y) in enumerate(zip(sensor_x_local, sensor_y_local)):
                        rx = self._Receiver(
                            name=f"rx_{rx_idx}",
                            position=[rx_x, rx_y, self.rx_height],
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
                    path_gain_db = 10 * np.log10(path_power + 1e-30)
                    
                    # Store in matrix
                    A[:, tx_idx] = path_gain_db
                    
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
            print(f"  Mean path gain: {np.mean(A[A > -199]):.1f} dB")
            print(f"  Range: [{np.min(A[A > -199]):.1f}, {np.max(A):.1f}] dB")
            
        return A
