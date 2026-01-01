import numpy as np
import scipy.io as sio
import yaml
import os
from pathlib import Path
from .base import PropagationModel
from ..tirem.main_tirem_pred import call_tirem_loss, Params
from ..tirem.common import build_arrays
from joblib import Parallel, delayed, cpu_count

def _compute_chunk(indices, width, rx_loc, current_side_len, sampling_interval, current_elev_map, params):
    """
    Compute TIREM loss for a chunk of grid points.
    Helper function for parallel execution.
    """
    results = []
    for i in indices:
        row = i // width
        col = i % width
        tx_loc = np.array([col, row])
        
        # Skip if Tx == Rx (approximate check for coincident points)
        if np.linalg.norm(rx_loc - tx_loc) < 0.5:
            # Coincident points: set to a default high gain (0 path loss)
            results.append((i, 1.0))
            continue
            
        # Build profile
        # build_arrays expects 1-based indexing for the map
        d_array, e_array = build_arrays(
            current_side_len, 
            sampling_interval, 
            tx_loc + 1, 
            rx_loc + 1, 
            current_elev_map
        )
        
        # Calculate loss
        try:
            loss_db = call_tirem_loss(d_array, e_array, params)
            # Convert to linear gain: Gain = 10^(-Loss/10)
            linear_gain = 10 ** (-loss_db / 10.0)
            results.append((i, linear_gain))
        except Exception:
            # In case of error, set to 0
            results.append((i, 0.0))
            
    return results


class TiremModel(PropagationModel):
    """
    TIREM propagation model wrapper.
    """
    def __init__(self, config_path):
        """
        Initialize TIREM model from config file.
        
        Parameters
        ----------
        config_path : str or Path
            Path to tirem_parameters.yaml
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.freq = self.config.get('frequency_mhz', 3534)
        self.tx_height = self.config.get('transmitter_height_m', 8.8)
        self.rx_height = self.config.get('receiver_height_m', 2.0)
        self.polarz = self.config.get('polarization', 'H')
        self.conduc = self.config.get('conductivity', 0.028)
        self.permit = self.config.get('permittivity', 15.0)
        self.humid = self.config.get('humidity', 10.0)
        self.refrac = self.config.get('surface_refractivity', 301.0)
        
        map_file = self.config.get('map_file', 'SLCmap_5May2022.mat')
        # Resolve map path relative to config file or CWD
        if not os.path.isabs(map_file):
            # Try relative to config file location
            map_path = Path(config_path).parent.parent / map_file
            if not map_path.exists():
                # Try relative to CWD
                map_path = Path.cwd() / map_file
            # ALWAYS resolve to absolute path to ensure consistency across execution contexts
            self.map_path = str(map_path.resolve())
        else:
            self.map_path = str(Path(map_file).resolve())
            
        self._load_map()
        
    def _load_map(self):
        """Load the SLC map."""
        if not os.path.exists(self.map_path):
            raise FileNotFoundError(f"Map file not found: {self.map_path}")
            
        map_struct = sio.loadmat(self.map_path)['SLC']
        self.SLC = map_struct[0][0]
        self.column_map = dict(zip([name for name in self.SLC.dtype.names], [i for i in range(len(self.SLC.dtype.names))]))
        
        # Extract elevation map (DEM + Building heights)
        # Assuming fusion map structure as in main_tirem_pred.py
        # slc_map = SLC[column_map['dem']] + 0.3048 * SLC[column_map['hybrid_bldg']]
        # Check if fields exist
        if 'dem' in self.column_map and 'hybrid_bldg' in self.column_map:
            self.elev_map = self.SLC[self.column_map['dem']] + 0.3048 * self.SLC[self.column_map['hybrid_bldg']]
        elif 'data' in self.column_map:
             self.elev_map = self.SLC[self.column_map['data']]
        else:
            raise ValueError("Map does not contain expected elevation fields (dem+hybrid_bldg or data)")
            
        # Get side_len from map metadata if possible, else default
        # cellsize is usually in the map struct
        if 'cellsize' in self.column_map:
            self.side_len = float(self.SLC[self.column_map['cellsize']][0][0])
        else:
            self.side_len = 30.0 # Default fallback
            
    def compute_propagation_matrix(self, sensor_locations, map_shape, scale=1.0, n_jobs=-1, verbose=True):
        import hashlib
        import json
        
        # Define cache directory - use absolute path based on this file's location
        _THIS_DIR = Path(__file__).parent.resolve()
        _PROJECT_ROOT = _THIS_DIR.parent.parent
        CACHE_DIR = _PROJECT_ROOT / "data" / "cache" / "tirem"
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Create a unique cache key based on all parameters that affect the result
        # 1. Sensor locations
        # 2. Map shape and scale
        # 3. TIREM configuration (frequency, heights, etc.)
        # 4. Map file path (in case map changes)
        
        cache_params = {
            'sensor_locations': sensor_locations.tolist() if isinstance(sensor_locations, np.ndarray) else sensor_locations,
            'map_shape': list(map_shape),
            'scale': float(scale),
            'tirem_config': self.config,
            'map_path': str(self.map_path)
        }
        
        # Create a stable string representation for hashing
        cache_string = json.dumps(cache_params, sort_keys=True)
        cache_hash = hashlib.md5(cache_string.encode('utf-8')).hexdigest()
        
        cache_file = CACHE_DIR / f"tirem_prop_matrix_{cache_hash}.npy"
        
        # --- Legacy Fallback Logic ---
        # If the canonical (absolute path) cache doesn't exist, check for the legacy (relative path) cache
        # This handles the case where the notebook created a cache using a relative path string
        if not cache_file.exists():
            if verbose:
                print(f"Canonical cache not found. Checking legacy cache...")
            
            # Construct legacy map_path (simulating notebook environment)
            # Notebook uses '../config/tirem_parameters.yaml', so map path resolution
            # results in '..\SLCmap_5May2022.mat' (on Windows)
            # We assume the map file is in the parent of the config parent, or similar.
            # Heuristic: try to reconstruct the relative path string that likely generated the legacy cache
            try:
                # Assuming standard structure where map is in project root
                # and notebook runs from notebooks/
                legacy_map_path = str(Path('..') / Path(self.map_path).name)
                
                legacy_params = cache_params.copy()
                legacy_params['map_path'] = legacy_map_path
                
                legacy_string = json.dumps(legacy_params, sort_keys=True)
                legacy_hash = hashlib.md5(legacy_string.encode('utf-8')).hexdigest()
                legacy_file = CACHE_DIR / f"tirem_prop_matrix_{legacy_hash}.npy"
                
                if legacy_file.exists():
                    if verbose:
                        print(f"✓ Found legacy cache: {legacy_file}")
                        print(f"Migrating to canonical location: {cache_file}")
                    
                    try:
                        data = np.load(legacy_file)
                        # Save to canonical location for future use
                        np.save(cache_file, data)
                        return data
                    except Exception as e:
                        print(f"Failed to migrate legacy cache: {e}")
            except Exception as e:
                print(f"Error checking legacy cache: {e}")

        if cache_file.exists():
            if verbose:
                print(f"Loading cached propagation matrix from: {cache_file}")
            try:
                return np.load(cache_file)
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")
        
        M = len(sensor_locations)
        height, width = map_shape
        N = height * width
        
        if verbose:
            print(f"Building propagation matrix (TIREM): {M} sensors × {N} grid points")
            print(f"Matrix size: {M}×{N} = {M*N:,} elements")
            
        A_model = np.zeros((M, N), dtype=np.float64)
        
        # Create Params object (reused)
        # We only need to update specific fields if they change, but here they are constant
        # except maybe tx/rx locations which are passed to build_arrays, not params
        # Wait, call_tirem_loss uses params.txheight, rxheight, etc.
        # It does NOT use params.bs_x/y for the calculation, it uses d_array/e_array
        
        # Create a dummy Params object with necessary values
        params = Params(
            bs_endpoint_name="dummy",
            bs_is_tx=1, # Doesn't matter for call_tirem_loss
            txheight=self.tx_height,
            rxheight=self.rx_height,
            bs_lon=0, bs_lat=0, bs_x=0, bs_y=0, # Dummy
            freq=self.freq,
            polarz=self.polarz,
            generate_features=0,
            map_type="fusion",
            map_filedir=self.map_path,
            gain=0,
            first_call=0,
            extsn=0,
            refrac=self.refrac,
            conduc=self.conduc,
            permit=self.permit,
            humid=self.humid,
            side_len=scale, # Use scale as side_len in Params too, though mostly for metadata
            sampling_interval=0.5 # Default
        )
        
        sampling_interval = 0.5
        
        from scipy.ndimage import zoom
        
        # Resize elevation map to match the requested map_shape
        # This ensures the map aligns perfectly with the reconstruction grid
        orig_height, orig_width = self.elev_map.shape
        target_height, target_width = map_shape
        
        if (orig_height, orig_width) != (target_height, target_width):
            if verbose:
                print(f"Resizing TIREM map from {self.elev_map.shape} to {map_shape}...")
            
            zoom_factors = (target_height / orig_height, target_width / orig_width)
            # Use order=1 (bilinear) for elevation data
            current_elev_map = zoom(self.elev_map, zoom_factors, order=1)
        else:
            current_elev_map = self.elev_map

        # Use the provided scale as the side_len for TIREM
        # This overrides any internal map resolution
        current_side_len = scale
        
        # Loop over sensors (Rx)
        for j, sensor in enumerate(sensor_locations):
            rx_col, rx_row = sensor
            rx_loc = np.array([rx_col, rx_row])
            
            # Parallelize over grid points (Tx)
            # We chunk the grid indices to reduce overhead
            all_indices = np.arange(N)
            
            # Determine number of chunks
            # If n_jobs is -1, use all CPUs. 
            n_workers = n_jobs if n_jobs > 0 else cpu_count()
            # Create enough chunks to keep workers busy, but not too many to cause overhead
            n_chunks = max(n_workers * 4, 16)
            chunks = np.array_split(all_indices, n_chunks)
            
            if verbose and j == 0:
                print(f"  Parallelizing with {n_workers} workers over {n_chunks} chunks per sensor...")

            results_list = Parallel(n_jobs=n_jobs)(
                delayed(_compute_chunk)(
                    chunk, width, rx_loc, current_side_len, sampling_interval, current_elev_map, params
                ) for chunk in chunks
            )
            
            # Aggregate results
            for res in results_list:
                for idx, val in res:
                    A_model[j, idx] = val
                    
            if verbose and (j + 1) % max(1, M // 10) == 0:
                print(f"  Processed {j+1}/{M} sensors...")
        
        # Save to cache
        if verbose:
            print(f"Saving propagation matrix to cache: {cache_file}")
        try:
            np.save(cache_file, A_model)
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")
                
        return A_model

    def compute_geometric_features(self, sensor_locations, map_shape, scale=1.0, n_jobs=-1, verbose=True):
        """
        Compute geometric features for all pairs of (sensor, grid_point).
        
        Features:
        0. Check for LOS (binary)
        1. Shadowing Angle (degrees) - angle of shadow from obstacle closest to Rx (0 if LOS)
        2. Number of obstacles (count)
        3. Distance (meters)
        
        Returns
        -------
        features : ndarray of shape (M, N, 4)
        """
        import hashlib
        import json
        
        # Define cache directory - use absolute path based on this file's location
        _THIS_DIR = Path(__file__).parent.resolve()
        _PROJECT_ROOT = _THIS_DIR.parent.parent
        CACHE_DIR = _PROJECT_ROOT / "data" / "cache" / "tirem"
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        cache_params = {
            'sensor_locations': sensor_locations.tolist() if isinstance(sensor_locations, np.ndarray) else sensor_locations,
            'map_shape': list(map_shape),
            'scale': float(scale),
            'tirem_config': self.config,
            'map_path': str(self.map_path),
            'type': 'geometric_features'
        }
        
        cache_string = json.dumps(cache_params, sort_keys=True)
        cache_hash = hashlib.md5(cache_string.encode('utf-8')).hexdigest()
        cache_file = CACHE_DIR / f"tirem_features_{cache_hash}.npy"
        
        # --- Legacy Fallback Logic ---
        if not cache_file.exists():
            if verbose:
                print(f"Canonical cache not found. Checking legacy cache...")
            
            try:
                legacy_map_path = str(Path('..') / Path(self.map_path).name)
                
                legacy_params = cache_params.copy()
                legacy_params['map_path'] = legacy_map_path
                
                legacy_string = json.dumps(legacy_params, sort_keys=True)
                legacy_hash = hashlib.md5(legacy_string.encode('utf-8')).hexdigest()
                legacy_file = CACHE_DIR / f"tirem_features_{legacy_hash}.npy"
                
                if legacy_file.exists():
                    if verbose:
                        print(f"✓ Found legacy cache: {legacy_file}")
                        print(f"Migrating to canonical location: {cache_file}")
                    
                    try:
                        data = np.load(legacy_file)
                        np.save(cache_file, data)
                        return data
                    except Exception as e:
                        print(f"Failed to migrate legacy cache: {e}")
            except Exception as e:
                print(f"Error checking legacy cache: {e}")
        
        if cache_file.exists():
            if verbose:
                print(f"Loading cached feature tensor from: {cache_file}")
            try:
                return np.load(cache_file)
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")
        
        M = len(sensor_locations)
        height, width = map_shape
        N = height * width
        n_features = 4  # LOS, Elev, Obstacles, Dist
        
        if verbose:
            print(f"Computing geometric features: {M} sensors × {N} grid points × {n_features} features")
            
        features = np.zeros((M, N, n_features), dtype=np.float32)
        
        params = Params(
            bs_endpoint_name="dummy", bs_is_tx=1,
            txheight=self.tx_height, rxheight=self.rx_height,
            bs_lon=0, bs_lat=0, bs_x=0, bs_y=0,
            freq=self.freq, polarz=self.polarz, generate_features=0,
            map_type="fusion", map_filedir=self.map_path,
            gain=0, first_call=0, extsn=0,
            refrac=self.refrac, conduc=self.conduc, permit=self.permit, humid=self.humid,
            side_len=scale, sampling_interval=0.5
        )
        sampling_interval = 0.5
        
        from scipy.ndimage import zoom
        orig_height, orig_width = self.elev_map.shape
        target_height, target_width = map_shape
        if (orig_height, orig_width) != (target_height, target_width):
            if verbose:
                print(f"Resizing TIREM map from {self.elev_map.shape} to {map_shape}...")
            current_elev_map = zoom(self.elev_map, (target_height / orig_height, target_width / orig_width), order=1)
        else:
            current_elev_map = self.elev_map
            
        current_side_len = scale
        
        for j, sensor in enumerate(sensor_locations):
            rx_col, rx_row = sensor
            rx_loc = np.array([rx_col, rx_row])
            
            all_indices = np.arange(N)
            n_workers = n_jobs if n_jobs > 0 else cpu_count()
            # More computation per chunk for features, so slightly smaller chunks might be better?
            # Or same logic. Stick to same logic.
            n_chunks = max(n_workers * 4, 16)
            chunks = np.array_split(all_indices, n_chunks)
            
            if verbose and j == 0:
                print(f"  Parallelizing with {n_workers} workers...")
            
            results_list = Parallel(n_jobs=n_jobs)(
                delayed(_compute_features_chunk)(
                    chunk, width, rx_loc, current_side_len, sampling_interval, current_elev_map, params
                ) for chunk in chunks
            )
            
            for res in results_list:
                for idx, feats in res:
                    features[j, idx, :] = feats
                    
            if verbose and (j + 1) % max(1, M // 10) == 0:
                print(f"  Processed {j+1}/{M} sensors...")
                
        if verbose:
            print(f"Saving features to cache: {cache_file}")
        try:
            np.save(cache_file, features)
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")
            
        return features


def _compute_features_chunk(indices, width, rx_loc, current_side_len, sampling_interval, current_elev_map, params):
    """
    Compute geometric features for a chunk of grid points.
    Returns list of (index, [los, elevation, obstacles, distance])
    """
    import math
    results = []
    
    # Pre-extract params
    tx_height = params.txheight
    rx_height = params.rxheight
    
    for i in indices:
        row = i // width
        col = i % width
        tx_loc = np.array([col, row]) # Grid coordinates (0-indexed)
        
        # Distance check
        dist_px = np.linalg.norm(rx_loc - tx_loc)
        if dist_px < 0.5:
             # Coincident: LOS=1, Elev=0, Obs=0, Dist=0
             results.append((i, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)))
             continue
             
        # Build profile (1-based indexing for build_arrays)
        d_array, e_array = build_arrays(
            current_side_len, 
            sampling_interval, 
            tx_loc + 1, 
            rx_loc + 1, 
            current_elev_map
        )
        
        # Logic from main_tirem_pred.py
        # Filter non-zero elevation points if any? build_arrays returns full path.
        # It's better to use all points returned.
        
        # d_array is distances from Tx along path
        # e_array is terrain elevation
        
        if len(d_array) < 2:
            results.append((i, np.array([1.0, 0.0, 0.0, d_array[-1] if len(d_array)>0 else 0.0], dtype=np.float32)))
            continue

        # Heights
        # Careful: e_array includes terrain elevation. 
        # e_array[0] approx tx_elevation, e_array[-1] approx rx_elevation
        tx_terrain = e_array[0]
        rx_terrain = e_array[-1]
        
        tx_total_height = tx_terrain + tx_height
        rx_total_height = rx_terrain + rx_height
        
        total_dist = d_array[-1]
        if total_dist == 0:
             results.append((i, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)))
             continue

        # Slope for LOS line
        slope = (rx_total_height - tx_total_height) / total_dist
        
        # Obstacles
        num_obstacles = 0.0
        # Check all intermediate points
        # LOS line height at distance d: h(d) = tx_total_height + slope * d
        # Terrain height at distance d: e_array
        # Obstacle if h(d) < e_array[i]
        
        # Vectorized check
        los_line_heights = tx_total_height + slope * d_array
        # Typically endpoints are LOS (antenna above ground), so check internals
        # But allow endpoints to be checked if antenna is burried (unlikely)
        # main_tirem_pred checks range(len - 1)
        
        # obstacles = (los_line_heights < e_array)
        # num_obstacles = np.sum(obstacles)
        
        # Refined check (ignoring start/end or very close to it to avoid artifacts)
        # Just check everything strictly
        obstacles_mask = los_line_heights < (e_array - 0.01) # Small epsilon
        num_obstacles = np.sum(obstacles_mask)
        
        is_los = 1.0 if num_obstacles == 0 else 0.0
        
        # Shadowing Angle for the obstacle closest to the receiver
        # Reference: main_tirem_pred.py line 357
        # If LOS (no obstacles), shadowing_angle = 0
        shadowing_angle = 0.0
        if num_obstacles > 0:
            # Find the last obstacle (closest to Rx) by traversing backwards from Rx
            for k in range(len(d_array) - 2, 0, -1):  # Skip endpoints
                if obstacles_mask[k]:
                    # Found the last obstacle (closest to Rx)
                    ke_height = e_array[k]
                    ke_dist = d_array[k]
                    dist_ke_to_rx = total_dist - ke_dist
                    if dist_ke_to_rx > 0.01:
                        # Angle from Rx looking up to the knife edge top
                        angle_rx_to_ke = math.degrees(math.atan((ke_height - rx_total_height) / dist_ke_to_rx))
                        # Add the slope angle (original LOS direction)
                        shadowing_angle = angle_rx_to_ke + math.degrees(math.atan(slope))
                    break
        
        results.append((i, np.array([is_los, shadowing_angle, num_obstacles, total_dist], dtype=np.float32)))
        
    return results
