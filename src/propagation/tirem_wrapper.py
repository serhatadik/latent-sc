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
            self.map_path = str(map_path)
        else:
            self.map_path = map_file
            
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
        
        # Define cache directory
        CACHE_DIR = Path("../data/cache/tirem")
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
