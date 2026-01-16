"""
Validation module for sparse reconstruction.

This module provides tools to validate the reconstruction algorithm by comparing
predicted signal strengths against a large set of widespread observations.
"""

import os
import hashlib
import numpy as np
import yaml
from pathlib import Path
import time
import joblib

from src.utils import get_sensor_locations_array, load_monitoring_locations
from src.sparse_reconstruction.propagation_matrix import compute_propagation_matrix
from src.sparse_reconstruction import linear_to_dbm, dbm_to_linear

class ReconstructionValidator:
    """
    Validator class for evaluating reconstruction performance on a test set.
    """
    
    def __init__(self, map_data, validation_config_path, validation_data_dir):
        """
        Initialize validator.
        
        Parameters
        ----------
        map_data : dict
            Map data dictionary (must contain 'shape' and georeferencing info).
        validation_config_path : str or Path
            Path to the YAML file defining validation locations.
        validation_data_dir : str or Path
            Directory where processed validation data (powers.npy) is stored.
        """
        self.map_data = map_data
        self.val_config_path = Path(validation_config_path)
        self.val_data_dir = Path(validation_data_dir)
        
        # Load validation locations
        print(f"Loading validation locations from {self.val_config_path}...")
        self.locations_config = load_monitoring_locations(
            config_path=str(self.val_config_path), 
            map_data=map_data
        )
        self.val_points = get_sensor_locations_array(self.locations_config)
        self.val_names = [loc['name'] for loc in self.locations_config['data_points']]
        
        print(f"Loaded {len(self.val_points)} validation points.")
        
        self.observed_powers_dBm = None
        self.prop_matrix = None
        self.prop_matrix_model_type = None

    def load_observed_data(self, file_prefix):
        """
        Load observed power data for the validation set.
        
        Parameters
        ----------
        file_prefix : str
            Prefix of the data files, e.g., 'validation_mario'.
            The loader looks for '{file_prefix}_avg_powers.npy'.
            
        Returns
        -------
        ndarray
            Observed powers in dBm.
        """
        p = self.val_data_dir / f"{file_prefix}_avg_powers.npy"
        if not p.exists():
            raise FileNotFoundError(f"Validation data not found: {p}")
            
        self.observed_powers_dBm = np.load(p)
        print(f"Loaded observed powers: {self.observed_powers_dBm.shape} samples")
        return self.observed_powers_dBm

    def get_propagation_matrix(self, model_type, model_config_path, scale=1.0, 
                             cache_dir='../data/cache', n_jobs=-1, verbose=True):
        """
        Get or compute the propagation matrix (A_val) for the validation set.
        
        This matrix maps the full grid (N) to the validation points (M_val).
        Size: (M_val, N).
        
        Parameters
        ----------
        model_type : str
            'tirem' or 'raytracing' or 'log_distance'.
        model_config_path : str
            Path to propagation model config.
        scale : float
            Pixel-to-meter scale.
        cache_dir : str
            Directory to store cached matrices.
            
        Returns
        -------
        ndarray
            Propagation matrix A_val.
        """
        # Ensure cache dir exists
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # Create a stable hash for the configuration
        # Hash includes: val_points coords, map_shape, model_type, config content
        hasher = hashlib.md5()
        hasher.update(self.val_points.tobytes())
        hasher.update(str(self.map_data['shape']).encode('utf-8'))
        hasher.update(str(scale).encode('utf-8'))
        hasher.update(model_type.encode('utf-8'))
        
        if model_config_path:
            with open(model_config_path, 'rb') as f:
                hasher.update(f.read())
        
        cache_key = hasher.hexdigest()
        cache_file = cache_path / f"prop_matrix_val_{cache_key}.npy"
        
        if cache_file.exists():
            if verbose:
                print(f"Loading cached propagation matrix from {cache_file}...")
            self.prop_matrix = np.load(cache_file)
            self.prop_matrix_model_type = model_type
            return self.prop_matrix
            
        if verbose:
            print(f"Computing propagation matrix for validation (M={len(self.val_points)}, N={np.prod(self.map_data['shape'])})...")
            print(f"  Model: {model_type}")
            print(f"  Config: {model_config_path}")
            
        start_time = time.time()
        
        # Compute matrix
        self.prop_matrix = compute_propagation_matrix(
            sensor_locations=self.val_points,
            map_shape=self.map_data['shape'],
            scale=scale,
            model_type=model_type,
            config_path=model_config_path,
            n_jobs=n_jobs,
            verbose=verbose
        )
        
        elapsed = time.time() - start_time
        if verbose:
            print(f"Computation finished in {elapsed:.2f}s. Saving to cache...")
            
        np.save(cache_file, self.prop_matrix)
        self.prop_matrix_model_type = model_type
        
        return self.prop_matrix

    def predict_rss(self, est_tx_map_linear):
        """
        Predict RSS at validation points given an estimated transmit power map.
        
        Parameters
        ----------
        est_tx_map_linear : ndarray
            Estimated transmit power map in linear scale (mW). Shape (H, W).
            
        Returns
        -------
        ndarray
            Predicted RSS in dBm.
        """
        if self.prop_matrix is None:
            raise RuntimeError("Propagation matrix not loaded. Call get_propagation_matrix() first.")
            
        # Linear prediction: p = A * t
        t_vec = est_tx_map_linear.ravel()
        rss_linear = self.prop_matrix @ t_vec
        
        return linear_to_dbm(rss_linear)

    def compute_metrics(self, predicted_dBm, verbose=True):
        """
        Compute validation metrics (MAE, RMSE) comparing predicted vs observed RSS.
        
        Parameters
        ----------
        predicted_dBm : ndarray
            Predicted RSS values (dBm).
            
        Returns
        -------
        dict
            Dictionary of metrics.
        """
        if self.observed_powers_dBm is None:
            raise RuntimeError("Observed data not loaded. Call load_observed_data() first.")
            
        # Error calculation
        error_dB = predicted_dBm - self.observed_powers_dBm
        mae = np.mean(np.abs(error_dB))
        rmse = np.sqrt(np.mean(error_dB**2))
        bias = np.mean(error_dB)
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'bias': bias,
            'n_samples': len(error_dB),
            'min_error': np.min(error_dB),
            'max_error': np.max(error_dB)
        }
        
        if verbose:
            print("\nValidation Metrics:")
            print(f"  MAE:  {mae:.2f} dB")
            print(f"  RMSE: {rmse:.2f} dB")
            print(f"  Bias: {bias:.2f} dB")
            
        return metrics

    def filter_out_of_bounds(self, verbose=True):
        """
        Filter out validation points that are outside the map extent.
        
        This should be called after load_observed_data().
        """
        if self.val_points is None:
            return
            
        height, width = self.map_data['shape']
        
        # val_points are [col, row]
        cols = self.val_points[:, 0]
        rows = self.val_points[:, 1]
        
        # Check bounds
        valid_mask = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
        
        n_total = len(self.val_points)
        n_valid = np.sum(valid_mask)
        n_removed = n_total - n_valid
        
        if n_removed > 0:
            if verbose:
                print(f"Filtering locations outside map extent:")
                print(f"  Original: {n_total}")
                print(f"  Valid:    {n_valid}")
                print(f"  Removed:  {n_removed}")
                
            self.val_points = self.val_points[valid_mask]
            self.val_names = [name for i, name in enumerate(self.val_names) if valid_mask[i]]
            
            if self.observed_powers_dBm is not None:
                self.observed_powers_dBm = self.observed_powers_dBm[valid_mask]
            
            # Reset prop matrix if it was computed with the old points (though unlikely if called in order)
            self.prop_matrix = None
        else:
            if verbose:
                print("All validation points are within map extent.")

    def plot_validation_map(self, ax=None, title="Validation Data"):
        """
        Visualize the extracted validation data on the map.
        
        Parameters
        ----------
        ax : matplotlib axis, optional
            Axis to draw on
        title : str, optional
            Plot title
            
        Returns
        -------
        ax : matplotlib axis
        """
        import matplotlib.pyplot as plt
        
        if self.val_points is None or self.observed_powers_dBm is None:
            print("No validation data to plot.")
            return

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Determine bounds
        height, width = self.map_data['shape']
        
        # Scatter plot of validation points
        # If val_points is [col, row], then x=col, y=row
        sc = ax.scatter(
            self.val_points[:, 0], 
            self.val_points[:, 1], 
            c=self.observed_powers_dBm, 
            cmap='viridis', 
            s=20, 
            alpha=0.8,
            edgecolors='k',
            linewidth=0.5
        )
        
        # Add colorbar
        cbar = plt.colorbar(sc, ax=ax, label='Observed RSS (dBm)')
        
        # Set limits
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        
        # Invert Y to match image coordinates usually
        # But wait, spatial_plots.py handles Y axis normally?
        # In spatial_plots.py: 
        # y = np.linspace(0, transmit_power_map.shape[0], ...)
        # Y grows upwards in plot unless inverted?
        # But image origin is top-left in matrices.
        # Let's check: plot_transmit_power_map uses meshgrid and contourf.
        # It does NOT call invert_yaxis(). 
        # This implies standard Cartesian behavior where Y=0 is bottom?
        # But pixels usually have (0,0) at top-left.
        # Actually, if meshgrid Y is 0..height, 0 is bottom.
        # If the map data (matrix) is oriented such that row 0 corresponds to Y=0 or Y=Height depends on loading.
        # Usually load_slc_map puts (0,0) at top-left.
        # If plot_transmit_power_map doesn't invert, maybe it plots upside down relative to image or just matches standard plot?
        # Let's trust standard scatter behavior (Y up) and see if it aligns with the 'shape' properly. 
        # If not, the user will see it flipped and we fix.
        # However, typically (col, row) -> (x, y) means row increases downwards in image.
        # To match map image layout, we usually invert y axis. 
        # I'll stick to mirroring spatial_plots behavior which is NO inversion explicitly seen for scatter.
        # BUT wait: spatial_plots relies on UTM ticks generation.
        # Let's assume standard behavior.
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # UTM Ticks if available
        if 'UTM_lat' in self.map_data and 'UTM_long' in self.map_data:
             UTM_lat = self.map_data['UTM_lat']
             UTM_long = self.map_data['UTM_long']
             
             # X ticks (Long/Easting)
             interval_x = max(1, len(UTM_long) // 5)
             ticks_x = list(range(0, len(UTM_long), interval_x))
             labels_x = [f"{UTM_long[i]:.0f}" for i in ticks_x]
             ax.set_xticks(ticks_x)
             ax.set_xticklabels(labels_x)
             ax.set_xlabel("UTM Easting (m)")

             # Y ticks (Lat/Northing)
             interval_y = max(1, len(UTM_lat) // 5)
             ticks_y = list(range(0, len(UTM_lat), interval_y))
             labels_y = [f"{UTM_lat[i]:.0f}" for i in ticks_y]
             ax.set_yticks(ticks_y)
             ax.set_yticklabels(labels_y)
             # Note: Typically UTM Northing increases upwards. 
             # If index 0 of UTM_lat corresponds to row 0, and UTM_lat increases or decreases?
             # Usually image row 0 is Top (Max Northing?).
             # If that's the case, Y axis 0->Height corresponds to Max->Min Northing?
             # Let's follow spatial_plots.py: 
             # tick_labels = ['{:.1f}'.format(lat) for lat in UTM_lat[::interval]]
             # plt.yticks(ticks=[0] + tick_values[1:]...)
             # It maps pixel indices directly to UTM array values.
             # So we should be consistent with that.
             
             ax.set_ylabel("UTM Northing (m)")
        
        ax.set_title(f"{title} ({len(self.val_points)} points)")
        
        return ax



