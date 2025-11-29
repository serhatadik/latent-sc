import numpy as np
from .base import PropagationModel
from ..utils.coordinates import euclidean_distance

def compute_linear_path_gain(distance, pi0=0, np_exponent=2, di0=1):
    """
    Compute linear path gain (not dB) from distance.
    """
    distance = np.array(distance, dtype=float)
    distance = np.maximum(distance, di0)  # Avoid log(0)

    # Path loss in dB
    path_loss_dB = pi0 + 10 * np_exponent * np.log10(distance / di0)

    # Convert to linear scale
    linear_gain = 10 ** (-path_loss_dB / 10)

    return linear_gain

class LogDistanceModel(PropagationModel):
    """
    Log-distance path loss model.
    """
    def __init__(self, np_exponent=2, pi0=0, di0=1, vectorized=True):
        self.np_exponent = np_exponent
        self.pi0 = pi0
        self.di0 = di0
        self.vectorized = vectorized

    def compute_propagation_matrix(self, sensor_locations, map_shape, scale=1.0, verbose=True):
        M = len(sensor_locations)
        height, width = map_shape
        N = height * width

        if verbose:
            print(f"Building propagation matrix (Log-Distance): {M} sensors × {N} grid points")
            print(f"Matrix size: {M}×{N} = {M*N:,} elements ({M*N*8/1e6:.1f} MB)")

        A_model = np.zeros((M, N), dtype=np.float64)

        if self.vectorized:
            # Vectorized computation
            rows_grid, cols_grid = np.mgrid[0:height, 0:width]
            rows_flat = rows_grid.ravel()
            cols_flat = cols_grid.ravel()

            for j, sensor in enumerate(sensor_locations):
                sensor_col, sensor_row = sensor

                dx = (cols_flat - sensor_col) * scale
                dy = (rows_flat - sensor_row) * scale
                distances = np.sqrt(dx**2 + dy**2)

                A_model[j, :] = compute_linear_path_gain(
                    distances, self.pi0, self.np_exponent, self.di0
                )

                if verbose and (j + 1) % max(1, M // 10) == 0:
                    print(f"  Processed {j+1}/{M} sensors...")
        else:
            # Loop-based computation
            for j, sensor in enumerate(sensor_locations):
                sensor_col, sensor_row = sensor
                for i in range(N):
                    row = i // width
                    col = i % width
                    grid_location = [col, row]
                    distance = euclidean_distance(grid_location, sensor, scale)
                    A_model[j, i] = compute_linear_path_gain(
                        distance, self.pi0, self.np_exponent, self.di0
                    )
                if verbose and (j + 1) % max(1, M // 10) == 0:
                    print(f"  Processed {j+1}/{M} sensors...")

        if verbose:
            print(f"Propagation matrix built successfully")
            print(f"  Min gain: {A_model.min():.2e}")
            print(f"  Max gain: {A_model.max():.2e}")
            print(f"  Mean gain: {A_model.mean():.2e}")

        return A_model
