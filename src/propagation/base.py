from abc import ABC, abstractmethod
import numpy as np

class PropagationModel(ABC):
    """
    Abstract base class for propagation models.
    """
    
    @abstractmethod
    def compute_propagation_matrix(self, sensor_locations, map_shape, scale=1.0, verbose=True):
        """
        Compute the propagation matrix A_model ∈ ℝ^(M×N).

        Parameters
        ----------
        sensor_locations : ndarray of shape (M, 2)
            Sensor coordinates in pixel space (col, row).
        map_shape : tuple of (height, width)
            Shape of the grid.
        scale : float, optional
            Scaling factor to convert pixels to meters, default: 1.0.
        verbose : bool, optional
            Print progress information.

        Returns
        -------
        A_model : ndarray of shape (M, N)
            Propagation matrix with linear path gains.
        """
        pass
