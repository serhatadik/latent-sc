"""
Distance calculation utilities.
Extracted from visualization scripts.
"""
import numpy as np
from typing import List, Tuple, Dict
from geopy import distance
import json


def load_transmitter_locations(transmitters_file: str) -> Dict[str, Tuple[float, float]]:
    """
    Load transmitter locations from JSON file.

    Args:
        transmitters_file: Path to transmitters.json file

    Returns:
        Dictionary mapping transmitter names to (latitude, longitude) tuples
    """
    with open(transmitters_file, 'r') as f:
        transmitters_data = json.load(f)

    transmitters = {}
    for key, value in transmitters_data.items():
        transmitters[key] = (value['latitude'], value['longitude'])

    return transmitters


def calculate_distances(
    coordinates: np.ndarray,
    transmitter_location: Tuple[float, float]
) -> List[float]:
    """
    Calculate distances from multiple coordinates to a transmitter.

    Args:
        coordinates: Nx2 array of [longitude, latitude] coordinates
        transmitter_location: (latitude, longitude) of transmitter

    Returns:
        List of distances in meters
    """
    distances = []

    for coord in coordinates:
        # Convert [lon, lat] to (lat, lon) for distance calculation
        dist_m = distance.distance(tuple(coord[::-1]), transmitter_location).m
        distances.append(dist_m)

    return distances


def calculate_distances_to_all_transmitters(
    coordinates: np.ndarray,
    transmitters: Dict[str, Tuple[float, float]]
) -> Dict[str, List[float]]:
    """
    Calculate distances from coordinates to all transmitters.

    Args:
        coordinates: Nx2 array of [longitude, latitude] coordinates
        transmitters: Dictionary mapping transmitter names to (lat, lon) tuples

    Returns:
        Dictionary mapping transmitter names to lists of distances
    """
    all_distances = {}

    for tx_name, tx_location in transmitters.items():
        all_distances[tx_name] = calculate_distances(coordinates, tx_location)

    return all_distances
