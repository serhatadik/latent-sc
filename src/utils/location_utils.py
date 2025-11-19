"""Utilities for loading and converting monitoring station locations."""

import yaml
import numpy as np
from .coordinates import latlon_to_pixel


def load_monitoring_locations(config_path, map_data):
    """
    Load monitoring station locations and convert to pixel coordinates.

    This function reads the monitoring_locations.yaml file which contains
    latitude/longitude coordinates and automatically converts them to
    pixel coordinates based on the SLC map's UTM extent and resolution.

    Parameters
    ----------
    config_path : str
        Path to monitoring_locations.yaml file
    map_data : dict
        Map data dictionary from load_slc_map() containing:
        - 'UTM_lat': Array of UTM northing values
        - 'UTM_long': Array of UTM easting values
        - 'shape': (height, width) of the map

    Returns
    -------
    dict
        Dictionary containing:
        - 'data_points': List of monitoring stations with pixel coordinates
        - 'utm_zone': UTM zone number
        - 'northern_hemisphere': Boolean indicating hemisphere

    Examples
    --------
    >>> map_data = load_slc_map("../", downsample_factor=10)
    >>> locations = load_monitoring_locations("config/monitoring_locations.yaml", map_data)
    >>> len(locations['data_points'])
    10
    >>> locations['data_points'][0]['name']
    'Bookstore'
    >>> locations['data_points'][0]['coordinates']
    [96, 299]

    Notes
    -----
    The input config file should contain lat/lon coordinates. This function
    automatically converts them to pixel coordinates using the map's UTM extent.
    """
    # Load YAML config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    utm_zone = config.get('utm_zone', 12)
    northern = config.get('northern_hemisphere', True)

    # Convert each location from lat/lon to pixel coordinates
    data_points = []

    for location in config['data_points']:
        name = location['name']
        lat = location['latitude']
        lon = location['longitude']

        # Convert to pixel coordinates
        col, row = latlon_to_pixel(lat, lon, map_data, utm_zone, northern)

        data_points.append({
            'name': name,
            'latitude': lat,
            'longitude': lon,
            'coordinates': [col, row]  # [x, y] format
        })

    return {
        'data_points': data_points,
        'utm_zone': utm_zone,
        'northern_hemisphere': northern
    }


def get_sensor_locations_array(locations_dict):
    """
    Extract sensor locations as a numpy array.

    Parameters
    ----------
    locations_dict : dict
        Dictionary from load_monitoring_locations()

    Returns
    -------
    ndarray
        Array of shape (n_sensors, 2) with [col, row] coordinates

    Examples
    --------
    >>> locations = load_monitoring_locations("config/monitoring_locations.yaml", map_data)
    >>> sensor_locs = get_sensor_locations_array(locations)
    >>> sensor_locs.shape
    (10, 2)
    """
    coords = [loc['coordinates'] for loc in locations_dict['data_points']]
    return np.array(coords)


def get_sensor_names(locations_dict):
    """
    Extract sensor names as a list.

    Parameters
    ----------
    locations_dict : dict
        Dictionary from load_monitoring_locations()

    Returns
    -------
    list of str
        List of sensor names

    Examples
    --------
    >>> locations = load_monitoring_locations("config/monitoring_locations.yaml", map_data)
    >>> names = get_sensor_names(locations)
    >>> names[0]
    'Bookstore'
    """
    return [loc['name'] for loc in locations_dict['data_points']]


def verify_coordinate_conversion(locations_dict, verbose=True):
    """
    Verify coordinate conversion by comparing to old pixel coordinates.

    This is useful for validating that the lat/lon to pixel conversion
    is working correctly by comparing to previously known pixel coordinates.

    Parameters
    ----------
    locations_dict : dict
        Dictionary from load_monitoring_locations()
    verbose : bool, optional
        If True, print comparison results (default: True)

    Returns
    -------
    list of dict
        List of dictionaries with conversion verification results

    Examples
    --------
    >>> locations = load_monitoring_locations("config/monitoring_locations.yaml", map_data)
    >>> results = verify_coordinate_conversion(locations)
    """
    # Reference coordinates from old config (for verification)
    old_coordinates = {
        'Bookstore': [96, 299],
        'EBC': [256, 376],
        'Guesthouse': [287, 344],
        'Moran': [262, 428],
        'WEB': [131, 383],
        'Sagepoint': [382, 266],
        'Law73': [24, 244],
        'Humanities': [171, 314],
        'Madsen': [285, 158],
        'Garage': [190, 239]
    }

    results = []

    if verbose:
        print("\n" + "="*80)
        print("COORDINATE CONVERSION VERIFICATION")
        print("="*80)
        print(f"{'Sensor':<15} {'Lat/Lon':<25} {'Old [x,y]':<15} {'New [x,y]':<15} {'Diff':<10}")
        print("-"*80)

    for location in locations_dict['data_points']:
        name = location['name']
        new_coords = location['coordinates']
        old_coords = old_coordinates.get(name, [None, None])

        lat_lon_str = f"({location['latitude']:.5f}, {location['longitude']:.5f})"

        if old_coords[0] is not None:
            diff_x = new_coords[0] - old_coords[0]
            diff_y = new_coords[1] - old_coords[1]
            diff_magnitude = np.sqrt(diff_x**2 + diff_y**2)

            result = {
                'name': name,
                'latitude': location['latitude'],
                'longitude': location['longitude'],
                'new_coordinates': new_coords,
                'old_coordinates': old_coords,
                'difference': [diff_x, diff_y],
                'difference_magnitude': diff_magnitude
            }

            if verbose:
                print(f"{name:<15} {lat_lon_str:<25} {str(old_coords):<15} "
                      f"{str(new_coords):<15} {diff_magnitude:>8.2f} px")
        else:
            result = {
                'name': name,
                'latitude': location['latitude'],
                'longitude': location['longitude'],
                'new_coordinates': new_coords,
                'old_coordinates': None,
                'difference': None,
                'difference_magnitude': None
            }

            if verbose:
                print(f"{name:<15} {lat_lon_str:<25} {'N/A':<15} {str(new_coords):<15} {'N/A':<10}")

        results.append(result)

    if verbose:
        print("="*80 + "\n")

    return results


def load_transmitter_locations(config_path, map_data):
    """
    Load transmitter station locations and convert to pixel coordinates.

    Parameters
    ----------
    config_path : str
        Path to transmitter_locations.yaml file
    map_data : dict
        Map data dictionary from load_slc_map()

    Returns
    -------
    dict
        Dictionary mapping transmitter names to their pixel coordinates.
        Each entry contains:
        - 'name': Transmitter name
        - 'latitude': Latitude in decimal degrees
        - 'longitude': Longitude in decimal degrees
        - 'height': Height in meters
        - 'type': Transmitter type (Dense or Rooftop)
        - 'coordinates': [col, row] pixel coordinates

    Examples
    --------
    >>> map_data = load_slc_map("../", downsample_factor=10)
    >>> tx_locs = load_transmitter_locations("config/transmitter_locations.yaml", map_data)
    >>> tx_locs['mario']['coordinates']
    [123, 456]
    """
    # Load YAML config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Default to Salt Lake City UTM zone
    utm_zone = 12
    northern = True

    transmitter_dict = {}

    for tx_name, tx_data in config['transmitters'].items():
        lat = tx_data['latitude']
        lon = tx_data['longitude']
        height = tx_data['height']
        tx_type = tx_data['type']

        # Convert to pixel coordinates
        col, row = latlon_to_pixel(lat, lon, map_data, utm_zone, northern)

        transmitter_dict[tx_name] = {
            'name': tx_name,
            'latitude': lat,
            'longitude': lon,
            'height': height,
            'type': tx_type,
            'coordinates': [col, row]  # [x, y] format
        }

    return transmitter_dict
