"""
Data loading utilities for IQ samples.
Extracted from read_samples.py, Fading_Analysis.py, and fading_analysis_rot.py
"""
import numpy as np
import pandas as pd
import os
from typing import Dict, List
from pathlib import Path


def load_iq_samples_from_directories(
    directories: List[Path],
    samples_to_skip: int = 2
) -> Dict[np.datetime64, np.ndarray]:
    """
    Load IQ samples from multiple directories.

    Each directory should contain .npy files with timestamps in their filenames.
    The last `samples_to_skip` files in each directory are skipped (potentially incomplete).

    Args:
        directories: List of directory paths containing IQ sample files
        samples_to_skip: Number of files to skip at the end of each directory

    Returns:
        Dictionary mapping timestamps to IQ sample arrays
    """
    data = {}

    for directory in directories:
        files = os.listdir(directory)
        print(f"Loading {len(files)} files from {directory}")

        for cnt, file in enumerate(files):
            # Skip last N files (potentially incomplete)
            if cnt < len(files) - samples_to_skip:
                # Extract timestamp from filename (format: YYYY-MM-DD-HH-MM-SS-IQ.npy)
                time = pd.to_datetime(file.split('-IQ')[0].split('.')[0])
                time = np.datetime64(time).astype('datetime64[s]')

                # Load IQ samples (take first element [0] as files contain nested arrays)
                data[time] = np.load(os.path.join(directory, file))[0]

    return data


def find_sample_directories_recursive(root_dir: Path, pattern: str = "samples_20*") -> List[Path]:
    """
    Recursively find all directories matching the pattern under root_dir.

    This searches through all subdirectories to find sample folders,
    since the walking/ and driving/ directories have subcategorization
    (e.g., by date and person).

    Args:
        root_dir: Root directory to search
        pattern: Glob pattern for sample directories

    Returns:
        List of paths to sample directories
    """
    sample_dirs = []

    # Use rglob to recursively search for matching directories
    for path in root_dir.rglob(pattern):
        if path.is_dir():
            sample_dirs.append(path)

    return sample_dirs


def load_mobile_iq_samples(
    walking_dir: Path,
    driving_dir: Path,
    sample_pattern: str = "samples_20*",
    samples_to_skip: int = 2
) -> Dict[np.datetime64, np.ndarray]:
    """
    Load IQ samples for mobile measurements (walking + driving).

    This function recursively searches for sample directories under
    walking_dir and driving_dir, since there may be subcategorization
    by date and person.

    Args:
        walking_dir: Directory containing walking measurement folders
        driving_dir: Directory containing driving measurement folders
        sample_pattern: Pattern for matching sample directories
        samples_to_skip: Number of files to skip at end of each directory

    Returns:
        Dictionary mapping timestamps to IQ sample arrays
    """
    # Recursively find all sample directories
    print(f"Searching for sample directories under {walking_dir}...")
    walking_folders = find_sample_directories_recursive(walking_dir, sample_pattern)
    print(f"Found {len(walking_folders)} walking sample directories")

    print(f"Searching for sample directories under {driving_dir}...")
    driving_folders = find_sample_directories_recursive(driving_dir, sample_pattern)
    print(f"Found {len(driving_folders)} driving sample directories")

    all_folders = walking_folders + driving_folders
    print(f"Total sample directories: {len(all_folders)}")

    # Load IQ samples from all directories
    data = load_iq_samples_from_directories(all_folders, samples_to_skip)

    return data


def load_stationary_iq_samples(
    stat_rot_dir: Path,
    folder_pattern: str = "stat",
    samples_to_skip: int = 2
) -> Dict[np.datetime64, np.ndarray]:
    """
    Load IQ samples for stationary measurements.

    Args:
        stat_rot_dir: Directory containing stationary/rotation measurement folders
        folder_pattern: Pattern for matching folders ("stat" or "rot")
        samples_to_skip: Number of files to skip at end of each directory

    Returns:
        Dictionary mapping timestamps to IQ sample arrays
    """
    # Find all stationary sample directories
    folders = [
        stat_rot_dir / name
        for name in os.listdir(stat_rot_dir)
        if name.startswith(folder_pattern)
    ]

    # Load IQ samples from all directories
    data = load_iq_samples_from_directories(folders, samples_to_skip)

    return data
