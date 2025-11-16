"""
Central configuration for RF measurement processing pipeline.
All hard-coded values from the original scripts are consolidated here.
"""
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent.parent
RAW_DATA_DIR = BASE_DIR / "raw_data"
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"
OUTPUT_DIR = Path(__file__).parent.parent / "output"

# Legacy output directory name (for backward compatibility)
LEGACY_OUTPUT_DIR = Path(__file__).parent.parent / "files_generated_by_process_data_scripts"

# RF Parameters
CENTER_FREQ = 3.534e9  # Hz
SAMPLE_RATE = 0.22e6   # Hz (220 kHz)

# RF Channel Definitions (min, max frequency pairs in Hz)
# These represent the 5 transmitter channels
RF_CHANNELS = {
    "TX1": (3533903750, 3533931250),
    "TX2": (3533945000, 3533972500),
    "TX3": (3533986250, 3534013750),
    "TX4": (3534027500, 3534055000),
    "TX5": (3534068750, 3534096250),
}

# Data source directories (relative to BASE_DIR)
DATA_DIRS = {
    "walking": RAW_DATA_DIR / "walking",
    "driving": RAW_DATA_DIR / "driving",
    "stat_rot": RAW_DATA_DIR / "stat_rot",
}

# GPS data directories
GPS_DATA_DIRS = {
    "mobile": RAW_DATA_DIR / "gps_data" / "all_gps_data",
    "stat_rot": RAW_DATA_DIR / "gps_data" / "stat_rot",
    "serhat": RAW_DATA_DIR / "gps_data_serhat",
}

# Algorithm Parameters
TIME_OFFSET_HOURS = -6  # Convert UTC to local time (UTC-6)
COORD_DEDUP_THRESHOLD_M = 20  # Merge coordinates within 20 meters
OUTLIER_DISTANCE_THRESHOLD_M = 10**2.7  # Distance threshold for outlier filtering
OUTLIER_POWER_THRESHOLD_DB = -88  # Power threshold for outlier filtering

# Data filtering
MIN_LATITUDE = 40  # Minimum valid latitude for GPS coordinates

# File patterns
IQ_SAMPLE_PATTERN = "samples_20*"  # Pattern for IQ sample directories
GPS_FILE_PATTERN_MOBILE = "*.txt"  # GPS files for mobile measurements
GPS_FILE_PATTERN_STAT = "*Stat.txt"  # GPS files for stationary measurements
GPX_FILE_PATTERN = "2023*"  # GPX files from Serhat's GPS data

# Processing parameters
SAMPLES_TO_SKIP = 2  # Skip last 2 files in each directory (potentially incomplete)
PROGRESS_PRINT_INTERVAL = 1000  # Print progress every N samples

# Transmitter split date (day 27 splits EBC from Ustar on TX1)
TX1_SPLIT_DAY = 27  # Day of month that separates EBC from Ustar measurements
