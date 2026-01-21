# Latent Spectrum Reconstruction for Signal Power Estimation

This repository contains code for estimating signal power across a geographic region using measurements from monitoring stations. It implements likelihood-based localization and spatial interpolation techniques, plus a general-purpose pipeline for processing raw IQ samples from Software Defined Radios (SDRs).

---

## Repository Structure

```
latent-sc/
├── config/                      # Configuration files
│   ├── parameters.yaml          # Algorithm parameters
│   └── monitoring_locations*.yaml  # Monitoring station configs
├── data/                        # Data directory
│   ├── rf_baseline/            # Paper data
│   └── processed/              # Processed custom data
├── notebooks/                   # Jupyter notebooks
│   ├── paper_reproduction.ipynb       # Paper analysis
│   └── custom_data_localization.ipynb # Custom data example
├── scripts/                     # Processing scripts
│   ├── process_raw_data_to_monitoring.py  # IQ → YAML
│   ├── 01_process_occupancy.py
│   ├── 02_estimate_signals.py
│   ├── 03_temporal_analysis.py
│   ├── 04_generate_figures.py
│   └── run_full_pipeline.py
├── src/                         # Source code
│   ├── data_processing/
│   │   ├── iq_processor.py     # IQ processing
│   │   ├── occupancy.py
│   │   └── temporal.py
│   ├── localization/           # Localization algorithms
│   ├── interpolation/          # IDW interpolation
│   ├── analysis/               # Statistical analysis
│   ├── utils/                  # Coordinate transforms, conversions
│   └── visualization/          # Plotting functions
├── tests/                       # Unit tests
├── CUSTOM_DATA_PIPELINE.md     # Custom data guide
└── EBC_USTAR_SEPARATION.md     # Transmitter guide
```

---

## Features

### Core Localization & Analysis
- **Likelihood-based Localization**: Estimate transmitter location and power
- **Signal Power Estimation**: Predict signal strength at unmeasured locations
- **Spatial Interpolation**: IDW with confidence bounds
- **Temporal Analysis**: Time-of-day and seasonal variations
- **Occupancy Metrics**: Duty cycle, average power, signal statistics
- **Visualization**: Spatial heatmaps, temporal plots, analysis figures

### Custom Data Processing Pipeline
- **Raw IQ Processing**: Convert IQ samples → monitoring locations
- **Multi-Transmitter**: 6 transmitters across 5 frequency bands
- **Date Filtering**: Automatic EBC (≤June 27) / USTAR (≥June 28) separation
- **Recursive Search**: Find `samples_*` directories at any depth
- **GPS Matching**: Nearest-neighbor with ±10s tolerance
- **Auto-Aggregation**: Group by location with 20m deduplication
- **YAML Generation**: Create configs compatible with notebooks

---

## Quick Start

### Option 1: Paper Data (RF Baseline)

**Run full pipeline:**
```bash
python scripts/run_full_pipeline.py
```

**Or step-by-step:**
```bash
python scripts/01_process_occupancy.py
python scripts/02_estimate_signals.py
python scripts/03_temporal_analysis.py
python scripts/04_generate_figures.py
```

**Interactive:**
```bash
jupyter notebook notebooks/paper_reproduction.ipynb
```

### Option 2: Process Your Own IQ Data

**Step 1: Process raw IQ samples**
```bash
python scripts/process_raw_data_to_monitoring.py \
    --input-dir "C:/Users/serha/raw_data/driving/" \
    --transmitter mario \
    --num-locations 10 \
    --output-yaml "config/monitoring_locations_mario.yaml"
```

**Step 2: Run localization**
```bash
jupyter notebook notebooks/custom_data_localization.ipynb
```

**See detailed guide**: [CUSTOM_DATA_PIPELINE.md](CUSTOM_DATA_PIPELINE.md)

---

## Custom Data Pipeline

### Supported Transmitters

| Transmitter | Channel | Frequency (MHz) | Date Range |
|-------------|---------|-----------------|------------|
| `ebc` | TX1 | 3533.904 - 3533.931 | ≤ June 27, 2023 |
| `ustar` | TX1 | 3533.904 - 3533.931 | ≥ June 28, 2023 |
| `guesthouse` | TX2 | 3533.945 - 3533.973 | All dates |
| `mario` | TX3 | 3533.986 - 3534.014 | All dates |
| `moran` | TX4 | 3534.028 - 3534.055 | All dates |
| `wasatch` | TX5 | 3534.069 - 3534.096 | All dates |

**Note**: EBC and USTAR share TX1 frequency band but are date-separated (transmitter replacement on June 28, 2023).

### Processing Workflow

```
Raw IQ Samples (.npy) + GPS Data (.txt/.gpx)
              ↓
  [Recursive Directory Search]
              ↓
  [Load IQ Samples & GPS Coordinates]
              ↓
  [Compute PSD, Extract Channel Power]
              ↓
  [Match with GPS (±10s tolerance)]
              ↓
  [Apply Date Filter (EBC/USTAR)]
              ↓
  [Aggregate by Location (20m radius)]
              ↓
monitoring_locations.yaml + Processed Data
              ↓
  [Localization & Power Estimation]
```

### Usage Examples

**Process stationary data:**
```bash
python scripts/process_raw_data_to_monitoring.py \
    --input-dir "C:/Users/serha/raw_data/stat_rot/stat/" \
    --transmitter mario \
    --num-locations 10
```

**Process driving data (EBC - June 27 only):**
```bash
python scripts/process_raw_data_to_monitoring.py \
    --input-dir "C:/Users/serha/raw_data/driving/" \
    --transmitter ebc \
    --num-locations 8
# Recursively finds all sample directories
# Automatically filters to keep only June 27 data
```

**Process walking data (USTAR - June 28+):**
```bash
python scripts/process_raw_data_to_monitoring.py \
    --input-dir "C:/Users/serha/raw_data/walking/" \
    --transmitter ustar \
    --num-locations 15
# Recursively finds all sample directories
# Automatically filters to keep only June 28+ data
```

### Key Features

**Recursive Search**: Finds `samples_*` at any nesting level
**Auto GPS Detection**: Detects GPS directory based on input path
**Date Filtering**: EBC/USTAR separated by June 28 split date
**Time Matching**: ±10 second tolerance for GPS/IQ alignment
**Location Grouping**: 20m deduplication radius
**Flexible Config**: All parameters customizable via CLI

### Example Output

**Command:**
```bash
python scripts/process_raw_data_to_monitoring.py \
    --input-dir "C:/Users/serha/raw_data/driving/" \
    --transmitter ebc \
    --num-locations 3
```

**Results:**
```
Found 11 sample directories
Loaded 9828 IQ samples
Date filter: samples before 2023-06-28T00:00:00 (EBC period)
Skipped 3863 samples outside EBC date range
Matched 5852 measurements with GPS coordinates
Aggregated to 337 locations with >=5 samples

Generated:
- config/monitoring_locations_ebc.yaml
- data/processed/ebc/ebc_avg_powers.npy
- data/processed/ebc/ebc_summary.csv
```

---

## Module Overview

### `src/data_processing/`
- **`iq_processor.py`**: Raw IQ processing
  - `load_iq_samples_from_directories()`: Load from .npy files
  - `load_gps_from_csv()`: Load GPS with timezone handling
  - `process_iq_sample()`: Compute PSD, extract power
  - `match_power_with_gps()`: Match with date filtering
  - `aggregate_measurements_by_location()`: Group by position
  - Transmitter-to-channel mapping with EBC/USTAR split
- `loader.py`: Monitoring data loading
- `occupancy.py`: Occupancy metrics
- `temporal.py`: Temporal filtering

### `src/localization/`
- `likelihood.py`: Likelihood-based transmitter localization
- `transmitter.py`: Transmitter location estimation
- `path_loss.py`: Path loss models

### `src/interpolation/`
- `idw.py`: Inverse Distance Weighting
- `confidence.py`: Confidence bound estimation

### `src/utils/`
- `coordinates.py`: Coordinate transformations (lat/lon ↔ UTM ↔ pixel)
- `conversions.py`: Unit conversions (dB ↔ linear)
- `map_utils.py`: Map loading
- `location_utils.py`: Monitoring location utilities

### `src/visualization/`
- `spatial_plots.py`: Spatial heatmaps and maps
- `temporal_plots.py`: Time series plots
- `analysis_plots.py`: Statistical analysis figures

### `scripts/`
- **`process_raw_data_to_monitoring.py`**: IQ processing CLI
  - Recursive directory search
  - Auto GPS detection
  - Date-based filtering
  - YAML generation

---

## Configuration

### Algorithm Parameters

Edit `config/parameters.yaml`:
- Spatial resolution and map settings
- Path loss exponents
- Localization parameters
- Interpolation settings
- Visualization preferences

### Monitoring Locations

**For paper data:**
```python
from src.utils import load_monitoring_locations, load_slc_map

map_data = load_slc_map("../", downsample_factor=10)
locations = load_monitoring_locations("config/monitoring_locations.yaml", map_data)
```

**For custom data:**
```bash
# Generate config
python scripts/process_raw_data_to_monitoring.py \
    --input-dir "path/to/data/" \
    --transmitter mario \
    --num-locations 10 \
    --output-yaml "config/monitoring_locations_mario.yaml"
```

```python
# Load in notebook
locations = load_monitoring_locations("config/monitoring_locations_mario.yaml", map_data)
observed_powers = np.load("data/processed/mario/mario_avg_powers.npy")
```

---

## IQ Data Processing Details

### Input Format

**IQ Samples:**
- Complex samples: I + jQ
- Sample rate: 220 kHz
- Center frequency: 3.534 GHz
- Format: NumPy arrays (.npy)

**GPS Data:**
- CSV format: `date time`, `latitude`, `longitude`
- Timestamps: UTC (auto-converted to local)
- Offset: -6 hours for UTC-6

### Processing Steps

1. **Recursive Search**: Find all `samples_*` directories
2. **Load IQ**: Read complex samples from .npy files
3. **Load GPS**: Read coordinates from CSV/GPX
4. **Hamming Window**: Reduce spectral leakage
5. **FFT**: Transform to frequency domain
6. **PSD**: Compute Power Spectral Density (dB)
7. **Extract Power**: Max power in transmitter band
8. **GPS Match**: Nearest-neighbor (±10s tolerance)
9. **Date Filter**: Apply EBC/USTAR separation
10. **Aggregate**: Group within 20m radius
11. **Export**: Generate YAML + processed data

### Output Files

- `monitoring_locations_<tx>.yaml` - Station coordinates
- `<tx>_avg_powers.npy` - Average power per location
- `<tx>_latitudes.npy` - Latitude values
- `<tx>_longitudes.npy` - Longitude values
- `<tx>_std_powers.npy` - Power standard deviations
- `<tx>_sample_counts.npy` - Samples per location
- `<tx>_summary.csv` - Human-readable summary

---

## Documentation

- **[CUSTOM_DATA_PIPELINE.md](CUSTOM_DATA_PIPELINE.md)**: Complete guide for processing raw IQ data
  - Detailed workflow explanation
  - Command-line arguments reference
  - Usage examples for different data sources
  - Troubleshooting guide

- **[EBC_USTAR_SEPARATION.md](EBC_USTAR_SEPARATION.md)**: EBC/USTAR transmitter separation
  - Date-based filtering details
  - Test results and validation
  - Usage recommendations

---

## Examples

### Process Custom Data End-to-End

```bash
# 1. Process raw IQ samples
python scripts/process_raw_data_to_monitoring.py \
    --input-dir "C:/Users/serha/raw_data/driving/" \
    --transmitter mario \
    --num-locations 10 \
    --output-yaml "config/monitoring_locations_mario.yaml" \
    --output-data "data/processed/mario/"

# 2. Use in Python
from src.utils import load_monitoring_locations, load_slc_map
from src.localization import estimate_transmit_power_map
import numpy as np

# Load map and locations
map_data = load_slc_map("../", downsample_factor=10)
locations = load_monitoring_locations("config/monitoring_locations_mario.yaml", map_data)

# Load processed measurements
observed_powers = np.load("data/processed/mario/mario_avg_powers.npy")
sensor_locs = get_sensor_locations_array(locations)

# Run localization
tx_power_map = estimate_transmit_power_map(
    map_shape=map_data['shape'],
    sensor_locations=sensor_locs,
    observed_powers=observed_powers,
    scale=5.0,
    np_exponent=2.0
)
```

---

## Changelog

### Version 1.1 (Latest)

**New Features:**
- Raw IQ sample processing pipeline
- Multi-transmitter support (6 transmitters, 5 frequency bands)
- EBC/USTAR date-based automatic separation
- Recursive directory search for nested structures
- GPS coordinate auto-matching with time tolerance
- Location aggregation with spatial deduplication
- Custom data localization notebook
- Comprehensive documentation (2 new guides)

**Improvements:**
- Enhanced coordinate conversion utilities
- Monitoring location loading with auto lat/lon → pixel conversion
- Better error handling and progress reporting

### Version 1.0

- Initial release with paper reproduction code
- Likelihood-based localization
- Signal power estimation
- Spatial interpolation (IDW)
- Temporal analysis
- Visualization tools

---

## Support

For questions or issues:
1. Check [CUSTOM_DATA_PIPELINE.md](CUSTOM_DATA_PIPELINE.md) for usage guidance
2. Review [EBC_USTAR_SEPARATION.md](EBC_USTAR_SEPARATION.md) for transmitter details
3. Examine example notebook: `notebooks/custom_data_localization.ipynb`
4. Run `python scripts/process_raw_data_to_monitoring.py --help`
