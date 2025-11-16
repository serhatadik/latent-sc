# RF Measurement Processing Pipeline

This directory contains a reorganized and refactored RF measurement processing pipeline. The code has been restructured to be more maintainable, eliminate duplication, and provide a clearer separation of concerns.

## Directory Structure

```
process/
├── config/              # Configuration files
│   ├── settings.py      # Central configuration (paths, RF params, etc.)
│   └── transmitters.json # Transmitter locations
│
├── core/                # Shared utilities
│   ├── signal_processing.py  # FFT/PSD calculations
│   ├── gps_utils.py          # GPS loading and coordinate operations
│   ├── data_loading.py       # IQ sample loading
│   └── distance_utils.py     # Distance calculations
│
├── acquisition/         # Data acquisition
│   └── usrp_receiver.py      # USRP radio interface for sample collection
│
├── processing/          # Data processing
│   ├── process_measurements.py  # Process IQ samples → power measurements
│   └── convert_data.py          # Convert to JSON or MATLAB format
│
├── validation/          # Data validation
│   └── validate_data.py      # Validate JSON data consistency
│
├── analysis/            # Data analysis and visualization
│   └── visualize_measurements.py  # Visualize power vs distance
│
├── output/              # Output directory for processed files
│
├── legacy/              # Original scripts (for reference)
│
└── README.md            # This file
```

## Pipeline Overview

The RF measurement pipeline consists of several stages:

1. **Data Acquisition** (`acquisition/`)
   - Collect IQ samples from USRP radio hardware
   - Save samples with GPS timestamps

2. **Data Processing** (`processing/`)
   - Load IQ samples and GPS coordinates
   - Calculate Power Spectral Density (PSD)
   - Extract power for each RF channel
   - Save processed measurements

3. **Data Conversion** (`processing/`)
   - Convert to JSON format (with transmitter metadata)
   - Convert to MATLAB .mat format (tabular)

4. **Data Validation** (`validation/`)
   - Verify consistency of receiver coordinates
   - Check data integrity

5. **Data Analysis** (`analysis/`)
   - Visualize power vs. log-distance
   - Calculate path loss exponents
   - Generate statistical plots

## Usage

### 1. Data Acquisition (Optional)

Collect IQ samples from USRP radio:

```bash
cd acquisition/
python usrp_receiver.py -f 3534e6 -g 70 -r 220e3 -o ../data
```

Options:
- `-f, --frequency`: Center frequency (Hz)
- `-g, --gain`: Rx gain (dB)
- `-r, --rate`: Sample rate (Hz)
- `-n, --nsamps`: Number of samples per capture
- `-o, --output_dir`: Output directory

### 2. Process Measurements

Process IQ samples to extract power measurements:

```bash
cd processing/

# Process mobile (walking/driving) measurements
python process_measurements.py --dataset-type mobile

# Process stationary measurements
python process_measurements.py --dataset-type stationary

# Process rotation measurements
python process_measurements.py --dataset-type rotation
```

**Output:** Creates `.npy` files in `output/` (or `files_generated_by_process_data_scripts/` for backward compatibility):
- `TX1EBC_pow_{suffix}.npy` - EBC transmitter power measurements
- `TX1Ustar_pow_{suffix}.npy` - USTAR transmitter power measurements
- `TX2-5_pow_{suffix}.npy` - Other transmitter power measurements
- `coordinates_{suffix}.npy` - Receiver coordinates
- `coordinates_ebc_{suffix}.npy` - EBC-specific coordinates
- `coordinates_ustar_{suffix}.npy` - USTAR-specific coordinates

Where `{suffix}` is:
- `test` for mobile measurements
- `stat` for stationary measurements
- `rot` for rotation measurements

### 3. Convert Data

Convert processed measurements to JSON or MATLAB format:

```bash
cd processing/

# Convert to JSON
python convert_data.py --format json --output ../processed_data/data.json

# Convert to MATLAB .mat format
python convert_data.py --format matlab --output dd_meas_data.mat

# Disable outlier filtering
python convert_data.py --format json --output data.json --no-outlier-filter
```

**JSON Format:**
```json
{
  "2023-04-27 10:15:30": {
    "pow_rx_tx": [
      [power_db, rx_lat, rx_lon, tx_lat, tx_lon],
      ...
    ],
    "metadata": ["walking"]
  }
}
```

**MATLAB Format:**
- Array with columns: `[TX1_pow, lat, lon, TX2_pow, lat, lon, ...]`

### 4. Validate Data

Validate JSON data consistency:

```bash
cd validation/
python validate_data.py --input ../processed_data/data.json --verbose
```

Checks that all receiver coordinates within each timestamp match.

### 5. Visualize Results

Visualize power vs. distance:

```bash
cd analysis/

# Visualize mobile measurements (6 transmitters)
python visualize_measurements.py --dataset-type mobile

# Visualize stationary measurements (5 transmitters, 2 plots)
python visualize_measurements.py --dataset-type stationary

# Visualize rotation measurements
python visualize_measurements.py --dataset-type rotation
```

## Configuration

All configuration is centralized in `config/`:

### `config/settings.py`

Edit this file to change:
- Data directory paths
- RF parameters (frequencies, sample rate)
- Algorithm parameters (deduplication threshold, outlier filtering)
- Output directories

### `config/transmitters.json`

Edit this file to change transmitter locations:

```json
{
  "ebc": {
    "name": "EBC",
    "latitude": 40.76702,
    "longitude": -111.83807,
    "description": "EBC transmitter location"
  },
  ...
}
```

## Key Improvements

### Eliminated Code Duplication

**Before:** 11 scripts with ~50% code duplication
**After:** 5 main scripts + utility modules

Consolidated:
- `read_samples.py`, `Fading_Analysis.py`, `fading_analysis_rot.py` → `process_measurements.py`
- `convert_to_json.py`, `tabular_loc_power.py` → `convert_data.py`
- `visualize_fading.py`, `visualize_fading_rot.py`, `visualize_RSSI_dist.py`, `fading_statistics.py` → `visualize_measurements.py`
- `usrp_receive_samples.py` → `acquisition/usrp_receiver.py`
- `check.py` → `validation/validate_data.py`

### Centralized Configuration

All hard-coded values moved to `config/settings.py` and `config/transmitters.json`:
- RF frequencies and parameters
- Data directory paths
- Transmitter locations
- Algorithm thresholds

### Shared Utilities

Common functions extracted to `core/`:
- Signal processing (FFT/PSD calculation)
- GPS loading and coordinate deduplication
- Distance calculations
- Data loading

### Better Naming

- Consistent snake_case naming
- Clear, descriptive function and variable names
- Removed cryptic abbreviations (e.g., "dd" suffix)

### Enhanced Error Handling

- Input validation
- Better error messages
- Exit codes for validation scripts

## Dataset Types

The pipeline supports three dataset types:

1. **Mobile (`--dataset-type mobile`)**
   - Walking and driving measurements
   - Uses both CSV and GPX GPS data
   - Splits TX1 channel by day (day 27: EBC, others: USTAR)
   - Output suffix: `test`

2. **Stationary (`--dataset-type stationary`)**
   - Static receiver measurements
   - Uses CSV GPS data from stat_rot/ directory
   - Splits TX1 channel by day
   - Output suffix: `stat`

3. **Rotation (`--dataset-type rotation`)**
   - Receiver rotation measurements
   - Uses CSV GPS data from stat_rot/ directory
   - TX1 channel is USTAR only (no EBC split)
   - Output suffix: `rot`

## Transmitters

The system tracks 6 transmitter locations:

1. **EBC** - Engineering Building Complex
2. **USTAR** - USTAR building
3. **Guesthouse** - Guesthouse location
4. **Mario** - Mario building
5. **Moran** - Moran Eye Center
6. **Wasatch** - Wasatch building

RF channels:
- TX1: 3533.90-3533.93 GHz (EBC or USTAR depending on date)
- TX2: 3533.95-3533.97 GHz (Guesthouse)
- TX3: 3533.99-3534.01 GHz (Mario)
- TX4: 3534.03-3534.05 GHz (Moran)
- TX5: 3534.07-3534.10 GHz (Wasatch)

## Dependencies

- `numpy` - Numerical operations
- `pandas` - Data loading and manipulation
- `matplotlib` - Visualization
- `scipy` - Signal processing and MATLAB export
- `geopy` - Distance calculations
- `gpxpy` - GPX file parsing
- `uhd` - USRP hardware driver (for acquisition only)

## Functional Equivalency

The new scripts produce **identical output** to the original scripts. All processing logic, filtering, and calculations are preserved exactly as they were.

To verify:
1. Run old scripts and save outputs
2. Run new scripts with same inputs
3. Compare outputs using `np.allclose()` for numerical data

## Legacy Scripts

Original scripts are preserved in `legacy/` directory for reference and comparison.

## License

See LICENSE file in repository root.
