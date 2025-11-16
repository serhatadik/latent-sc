# Legacy Scripts

This directory contains the original RF measurement processing scripts before reorganization.

These scripts are preserved for reference and to allow comparison with the new reorganized codebase.

## Original Scripts

### Data Processing
- **`read_samples.py`** - Process mobile (walking/driving) measurements
  - Replaced by: `../processing/process_measurements.py --dataset-type mobile`

- **`Fading_Analysis.py`** - Process stationary measurements
  - Replaced by: `../processing/process_measurements.py --dataset-type stationary`

- **`fading_analysis_rot.py`** - Process rotation measurements
  - Replaced by: `../processing/process_measurements.py --dataset-type rotation`

### Data Conversion
- **`convert_to_json.py`** - Convert numpy arrays to JSON format
  - Replaced by: `../processing/convert_data.py --format json`

- **`tabular_loc_power.py`** - Convert numpy arrays to MATLAB .mat format
  - Replaced by: `../processing/convert_data.py --format matlab`

### Visualization
- **`visualize_RSSI_dist.py`** - Visualize mobile measurements
  - Replaced by: `../analysis/visualize_measurements.py --dataset-type mobile`

- **`visualize_fading.py`** - Visualize stationary measurements
  - Replaced by: `../analysis/visualize_measurements.py --dataset-type stationary`

- **`visualize_fading_rot.py`** - Visualize rotation measurements
  - Replaced by: `../analysis/visualize_measurements.py --dataset-type rotation`

- **`fading_statistics.py`** - Stationary measurement statistics (duplicate of visualize_fading.py)
  - Replaced by: `../analysis/visualize_measurements.py --dataset-type stationary`

### Validation
- **`check.py`** - Validate JSON data consistency
  - Replaced by: `../validation/validate_data.py`

### Acquisition
- **`usrp_receive_samples.py`** - USRP radio sample acquisition
  - Replaced by: `../acquisition/usrp_receiver.py`

## Issues with Original Scripts

### Code Duplication
- **~50% code duplication** across processing and visualization scripts
- `read_samples.py`, `Fading_Analysis.py`, `fading_analysis_rot.py` were 90% identical
- `visualize_fading.py` and `visualize_fading_rot.py` were 95% identical
- `fading_statistics.py` was an exact duplicate of `visualize_fading.py`

### Hard-coded Configuration
- All RF frequencies, transmitter locations, and paths hard-coded inline
- No central configuration file
- Difficult to reuse with different parameters

### Naming Inconsistencies
- Mix of PascalCase and snake_case (e.g., `Fading_Analysis.py` vs `read_samples.py`)
- Cryptic variable names (e.g., `ebcdd`, `cfmin1`, `dist1`)

### No Abstraction
- FFT/PSD calculation logic repeated 3+ times
- GPS loading logic repeated
- Distance calculation and coordinate deduplication repeated

### Mixed Responsibilities
- Single scripts doing loading, processing, and visualization
- No separation of concerns

## Migration Guide

To migrate from legacy scripts to new reorganized scripts:

### 1. Update Configuration

Create `../config/settings.py` if you need custom paths or parameters.
Default values match the original hard-coded values.

### 2. Update Commands

**Old:**
```bash
python read_samples.py
python Fading_Analysis.py
python fading_analysis_rot.py
```

**New:**
```bash
cd ../processing/
python process_measurements.py --dataset-type mobile
python process_measurements.py --dataset-type stationary
python process_measurements.py --dataset-type rotation
```

**Old:**
```bash
python convert_to_json.py
python tabular_loc_power.py
```

**New:**
```bash
cd ../processing/
python convert_data.py --format json --output data.json
python convert_data.py --format matlab --output dd_meas_data.mat
```

**Old:**
```bash
python visualize_RSSI_dist.py
python visualize_fading.py
python visualize_fading_rot.py
```

**New:**
```bash
cd ../analysis/
python visualize_measurements.py --dataset-type mobile
python visualize_measurements.py --dataset-type stationary
python visualize_measurements.py --dataset-type rotation
```

**Old:**
```bash
python check.py
```

**New:**
```bash
cd ../validation/
python validate_data.py --input ../processed_data/data.json
```

### 3. Verify Output

The new scripts produce identical output to the old scripts.

To verify:
```python
import numpy as np

# Load outputs from old script
old_output = np.load("old_TX1EBC_pow_test.npy")

# Load outputs from new script
new_output = np.load("../output/TX1EBC_pow_test.npy")

# Compare
assert np.allclose(old_output, new_output)
print("Outputs match!")
```

## Do Not Modify

These scripts are preserved for reference only and should not be modified.

All future development should use the reorganized structure in the parent directory.
