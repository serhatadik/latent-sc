"""
Test script to verify functional equivalency between old and new scripts.

This script compares outputs from the legacy scripts with outputs from the
new reorganized scripts to ensure they produce identical results.
"""
import sys
import numpy as np
from pathlib import Path

# Test configuration
TOLERANCE = 1e-10  # Numerical tolerance for floating point comparison
TEST_PASSED = []
TEST_FAILED = []


def test_file_equivalency(file1: Path, file2: Path, description: str):
    """
    Test if two numpy files are equivalent.

    Args:
        file1: Path to first file
        file2: Path to second file
        description: Description of what's being tested
    """
    try:
        data1 = np.load(file1)
        data2 = np.load(file2)

        # Check shapes match
        if data1.shape != data2.shape:
            TEST_FAILED.append(f"{description}: Shape mismatch ({data1.shape} vs {data2.shape})")
            return False

        # Check values are close
        if np.allclose(data1, data2, rtol=TOLERANCE, atol=TOLERANCE):
            TEST_PASSED.append(f"{description}: PASS")
            print(f"[PASS] {description}")
            return True
        else:
            max_diff = np.max(np.abs(data1 - data2))
            TEST_FAILED.append(f"{description}: Values differ (max diff: {max_diff})")
            print(f"[FAIL] {description} - Max difference: {max_diff}")
            return False

    except FileNotFoundError as e:
        TEST_FAILED.append(f"{description}: File not found - {e}")
        print(f"[FAIL] {description} - File not found: {e}")
        return False
    except Exception as e:
        TEST_FAILED.append(f"{description}: Error - {e}")
        print(f"[FAIL] {description} - Error: {e}")
        return False


def test_imports():
    """Test that all new modules can be imported."""
    print("\n" + "=" * 60)
    print("Testing Module Imports")
    print("=" * 60)

    modules_to_test = [
        "config.settings",
        "core.signal_processing",
        "core.gps_utils",
        "core.data_loading",
        "core.distance_utils",
    ]

    all_passed = True
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"[PASS] {module_name}")
            TEST_PASSED.append(f"Import {module_name}: PASS")
        except ImportError as e:
            print(f"[FAIL] {module_name}: {e}")
            TEST_FAILED.append(f"Import {module_name}: FAIL - {e}")
            all_passed = False

    return all_passed


def test_signal_processing():
    """Test signal processing functions."""
    print("\n" + "=" * 60)
    print("Testing Signal Processing Functions")
    print("=" * 60)

    from core.signal_processing import find_indices_outside, compute_psd, extract_channel_power
    from config import settings

    # Test find_indices_outside
    arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    indices = find_indices_outside(arr, 3, 7)
    expected = [0, 1, 2, 6, 7, 8, 9]  # Indices where value is NOT strictly in open interval (3, 7)
    # Note: 3 and 7 are NOT in the open interval (3, 7), so indices 2 and 6 are included

    if indices == expected:
        print("[PASS] find_indices_outside works correctly")
        TEST_PASSED.append("find_indices_outside: PASS")
    else:
        print(f"[FAIL] find_indices_outside failed: got {indices}, expected {expected}")
        TEST_FAILED.append("find_indices_outside: FAIL")

    # Test compute_psd with synthetic data
    np.random.seed(42)
    iq_samples = np.random.randn(1024) + 1j * np.random.randn(1024)
    freq, psd = compute_psd(iq_samples, settings.SAMPLE_RATE, settings.CENTER_FREQ)

    if freq.shape == psd.shape and len(freq) == 1024:
        print("[PASS] compute_psd produces correct output shape")
        TEST_PASSED.append("compute_psd shape: PASS")
    else:
        print(f"[FAIL] compute_psd shape mismatch")
        TEST_FAILED.append("compute_psd shape: FAIL")


def test_data_structures():
    """Test that data structures are correct."""
    print("\n" + "=" * 60)
    print("Testing Data Structures")
    print("=" * 60)

    from config import settings
    import json

    # Test transmitter locations load correctly
    transmitters_file = Path("process/config/transmitters.json")
    if not transmitters_file.exists():
        transmitters_file = Path("config/transmitters.json")

    try:
        with open(transmitters_file, 'r') as f:
            transmitters = json.load(f)

        if len(transmitters) == 6:  # Should have 6 transmitters
            print(f"[PASS] Transmitters JSON has correct number of entries ({len(transmitters)})")
            TEST_PASSED.append("Transmitters count: PASS")
        else:
            print(f"[FAIL] Transmitters JSON has wrong number of entries ({len(transmitters)}, expected 6)")
            TEST_FAILED.append("Transmitters count: FAIL")

        # Check all required fields
        required_fields = ["name", "latitude", "longitude", "description"]
        for tx_name, tx_data in transmitters.items():
            if all(field in tx_data for field in required_fields):
                print(f"[PASS] Transmitter '{tx_name}' has all required fields")
            else:
                print(f"[FAIL] Transmitter '{tx_name}' missing required fields")
                TEST_FAILED.append(f"Transmitter {tx_name} fields: FAIL")

    except Exception as e:
        print(f"[FAIL] Error loading transmitters: {e}")
        TEST_FAILED.append(f"Transmitters loading: FAIL - {e}")

    # Test RF channels
    if len(settings.RF_CHANNELS) == 5:
        print(f"[PASS] RF_CHANNELS has correct number of entries ({len(settings.RF_CHANNELS)})")
        TEST_PASSED.append("RF_CHANNELS count: PASS")
    else:
        print(f"[FAIL] RF_CHANNELS has wrong number of entries ({len(settings.RF_CHANNELS)}, expected 5)")
        TEST_FAILED.append("RF_CHANNELS count: FAIL")


def compare_existing_outputs():
    """
    Compare existing outputs from legacy scripts with what would be produced by new scripts.
    This only works if both old and new outputs exist.
    """
    print("\n" + "=" * 60)
    print("Comparing Existing Output Files")
    print("=" * 60)

    # Define file pairs to compare (if they exist)
    output_dir = Path("process/files_generated_by_process_data_scripts")
    new_output_dir = Path("process/output")

    # These are the main output files
    files_to_compare = [
        "TX1EBC_pow_test.npy",
        "TX1Ustar_pow_test.npy",
        "TX2_pow_test.npy",
        "TX3_pow_test.npy",
        "TX4_pow_test.npy",
        "TX5_pow_test.npy",
        "coordinates_test.npy",
        "coordinates_ebc_test.npy",
        "coordinates_ustar_test.npy",
    ]

    # Check if new output exists
    has_new_output = False
    for filename in files_to_compare:
        if (new_output_dir / filename).exists():
            has_new_output = True
            break

    if not has_new_output:
        print("[INFO] No new output files found in process/output/")
        print("  Run the new processing scripts first to compare outputs.")
        print("  Example:")
        print("    cd process/processing/")
        print("    python process_measurements.py --dataset-type mobile")
        return

    # Compare files
    for filename in files_to_compare:
        old_file = output_dir / filename
        new_file = new_output_dir / filename

        if old_file.exists() and new_file.exists():
            test_file_equivalency(old_file, new_file, filename)
        elif not new_file.exists():
            print(f"[INFO] {filename} - New output not yet generated")
        elif not old_file.exists():
            print(f"[INFO] {filename} - Legacy output not found")


def print_summary():
    """Print test summary."""
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Passed: {len(TEST_PASSED)}")
    print(f"Failed: {len(TEST_FAILED)}")

    if TEST_FAILED:
        print("\nFailed Tests:")
        for failure in TEST_FAILED:
            print(f"  [FAIL] {failure}")

    print("\n" + "=" * 60)
    if TEST_FAILED:
        print("RESULT: SOME TESTS FAILED")
        return False
    else:
        print("RESULT: ALL TESTS PASSED")
        return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("RF Measurement Processing - Functional Equivalency Tests")
    print("=" * 60)

    # Change to process directory if needed
    if Path("process").exists():
        import os
        os.chdir("process")
        print("Working directory: process/")

    # Run tests
    test_imports()
    test_signal_processing()
    test_data_structures()
    compare_existing_outputs()

    # Print summary
    success = print_summary()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
