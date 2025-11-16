#!/usr/bin/env python
"""
Full Pipeline Runner

This script runs the complete analysis pipeline from raw data to final figures.

Pipeline Steps:
    01. Process occupancy metrics
    02. Estimate signal strength via localization
    03. Perform temporal analysis
    04. Generate all figures

Usage:
    python run_full_pipeline.py --config config/parameters.yaml
    python run_full_pipeline.py --config config/parameters.yaml --skip-step 1
    python run_full_pipeline.py --config config/parameters.yaml --band "3610-3650"
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd, description):
    """
    Run a shell command and report status.

    Parameters
    ----------
    cmd : list
        Command and arguments as list
    description : str
        Description of the step

    Returns
    -------
    success : bool
        True if command succeeded
    """
    print(f"\n{'='*70}")
    print(f"STEP: {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}\n")

    start_time = time.time()

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        elapsed = time.time() - start_time
        print(f"\n✓ {description} completed successfully in {elapsed:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n✗ Error: Command not found. Make sure Python is in your PATH.")
        return False


def main(args):
    """Run the full analysis pipeline."""
    print(f"\n{'#'*70}")
    print("# FULL PIPELINE EXECUTION")
    print(f"{'#'*70}\n")

    # Get the scripts directory
    scripts_dir = Path(__file__).parent

    # Build common arguments
    common_args = [
        '--config', args.config,
        '--locations', args.locations,
    ]

    if args.band:
        common_args.extend(['--band', args.band])

    # Track success of each step
    steps_success = []

    # Step 1: Process Occupancy
    if 1 not in args.skip_steps:
        cmd = [
            sys.executable,
            str(scripts_dir / '01_process_occupancy.py'),
            *common_args,
            '--data-path', args.data_path,
            '--output-dir', args.processed_dir,
        ]

        if args.monitor:
            cmd.extend(['--monitor', args.monitor])

        success = run_command(cmd, "01: Process Occupancy Metrics")
        steps_success.append(('01_process_occupancy', success))

        if not success and not args.continue_on_error:
            print("\n✗ Pipeline stopped due to error. Use --continue-on-error to proceed anyway.")
            return 1
    else:
        print(f"\nSkipping Step 1: Process Occupancy Metrics")

    # Step 2: Estimate Signals
    if 2 not in args.skip_steps:
        cmd = [
            sys.executable,
            str(scripts_dir / '02_estimate_signals.py'),
            *common_args,
            '--map-dir', args.map_dir,
            '--metrics-path', f'{args.processed_dir}/occupancy_metrics.npz',
            '--output-dir', args.processed_dir,
        ]

        success = run_command(cmd, "02: Estimate Signal Strength")
        steps_success.append(('02_estimate_signals', success))

        if not success and not args.continue_on_error:
            print("\n✗ Pipeline stopped due to error. Use --continue-on-error to proceed anyway.")
            return 1
    else:
        print(f"\nSkipping Step 2: Estimate Signal Strength")

    # Step 3: Temporal Analysis
    if 3 not in args.skip_steps:
        cmd = [
            sys.executable,
            str(scripts_dir / '03_temporal_analysis.py'),
            *common_args,
            '--data-path', args.data_path,
            '--metrics-path', f'{args.processed_dir}/occupancy_metrics.npz',
            '--output-dir', args.processed_dir,
        ]

        if args.monitor:
            cmd.extend(['--monitor', args.monitor])

        success = run_command(cmd, "03: Temporal Analysis")
        steps_success.append(('03_temporal_analysis', success))

        if not success and not args.continue_on_error:
            print("\n✗ Pipeline stopped due to error. Use --continue-on-error to proceed anyway.")
            return 1
    else:
        print(f"\nSkipping Step 3: Temporal Analysis")

    # Step 4: Generate Figures
    if 4 not in args.skip_steps:
        cmd = [
            sys.executable,
            str(scripts_dir / '04_generate_figures.py'),
            *common_args,
            '--map-dir', args.map_dir,
            '--data-path', args.data_path,
            '--processed-dir', args.processed_dir,
            '--output-dir', args.output_dir,
        ]

        if args.figure:
            cmd.extend(['--figure', str(args.figure)])

        success = run_command(cmd, "04: Generate Figures")
        steps_success.append(('04_generate_figures', success))

        if not success and not args.continue_on_error:
            print("\n✗ Pipeline stopped due to error. Use --continue-on-error to proceed anyway.")
            return 1
    else:
        print(f"\nSkipping Step 4: Generate Figures")

    # Print summary
    print(f"\n\n{'#'*70}")
    print("# PIPELINE SUMMARY")
    print(f"{'#'*70}\n")

    all_success = True
    for step_name, success in steps_success:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {step_name:30s} {status}")
        if not success:
            all_success = False

    print(f"\n{'#'*70}")

    if all_success:
        print("# ✓ ALL STEPS COMPLETED SUCCESSFULLY!")
        print(f"# Results saved to: {args.output_dir}")
    else:
        print("# ✗ SOME STEPS FAILED")
        print("# Check the output above for error details")

    print(f"{'#'*70}\n")

    return 0 if all_success else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run the complete spectrum occupancy analysis pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline on all bands
  python run_full_pipeline.py

  # Run only for CBRS GAA band
  python run_full_pipeline.py --band "3610-3650"

  # Skip step 1 (use existing occupancy metrics)
  python run_full_pipeline.py --skip-step 1

  # Generate only Figure 3
  python run_full_pipeline.py --skip-step 1 --skip-step 2 --skip-step 3 --figure 3

  # Continue even if a step fails
  python run_full_pipeline.py --continue-on-error
        """
    )

    parser.add_argument('--config', type=str, default='config/parameters.yaml',
                       help='Path to configuration file')
    parser.add_argument('--locations', type=str, default='config/monitoring_locations.yaml',
                       help='Path to monitoring locations file')
    parser.add_argument('--map-dir', type=str, default='./',
                       help='Directory containing SLC map file')
    parser.add_argument('--data-path', type=str, default='./data/raw/rfbaseline/',
                       help='Path to raw data directory')
    parser.add_argument('--processed-dir', type=str, default='./data/processed/',
                       help='Directory for processed intermediate results')
    parser.add_argument('--output-dir', type=str, default='./data/results/',
                       help='Output directory for final results and figures')

    # Filtering options
    parser.add_argument('--band', type=str, default=None,
                       help='Process only specific band (e.g., "3610-3650")')
    parser.add_argument('--monitor', type=str, default=None,
                       help='Process only specific monitor (e.g., Bookstore)')
    parser.add_argument('--figure', type=int, default=None,
                       help='Generate only specific figure (2-7)')

    # Execution control
    parser.add_argument('--skip-step', type=int, action='append', dest='skip_steps', default=[],
                       help='Skip specific step (can be used multiple times: --skip-step 1 --skip-step 2)')
    parser.add_argument('--continue-on-error', action='store_true',
                       help='Continue pipeline execution even if a step fails')

    args = parser.parse_args()

    # Convert skip_steps to set for easier checking
    args.skip_steps = set(args.skip_steps)

    exit_code = main(args)
    sys.exit(exit_code)
