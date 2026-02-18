# `scripts/sweep/` — Comprehensive Parameter Sweep Package

This package contains the factored modules of the comprehensive parameter sweep pipeline. The original monolithic `scripts/comprehensive_parameter_sweep.py` (~3500 lines) was split into focused modules organized by responsibility.

## Entry Point

The CLI entry point remains **`scripts/comprehensive_parameter_sweep.py`**. It sets threading environment variables, defines `main()` with argparse, and re-exports all public names from this package via `from scripts.sweep import *` for backward compatibility with downstream scripts.

```
python scripts/comprehensive_parameter_sweep.py --help
python scripts/comprehensive_parameter_sweep.py --test --tx-counts 1 --max-dirs 1
```

## Module Overview

### `constants.py`
Shared constants used across multiple modules:
- `KNOWN_TRANSMITTERS` — canonical list of transmitter names
- `AVAILABLE_WHITENING_CONFIGS` — whitening method configurations to sweep
- `POWER_DENSITY_THRESHOLDS` — power density thresholds to sweep
- `DESIRED_COLUMN_ORDER` — column ordering for results CSV
- `TIREM_CACHE_DIR` — path to TIREM propagation cache
- `_PROJECT_ROOT` — resolved project root (3 levels up from this file)

### `discovery.py`
Data directory discovery and pre-processing:
- `parse_directory_name()` — extract transmitter names, nloc, and seed from directory names
- `discover_data_directories()` — scan `data/processed/` and group directories by TX count
- `define_sigma_noise_strategies()` — build sigma noise strategies from observed power data
- `check_tirem_cache_exists()` — verify TIREM cache files exist before running experiments

### `experiment.py`
Core computational functions:
- `run_single_experiment()` — run one reconstruction experiment end-to-end (GLRT localization, candidate filtering, combinatorial BIC selection, power recomputation, per-TX exponent refit, reconstruction error)
- `save_glrt_visualization()` — generate GLRT iteration maps and final selection plots

Sets `matplotlib.use('Agg')` at module level for headless rendering.

### `orchestration.py`
Multiprocessing orchestration:
- `process_single_directory()` — process all parameter combinations for one data directory (top-level function, pickleable for multiprocessing)
- `run_comprehensive_sweep()` — main loop that dispatches directories to workers (sequential or parallel via `ProcessPoolExecutor`)
- `_worker_init()` — disables numpy threading in worker processes

### `results_io.py`
CSV I/O:
- `append_results_to_csv()` — incrementally append experiment results to `all_results.csv`
- `save_bic_results_csv()` — save a simplified BIC-only results CSV

### `analysis.py`
Aggregation and analysis (pure pandas/numpy, no internal sweep imports):
- `analyze_by_tx_count()` — group and summarize results per TX count
- `analyze_universal()` — summarize results across all TX counts
- `analyze_by_tx_set()` — summarize results by specific transmitter set
- `analyze_glrt_score_correlation()` — correlate GLRT scores with localization performance
- `create_final_results()` — select best strategy per directory by lowest BIC
- `cleanup_visualizations_for_best_only()` — remove visualization dirs for non-best strategies

### `reporting.py`
Markdown report generators:
- `generate_bic_analysis_report()` — BIC combinatorial selection analysis (`analysis_report_bic.md`)
- `generate_final_analysis_report()` — best-per-directory results (`analysis_report_final.md`)
- `generate_analysis_report()` — full comprehensive report (`analysis_report.md`)

### `plotting.py`
Visualization plots:
- `generate_plots()` — summary bar charts, strategy heatmap, threshold sensitivity plot

Sets `matplotlib.use('Agg')` at module level for headless rendering.

### `__init__.py`
Re-exports all public functions and constants from every submodule. Also ensures `_PROJECT_ROOT` is on `sys.path` so `src.*` imports resolve in subprocesses.

## Dependency Graph

```
constants         (no internal deps)
discovery         → constants
experiment        → constants, external (src.*, candidate_analysis)
results_io        → constants
orchestration     → constants, discovery, experiment, results_io
analysis          → (no internal deps)
reporting         → analysis
plotting          → (no internal deps)
__init__          → all above (re-exports only)
```

No circular dependencies.

## Backward Compatibility

Three downstream scripts import from `scripts.comprehensive_parameter_sweep`:

| Script | Imports |
|--------|---------|
| `rerun_analysis.py` | `analyze_by_tx_count`, `analyze_universal`, `generate_analysis_report` |
| `reanalyze_legacy_results.py` | `analyze_by_tx_count`, `analyze_universal`, `analyze_by_tx_set`, `generate_analysis_report`, `generate_plots` |
| `run_bic_analysis.py` | `generate_bic_analysis_report`, `save_bic_results_csv`, `create_final_results`, `generate_final_analysis_report`, `cleanup_visualizations_for_best_only` |

All of these continue to work because the entry-point script does `from scripts.sweep import *`.
