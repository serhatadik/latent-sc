"""
Comprehensive parameter sweep package.

Re-exports all public functions for convenient ``from scripts.sweep import ...`` access.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so that ``src.*`` and ``scripts.*`` imports work
# when this package is loaded from within a subprocess or standalone script.
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# --- constants ---
from .constants import (
    KNOWN_TRANSMITTERS,
    AVAILABLE_WHITENING_CONFIGS,
    POWER_DENSITY_THRESHOLDS,
    DESIRED_COLUMN_ORDER,
    TIREM_CACHE_DIR,
)

# --- discovery ---
from .discovery import (
    check_tirem_cache_exists,
    parse_directory_name,
    discover_data_directories,
    define_sigma_noise_strategies,
)

# --- experiment ---
from .experiment import (
    save_glrt_visualization,
    run_single_experiment,
)

# --- orchestration ---
from .orchestration import (
    process_single_directory,
    run_comprehensive_sweep,
)

# --- results_io ---
from .results_io import (
    append_results_to_csv,
    save_bic_results_csv,
)

# --- analysis ---
from .analysis import (
    analyze_by_tx_count,
    analyze_universal,
    analyze_by_tx_set,
    analyze_glrt_score_correlation,
    create_final_results,
    cleanup_visualizations_for_best_only,
)

# --- reporting ---
from .reporting import (
    generate_bic_analysis_report,
    generate_final_analysis_report,
    generate_analysis_report,
)

# --- plotting ---
from .plotting import generate_plots
