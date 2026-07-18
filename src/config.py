"""
Central configuration for the Multi-Hospital Admission Forecasting project.
Dataset Type: Synthetic
"""
import os

# ── Project root ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Paths ──
DATA_RAW_DIR = PROJECT_ROOT
DATA_CLEAN_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
EDA_DIR = os.path.join(OUTPUT_DIR, "eda")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
FORECASTS_DIR = os.path.join(OUTPUT_DIR, "forecasts")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")
DASHBOARD_DIR = os.path.join(OUTPUT_DIR, "dashboard")

# Ensure dirs exist
for d in [DATA_CLEAN_DIR, EDA_DIR, MODELS_DIR, FORECASTS_DIR, REPORTS_DIR, DASHBOARD_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Location mapping (file index → neutral label) ──
LOCATION_MAP = {
    1: "Hospital_1",
    2: "Hospital_2",
    3: "Hospital_3",
    4: "Hospital_4",
    5: "Hospital_5",
    6: "Hospital_6",
}

# Source files
SOURCE_FILES = {
    loc: os.path.join(DATA_RAW_DIR, f"Synthetic Data {idx}.xlsx")
    for idx, loc in LOCATION_MAP.items()
}

# ── Train / Test split ──
TRAIN_END = "2025-12-31"
TEST_START = "2026-01-01"
TEST_END = "2026-06-30"
FORECAST_START = "2026-07-01"
FORECAST_END = "2026-12-31"
FORECAST_MONTHS = 6

# ── Color palette (premium, harmonious) ──
COLORS = {
    "Hospital_1": "#6366F1",   # Indigo
    "Hospital_2": "#8B5CF6",   # Violet
    "Hospital_3": "#EC4899",   # Pink
    "Hospital_4": "#F59E0B",   # Amber
    "Hospital_5": "#10B981",   # Emerald
    "Hospital_6": "#06B6D4",   # Cyan
}

LOCATION_ORDER = list(LOCATION_MAP.values())

# ── Plot style ──
PLOT_STYLE = {
    "figure.facecolor": "#0F172A",
    "axes.facecolor": "#1E293B",
    "axes.edgecolor": "#334155",
    "axes.labelcolor": "#E2E8F0",
    "text.color": "#E2E8F0",
    "xtick.color": "#94A3B8",
    "ytick.color": "#94A3B8",
    "grid.color": "#334155",
    "grid.alpha": 0.5,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "figure.titlesize": 16,
    "figure.titleweight": "bold",
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.facecolor": "#0F172A",
}

WATERMARK = "Dataset Type: Synthetic"
