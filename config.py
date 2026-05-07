"""
config.py
---------
All project-wide constants for the ST446 MultiAsset Decision Pipeline.
"""

# ── GCS paths ──────────────────────────────────────────────────────────────
GCS_BUCKET   = "gs://st446-covid-sujay-2026-9f3a" ### bucket used on GCP
DATA_ROOT    = GCS_BUCKET + "/pipeline_v2"
BRONZE_PATH  = DATA_ROOT + "/bronze"
SILVER_PATH  = DATA_ROOT + "/silver"
GOLD_PATH    = DATA_ROOT + "/gold"
RESULTS_PATH = DATA_ROOT + "/results"

# ── Date range ─────────────────────────────────────────────────────────────
START_DATE = "2005-01-01"
END_DATE   = "2025-01-01"

# ── Walk-forward training ──────────────────────────────────────────────────
INITIAL_TRAIN_DAYS = 504   # 2 years of trading days
REFIT_EVERY        = 63    # refit every ~quarter
MIN_OBS            = 252   # minimum obs required for any model

# ── Prediction horizon ─────────────────────────────────────────────────────
HOLD_DAYS       = 5
DECISION_THRESH = 0.50

# ── Risk management ────────────────────────────────────────────────────────
STOP_LOSS_PCT      = -0.02    # -2% log-return stop-loss
TRAILING_STOP_PCT  = -0.015   # -1.5% trailing stop

# ── Graph / GNN ────────────────────────────────────────────────────────────
CORR_EDGE_THRESHOLD = 0.30
PREGEL_LAYERS       = 3
GNN_EPOCHS          = 20
GNN_LR              = 1e-3
GNN_HIDDEN          = 32
GNN_LAYERS          = 2

# ── Bayesian PEFT priors ───────────────────────────────────────────────────
SIGMA_SQ_INIT = 1e-3 ### Alterable based on prior requirements
DELTA_INIT    = 1.0 ### Alterable based on prior requirements

# ── Evaluation ─────────────────────────────────────────────────────────────
TRANSACTION_COSTS_BPS  = [0, 5, 10, 20]
BOOTSTRAP_N            = 1000
ANNUALISATION_FACTOR   = 252

# ── Dev filter (set to None for full universe) ─────────────────────────────
DEV_FILTER = None
# DEV_FILTER = ["SPY", "AAPL", "EURUSD=X", "GC=F", "ES=F"]

# ── Assets excluded from evaluation metrics ────────────────────────────────
# VX=F has inverted economics (steep contango) so excluded from perf metrics
EVAL_EXCLUDE = {"VX=F"}
