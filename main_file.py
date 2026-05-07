"""
main.py
=======
ST446 MultiAsset Decision Pipeline — full end-to-end runner.

Layers:
  Bronze → Silver → Gold → Models → Evaluation → Benchmark → Visualisation

Folder structure expected:
  config.py
  assets.py
  spark_session.py
  bronze/downloader.py
  silver/silver_layer.py
  gold/gold_layer.py
  models/walk_forward.py
  models/model_suite.py
  evaluation/metrics.py
  benchmark/spark_benchmark.py
  visualisation/plots.py

Run:
  python main.py
"""

import sys, os, warnings, subprocess
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Install dependencies ───────────────────────────────────────────────────
for pkg in ["yfinance", "xgboost"]:
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])

# ── Add sub-packages to path ───────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# ── Imports ────────────────────────────────────────────────────────────────
from config import (
    GOLD_PATH, RESULTS_PATH,
    INITIAL_TRAIN_DAYS, REFIT_EVERY, HOLD_DAYS, DECISION_THRESH,
)
from assets import (
    TICKERS, ASSET_CLASS_MAP,
    FEATURE_COLS, BAYESIAN_FEATURE_COLS, GNN_NODE_FEATURES,
    EVAL_EXCLUDE,
)
from spark_session import get_spark

from bronze.downloader import run_bronze
from silver.silver_layer import run_silver
from gold.gold_layer import run_gold

from models.walk_forward import run_model_distributed, RESULT_SCHEMA
from models.model_suite import (
    make_always_hold_fn, xgb_fn,
    make_bayesian_peft_fn, calibrate_prior,
    make_pregel_signal_fn, HAS_TORCH,
)

from evaluation.metrics import (
    compute_sharpe, run_full_evaluation,
    run_tc_sensitivity, run_dm_tests,
)
from benchmark.spark_benchmark import (
    run_spark_benchmark, run_pandas_vs_spark_comparison,
)
from visualisation.plots import (
    plot_equity_and_drawdown,
    plot_win_rate_and_stoploss,
    plot_spy_signals,
)

from pyspark.sql import functions as F

print("All imports OK.")


# =============================================================================
# SETUP: Spark session
# =============================================================================
# Spark is configured with:
#   - Adaptive Query Execution (AQE) for dynamic partition coalescing
#   - Arrow-based pandas UDF serialisation (columnar, fast)
#   - UTC timezone for consistent date handling across partitions
#   - GraphFrames JAR for Pregel distributed graph computation

spark = get_spark()
print(f"Spark {spark.version} | parallelism={spark.sparkContext.defaultParallelism}")

print(f"Universe: {len(TICKERS)} tickers across "
      f"{len(set(ASSET_CLASS_MAP.values()))} asset classes")


# =============================================================================
# PART 1 — Bronze Layer: Raw OHLCV ingestion
# =============================================================================
# Downloads raw price data for all 119 instruments.
#
# Retry logic:
#   Up to 3 attempts with exponential backoff (2^attempt seconds) per ticker.
#
# Delisted removal:
#   Tickers with < 252 valid rows in the trailing 2-year window are excluded.
#   This keeps the universe to instruments actively trading at analysis time.
#
# Storage:
#   Gzip-compressed Parquet, partitioned by asset_class on GCS.
#   Bronze is the single source of truth for all downstream layers.

raw_frames = run_bronze(spark)


# =============================================================================
# PART 2 — Silver Layer: 35-dimensional feature engineering
# =============================================================================
# Two approaches used:
#
# (A) Spark window functions (equal-weight aggregations):
#   ret_1d/5d/21d/63d/252d, mom_12_1, vol_20/60, vol_ratio, vol_of_vol,
#   atr_14 = mean(max(H-L, |H-prev_c|, |L-prev_c|)) over 14 days
#   sma_20/60/200, dist_sma_*, trend_strength = |dist_sma_20|
#   Bollinger: bb_pctb = (price - lower) / (upper - lower)  [0.5 on zero width]
#              bb_bandwidth = (upper - lower) / mid
#   amihud = |ret_1d| / dollar_volume  [Amihud illiquidity proxy]
#   turnover_z_60, vol_z_60, hl_range, intraday_ret, gap
#   drawdown = adj_close / rolling_max_252 - 1
#   max_drawdown_252, drawdown_speed = drawdown - lag(drawdown, 5)
#   Calendar: dow, month, is_month_end, is_quarter_end
#
# (B) applyInPandas (recursive, one ticker per executor):
#   ema_12/26: ewm(span, adjust=False) — true exponential decay
#   macd = ema_12 - ema_26
#   macd_signal = ewm(span=9) of macd
#   rsi_14: Wilder smoothing alpha=1/14 — correct industry definition
#   downside_vol_20: rolling std of negative returns only, annualised
#   skew_60, kurt_60: rolling distributional shape features
#
# Amihud winsorised at 99th percentile (approxQuantile) to prevent
# illiquidity spikes in thinly traded instruments from distorting training.

run_silver(spark)


# =============================================================================
# PART 3 — Gold Layer: Cross-sectional features, targets, Pregel GCN
# =============================================================================
# Gold breaks the per-ticker isolation of Silver in two ways:
#
# (A) Cross-sectional features (require partitionBy date — full shuffle):
#   corr_spy_60: rolling 60-day correlation with SPY
#   beta_60 = corr * (vol_asset / vol_spy)
#   idio_vol_60 = sqrt(max(vol^2 - beta^2 * spy_vol^2, 0)) * sqrt(252)
#   xs_ret_rank, xs_vol_rank, xs_mom_rank = percent_rank across all tickers
#   ret_z_60 = ret_1d / (vol_20 / sqrt(252))  [daily Sharpe-like signal]
#
# (B) Leakage-free 5-day target:
#   CORRECT:   target_ret_5d = ln(adj_close_{t+5} / adj_close_t) via F.lead
#   INCORRECT: shift(ret_5d, -5) — mixes features with overlapping future returns
#   target_5d = 1 (hold) if target_ret_5d > 0, else 0 (exit)
#   Last HOLD_DAYS rows per ticker dropped (no forward price available)
#
# (C) Pregel distributed GCN:
#   Graph construction: pairwise correlation over trailing 90 days
#   Edges drawn where |rho| > CORR_EDGE_THRESHOLD (default 0.30)
#   Message passing (n_layers supersteps):
#     h_v^(l+1) = tanh( sum_{u in N(v)} w_uv * h_u^(l) / sum_weights )
#   pregel_score = post-message-passing node embedding
#   Any ticker absent from a refit-date graph receives neutral score 0.5

run_gold(spark)


# =============================================================================
# PART 4 — Load Gold and prepare feature matrices
# =============================================================================
# Gold is loaded from GCS and cached because it is reused throughout model exec.
# Feature lists are filtered against actual Gold schema (protective — optional
# upstream features missing do not crash downstream models).
#
# Unconditional hold rate = empirical P(positive 5-day return) across the panel.
# This is the baseline win rate for Always-Hold.
# Active models must beat passive market exposure, not a 50/50 coin flip.
#
# RESULT_SCHEMA (common output for all models):
#   ticker, date, asset_class, signal, p_hold, ret_5d_fwd, model

gold = spark.read.parquet(GOLD_PATH).cache()
gold = gold.withColumn("ret_5d_fwd", F.col("target_ret_5d"))

feat_cols  = [c for c in FEATURE_COLS       if c in gold.columns]
bay_cols   = [c for c in BAYESIAN_FEATURE_COLS if c in gold.columns]
gnn_cols   = [c for c in GNN_NODE_FEATURES  if c in gold.columns]

gold_pd = gold.toPandas()
gold_pd["date"] = pd.to_datetime(gold_pd["date"])
gold_pd = gold_pd.sort_values(["ticker", "date"]).reset_index(drop=True)

hold_rate = gold_pd["target_5d"].mean()
print(f"Gold: {len(gold_pd):,} rows | {gold_pd['ticker'].nunique()} tickers")
print(f"Date range: {gold_pd['date'].min().date()} → {gold_pd['date'].max().date()}")
print(f"Unconditional hold rate: {hold_rate:.1%}  (Always-Hold baseline win rate)")

gold_filtered = gold.filter(F.col("target_5d").isNotNull())


# =============================================================================
# PART 5 — Model Suite
# =============================================================================

# ── Model 1: Always-Hold baseline ────────────────────────────────────────
# Always emits signal=1. P(hold) = unconditional positive 5-day target rate.
# The null trading model — earns the market return with zero active decisions.
# Stronger than random guessing because assets have positive drift.

always_hold_fn = make_always_hold_fn(hold_rate)
results_always_hold = run_model_distributed(
    always_hold_fn, "always_hold", [feat_cols[0]], gold_filtered
)
print(f"Always-Hold predictions: {len(results_always_hold):,}")

# Worker package check (ensures xgboost is available on Spark executors)
def _check_worker(_):
    try:
        import xgboost
        return f"ok: xgb={xgboost.__version__}"
    except ImportError as e:
        return f"missing: {e}"

worker_check = spark.sparkContext.parallelize(range(4), 4).map(_check_worker).collect()
print(f"Worker check: {set(worker_check)}")

# ── Model 2: XGBoost ──────────────────────────────────────────────────────
# Non-linear additive tree ensemble on full feature set X^{full}.
# Fitted independently inside each ticker's walk-forward loop.
# Moderate regularisation: n_estimators=100, max_depth=4, lr=0.05.

results_xgb = run_model_distributed(
    xgb_fn, "xgboost", feat_cols, gold_filtered
)
print(f"XGBoost predictions: {len(results_xgb):,}")

# ── Model 3: Bayesian Hybrid PEFT ─────────────────────────────────────────
# Sequential online Bayesian linear classifier.
# Sherman-Morrison rank-1 posterior update at each observation.
# Posterior predictive probability via probit approximation:
#   p = sigma(mu / sqrt(1 + pi/8 * sigma^2))
# Prior calibrated from Gold panel:
#   sigma_sq = var(target_ret_5d)
#   delta    = sigma_sq / mean_feature_variance

sigma_sq, delta = calibrate_prior(gold_pd, bay_cols)
bayes_fn = make_bayesian_peft_fn(bay_cols, sigma_sq, delta)
results_bayes = run_model_distributed(
    bayes_fn, "bayesian_peft",
    bay_cols + ["target_ret_5d"], gold_filtered,
)
print(f"Bayesian PEFT predictions: {len(results_bayes):,}")

# ── Model 4A: Pregel GNN ──────────────────────────────────────────────────
# Logistic regression on Pregel-augmented features.
# pregel_score encodes distributed graph centrality from the Gold layer.
# Tests whether cross-asset correlation network adds predictive value.

pregel_fn = make_pregel_signal_fn(gnn_cols)
results_pregel = run_model_distributed(
    pregel_fn, "pregel_gnn",
    gnn_cols + ["pregel_score"], gold_filtered,
)
print(f"Pregel GNN predictions: {len(results_pregel):,}")

# ── Model 4B: GraphSAGE (conditional on PyTorch) ─────────────────────────
# Trainable GNN with neighbourhood aggregation:
#   h_v^(k) = sigma(W^(k) · CONCAT(h_v^(k-1), MEAN_{u in N(v)} h_u^(k-1)))
# Refits every REFIT_EVERY * 5 days; trains on up to 50 daily graphs.

if HAS_TORCH:
    from models.model_suite import run_graphsage_walkforward
    results_graphsage = run_graphsage_walkforward(gold_pd, gnn_cols)
    print(f"GraphSAGE predictions: {len(results_graphsage):,}")
else:
    results_graphsage = pd.DataFrame(
        columns=["ticker","date","signal","p_hold","ret_5d_fwd","model","asset_class"]
    )
    print("PyTorch not available — GraphSAGE skipped.")

# ── Save results to GCS ───────────────────────────────────────────────────
results_always_hold.to_parquet(f"{RESULTS_PATH}/results_always_hold.parquet", index=False)
results_xgb.to_parquet(f"{RESULTS_PATH}/results_xgb.parquet",         index=False)
results_bayes.to_parquet(f"{RESULTS_PATH}/results_bayes.parquet",       index=False)
results_pregel.to_parquet(f"{RESULTS_PATH}/results_pregel_gnn.parquet", index=False)
print("All results saved.")

# ── Combine for evaluation ────────────────────────────────────────────────
all_results = pd.concat(
    [results_always_hold, results_xgb, results_bayes, results_pregel],
    ignore_index=True,
)
all_results["date"]       = pd.to_datetime(all_results["date"])
all_results["ret_5d_fwd"] = pd.to_numeric(all_results["ret_5d_fwd"], errors="coerce")
all_results["signal"]     = pd.to_numeric(all_results["signal"],     errors="coerce").astype("Int64")
all_results["p_hold"]     = pd.to_numeric(all_results["p_hold"],     errors="coerce")
all_results["signal_ret"] = all_results["ret_5d_fwd"] * all_results["signal"].fillna(0)

print(f"Total predictions: {len(all_results):,}")


# =============================================================================
# PART 6 — Financial Evaluation
# =============================================================================
# Sharpe ratio:
#   Sharpe = mean(R_p) / std(R_p) * sqrt(annualisation_factor / REFIT_EVERY)
#
# Sortino ratio:
#   Sortino = mean(R_p) / std(R_p[R_p < 0]) * sqrt(annualisation_factor / REFIT_EVERY)
#
# Calmar ratio:
#   Calmar = annualised_return / |max_drawdown|
#
# Max Drawdown:
#   MDD = min_t( W_t / max_{tau<=t} W_tau - 1 )
#
# Stop-loss simulation:
#   For each signal=1 date, monitor for HOLD_DAYS days.
#   Exit early if cumulative log-return < STOP_LOSS_PCT (-2%).
#
# DM test (Diebold-Mariano 1995):
#   H0: E[L(e1) - L(e2)] = 0   where L = squared loss
#   Negative DM stat → active model has lower loss → wins
#   Newey-West correction (h-1 lags) for overlapping 5-day returns.
#
# Bootstrap CI:
#   95% CI for Sharpe via 1000 bootstrap resamples with replacement.

MODELS_TO_EVAL = ["always_hold", "xgboost", "bayesian_peft", "pregel_gnn"]

print("\nRunning full evaluation suite...")
metrics_df = run_full_evaluation(all_results, gold_pd, MODELS_TO_EVAL)
print("\nFull metrics table:")
print(metrics_df.set_index("model").to_string())

print("\nTransaction cost sensitivity:")
tc_df = run_tc_sensitivity(all_results, MODELS_TO_EVAL)
print(tc_df.round(3))

print("\nDiebold-Mariano tests vs Always-Hold:")
dm_df = run_dm_tests(all_results, MODELS_TO_EVAL)


# =============================================================================
# PART 7 — Spark Distributed Computing Benchmark
# =============================================================================
# Benchmarks the Silver feature pipeline under different partition counts.
# Transformations benchmarked:
#   (1) vol_ratio     = vol_20 / vol_60          [per-row, no shuffle]
#   (2) xs_ret_rank   = percent_rank(ret_1d)      [per-date, full shuffle]
#   (3) ret_z         = ret_1d / (vol_20/sqrt252) [per-row, no shuffle]
#
# Also compares single-node pandas vs Spark on the same workload.
# Results saved to GCS for the scaling table in the report.

bench_df = run_spark_benchmark(spark)
comparison = run_pandas_vs_spark_comparison(spark)


# =============================================================================
# PART 8 — Visualisation
# =============================================================================

plot_equity_and_drawdown(all_results, metrics_df, MODELS_TO_EVAL)
plot_win_rate_and_stoploss(all_results, metrics_df, MODELS_TO_EVAL)
plot_spy_signals(all_results, MODELS_TO_EVAL)

print("\n========== PIPELINE COMPLETE ==========")
