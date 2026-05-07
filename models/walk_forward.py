"""
models/walk_forward.py
----------------------
Part 4: Common walk-forward harness used by every model.

make_walk_forward_fn  – wraps a model function in a ticker-level
                        chronological walk-forward loop.
run_model_distributed – executes the walk-forward via Spark applyInPandas,
                        one ticker per executor partition.

Walk-forward design:
  At each prediction date t:
    train  = pdf.iloc[:t - HOLD_DAYS]   (label cutoff: only closed labels)
    feat   = pdf.iloc[t:t+1]            (feature row at prediction date)
  Predictions are generated every REFIT_EVERY days after INITIAL_TRAIN_DAYS.

Label cutoff:
  A row at t-2 has a target ending at t+3 — not yet observable at time t.
  The training set therefore includes only rows whose forward return window
  has fully closed before the prediction date:
    label_cutoff = t - HOLD_DAYS

Leakage control:
  pred_indices = range(INITIAL_TRAIN_DAYS, len(pdf) - HOLD_DAYS, REFIT_EVERY)
  This creates a strictly causal walk-forward: no model sees labels that
  would not have existed at the prediction date.

Parallelism:
  groupBy("ticker").applyInPandas(fn, schema=RESULT_SCHEMA)
  Each ticker is one independent chronological sequence — the correct unit
  of parallelism because each ticker has a moderate number of rows and
  the full panel has many independent ticker histories.
"""

import time
import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType,
)

from config import INITIAL_TRAIN_DAYS, REFIT_EVERY, HOLD_DAYS, MIN_OBS
from assets import EVAL_EXCLUDE


# Common output schema for every model
RESULT_SCHEMA = StructType([
    StructField("ticker",      StringType(),  False),
    StructField("date",        StringType(),  False),
    StructField("asset_class", StringType(),  True),
    StructField("signal",      IntegerType(), True),
    StructField("p_hold",      DoubleType(),  True),
    StructField("ret_5d_fwd",  DoubleType(),  True),
    StructField("model",       StringType(),  False),
])


def make_walk_forward_fn(model_fn, model_name: str, feat_cols: list):
    """
    Build a ticker-level walk-forward function around model_fn.

    model_fn signature: (train_X, train_y, feat_X) -> (signal: int, p_hold: float)

    Parameters
    ----------
    model_fn   : callable  – model-specific fit-predict function
    model_name : str       – label written to the 'model' column
    feat_cols  : list[str] – feature columns to pass to the model

    Returns
    -------
    Callable that accepts one ticker's full pd.DataFrame and returns
    a pd.DataFrame matching RESULT_SCHEMA.
    """
    def _run(pdf: pd.DataFrame) -> pd.DataFrame:
        pdf    = pdf.sort_values("date").reset_index(drop=True)
        ticker = pdf["ticker"].iloc[0] if "ticker" in pdf.columns else "?"

        if len(pdf) < INITIAL_TRAIN_DAYS + HOLD_DAYS + 1:
            return pd.DataFrame(columns=RESULT_SCHEMA.fieldNames())

        ret_col = (
            "target_ret_5d" if "target_ret_5d" in pdf.columns
            else "ret_5d_fwd" if "ret_5d_fwd" in pdf.columns
            else None
        )

        rows = []
        pred_indices = list(range(INITIAL_TRAIN_DAYS, len(pdf) - HOLD_DAYS, REFIT_EVERY))

        for t in pred_indices:
            # label_cutoff: only include rows whose 5-day label has closed
            label_cutoff = t - HOLD_DAYS
            train  = pdf.iloc[:label_cutoff].copy()
            feat   = pdf.iloc[t:t+1].copy()

            train_X = train[feat_cols].fillna(0.0)
            train_y = train["target_5d"].fillna(0)
            feat_X  = feat[feat_cols].fillna(0.0)

            if train_X.shape[0] < MIN_OBS or feat_X.isnull().all().all():
                continue

            try:
                signal, p_hold = model_fn(train_X, train_y, feat_X)
            except Exception:
                continue

            raw_date = pdf["date"].iloc[t]
            date_str = (raw_date.strftime("%Y-%m-%d")
                        if hasattr(raw_date, "strftime") else str(raw_date))
            ret_val  = float(pdf[ret_col].iloc[t]) if ret_col else 0.0

            rows.append({
                "ticker":      ticker,
                "date":        date_str,
                "asset_class": pdf["asset_class"].iloc[t] if "asset_class" in pdf.columns else "",
                "signal":      int(signal),
                "p_hold":      float(p_hold),
                "ret_5d_fwd":  ret_val,
                "model":       model_name,
            })

        if not rows:
            return pd.DataFrame(columns=RESULT_SCHEMA.fieldNames())
        return pd.DataFrame(rows)

    return _run


def run_model_distributed(model_fn,
                           model_name: str,
                           feat_cols: list,
                           gold_sdf: DataFrame) -> pd.DataFrame:
    """
    Execute a walk-forward model in Spark via groupBy/applyInPandas.

    Each ticker's full history is sent to one executor as a pandas DataFrame.
    Results are collected back to the driver as a unified pandas DataFrame.

    Parameters
    ----------
    model_fn   : callable
    model_name : str
    feat_cols  : list[str]
    gold_sdf   : Spark DataFrame (Gold layer, target_5d not null)

    Returns
    -------
    pd.DataFrame with RESULT_SCHEMA columns.
    """
    print(f"Running {model_name} (distributed via applyInPandas)...")
    t0 = time.time()

    fn = make_walk_forward_fn(model_fn, model_name, feat_cols)

    result_sdf = (gold_sdf
        .filter(~F.col("ticker").isin(list(EVAL_EXCLUDE)))
        .groupBy("ticker")
        .applyInPandas(fn, schema=RESULT_SCHEMA))

    result_pd = result_sdf.toPandas()
    result_pd["date"] = pd.to_datetime(result_pd["date"])

    elapsed = time.time() - t0
    print(f"  {model_name}: {len(result_pd):,} predictions · "
          f"{result_pd['ticker'].nunique()} tickers · {elapsed:.1f}s")
    return result_pd
