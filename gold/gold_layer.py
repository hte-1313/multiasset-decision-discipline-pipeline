
"""
gold/gold_layer.py
------------------
Part 3: Gold layer — cross-sectional features, leakage-free targets,
         Pregel distributed GCN, and GCS write.

Three sub-stages:
  1. Cross-sectional features  — require the full panel (partitionBy date)
  2. Leakage-free 5-day target — ln(P_{t+5}/P_t) binarised as hold/exit
  3. Pregel GCN embeddings     — distributed message passing on correlation graph
"""

import numpy as np
import pandas as pd
from functools import reduce

from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

from config import (SILVER_PATH, GOLD_PATH, HOLD_DAYS,
                    INITIAL_TRAIN_DAYS, REFIT_EVERY,
                    PREGEL_LAYERS, CORR_EDGE_THRESHOLD)
from assets import GNN_NODE_FEATURES
from silver.silver_layer import safe_div


# ── Cross-sectional features ───────────────────────────────────────────────

def run_gold_cross_sectional(silver: DataFrame) -> DataFrame:
    """
    Compute features that require the full asset panel on each date.

    These cannot be computed in Silver (per-ticker only) because they need
    partitionBy("date") windows which shuffle data across tickers.

    Features added:
      corr_spy_60  : rolling 60-day correlation with SPY return
      beta_60      : rolling 60-day market beta vs SPY
      idio_vol_60  : idiosyncratic volatility after stripping market component
                     idio_vol = sqrt(vol^2 - beta^2 * spy_vol^2) * sqrt(252)
      ret_z_60     : return normalised by daily vol (Sharpe-like signal)
      xs_ret_rank  : percent_rank of daily return across all tickers (per date)
      xs_vol_rank  : percent_rank of vol_20 across all tickers (per date)
      xs_mom_rank  : percent_rank of mom_12_1 across all tickers (per date)
      vol_ratio    : vol_20 / vol_60 (regime indicator)
      vol_of_vol   : rolling std of vol_20 (second-order uncertainty)

    Cross-sectional ranks are normalised to [0,1] via percent_rank() so they
    are comparable across dates when active instrument count changes.
    """
    # SPY reference series (small, cached)
    spy = (silver
        .filter(F.col("ticker") == "SPY")
        .select("date",
                F.col("ret_1d").alias("spy_ret_1d"),
                F.col("vol_20").alias("spy_vol_20"))
        .cache())

    df = silver.join(spy, on="date", how="left")

    # applyInPandas: per-ticker rolling corr, beta, idio_vol
    base_schema = df.schema
    extra_fields = [
        T.StructField("corr_spy_60", T.DoubleType(), True),
        T.StructField("beta_60",     T.DoubleType(), True),
        T.StructField("idio_vol_60", T.DoubleType(), True),
    ]
    cross_schema = T.StructType(base_schema.fields + extra_fields)

    def _add_cross(pdf: pd.DataFrame) -> pd.DataFrame:
        pdf = pdf.sort_values("date").copy()
        ret     = pdf["ret_1d"].fillna(0)
        spy_ret = pdf["spy_ret_1d"].fillna(0)
        spy_vol = pdf["spy_vol_20"].fillna(0)
        vol     = pdf["vol_20"].fillna(0)

        # Rolling 60-day correlation with SPY
        corr = ret.rolling(60, min_periods=20).corr(spy_ret)
        pdf["corr_spy_60"] = corr

        # Beta = corr * (vol_asset / vol_spy)
        pdf["beta_60"] = corr * vol / spy_vol.replace(0, np.nan)

        # Idiosyncratic vol = sqrt(max(vol^2 - beta^2 * spy_vol^2, 0))
        beta2   = pdf["beta_60"].fillna(0) ** 2
        spy_vol2 = spy_vol ** 2
        idio2   = (vol ** 2 - beta2 * spy_vol2).clip(lower=0)
        pdf["idio_vol_60"] = np.sqrt(idio2)
        return pdf

    df = df.groupBy("ticker").applyInPandas(_add_cross, schema=cross_schema)

    # Cross-sectional percent ranks (per date, all tickers)
    w_date_ret = Window.partitionBy("date").orderBy("ret_1d")
    w_date_vol = Window.partitionBy("date").orderBy("vol_20")
    w_date_mom = Window.partitionBy("date").orderBy("mom_12_1")
    df = df.withColumn("xs_ret_rank", F.percent_rank().over(w_date_ret))
    df = df.withColumn("xs_vol_rank", F.percent_rank().over(w_date_vol))
    df = df.withColumn("xs_mom_rank", F.percent_rank().over(w_date_mom))

    # Derived features
    w20t = Window.partitionBy("ticker").orderBy("date").rowsBetween(-19, 0)
    df = df.withColumn("vol_ratio",  safe_div(F.col("vol_20"), F.col("vol_60")))
    df = df.withColumn("vol_of_vol", F.stddev("vol_20").over(w20t))
    # ret_z_60: daily Sharpe-like signal
    df = df.withColumn("ret_z_60",
        safe_div(F.col("ret_1d"), F.col("vol_20") / F.sqrt(F.lit(252.0))))
    df = df.drop("spy_ret_1d", "spy_vol_20")

    spy.unpersist()
    return df


# ── Leakage-free targets ───────────────────────────────────────────────────

def add_leakage_free_targets(df: DataFrame) -> DataFrame:
    """
    Construct the 5-day forward return target with zero information leakage.

    Correct approach:
      ret_5d_fwd = ln(adj_close_{t+5} / adj_close_t)

    The CORRECT method uses F.lead(adj_close, 5) to look forward in time.
    The INCORRECT method uses shift(ret_5d, -5) which shifts the already-computed
    5-day return backwards — this mixes current features with future returns that
    overlap the feature window by up to 5 days.

    Binary classification target:
      target_5d = 1 (hold)  if ret_5d_fwd > 0
      target_5d = 0 (exit)  otherwise

    Rows where adj_close_{t+5} does not exist (last HOLD_DAYS rows per ticker)
    are dropped — there is nothing to predict against.

    Also stores: target_ret_5d (continuous log return for Bayesian PEFT updates)
    """
    w = Window.partitionBy("ticker").orderBy("date")

    df = df.withColumn("adj_close_fwd5", F.lead("adj_close", HOLD_DAYS).over(w))
    df = df.withColumn("target_ret_5d",
        F.when(
            F.col("adj_close_fwd5").isNotNull() & (F.col("adj_close") > 0),
            F.log(F.col("adj_close_fwd5") / F.col("adj_close"))
        ))
    df = df.withColumn("target_5d",
        F.when(F.col("target_ret_5d") > 0, 1).otherwise(0))

    # Drop rows where target cannot be computed (last HOLD_DAYS rows per ticker)
    df = df.filter(F.col("target_ret_5d").isNotNull())
    df = df.drop("adj_close_fwd5")

    # Alias for compatibility
    df = df.withColumn("ret_5d_fwd", F.col("target_ret_5d"))
    return df


# ── Pregel distributed GCN ────────────────────────────────────────────────

def compute_pregel_features(gold_sdf: DataFrame,
                             refit_dates: list,
                             n_layers: int = PREGEL_LAYERS,
                             threshold: float = CORR_EDGE_THRESHOLD,
                             spark: SparkSession = None) -> DataFrame:
    """
    Distributed GCN node embeddings via Pregel-style message passing.

    Mathematical formulation (one superstep = one GCN layer):
      h_v^(l+1) = tanh( sum_{u in N(v)} w_uv * h_u^(l) / sum_weights )

    In each superstep:
      Message from u to v: h_u^(l) * edge_weight
      Aggregation at v: weighted average = sum(messages) / sum(weights)
      Update: tanh(aggregated)

    Graph construction:
      - Pulls 90-day trailing return panel for each refit date.
      - Computes full pairwise correlation matrix.
      - Draws edges where |rho| > threshold, weighted by |rho|.
      - Runs n_layers message-passing iterations in pandas (per refit date).

    Returns a Spark DataFrame with columns: ticker, date, pregel_score.
    Any ticker absent from a refit date graph receives neutral score 0.5.
    """
    all_emb_frames = []

    print(f"Computing Pregel features for {len(refit_dates)} refit dates...")
    for i, rd in enumerate(refit_dates):
        if i % 50 == 0:
            print(f"  Pregel: {i}/{len(refit_dates)} dates processed")
        try:
            # Node features snapshot
            snap = (gold_sdf
                .filter(F.col("date") == rd)
                .select(["ticker"] + [c for c in GNN_NODE_FEATURES
                                       if c in gold_sdf.columns])
                .dropna()
                .toPandas())

            if len(snap) < 5:
                continue

            feat_cols = [c for c in GNN_NODE_FEATURES if c in snap.columns]
            for col in feat_cols:
                mu = snap[col].mean(); sd = snap[col].std()
                snap[col] = (snap[col] - mu) / (sd + 1e-9)
            snap["score"] = snap[feat_cols].mean(axis=1)

            # Build edges from 90-day return correlation
            ret_wide = (gold_sdf
                .filter(
                    (F.col("date") >= F.date_sub(F.lit(str(rd)), 90)) &
                    (F.col("date") <= F.lit(str(rd)))
                )
                .select("date", "ticker", "ret_1d")
                .toPandas()
                .pivot(index="date", columns="ticker", values="ret_1d")
                .dropna(how="all"))

            corr_mat = ret_wide.corr()
            edge_rows = []
            for t1 in corr_mat.columns:
                for t2 in corr_mat.columns:
                    if t1 != t2 and abs(corr_mat.loc[t1, t2]) > threshold:
                        edge_rows.append({
                            "src": t1, "dst": t2,
                            "weight": float(abs(corr_mat.loc[t1, t2]))
                        })

            if not edge_rows:
                continue

            edges_pdf = pd.DataFrame(edge_rows)
            nodes_pdf = snap[["ticker", "score"]].copy()

            # n_layers message-passing supersteps
            for _ in range(n_layers):
                merged = edges_pdf.merge(
                    nodes_pdf.rename(columns={"ticker": "src", "score": "src_score"}),
                    on="src"
                )
                merged["weighted_msg"] = merged["src_score"] * merged["weight"]
                agg = merged.groupby("dst").agg(
                    msg_sum=("weighted_msg", "sum"),
                    weight_sum=("weight", "sum")
                ).reset_index()
                agg["new_score"] = np.tanh(
                    agg["msg_sum"] / agg["weight_sum"].replace(0, np.nan)
                )
                nodes_pdf = nodes_pdf.merge(
                    agg[["dst", "new_score"]].rename(columns={"dst": "ticker"}),
                    on="ticker", how="left"
                )
                nodes_pdf["score"] = nodes_pdf["new_score"].fillna(nodes_pdf["score"])
                nodes_pdf = nodes_pdf.drop(columns=["new_score"])

            nodes_pdf["date"]         = rd
            nodes_pdf["pregel_score"] = nodes_pdf["score"]
            emb_sdf = spark.createDataFrame(
                nodes_pdf[["ticker", "date", "pregel_score"]]
            )
            all_emb_frames.append(emb_sdf)

        except Exception as e:
            print(f"  Pregel skipped for {rd}: {e}")
            continue

    if not all_emb_frames:
        print("No Pregel embeddings computed — pregel_score = 0.5")
        return gold_sdf.select("ticker", "date").withColumn("pregel_score", F.lit(0.5))

    print(f"Pregel complete: {len(all_emb_frames)} dates computed.")
    return reduce(lambda a, b: a.union(b), all_emb_frames)


# ── Gold orchestrator ──────────────────────────────────────────────────────

def run_gold(spark: SparkSession) -> None:
    """
    Full Gold layer:
      1. Load Silver from GCS.
      2. Add cross-sectional features (corr, beta, idio_vol, xs_ranks).
      3. Add leakage-free 5-day targets.
      4. Compute Pregel GCN embeddings on refit dates.
      5. Join Pregel scores back onto Gold.
      6. Write to GCS partitioned by asset_class.
      7. Unpersist Silver to free executor memory.
    """
    print("Loading Silver layer...")
    silver = spark.read.parquet(SILVER_PATH).cache()

    print("Computing cross-sectional features...")
    gold = run_gold_cross_sectional(silver)

    print("Adding 5-day targets...")
    gold = add_leakage_free_targets(gold)

    # Refit dates: every REFIT_EVERY trading days after initial training window
    dates_pd = (gold
        .filter(F.col("ticker") == "SPY")
        .select("date").orderBy("date")
        .toPandas()["date"].sort_values().tolist())
    refit_dates = dates_pd[INITIAL_TRAIN_DAYS::REFIT_EVERY]
    print(f"Computing Pregel on {len(refit_dates)} refit dates...")

    pregel_sdf = compute_pregel_features(gold, refit_dates, spark=spark)

    gold = gold.join(pregel_sdf, on=["ticker", "date"], how="left")
    gold = gold.withColumn("pregel_score",
        F.coalesce(F.col("pregel_score"), F.lit(0.5)))

    (gold.write
        .mode("overwrite")
        .partitionBy("asset_class")
        .parquet(GOLD_PATH))

    n = spark.read.parquet(GOLD_PATH).count()
    print(f"Gold saved: {n:,} rows → {GOLD_PATH}")
    silver.unpersist()
