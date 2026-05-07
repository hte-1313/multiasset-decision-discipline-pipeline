
"""
silver/silver_layer.py
----------------------
Part 2: Silver layer — 35-dimensional feature engineering.

Two complementary approaches:
  1. Native Spark window functions for equal-weight aggregations
     (returns, SMAs, Bollinger, Amihud, liquidity, drawdown, calendar).
  2. applyInPandas for recursive/state-dependent computations
     (EMA, MACD, RSI with Wilder smoothing, downside vol, skew, kurt).

Why two approaches?
  Spark Window functions are limited to equal-weight aggregations.
  True EMA, Wilder-smoothed RSI, and MACD require exponential decay —
  recursive computations only possible via pandas ewm().

Window specifications (all partitionBy ticker):
  w    → unbounded (lag/lead only)
  w14  → 14 days  (oscillators, ATR)
  w20  → 20 days  (short-term vol, Bollinger)
  w60  → 60 days  (medium-term vol, liquidity z-scores)
  w200 → 200 days (long-term trend, SMA200)
  w252 → 252 days (drawdown, cumulative index)

After all features: Amihud is winsorised at its 99th percentile
using approxQuantile to remove extreme illiquidity spikes.
"""

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

from config import BRONZE_PATH, SILVER_PATH


def safe_div(num, den, default=None):
    """
    Null-safe division. Replaces F.try_divide (requires Spark 3.4+).
    Returns null on zero/null denominator, or a specified default value.
    """
    cond = den.isNotNull() & (den != 0)
    if default is None:
        return F.when(cond, num / den)
    return F.when(cond, num / den).otherwise(F.lit(default))


def _add_pandas_features(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    applyInPandas function: adds EMA, MACD, RSI, downside vol, skew, kurt.

    Called once per ticker on a single executor with the full sorted history.
    All computations here are recursive and cannot be expressed as Spark windows.

    EMA:  exponential weighted mean with span parameter (adjust=False for true EMA)
    MACD: ema_12 - ema_26, signal = ewm(span=9) of MACD
    RSI:  Wilder smoothing alpha = 1/14 (standard industry definition)
    Downside vol: rolling std of negative returns only, annualised
    Skew/Kurt: rolling 60-day distributional shape features
    """
    pdf = pdf.sort_values("date").copy()
    c = pdf["adj_close"]
    r = pdf["ret_1d"]

    # True EMA (exponential, not simple moving average)
    pdf["ema_12"] = c.ewm(span=12, adjust=False).mean()
    pdf["ema_26"] = c.ewm(span=26, adjust=False).mean()
    pdf["macd"]         = pdf["ema_12"] - pdf["ema_26"]
    pdf["macd_signal"]  = pdf["macd"].ewm(span=9, adjust=False).mean()
    pdf["macd_hist"]    = pdf["macd"] - pdf["macd_signal"]

    # RSI with Wilder smoothing (alpha = 1/14)
    delta = c.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    pdf["rsi_14"] = 100 - (100 / (1 + rs))

    # Downside volatility: std of negative returns only, annualised
    pdf["downside_vol_20"] = (
        r.where(r < 0).rolling(20, min_periods=5).std() * np.sqrt(252)
    )

    # Tail shape
    pdf["skew_60"] = r.rolling(60, min_periods=20).skew()
    pdf["kurt_60"] = r.rolling(60, min_periods=20).kurt()

    return pdf


def run_silver(spark: SparkSession) -> None:
    """
    Read Bronze, compute all 35 features, write Silver to GCS.

    Feature groups built with Spark windows:
      - Returns & momentum (ret_1d, ret_5d, ret_21d, ret_63d, ret_252d, mom_12_1)
      - Volatility (vol_20, vol_60, vol_ratio, vol_of_vol, atr_14)
      - Trend (sma_20, sma_60, sma_200, dist_sma_*, trend_strength)
      - Bollinger Bands (bb_mid, bb_up, bb_low_band, bb_pctb, bb_bandwidth)
      - Liquidity (dollar_volume, amihud, turnover_z_60, vol_z_60, hl_range,
                   intraday_ret, gap)
      - Drawdown (drawdown, max_drawdown_252, drawdown_speed)
      - Calendar (dow, month, is_month_end, is_quarter_end)

    applyInPandas adds: ema_12, ema_26, macd, macd_signal, macd_hist,
                        rsi_14, downside_vol_20, skew_60, kurt_60
    """
    bronze = spark.read.parquet(BRONZE_PATH)

    # Window specs (all partitioned by ticker)
    w    = Window.partitionBy("ticker").orderBy("date")
    w14  = Window.partitionBy("ticker").orderBy("date").rowsBetween(-13, 0)
    w20  = Window.partitionBy("ticker").orderBy("date").rowsBetween(-19, 0)
    w60  = Window.partitionBy("ticker").orderBy("date").rowsBetween(-59, 0)
    w200 = Window.partitionBy("ticker").orderBy("date").rowsBetween(-199, 0)
    w252 = Window.partitionBy("ticker").orderBy("date").rowsBetween(-251, 0)

    df = bronze

    # Returns & momentum
    df = df.withColumn("ret_1d",   F.log(F.col("adj_close") / F.lag("adj_close", 1).over(w)))
    df = df.withColumn("ret_5d",   F.log(F.col("adj_close") / F.lag("adj_close", 5).over(w)))
    df = df.withColumn("ret_21d",  F.log(F.col("adj_close") / F.lag("adj_close", 21).over(w)))
    df = df.withColumn("ret_63d",  F.log(F.col("adj_close") / F.lag("adj_close", 63).over(w)))
    df = df.withColumn("ret_252d", F.log(F.col("adj_close") / F.lag("adj_close", 252).over(w)))
    # mom_12_1: 12-month minus 1-month momentum (skip-month)
    df = df.withColumn("mom_12_1",
        safe_div(F.lag("adj_close", 21).over(w), F.lag("adj_close", 252).over(w)) - F.lit(1))

    # Volatility
    df = df.withColumn("vol_20",    F.stddev("ret_1d").over(w20) * F.lit(np.sqrt(252)))
    df = df.withColumn("vol_60",    F.stddev("ret_1d").over(w60) * F.lit(np.sqrt(252)))
    df = df.withColumn("vol_ratio", safe_div(F.col("vol_20"), F.col("vol_60")))
    df = df.withColumn("vol_of_vol", F.stddev("vol_20").over(w20))
    # ATR: max(H-L, |H-prev_close|, |L-prev_close|)
    prev_close = F.lag("close", 1).over(w)
    true_range = F.greatest(
        F.col("high") - F.col("low"),
        F.abs(F.col("high") - prev_close),
        F.abs(F.col("low")  - prev_close),
    )
    df = df.withColumn("atr_14", F.avg(true_range).over(w14))

    # Trend
    df = df.withColumn("sma_20",  F.avg("adj_close").over(w20))
    df = df.withColumn("sma_60",  F.avg("adj_close").over(w60))
    df = df.withColumn("sma_200", F.avg("adj_close").over(w200))
    df = df.withColumn("dist_sma_20",  safe_div(F.col("adj_close"), F.col("sma_20"))  - F.lit(1))
    df = df.withColumn("dist_sma_60",  safe_div(F.col("adj_close"), F.col("sma_60"))  - F.lit(1))
    df = df.withColumn("dist_sma_200", safe_div(F.col("adj_close"), F.col("sma_200")) - F.lit(1))
    df = df.withColumn("trend_strength", F.abs(F.col("dist_sma_20")))

    # Bollinger Bands (±2σ around sma_20)
    bb_std = F.stddev("adj_close").over(w20)
    df = df.withColumn("bb_mid",      F.col("sma_20"))
    df = df.withColumn("bb_up",       F.col("sma_20") + 2 * bb_std)
    df = df.withColumn("bb_low_band", F.col("sma_20") - 2 * bb_std)
    bb_width = F.col("bb_up") - F.col("bb_low_band")
    # bb_pctb: position within band (0.5 on zero-width band)
    df = df.withColumn("bb_pctb",
        safe_div(F.col("adj_close") - F.col("bb_low_band"), bb_width, default=0.5))
    df = df.withColumn("bb_bandwidth", safe_div(bb_width, F.col("bb_mid")))

    # Liquidity & microstructure
    df = df.withColumn("dollar_volume", F.col("adj_close") * F.col("volume"))
    # Amihud illiquidity: |ret| / dollar_volume (price impact proxy)
    df = df.withColumn("amihud", safe_div(F.abs(F.col("ret_1d")), F.col("dollar_volume")))
    dv_mean = F.avg("dollar_volume").over(w60)
    dv_std  = F.stddev("dollar_volume").over(w60)
    df = df.withColumn("turnover_z_60", safe_div(F.col("dollar_volume") - dv_mean, dv_std))
    v_mean = F.avg("volume").over(w60)
    v_std  = F.stddev("volume").over(w60)
    df = df.withColumn("vol_z_60",  safe_div(F.col("volume") - v_mean, v_std))
    df = df.withColumn("hl_range",  safe_div(F.col("high") - F.col("low"), F.col("adj_close")))
    df = df.withColumn("intraday_ret", safe_div(F.col("close") - F.col("open"), F.col("open")))
    df = df.withColumn("gap",
        safe_div(F.col("open"), F.lag("adj_close", 1).over(w)) - F.lit(1))

    # Drawdown
    df = df.withColumn("cum_index", F.exp(F.sum("ret_1d").over(w252)))
    rolling_max = F.max("adj_close").over(w252)
    df = df.withColumn("drawdown",  safe_div(F.col("adj_close"), rolling_max) - F.lit(1))
    df = df.withColumn("max_drawdown_252", F.min("drawdown").over(w252))
    df = df.withColumn("drawdown_speed", F.col("drawdown") - F.lag("drawdown", 5).over(w))

    # Calendar features
    df = df.withColumn("dow",   F.dayofweek("date"))
    df = df.withColumn("month", F.month("date"))
    df = df.withColumn("is_month_end",
        (F.last_day("date") == F.col("date")).cast("int"))
    df = df.withColumn("is_quarter_end",
        (F.month("date").isin([3, 6, 9, 12]) &
         (F.last_day("date") == F.col("date"))).cast("int"))

    # applyInPandas for recursive features
    base_schema = df.schema
    extra_fields = [
        T.StructField("ema_12",          T.DoubleType(), True),
        T.StructField("ema_26",          T.DoubleType(), True),
        T.StructField("macd",            T.DoubleType(), True),
        T.StructField("macd_signal",     T.DoubleType(), True),
        T.StructField("macd_hist",       T.DoubleType(), True),
        T.StructField("rsi_14",          T.DoubleType(), True),
        T.StructField("downside_vol_20", T.DoubleType(), True),
        T.StructField("skew_60",         T.DoubleType(), True),
        T.StructField("kurt_60",         T.DoubleType(), True),
    ]
    full_schema = T.StructType(base_schema.fields + extra_fields)

    df = df.groupBy("ticker").applyInPandas(_add_pandas_features, schema=full_schema)

    # Winsorise Amihud at 99th percentile
    amihud_99 = df.approxQuantile("amihud", [0.99], 0.01)[0]
    df = df.withColumn("amihud",
        F.when(F.col("amihud") > amihud_99, amihud_99).otherwise(F.col("amihud")))

    (df.write
       .mode("overwrite")
       .partitionBy("asset_class")
       .parquet(SILVER_PATH))

    n = spark.read.parquet(SILVER_PATH).count()
    print(f"Silver saved: {n:,} rows → {SILVER_PATH}")
