"""
bronze/downloader.py
--------------------
Part 1: Bronze layer — raw OHLCV ingestion from yfinance.

_download_one   – downloads one ticker with exponential-backoff retry.
_is_delisted    – flags tickers with fewer than 252 days of recent data.
run_bronze      – parallel download, delisted removal, GCS write.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from config import START_DATE, END_DATE, BRONZE_PATH
from assets import TICKERS, ASSET_CLASS_MAP, SECTOR_MAP


def _download_one(ticker: str) -> pd.DataFrame:
    """
    Download OHLCV for a single ticker from yfinance.

    Retry logic:
      - Up to 3 attempts with exponential backoff (2^attempt seconds).
      - Flattens MultiIndex columns, guessing which level holds price names.
      - Lowercases and snake_cases all column names.
      - Falls back to close as adj_close if adjusted price is missing.
      - Drops non-price columns (dividends, stock_splits, capital_gains).
      - Coerces OHLCV to numeric, drops rows where close or adj_close is null/zero.
      - Attaches ticker, asset_class, sector, ingested_at metadata.
      - Returns empty DataFrame on any unhandled exception.
    """
    for attempt in range(3):
        try:
            df = yf.Ticker(ticker).history(
                start=START_DATE, end=END_DATE,
                auto_adjust=False, actions=False,
            )
            if df is None or df.empty:
                time.sleep(2 ** attempt)
                continue

            if isinstance(df.columns, pd.MultiIndex):
                price_cols = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
                l0 = set(df.columns.get_level_values(0))
                if len(l0 & price_cols) >= 3:
                    df.columns = df.columns.get_level_values(0)
                else:
                    df.columns = df.columns.get_level_values(1)

            df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]

            if "adj_close" not in df.columns:
                if "close" in df.columns:
                    df["adj_close"] = df["close"]
                else:
                    return pd.DataFrame()

            df = df.reset_index()
            df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
            if "date" not in df.columns:
                df.rename(columns={"index": "date"}, inplace=True)

            df.drop(
                columns=[c for c in df.columns
                          if c in ("dividends", "stock_splits", "capital_gains")],
                errors="ignore", inplace=True,
            )

            for col in ["open", "high", "low", "close", "adj_close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.dropna(subset=["close", "adj_close"])
            df = df[(df["adj_close"] > 0) & (df["close"] > 0)]
            if df.empty:
                return pd.DataFrame()

            df["ticker"]      = ticker
            df["asset_class"] = ASSET_CLASS_MAP.get(ticker, "unknown")
            df["sector"]      = SECTOR_MAP.get(ticker, "unknown")
            df["ingested_at"] = datetime.now(timezone.utc).isoformat()
            df["date"]        = pd.to_datetime(df["date"]).dt.date
            return df

        except Exception:
            time.sleep(2 ** attempt)
    return pd.DataFrame()


def _is_delisted(df: pd.DataFrame, end_date: str = END_DATE,
                 min_days: int = 252) -> bool:
    """
    Return True if the ticker should be treated as delisted.

    Cutoff = 2 years (730 days) before end_date.
    Delisted = fewer than min_days valid rows in that trailing window,
    which is roughly one full trading year as the minimum activity threshold.
    """
    cutoff = pd.to_datetime(end_date) - pd.Timedelta(days=730)
    recent = df[pd.to_datetime(df["date"]) >= cutoff]
    return len(recent) < min_days


def run_bronze(spark: SparkSession, max_workers: int = 8) -> dict:
    """
    Download all tickers in parallel, remove delisted assets, write to GCS.

    Returns raw_frames dict {ticker: pd.DataFrame} for optional downstream use.
    Saves Gzip-compressed Parquet partitioned by asset_class to BRONZE_PATH.
    """
    frames, failed, delisted = {}, [], []

    print(f"Downloading {len(TICKERS)} tickers ({max_workers} threads)...")
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_download_one, t): t for t in TICKERS}
        for i, fut in enumerate(futures.values()):
            t   = list(futures.keys())[i]  # preserve mapping
            t   = futures[fut]
            df  = fut.result()
            pct = (i + 1) / len(TICKERS) * 100
            if df.empty:
                failed.append(t)
            else:
                frames[t] = df
            if (i + 1) % 20 == 0:
                print(f"  {pct:5.1f}%  | {len(frames)} ok | {len(failed)} failed")

    for t in list(frames):
        if _is_delisted(frames[t]):
            delisted.append(t)
            del frames[t]

    print(f"\nResults:")
    print(f"  Downloaded : {len(frames)}")
    print(f"  Failed     : {len(failed)} → {failed}")
    print(f"  Delisted   : {len(delisted)} → {delisted}")

    if not frames:
        raise RuntimeError("No data downloaded. Check yfinance connectivity.")

    import pandas as pd
    all_pd = pd.concat(frames.values(), ignore_index=True)
    print(f"  Total rows : {len(all_pd):,}")

    bronze_sdf = spark.createDataFrame(all_pd)
    bronze_sdf = (
        bronze_sdf
        .withColumn("date", F.col("date").cast("date"))
        .filter(F.col("adj_close").isNotNull() & (F.col("adj_close") > 0))
        .sort("ticker", "date")
    )

    (bronze_sdf
        .write
        .mode("overwrite")
        .partitionBy("asset_class")
        .parquet(BRONZE_PATH))

    n = spark.read.parquet(BRONZE_PATH).count()
    print(f"  Saved {n:,} rows → {BRONZE_PATH}")
    return frames
