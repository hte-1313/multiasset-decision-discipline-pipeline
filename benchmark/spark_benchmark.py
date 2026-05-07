
"""
benchmark/spark_benchmark.py
-----------------------------
Part 7: Spark distributed computing scaling benchmark.

Benchmarks the Silver feature pipeline under different Spark partition counts.
Applies three representative distributed transformations:
  1. Per-row volatility ratio:         vol_20 / vol_60
  2. Date-level cross-sectional rank:  percent_rank of ret_1d per date
  3. Volatility-normalised return:     ret_1d / (vol_20 / sqrt(252))

Reports elapsed time, rows/sec, and relative speedup.
Also runs a single-node pandas baseline for direct comparison.

The benchmark is saved to GCS Parquet so the scaling table can be
referenced in the report without re-running.
"""

import time
import pandas as pd
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

from config import SILVER_PATH, RESULTS_PATH


def run_spark_benchmark(spark: SparkSession) -> pd.DataFrame:
    """
    Benchmark Silver pipeline across 2, 4, and default-parallelism partitions.

    Returns pd.DataFrame with columns:
      n_partitions, simulated_workers, elapsed_seconds,
      rows_processed, rows_per_second, speedup
    """
    print("=== Spark Distributed Benchmark ===")
    print(f"Cluster default parallelism: {spark.sparkContext.defaultParallelism} cores")

    silver_bench = spark.read.parquet(SILVER_PATH).cache()
    n_rows = silver_bench.count()  # warm up cache
    print(f"Dataset: {n_rows:,} rows cached\n")

    results = []
    partition_counts = [2, 4, spark.sparkContext.defaultParallelism]

    for n_parts in partition_counts:
        print(f"  Running with {n_parts} partitions...")
        df_part = silver_bench.repartition(n_parts, "ticker")
        t0 = time.time()

        n = (df_part
             .withColumn("vol_ratio",
                 F.when(F.col("vol_60") > 0, F.col("vol_20") / F.col("vol_60")))
             .withColumn("xs_ret_rank",
                 F.percent_rank().over(
                     Window.partitionBy("date").orderBy("ret_1d")))
             .withColumn("ret_z",
                 F.when(F.col("vol_20") > 0,
                        F.col("ret_1d") / (F.col("vol_20") / F.sqrt(F.lit(252.0)))))
             .count())

        elapsed      = time.time() - t0
        rows_per_sec = round(n / elapsed, 0)
        results.append({
            "n_partitions":      n_parts,
            "simulated_workers": max(1, n_parts // 2),
            "elapsed_seconds":   round(elapsed, 2),
            "rows_processed":    n,
            "rows_per_second":   rows_per_sec,
        })
        print(f"    {n_parts} partitions: {elapsed:.1f}s | {rows_per_sec:,.0f} rows/sec")

    bench_df = pd.DataFrame(results)
    base_time = bench_df["elapsed_seconds"].max()
    bench_df["speedup"] = (base_time / bench_df["elapsed_seconds"]).round(2)

    print("\nDistributed Scaling Results:")
    print(bench_df[["n_partitions", "simulated_workers",
                     "elapsed_seconds", "rows_per_second", "speedup"]].to_string(index=False))
    print(f"\nSpeedup: {bench_df['speedup'].max():.2f}x from "
          f"{bench_df['n_partitions'].min()} to "
          f"{bench_df['n_partitions'].max()} partitions")

    bench_df.to_parquet(f"{RESULTS_PATH}/spark_benchmark.parquet", index=False)
    print("Benchmark saved.")
    return bench_df


def run_pandas_vs_spark_comparison(spark: SparkSession) -> dict:
    """
    Compare single-node pandas vs distributed Spark on the same workload.

    Returns dict with single_time, single_rps, spark_time, spark_rps, speedup.
    """
    # Single-node pandas
    print("=== Single-node pandas baseline ===")
    silver_pd = spark.read.parquet(SILVER_PATH).toPandas()
    t0 = time.time()
    silver_pd["vol_ratio"]   = silver_pd["vol_20"] / silver_pd["vol_60"]
    silver_pd["xs_ret_rank"] = silver_pd.groupby("date")["ret_1d"].rank(pct=True)
    single_time = time.time() - t0
    single_rps  = len(silver_pd) / single_time
    print(f"Pandas : {single_time:.2f}s | {single_rps:,.0f} rows/sec")

    # Distributed Spark (4 partitions)
    print("\n=== Distributed Spark ===")
    silver_sdf = spark.read.parquet(SILVER_PATH).repartition(4, "ticker").cache()
    silver_sdf.count()
    t0 = time.time()
    n = (silver_sdf
         .withColumn("vol_ratio", F.col("vol_20") / F.col("vol_60"))
         .withColumn("xs_ret_rank", F.percent_rank().over(
             Window.partitionBy("date").orderBy("ret_1d")))
         .count())
    spark_time = time.time() - t0
    spark_rps  = n / spark_time
    print(f"Spark  : {spark_time:.2f}s | {spark_rps:,.0f} rows/sec")
    print(f"\nSpeedup: {single_time / spark_time:.2f}x")

    return {
        "single_time": single_time, "single_rps": single_rps,
        "spark_time":  spark_time,  "spark_rps":  spark_rps,
        "speedup":     single_time / spark_time,
    }
