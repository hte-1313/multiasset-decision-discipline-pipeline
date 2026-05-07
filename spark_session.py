"""
spark_session.py
----------------
Creates and returns the shared SparkSession for the pipeline.
Configured for GCS, Arrow, Adaptive Query Execution, and GraphFrames.
"""

from pyspark.sql import SparkSession


def get_spark(app_name: str = "MultiAsset-Pipeline-v2") -> SparkSession:
    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.graphx.pregel.checkpointInterval", "5")
        .config("spark.jars", "/usr/lib/spark/jars/graphframes.jar")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark
