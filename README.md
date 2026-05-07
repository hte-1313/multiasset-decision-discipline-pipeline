# MultiAsset Decision Pipeline — ST446 Distributed Computing for Big Data

A distributed machine learning pipeline for multi-asset trading signal generation, built on Apache Spark and Google Cloud Storage. The project was developed as part of the ST446 Distributed Computing for Big Data course at LSE.

The core question it tries to answer: *can a distributed, graph-aware ML system generate better hold/exit signals across a diverse universe of financial assets than a simple always-hold baseline?*

---

## What This Project Does

The pipeline processes 119 instruments across five asset classes — equities, ETFs, FX pairs, commodity futures, and index futures — covering roughly 20 years of daily OHLCV data (2005–2025). It's structured as a classic medallion architecture across three layers, then feeds into a model suite and evaluation framework.

### The Pipeline at a Glance

| Layer | What Happens |
|-------|-------------|
| **Bronze** | Raw OHLCV ingested in parallel from `yfinance` · Hive-partitioned by asset class on GCS · Delisted assets filtered out |
| **Silver** | 35-dimensional feature vector per ticker-date · Mix of Spark window functions and `applyInPandas` for recursive features (EMA, RSI, MACD) |
| **Gold** | Cross-sectional normalisation · Pregel-based Graph Convolutional Network embeddings built from rolling return correlations · Leakage-free 5-day forward targets |
| **Models** | Always-Hold baseline · XGBoost · Bayesian Hybrid PEFT (online learning) · GNN (Pregel GCN + optional PyTorch GraphSAGE) |
| **Evaluation** | Walk-forward backtesting · Sharpe · Sortino · Calmar · Max Drawdown · Win Rate · Stop-loss and time exit overlays · Diebold-Mariano significance tests · Transaction cost sensitivity |

### Why Two Feature Engineering Approaches?

Spark's native `Window` functions can only do equal-weight aggregations. Any feature with row-by-row state — true EMA, Wilder-smoothed RSI, MACD — has to be dispatched to pandas via `applyInPandas`. This gives each ticker's full history to a single executor for recursive computation, while everything else runs natively distributed.

### The Graph Layer

One of the more unusual aspects of this pipeline is the use of Pregel message-passing (via GraphFrames) to build graph-based embeddings. Assets are connected by a rolling 60-day correlation graph with a threshold of 0.30 — if two assets are sufficiently correlated, they share an edge. The Pregel GCN then propagates signals across this graph over three iterations, so each asset's embedding reflects not just its own features but the weighted average of its neighbours'. The idea is that correlated assets carry useful information about each other's future returns.

---

## Technical Stack

- **Apache Spark** (3.5) on Google Cloud Dataproc
- **GraphFrames** for distributed graph computation (Pregel)
- **Google Cloud Storage** for all intermediate and output data (Parquet, gzip-compressed)
- **yfinance** for data ingestion
- **XGBoost** via `applyInPandas` for distributed batch scoring
- **PyTorch + PyTorch Geometric** (optional) for GraphSAGE
- **scipy** for the Diebold-Mariano test

---

## Project Structure

```
pipeline_v2/
├── bronze/          # Raw OHLCV, partitioned by asset_class
├── silver/          # Engineered features (35 columns per ticker-date)
├── gold/            # Cross-sectional features + graph embeddings + targets
└── results/         # Walk-forward predictions, metrics, benchmark output
```

---

## Limitations

There are a few things worth being upfront about here.

**Data quality is entirely dependent on yfinance.** The pipeline includes retry logic and delisting detection, but yfinance is an unofficial API — adjusted close prices, corporate action handling, and data completeness vary by ticker and can change without warning. For anything production-grade, you'd want a proper data vendor.

**The graph construction is static within each refit window.** Correlation edges are computed once per walk-forward window. In practice, asset correlations shift — sometimes dramatically, as they do in a crisis — and a dynamic graph that updates more frequently would likely capture these regime changes better.

**Pregel GCN is an approximation of a proper GNN.** The message-passing here is a simplified graph convolution built on top of Spark's Pregel API — it's not the same as running a full GNN training loop. It gets the distributed computation aspect right, but the learned representations are shallower than what a properly trained GraphSAGE or GAT model would produce. The optional PyTorch GraphSAGE path improves on this, but requires the cluster to have GPU resources available.

**Walk-forward is necessary but not sufficient.** The evaluation is walk-forward, which avoids look-ahead bias in the model itself. However, the feature engineering was developed with knowledge of the full dataset, which introduces a subtle form of indirect data snooping. A proper out-of-sample test on held-out post-2025 data hasn't been run yet.

**Transaction costs are estimated, not simulated.** The TC sensitivity analysis assumes a fixed cost per trade at various basis point levels. Real-world costs depend on instrument, venue, size, and market impact — none of which is modelled here. The results at 0 bps should be treated with appropriate scepticism.

**No position sizing.** Every hold signal is treated as an equal-weight position. A proper portfolio construction layer (mean-variance, risk parity, etc.) would likely change the performance characteristics significantly.

**VX=F is in the universe but excluded from evaluation** — volatility futures have inverted economics and steep contango that make them incomparable to the other instruments on standard return metrics.

---

## Future Work

A few things I'm planning to extend this with:

**Dynamic correlation graphs.** Rather than a static graph per window, build the edge set using an exponentially weighted rolling correlation so the graph topology updates continuously. This should make the GNN embeddings more responsive to correlation breakdowns during stress periods.

**Proper out-of-sample evaluation.** Hold out 2025 data entirely and evaluate there. The current setup ends at January 2025, so there's already a natural holdout available — it just hasn't been run.

**Richer position sizing.** Attach a simple risk parity or volatility-targeting layer on top of the binary signals. The current always-equal-weight approach ignores the very different volatility profiles of, say, an FX pair vs a single-name equity.

**Online learning improvements.** The Bayesian PEFT model is designed for online updating, but right now it refits on the same cadence as everything else. The real use case is streaming inference, where each new day's data updates the posterior without a full refit — that's the natural next step.

**Streaming ingestion.** Replace the batch yfinance download with a Spark Structured Streaming job connected to a live market data source. The Bronze schema is already designed to support this, but the actual streaming connector isn't implemented.

**Hyperparameter search.** XGBoost parameters are currently fixed. A proper time-series-aware hyperparameter search (no future leakage in cross-validation) could meaningfully improve performance.

**Explainability.** Add SHAP values for the XGBoost predictions and attention weights for the GNN, so there's some understanding of which features are actually driving the signals rather than just reporting the downstream metrics.

---

## Notes on the Benchmark

Part 7 of the notebook benchmarks the Silver feature pipeline across different Spark partition counts. It measures elapsed time, rows per second, and speedup relative to a single-node pandas baseline. The results are saved to `results/spark_benchmark.parquet` for reference. This is primarily for coursework demonstration of the distributed aspects — the numbers are cluster-specific and won't reproduce on a different Dataproc configuration.

---

## Course Context

This project was submitted for ST446 Distributed Computing for Big Data at the London School of Economics. The brief required demonstrated use of distributed data processing (Spark ETL, `applyInPandas` model execution), distributed graph computation (Pregel via GraphFrames), and a scaling benchmark. The financial application is real and the methodology is reasonable, but this is academic work — it is not investment advice and should not be used for live trading.
