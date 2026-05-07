"""
visualisation/plots.py
----------------------
Part 8: All visualisation functions for the pipeline results.

Plots produced:
  1. Cumulative equity curves (log scale)
  2. Portfolio drawdown over time
  3. Sharpe ratio by asset class (bar chart)
  4. Win rate heatmap by asset class
  5. Stop-loss impact on Sharpe (bar comparison)
  6. SPY signal decisions and cumulative return
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import ANNUALISATION_FACTOR, REFIT_EVERY

MODEL_LABELS = {
    "always_hold":   "Always-Hold (Baseline)",
    "xgboost":       "XGBoost (Batch)",
    "bayesian_peft": "Bayesian Hybrid PEFT (Online)",
    "pregel_gnn":    "Pregel GCN (Distributed)",
}
COLOURS = {
    "always_hold":   "#999999",
    "xgboost":       "#E24B4A",
    "bayesian_peft": "#1D9E75",
    "pregel_gnn":    "#534AB7",
}


def plot_equity_and_drawdown(all_results: pd.DataFrame,
                              metrics_df: pd.DataFrame,
                              models_to_eval: list) -> None:
    """
    Three-panel figure:
      Panel 1: Cumulative equity curves (log scale)
      Panel 2: Portfolio drawdown
      Panel 3: Sharpe ratio by asset class (grouped bar chart)
    """
    fig, axes = plt.subplots(3, 1, figsize=(16, 14))
    fig.suptitle(
        "MultiAsset Decision Pipeline — 5-Day Return Strategy Performance",
        fontsize=14, fontweight="bold", y=0.98,
    )

    # Panel 1: Equity curve
    ax1 = axes[0]
    for m in models_to_eval:
        sub  = all_results[all_results["model"] == m].copy()
        if sub.empty: continue
        port = sub.groupby("date")["signal_ret"].mean().sort_index().fillna(0)
        cum  = (1 + port).cumprod()
        ax1.plot(cum.index, cum.values,
                 label=MODEL_LABELS.get(m, m),
                 color=COLOURS.get(m, "blue"), linewidth=1.5)
    ax1.set_title("Cumulative Equity Curve", fontsize=11)
    ax1.set_ylabel("Cumulative Return (log scale)")
    ax1.set_yscale("log")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3)

    # Panel 2: Drawdown
    ax2 = axes[1]
    for m in models_to_eval:
        sub  = all_results[all_results["model"] == m].copy()
        if sub.empty: continue
        port = sub.groupby("date")["signal_ret"].mean().sort_index().fillna(0)
        cum  = (1 + port).cumprod()
        dd   = ((cum - cum.cummax()) / cum.cummax()).fillna(0)
        ax2.fill_between(cum.index, dd.values.astype(float), 0,
                          alpha=0.25, color=COLOURS.get(m, "blue"))
        ax2.plot(cum.index, dd.values.astype(float),
                  linewidth=0.8, label=MODEL_LABELS.get(m, m),
                  color=COLOURS.get(m, "blue"))
    ax2.set_title("Portfolio Drawdown", fontsize=11)
    ax2.set_ylabel("Drawdown (%)")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(alpha=0.3)

    # Panel 3: Asset-class Sharpe bars
    ax3 = axes[2]
    ac_cols   = ["sharpe_ETF", "sharpe_EQ", "sharpe_FX", "sharpe_FUT_COM", "sharpe_FUT_IDX"]
    ac_labels = ["ETF", "EQ", "FX", "FUT_COM", "FUT_IDX"]
    x = np.arange(len(ac_labels))
    w = 0.18
    for i, m in enumerate(models_to_eval):
        row  = metrics_df[metrics_df["model"] == m]
        if row.empty: continue
        vals = [float(row[c].values[0]) if c in row.columns and pd.notna(row[c].values[0])
                else 0.0 for c in ac_cols]
        ax3.bar(x + i * w, vals, w,
                label=MODEL_LABELS.get(m, m),
                color=COLOURS.get(m, "blue"), alpha=0.8)
    ax3.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax3.set_xticks(x + w * (len(models_to_eval) - 1) / 2)
    ax3.set_xticklabels(ac_labels, fontsize=10)
    ax3.set_title("Sharpe Ratio by Asset Class", fontsize=11)
    ax3.set_ylabel("Sharpe Ratio")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def plot_win_rate_and_stoploss(all_results: pd.DataFrame,
                                metrics_df: pd.DataFrame,
                                models_to_eval: list) -> None:
    """
    Two-panel figure:
      Left:  Win rate heatmap by asset class
      Right: Stop-loss impact on Sharpe ratio (bar comparison)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=120)
    fig.suptitle("Decision Quality: Win Rate & Stop-Loss Impact",
                  fontsize=12, fontweight="bold")

    # Win rate heatmap
    win_data = {}
    for m in models_to_eval:
        sub = all_results[all_results["model"] == m].copy()
        sub["win"] = ((sub["ret_5d_fwd"] > 0) & (sub["signal"] == 1)).astype(float)
        win_data[MODEL_LABELS.get(m, m)] = (
            sub.groupby("asset_class")["win"].mean()
               .reindex(["ETF", "EQ", "FX", "FUT_COM", "FUT_IDX"])
               .astype(float)
        )
    win_df = pd.DataFrame(win_data).astype(float)
    sns.heatmap(win_df.T, annot=True, fmt=".1%", cmap="RdYlGn",
                center=0.5, ax=axes[0], cbar_kws={"label": "Win Rate"})
    axes[0].set_title("Win Rate by Asset Class", fontsize=10)
    axes[0].set_xlabel("Asset Class")

    # Stop-loss impact
    sl_compare = pd.DataFrame({
        "No Stop-Loss": {
            m: float(metrics_df[metrics_df["model"] == m]["sharpe"].values[0])
            for m in models_to_eval
            if not metrics_df[metrics_df["model"] == m].empty
        },
        "With Stop-Loss": {
            m: float(metrics_df[metrics_df["model"] == m]["sharpe_stoploss"].values[0])
            for m in models_to_eval
            if not metrics_df[metrics_df["model"] == m].empty
        },
    }).rename(index={m: MODEL_LABELS.get(m, m) for m in models_to_eval})

    sl_compare.plot(kind="bar", ax=axes[1],
                    color=["#185FA5", "#1D9E75"], alpha=0.85, edgecolor="white")
    axes[1].axhline(0, color="black", linewidth=0.7, linestyle="--")
    axes[1].set_title("Stop-Loss Impact on Sharpe Ratio", fontsize=10)
    axes[1].set_ylabel("Sharpe Ratio")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_spy_signals(all_results: pd.DataFrame,
                      models_to_eval: list) -> None:
    """
    Two-panel figure for SPY (last 60 prediction periods):
      Panel 1: XGBoost hold/exit signals overlaid on 5-day returns
      Panel 2: Cumulative return comparison across all models
    """
    spy = all_results[all_results["ticker"] == "SPY"].copy()
    spy = spy.sort_values("date").tail(60)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle("SPY — Signal Decisions & Cumulative Return (Last 60 Periods)",
                  fontsize=12, fontweight="bold")

    # Signal overlay
    ax1 = axes[0]
    xgb_spy = spy[spy["model"] == "xgboost"].copy()
    if not xgb_spy.empty:
        ax1.bar(xgb_spy[xgb_spy["signal"] == 1]["date"],
                xgb_spy[xgb_spy["signal"] == 1]["ret_5d_fwd"],
                color="green", alpha=0.6, width=3, label="Hold signal")
        ax1.bar(xgb_spy[xgb_spy["signal"] == 0]["date"],
                xgb_spy[xgb_spy["signal"] == 0]["ret_5d_fwd"],
                color="red", alpha=0.6, width=3, label="Exit signal")
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_title("XGBoost Hold/Exit Signals on SPY — 5-Day Returns", fontsize=10)
    ax1.set_ylabel("5-Day Return")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # Cumulative return per model
    ax2 = axes[1]
    for m in models_to_eval:
        sub = spy[spy["model"] == m].copy().sort_values("date")
        if sub.empty: continue
        sub["sig_ret"] = sub["ret_5d_fwd"] * sub["signal"].fillna(0)
        cum = (1 + sub["sig_ret"].fillna(0)).cumprod()
        ax2.plot(sub["date"].values, cum.values,
                  label=MODEL_LABELS.get(m, m),
                  color=COLOURS.get(m, "blue"), linewidth=1.5)
    ax2.set_title("Cumulative Return on SPY by Model", fontsize=10)
    ax2.set_ylabel("Cumulative Return")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

