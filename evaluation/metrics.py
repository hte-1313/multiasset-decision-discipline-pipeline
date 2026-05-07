
"""
evaluation/metrics.py
---------------------
Part 6: Full quantitative evaluation suite.

Metrics:
  Sharpe ratio     : annualised risk-adjusted return (rf ≈ 0)
  Sortino ratio    : Sharpe using downside deviation only
  Calmar ratio     : annualised return / |max_drawdown|
  Win rate         : fraction of active signals with positive return
  Profit factor    : sum(wins) / |sum(losses)| for active signals
  Max drawdown     : worst peak-to-trough equity loss
  Stop-loss sim    : -2% cumulative log-return exit during holding period
  Time-based exit  : always exit after HOLD_DAYS trading days
  DM test          : Diebold-Mariano (1995) test for equal predictive accuracy
  Bootstrap CI     : 95% confidence intervals on Sharpe (1000 samples)

Transaction cost sensitivity:
  For each model and TC level (0, 5, 10, 20 bps):
    turnover = signal flip indicator (signal != lagged signal)
    ret_tc   = signal_ret - turnover * tc_per_trade
"""

import numpy as np
import pandas as pd
from scipy import stats

from config import (ANNUALISATION_FACTOR, REFIT_EVERY, HOLD_DAYS,
                    STOP_LOSS_PCT, BOOTSTRAP_N, TRANSACTION_COSTS_BPS)


# ── Core financial metrics ─────────────────────────────────────────────────

def compute_sharpe(returns: pd.Series,
                   ann: int = ANNUALISATION_FACTOR) -> float:
    """Annualised Sharpe ratio. Risk-free rate assumed zero (excess return)."""
    returns = returns.dropna()
    if len(returns) < 10 or returns.std() < 1e-12:
        return np.nan
    return float(returns.mean() / returns.std() * np.sqrt(ann / REFIT_EVERY))


def compute_sortino(returns: pd.Series,
                    ann: int = ANNUALISATION_FACTOR) -> float:
    """Annualised Sortino ratio using downside deviation only."""
    returns = returns.dropna()
    neg = returns[returns < 0]
    if len(neg) < 2 or neg.std() < 1e-12:
        return np.nan
    return float(returns.mean() / neg.std() * np.sqrt(ann / REFIT_EVERY))


def compute_max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown from cumulative return series."""
    returns = returns.fillna(0.0)
    cum     = (1 + returns).cumprod()
    dd      = (cum - cum.cummax()) / cum.cummax()
    return float(dd.min())


def compute_calmar(returns: pd.Series,
                   ann: int = ANNUALISATION_FACTOR) -> float:
    """Calmar ratio: annualised return / abs(max_drawdown)."""
    ann_ret = returns.mean() * (ann / REFIT_EVERY)
    mdd     = abs(compute_max_drawdown(returns))
    return float(ann_ret / mdd) if mdd > 1e-6 else np.nan


def compute_win_rate(returns: pd.Series, signals: pd.Series) -> float:
    """Win rate: fraction of active signals (signal=1) with positive return."""
    active = returns[signals == 1].dropna()
    return float((active > 0).mean()) if len(active) > 0 else np.nan


def compute_profit_factor(returns: pd.Series, signals: pd.Series) -> float:
    """Profit factor: sum(wins) / abs(sum(losses)) for active signals."""
    active = returns[signals == 1].dropna()
    wins   = active[active > 0].sum()
    losses = abs(active[active < 0].sum())
    return float(wins / losses) if losses > 1e-8 else np.nan


def apply_stop_loss_and_time_exit(group: pd.DataFrame,
                                   daily_rets: pd.Series,
                                   stop_loss: float = STOP_LOSS_PCT,
                                   hold_days: int   = HOLD_DAYS) -> pd.Series:
    """
    Simulate stop-loss and time-based exit for active positions.

    For each signal=1 row:
      - Open long at close of signal date.
      - Monitor daily for hold_days trading days.
      - Exit early if cumulative log-return < stop_loss.
      - Always exit at day hold_days (time-based exit).

    Returns realised_returns as a pd.Series indexed by group.index.
    """
    group    = group.sort_values("date").copy()
    dates    = group["date"].values
    sigs     = group["signal"].values
    n        = len(dates)
    realised = np.full(n, np.nan)

    for i in range(n):
        if sigs[i] != 1:
            realised[i] = 0.0
            continue
        cum_ret = 0.0
        exit_ret = 0.0
        for j in range(1, hold_days + 1):
            if i + j >= n:
                break
            r = daily_rets.get(pd.Timestamp(dates[i + j]), 0.0)
            cum_ret += r
            if cum_ret < stop_loss:
                exit_ret = stop_loss  # approximate stop-loss exit
                break
            if j == hold_days:
                exit_ret = cum_ret
        realised[i] = exit_ret

    return pd.Series(realised, index=group.index)


# ── Diebold-Mariano test ───────────────────────────────────────────────────

def diebold_mariano_test(e1: np.ndarray, e2: np.ndarray,
                          h: int = HOLD_DAYS) -> dict:
    """
    Diebold-Mariano (1995) test for equal predictive accuracy.

    H0: E[d_t] = 0 where d_t = L(e1_t) - L(e2_t) and L = squared loss.
    Negative DM stat → e1 has higher loss → model 2 (active) wins.

    Newey-West variance correction (h-1 lags) accounts for autocorrelation
    in overlapping HOLD_DAYS-period returns.

    Returns dict: dm_stat, p_value, result string.
    """
    d      = e1 ** 2 - e2 ** 2
    T      = len(d)
    if T < 20:
        return {"dm_stat": np.nan, "p_value": np.nan,
                "result": "insufficient data"}

    d_mean = d.mean()
    nw_var = d.var(ddof=1) / T
    for lag in range(1, h):
        gamma_l = np.cov(d[lag:], d[:-lag])[0, 1] if len(d) > lag + 1 else 0.0
        nw_var += 2 * (1 - lag / h) * gamma_l / T
    nw_var  = max(nw_var, 1e-12)
    dm_stat = d_mean / np.sqrt(nw_var)
    p_val   = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    result = ("model_1_better"  if dm_stat < 0 and p_val < 0.05 else
              "model_2_better"  if dm_stat > 0 and p_val < 0.05 else
              "no_significant_difference")
    return {"dm_stat": float(dm_stat), "p_value": float(p_val), "result": result}


# ── Bootstrap confidence intervals ────────────────────────────────────────

def bootstrap_sharpe_ci(returns: pd.Series,
                          n_boot: int = BOOTSTRAP_N,
                          ci: float   = 0.95) -> tuple:
    """
    Bootstrap 95% confidence interval for Sharpe ratio.

    Resamples with replacement. Valid without block-bootstrap for
    5-day non-overlapping returns (approximately iid).
    """
    returns = returns.dropna().values
    if len(returns) < 20:
        return (np.nan, np.nan)
    boot = [
        (s := np.random.choice(returns, len(returns), replace=True),
         s.mean() / s.std() * np.sqrt(ANNUALISATION_FACTOR / REFIT_EVERY)
         if s.std() > 1e-12 else 0.0)[1]
        for _ in range(n_boot)
    ]
    lo = np.percentile(boot, (1 - ci) / 2 * 100)
    hi = np.percentile(boot, (1 - (1 - ci) / 2) * 100)
    return float(lo), float(hi)


# ── Full model evaluation ──────────────────────────────────────────────────

def safe_float(x, default=np.nan):
    try:
        return float(x) if x is not None and not pd.isna(x) else default
    except Exception:
        return default


def evaluate_model(model_results: pd.DataFrame,
                    model_name:    str,
                    gold_pd_ref:   pd.DataFrame,
                    txn_cost_bps:  float = 0.0) -> dict:
    """
    Full evaluation of one model's predictions.

    Computes: Sharpe, Sortino, Calmar, Max DD, Win Rate, Profit Factor,
              transaction-cost-adjusted Sharpe, stop-loss Sharpe,
              turnover, hold rate, asset-class Sharpe breakdown,
              and bootstrap CIs.
    """
    df = model_results[model_results["model"] == model_name].copy()
    if df.empty:
        return {"model": model_name, "n_predictions": 0}

    df = df.sort_values("date")
    df["signal_ret"] = df["ret_5d_fwd"] * df["signal"].fillna(0)

    # Transaction cost drag
    tc_per_trade    = txn_cost_bps / 10000.0
    df["signal_lag"] = df.groupby("ticker")["signal"].shift(1)
    df["turnover"]   = (df["signal"] != df["signal_lag"]).astype(float)
    df["signal_ret_tc"] = df["signal_ret"] - df["turnover"] * tc_per_trade

    # Stop-loss simulation
    daily_ref = (gold_pd_ref
        .groupby(["ticker", "date"])["ret_1d"].first()
        .groupby(level=0).apply(lambda x: x.droplevel(0)))

    sl_parts = []
    for tick, grp in df.groupby("ticker"):
        dr = daily_ref.get(tick, pd.Series(dtype=float))
        if isinstance(dr, pd.DataFrame):
            dr = dr.iloc[:, 0]
        dr.index = pd.to_datetime(dr.index)
        sl_parts.append(apply_stop_loss_and_time_exit(grp, dr))
    df["sl_ret"] = pd.concat(sl_parts).reindex(df.index).fillna(0.0)

    rets    = df["signal_ret"].dropna()
    rets_tc = df["signal_ret_tc"].dropna()
    rets_sl = df["sl_ret"].dropna()
    sharpe  = compute_sharpe(rets)
    ci_lo, ci_hi = bootstrap_sharpe_ci(rets)

    wins = df.loc[(df["signal"] == 1) & (df["signal_ret"] > 0), "signal_ret"]
    loss = df.loc[(df["signal"] == 1) & (df["signal_ret"] < 0), "signal_ret"]

    return {
        "model":              model_name,
        "n_predictions":      len(df),
        "n_tickers":          df["ticker"].nunique(),
        "ann_return_pct":     safe_float(rets.mean() * (ANNUALISATION_FACTOR / REFIT_EVERY) * 100),
        "sharpe":             safe_float(sharpe),
        "sharpe_ci_lo":       safe_float(ci_lo),
        "sharpe_ci_hi":       safe_float(ci_hi),
        "sortino":            safe_float(compute_sortino(rets)),
        "calmar":             safe_float(compute_calmar(rets)),
        "max_drawdown_pct":   safe_float(compute_max_drawdown(rets) * 100),
        "vol_ann_pct":        safe_float(rets.std() * np.sqrt(ANNUALISATION_FACTOR / REFIT_EVERY) * 100),
        "win_rate_pct":       safe_float(compute_win_rate(df["signal_ret"], df["signal"]) * 100),
        "profit_factor":      safe_float(compute_profit_factor(df["signal_ret"], df["signal"])),
        "avg_win_pct":        safe_float(wins.mean() * 100),
        "avg_loss_pct":       safe_float(loss.mean() * 100),
        "sharpe_10bps":       safe_float(compute_sharpe(rets_tc)),
        "ann_return_10bps":   safe_float(rets_tc.mean() * (ANNUALISATION_FACTOR / REFIT_EVERY) * 100),
        "sharpe_stoploss":    safe_float(compute_sharpe(rets_sl)),
        "ann_return_sl_pct":  safe_float(rets_sl.mean() * (ANNUALISATION_FACTOR / REFIT_EVERY) * 100),
        "turnover_per_yr":    safe_float(df["turnover"].mean() * (ANNUALISATION_FACTOR / REFIT_EVERY)),
        "hold_rate_pct":      safe_float(df["signal"].mean() * 100),
        "sharpe_ETF":         safe_float(compute_sharpe(df.loc[df["asset_class"] == "ETF",     "signal_ret"])),
        "sharpe_EQ":          safe_float(compute_sharpe(df.loc[df["asset_class"] == "EQ",      "signal_ret"])),
        "sharpe_FX":          safe_float(compute_sharpe(df.loc[df["asset_class"] == "FX",      "signal_ret"])),
        "sharpe_FUT_COM":     safe_float(compute_sharpe(df.loc[df["asset_class"] == "FUT_COM", "signal_ret"])),
        "sharpe_FUT_IDX":     safe_float(compute_sharpe(df.loc[df["asset_class"] == "FUT_IDX", "signal_ret"])),
    }


def run_full_evaluation(all_results: pd.DataFrame,
                         gold_pd: pd.DataFrame,
                         models_to_eval: list) -> pd.DataFrame:
    """
    Run evaluate_model for every model and return a summary DataFrame.
    """
    rows = []
    for m in models_to_eval:
        r = evaluate_model(all_results, m, gold_pd, txn_cost_bps=10)
        rows.append(r)
        print(f"  {m:25s} | Sharpe: {r.get('sharpe', np.nan):.3f} "
              f"| Win%: {r.get('win_rate_pct', np.nan):.1f} "
              f"| MDD%: {r.get('max_drawdown_pct', np.nan):.1f}")
    return pd.DataFrame(rows)


def run_tc_sensitivity(all_results: pd.DataFrame,
                        models_to_eval: list) -> pd.DataFrame:
    """
    Sharpe ratio at each transaction-cost level for every model.
    """
    tc_results = {}
    for bps in TRANSACTION_COSTS_BPS:
        tc_row = {}
        for m in models_to_eval:
            sub = all_results[all_results["model"] == m].copy()
            if sub.empty:
                tc_row[m] = np.nan
                continue
            tc_per_trade  = bps / 10000.0
            sub["lag"]    = sub.groupby("ticker")["signal"].shift(1)
            sub["to"]     = (sub["signal"] != sub["lag"]).astype(float)
            sub["ret_tc"] = sub["signal_ret"] - sub["to"] * tc_per_trade
            tc_row[m]     = compute_sharpe(sub["ret_tc"].dropna())
        tc_results[bps] = tc_row
    df = pd.DataFrame(tc_results).T
    df.index.name = "TC (bps)"
    return df


def run_dm_tests(all_results: pd.DataFrame,
                  models_to_eval: list,
                  baseline: str = "always_hold") -> pd.DataFrame:
    """
    Diebold-Mariano pairwise tests: each model vs the always-hold baseline.
    """
    ah = (all_results[all_results["model"] == baseline]
          .copy().sort_values(["ticker", "date"]))
    ah["signal_ret"] = ah["ret_5d_fwd"] * ah["signal"].fillna(0)

    rows = []
    for m in models_to_eval:
        if m == baseline:
            continue
        mod = (all_results[all_results["model"] == m]
               .copy().sort_values(["ticker", "date"]))
        mod["signal_ret"] = mod["ret_5d_fwd"] * mod["signal"].fillna(0)
        merged = ah[["ticker", "date", "signal_ret"]].merge(
            mod[["ticker", "date", "signal_ret"]],
            on=["ticker", "date"], suffixes=("_ah", "_mod")
        )
        if merged.empty:
            continue
        out = diebold_mariano_test(merged["signal_ret_ah"].values,
                                    merged["signal_ret_mod"].values)
        rows.append({"model": m, **out})
        print(f"  {m:25s} | DM={out['dm_stat']:+.3f} "
              f"| p={out['p_value']:.4f} | {out['result']}")

    return pd.DataFrame(rows)
