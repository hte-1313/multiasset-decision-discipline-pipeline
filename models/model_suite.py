"""
models/model_suite.py
---------------------
Part 5: All four models in the suite.

Model 1: Always-Hold baseline
  signal = 1 always. P(hold) = unconditional hold rate from Gold panel.
  This is the null model — it earns the market return with no active decisions.
  Beats a 50/50 coin flip because financial assets have positive drift.
  Active models must beat this, not a random classifier.

Model 2: XGBoost
  Non-linear tabular model on the full feature set X^{full}.
  Additive ensemble of decision trees:
    p_hat = sigma( sum_{m=1}^{M} f_m(X^{full}) )
  Moderate regularisation: n_estimators=100, max_depth=4, lr=0.05,
  subsample=0.8, colsample_bytree=0.8.

Model 3: Bayesian Hybrid PEFT
  Sequential Gaussian posterior over a linear weight vector.
  Update (Sherman-Morrison rank-1 correction):
    k      = P * x / (sigma_sq + x' P x)
    w_new  = w + k * (y - x' w)
    P_new  = P - k * (P * x)'
  Posterior predictive probability uses probit approximation:
    p = sigma(mu / sqrt(1 + pi/8 * sigma^2))
  Uses a narrower feature set than XGBoost (linear model, fewer = more stable).

Model 4A: Pregel GNN (logistic regression on Pregel-augmented features)
  pregel_score from the Gold layer already encodes graph-structural information.
  Logistic regression maps Pregel + GNN node features to hold probability.
  Tests whether cross-asset dependence adds predictive value.

Model 4B: GraphSAGE (conditional on HAS_TORCH)
  Trainable GNN with neighbourhood aggregation:
    h_v^(k) = sigma(W^(k) · CONCAT(h_v^(k-1), MEAN_{u in N(v)} h_u^(k-1)))
  Trained on daily correlation graphs, walk-forward less frequently.
  Skipped if PyTorch / PyTorch Geometric unavailable.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from config import DECISION_THRESH, REFIT_EVERY, INITIAL_TRAIN_DAYS, HOLD_DAYS
from assets import GNN_NODE_FEATURES, GNN_HIDDEN, GNN_LAYERS, GNN_EPOCHS, GNN_LR
from config import CORR_EDGE_THRESHOLD

# Optional PyTorch
try:
    import torch
    import torch.nn as nn
    from torch_geometric.data import Data as PyGData
    from torch_geometric.nn import SAGEConv
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from assets import EVAL_EXCLUDE


# ── Model 1: Always-Hold ───────────────────────────────────────────────────

def make_always_hold_fn(unconditional_hold_rate: float):
    """
    Returns the always-hold model function.
    Calibrates p_hold to the unconditional positive-5d target rate so the
    baseline probability is empirically grounded, not arbitrarily 0.5.
    """
    def always_hold_fn(train_X, train_y, feat_X):
        return 1, unconditional_hold_rate
    return always_hold_fn


# ── Model 2: XGBoost ──────────────────────────────────────────────────────

def xgb_fn(train_X: pd.DataFrame, train_y: pd.Series,
            feat_X: pd.DataFrame):
    """
    XGBoost classifier with moderate regularisation.
    Imports xgboost inside the function so it works on Spark workers
    where the package may not be on the default PYTHONPATH.
    """
    import importlib
    if importlib.util.find_spec("xgboost") is None:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "xgboost", "-q", "--no-deps",
                               "--target", "/tmp/xgb_pkg"])
        sys.path.insert(0, "/tmp/xgb_pkg")
    import xgboost as xgb_lib

    valid_cols = [c for c in train_X.columns if train_X[c].notna().sum() > 10]
    if not valid_cols or feat_X[valid_cols].isnull().all().all():
        return 0, 0.5

    X_tr = train_X[valid_cols].fillna(0.0).values
    y_tr = train_y.fillna(0).astype(int).values
    X_te = feat_X[valid_cols].fillna(0.0).values

    if len(np.unique(y_tr)) < 2:
        return int(y_tr[-1]), float(y_tr[-1])

    model = xgb_lib.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", verbosity=0, n_jobs=1,
    )
    model.fit(X_tr, y_tr)
    p = float(model.predict_proba(X_te)[0, 1])
    return int(p >= DECISION_THRESH), p


# ── Model 3: Bayesian Hybrid PEFT ─────────────────────────────────────────

class BayesianHybridPEFT:
    """
    Online Bayesian linear classifier with Gaussian posterior.

    Maintains posterior over weight vector w:
      Prior:    w ~ N(0, delta * I)
      Likelihood: y_t ~ Bernoulli(sigma(x_t' w))

    Update uses Sherman-Morrison rank-1 correction (exact for Gaussian):
      k      = P x / (sigma_sq + x' P x)
      w_new  = w + k * (y - x' w)
      P_new  = P - k (P x)'

    Posterior predictive probability (probit approximation):
      mu    = x' w
      var   = x' P x + sigma_sq
      scale = sqrt(1 + pi/8 * var)
      p     = sigma(mu / scale)
    """

    def __init__(self, d: int, delta: float = 1.0, sigma_sq: float = 1e-3):
        self.d        = d
        self.w        = np.zeros(d)
        self.P        = np.eye(d) * delta
        self.sigma_sq = sigma_sq
        self.n_obs    = 0

    def update(self, x: np.ndarray, y: float) -> None:
        x   = np.asarray(x, dtype=float)
        Px  = self.P @ x
        k   = Px / (self.sigma_sq + float(x @ Px))
        e   = float(y) - float(x @ self.w)
        self.w  = self.w + k * e
        self.P  = self.P - np.outer(k, Px)
        self.n_obs += 1

    def predict_proba(self, x: np.ndarray) -> float:
        x     = np.asarray(x, dtype=float)
        mu    = float(x @ self.w)
        var   = float(x @ self.P @ x) + self.sigma_sq
        scale = np.sqrt(1.0 + np.pi / 8.0 * var)
        return float(1.0 / (1.0 + np.exp(-mu / scale)))

    def reset(self, delta: float, sigma_sq: float = None) -> None:
        self.w        = np.zeros(self.d)
        self.P        = np.eye(self.d) * delta
        self.sigma_sq = sigma_sq or self.sigma_sq
        self.n_obs    = 0


def calibrate_prior(gold_pd: pd.DataFrame, feat_cols: list) -> tuple:
    """
    Calibrate sigma_sq and delta from the Gold panel before model execution.

    sigma_sq = var(target_ret_5d)  clipped to [1e-6, 1.0]
    delta    = sigma_sq / mean_feature_variance  clipped to [0.01, 10.0]
    """
    ret_col  = "target_ret_5d" if "target_ret_5d" in gold_pd.columns else "ret_5d_fwd"
    ret_vals = gold_pd[ret_col].dropna().values
    sigma_sq = np.clip(float(np.var(ret_vals)) if len(ret_vals) > 10 else 1e-3,
                       1e-6, 1.0)
    feat_vars = gold_pd[feat_cols].var(skipna=True).mean()
    delta     = np.clip(float(sigma_sq / (feat_vars + 1e-8)), 0.01, 10.0)
    print(f"Bayesian prior calibration: sigma_sq={sigma_sq:.6f}  delta={delta:.4f}")
    return sigma_sq, delta


def make_bayesian_peft_fn(bayesian_feature_cols: list,
                           sigma_sq: float, delta: float):
    """Returns the Bayesian PEFT model function with calibrated priors."""
    def bayesian_peft_fn(train_X: pd.DataFrame, train_y: pd.Series,
                          feat_X: pd.DataFrame):
        cols = [c for c in bayesian_feature_cols if c in train_X.columns]
        if not cols:
            return 0, 0.5

        model = BayesianHybridPEFT(d=len(cols), delta=delta, sigma_sq=sigma_sq)
        X_tr  = train_X[cols].fillna(0.0).values

        ret_col = ("target_ret_5d" if "target_ret_5d" in train_X.columns
                   else "ret_5d_fwd" if "ret_5d_fwd" in train_X.columns
                   else None)
        y_tr = (train_X[ret_col].fillna(0.0).values if ret_col
                else train_y.fillna(0).astype(float).values * 0.01)

        for x_i, y_i in zip(X_tr[::REFIT_EVERY], y_tr[::REFIT_EVERY]):
            model.update(x_i, y_i)

        x_pred = feat_X[cols].fillna(0.0).values[0]
        p = model.predict_proba(x_pred)
        return int(p >= DECISION_THRESH), p

    return bayesian_peft_fn


# ── Model 4A: Pregel GNN (logistic on Pregel features) ────────────────────

def make_pregel_signal_fn(gnn_feature_cols: list):
    """
    Logistic regression on Pregel-augmented features.
    pregel_score encodes distributed graph centrality from the Gold layer.
    Tests whether cross-asset dependence adds value over asset-specific features.
    """
    def pregel_signal_fn(train_X: pd.DataFrame, train_y: pd.Series,
                          feat_X: pd.DataFrame):
        cols = [c for c in (gnn_feature_cols + ["pregel_score"])
                if c in train_X.columns]
        if not cols:
            return 0, 0.5

        X_tr = train_X[cols].fillna(0.0).values
        y_tr = train_y.fillna(0).astype(int).values
        X_te = feat_X[cols].fillna(0.0).values

        if len(np.unique(y_tr)) < 2:
            return int(y_tr[-1]), float(y_tr[-1])

        clf = LogisticRegression(C=1.0, max_iter=200, random_state=42)
        clf.fit(X_tr, y_tr)
        p = float(clf.predict_proba(X_te)[0, 1])
        return int(p >= DECISION_THRESH), p

    return pregel_signal_fn


# ── Model 4B: GraphSAGE (optional, requires PyTorch) ──────────────────────

if HAS_TORCH:
    class GraphSAGEModel(nn.Module):
        """
        GraphSAGE for binary node classification.

        Neighbourhood aggregation:
          h_v^(k) = sigma(W^(k) · CONCAT(h_v^(k-1), MEAN_{u in N(v)} h_u^(k-1)))

        Parameters
        ----------
        in_ch    : input feature dimension
        hidden   : hidden layer dimension
        n_layers : number of SAGE convolutional layers
        """
        def __init__(self, in_ch: int, hidden: int = 32, n_layers: int = 2):
            super().__init__()
            self.convs = nn.ModuleList()
            self.convs.append(SAGEConv(in_ch, hidden))
            for _ in range(n_layers - 2):
                self.convs.append(SAGEConv(hidden, hidden))
            self.convs.append(SAGEConv(hidden, 1))
            self.dropout = nn.Dropout(0.3)

        def forward(self, x, edge_index):
            for conv in self.convs[:-1]:
                x = conv(x, edge_index)
                x = torch.relu(x)
                x = self.dropout(x)
            return self.convs[-1](x, edge_index).squeeze(-1)

    def build_daily_graph(pdf_date: pd.DataFrame, feat_cols: list,
                           corr_threshold: float = CORR_EDGE_THRESHOLD):
        """Build a PyG graph for a single trading day's asset panel."""
        pdf_date = pdf_date.dropna(subset=feat_cols)
        tickers  = pdf_date["ticker"].tolist()
        if len(tickers) < 3:
            return None, []

        X = torch.tensor(
            pdf_date.set_index("ticker").reindex(tickers)[feat_cols]
                    .fillna(0.0).values,
            dtype=torch.float32,
        )
        y = torch.tensor(
            pdf_date.set_index("ticker").reindex(tickers)["target_5d"]
                    .fillna(0).astype(int).values,
            dtype=torch.float32,
        )

        corr_vals = pdf_date.set_index("ticker")["corr_spy_60"].fillna(0).to_dict()
        edges_src, edges_dst, edge_w = [], [], []
        for i, ti in enumerate(tickers):
            for j, tj in enumerate(tickers):
                if i >= j: continue
                c = abs((corr_vals.get(ti, 0) * corr_vals.get(tj, 0)) ** 0.5)
                if c > corr_threshold:
                    edges_src += [i, j]; edges_dst += [j, i]
                    edge_w    += [c, c]

        if not edges_src:  # star fallback
            edges_src = list(range(1, len(tickers)))
            edges_dst = [0] * (len(tickers) - 1)
            edge_w    = [0.1] * (len(tickers) - 1)
            edges_src += edges_dst; edges_dst += edges_src[:len(edges_src)//2]
            edge_w    *= 2

        ei = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        ew = torch.tensor(edge_w, dtype=torch.float32)
        return PyGData(x=X, edge_index=ei, edge_attr=ew, y=y), tickers

    def run_graphsage_walkforward(gold_pd: pd.DataFrame,
                                   feat_cols: list) -> pd.DataFrame:
        """
        Full walk-forward for PyTorch GraphSAGE.

        Refits every REFIT_EVERY * 5 days for stability and runtime control.
        Trains on up to 50 daily graphs from the trailing 252-day window.
        Each graph uses the daily correlation structure to build edges.
        """
        device    = torch.device("cpu")
        all_dates = sorted(gold_pd["date"].unique())
        model     = None
        results   = []
        last_fit  = -999

        if len(all_dates) < INITIAL_TRAIN_DAYS + HOLD_DAYS + 1:
            return pd.DataFrame(columns=["ticker","date","signal","p_hold",
                                          "ret_5d_fwd","model","asset_class"])

        for t_idx, date in enumerate(all_dates[INITIAL_TRAIN_DAYS:],
                                      start=INITIAL_TRAIN_DAYS):

            do_refit = (t_idx - last_fit) >= (REFIT_EVERY * 5)
            if do_refit:
                train_dates = all_dates[max(0, t_idx - 252):t_idx]
                graphs = []
                for td in train_dates[::5]:
                    snap = gold_pd[gold_pd["date"] == td].copy()
                    g, _ = build_daily_graph(snap, feat_cols)
                    if g is not None and 0 < g.y.sum() < len(g.y):
                        graphs.append(g)

                if len(graphs) < 5:
                    continue

                model = GraphSAGEModel(len(feat_cols), GNN_HIDDEN, GNN_LAYERS).to(device)
                opt   = torch.optim.Adam(model.parameters(), lr=GNN_LR)
                crit  = nn.BCEWithLogitsLoss()
                model.train()
                for _ in range(GNN_EPOCHS):
                    np.random.shuffle(graphs)
                    for g in graphs[:50]:
                        g    = g.to(device)
                        mask = ~torch.isnan(g.y)
                        if mask.sum() == 0: continue
                        opt.zero_grad()
                        out  = model(g.x, g.edge_index)
                        loss = crit(out[mask], g.y[mask])
                        loss.backward()
                        opt.step()
                last_fit = t_idx

            if model is None: continue
            snap = gold_pd[gold_pd["date"] == date].copy()
            if snap.empty: continue
            g, tickers = build_daily_graph(snap, feat_cols)
            if g is None: continue

            model.eval()
            with torch.no_grad():
                logits = model(g.x.to(device), g.edge_index.to(device))
                probs  = torch.sigmoid(logits).cpu().numpy()

            date_snap = snap.set_index("ticker").reindex(tickers)
            for i, tick in enumerate(tickers):
                if tick in EVAL_EXCLUDE: continue
                p  = float(probs[i])
                r  = date_snap.loc[tick, "ret_5d_fwd"] if tick in date_snap.index else 0.0
                ac = date_snap.loc[tick, "asset_class"] if "asset_class" in date_snap.columns else ""
                results.append({
                    "ticker": tick,
                    "date": str(date.date() if hasattr(date, "date") else date),
                    "signal": int(p >= DECISION_THRESH),
                    "p_hold": p,
                    "ret_5d_fwd": float(r) if pd.notna(r) else 0.0,
                    "model": "graphsage",
                    "asset_class": ac,
                })

        return pd.DataFrame(results)
