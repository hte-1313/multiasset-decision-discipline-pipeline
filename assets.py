"""
assets.py
---------
Full 119-instrument asset universe across 5 asset classes,
asset-class map, sector map, and feature column lists.
"""

from config import DEV_FILTER

# ── Tickers by class ───────────────────────────────────────────────────────
FX = [
    "EURUSD=X","GBPUSD=X","USDJPY=X","AUDUSD=X","USDCAD=X","USDCHF=X",
    "NZDUSD=X","USDCNH=X","USDSEK=X","USDNOK=X","USDBRL=X","USDMXN=X",
    "USDZAR=X","USDINR=X","USDKRW=X","EURGBP=X","EURJPY=X","GBPJPY=X",
    "AUDJPY=X","EURCAD=X",
]
ETF = [
    "SPY","QQQ","IWM","DIA","VTI","EFA","EEM","VWO",
    "XLF","XLE","XLK","XLV","XLI","XLU","XLRE","XLB","XLC","XLP",
    "TLT","IEF","SHY","HYG","LQD","TIP",
    "GLD","SLV","USO","UNG","PDBC",
]
EQ = [
    "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","AVGO","AMD","INTC",
    "JPM","BAC","GS","MS","WFC","C","BLK","AXP",
    "JNJ","UNH","LLY","PFE","ABBV","MRK","CVS",
    "XOM","CVX","COP","SLB","OXY",
    "WMT","COST","HD","MCD","NKE","KO","PEP","PG",
    "BA","CAT","GE","HON","UPS","LMT","NEE","T","VZ",
]
FUT_COM = [
    "GC=F","SI=F","HG=F","PL=F","PA=F",
    "CL=F","BZ=F","NG=F","HO=F","RB=F",
    "ZC=F","ZW=F","ZS=F","ZL=F","ZM=F",
    "KC=F","CT=F","SB=F","CC=F","LE=F",
]
FUT_IDX = ["ES=F","NQ=F","RTY=F","YM=F","VX=F"]

ALL_TICKERS = FX + ETF + EQ + FUT_COM + FUT_IDX

# ── Asset-class map ────────────────────────────────────────────────────────
ASSET_CLASS_MAP = (
    {t: "FX"      for t in FX}
    | {t: "ETF"   for t in ETF}
    | {t: "EQ"    for t in EQ}
    | {t: "FUT_COM" for t in FUT_COM}
    | {t: "FUT_IDX" for t in FUT_IDX}
)

# ── Sector map ─────────────────────────────────────────────────────────────
SECTOR_MAP = {
    **{t: "FX_Major" for t in FX[:7]},
    **{t: "FX_EM" for t in ["USDCNH=X","USDBRL=X","USDMXN=X","USDZAR=X","USDINR=X","USDKRW=X"]},
    **{t: "FX_Minor" for t in ["USDSEK=X","USDNOK=X"]},
    **{t: "FX_Cross" for t in FX[15:]},
    "SPY":"Broad_Eq","QQQ":"Broad_Eq","IWM":"Broad_Eq","DIA":"Broad_Eq","VTI":"Broad_Eq",
    "EFA":"Intl","EEM":"Intl","VWO":"Intl",
    **{t: "Sector_ETF" for t in ["XLF","XLE","XLK","XLV","XLI","XLU","XLRE","XLB","XLC","XLP"]},
    **{t: "FixedInc"  for t in ["TLT","IEF","SHY","HYG","LQD","TIP"]},
    **{t: "Cmdty_ETF" for t in ["GLD","SLV","USO","UNG","PDBC"]},
    **{t: "Tech" for t in ["AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","AVGO","AMD","INTC"]},
    **{t: "Fin"  for t in ["JPM","BAC","GS","MS","WFC","C","BLK","AXP"]},
    **{t: "Health" for t in ["JNJ","UNH","LLY","PFE","ABBV","MRK","CVS"]},
    **{t: "Energy_EQ" for t in ["XOM","CVX","COP","SLB","OXY"]},
    **{t: "Consumer"  for t in ["WMT","COST","HD","MCD","NKE","KO","PEP","PG"]},
    **{t: "Indust" for t in ["BA","CAT","GE","HON","UPS","LMT","NEE","T","VZ"]},
    **{t: "Metals"    for t in ["GC=F","SI=F","HG=F","PL=F","PA=F"]},
    **{t: "Energy_F"  for t in ["CL=F","BZ=F","NG=F","HO=F","RB=F"]},
    **{t: "Grains"    for t in ["ZC=F","ZW=F","ZS=F","ZL=F","ZM=F"]},
    **{t: "Softs"     for t in ["KC=F","CT=F","SB=F","CC=F","LE=F"]},
    **{t: "Idx_Fut"   for t in ["ES=F","NQ=F","RTY=F","YM=F"]},
    "VX=F": "Vol_Fut",
}

# ── Active ticker list (respects DEV_FILTER) ──────────────────────────────
TICKERS = DEV_FILTER if DEV_FILTER else ALL_TICKERS
ASSET_CLASS_MAP = {t: ASSET_CLASS_MAP[t] for t in TICKERS if t in ASSET_CLASS_MAP}
SECTOR_MAP      = {t: SECTOR_MAP.get(t, "Unknown") for t in TICKERS}

# ── Feature column lists ───────────────────────────────────────────────────
# All-model features used by XGBoost and Pregel GNN
FEATURE_COLS = [
    "ret_z_60", "ret_5d", "ret_21d", "mom_12_1",        # returns & momentum
    "vol_ratio", "vol_of_vol", "vol_20",                 # volatility regime
    "dist_sma_20", "dist_sma_60", "dist_sma_200", "trend_strength",  # trend
    "bb_pctb", "bb_bandwidth", "rsi_14", "macd_hist",   # oscillators
    "drawdown", "drawdown_speed", "max_drawdown_252",    # risk
    "xs_ret_rank", "xs_vol_rank", "xs_mom_rank",         # cross-sectional ranks
    "beta_60", "corr_spy_60", "idio_vol_60",             # market structure
    "amihud", "hl_range", "turnover_z_60",               # microstructure
    "is_month_end", "is_quarter_end",                    # calendar
    "pregel_score",                                      # graph structure
]

# Bayesian PEFT uses a narrower set (linear model — fewer features = less instability)
BAYESIAN_FEATURE_COLS = [
    "ret_z_60", "vol_ratio", "vol_of_vol",
    "drawdown", "drawdown_speed", "skew_60", "kurt_60",
    "trend_strength", "bb_bandwidth", "pregel_score",
]

# GNN node features (used by Pregel message passing and GraphSAGE)
GNN_NODE_FEATURES = [
    "xs_ret_rank", "xs_vol_rank", "xs_mom_rank",
    "dist_sma_20", "dist_sma_60", "bb_pctb",
    "ret_z_60", "vol_ratio", "drawdown", "trend_strength",
]
