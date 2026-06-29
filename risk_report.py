"""
Verifiable TDA Trading Strategy Backtest & Risk Report
=======================================================
Produces a 6-panel PDF plus console summary table.

Strategies compared
-------------------
  A) Sector Laplacian signal + Fiedler-value regime filter
       (spectral graph proxy for persistent homology — validated in
        Section 11: Fiedler value correlates −0.991 with topology CV,
        computes 50× faster than ripser)
  B) Sector Laplacian signal + 20-day realized-vol regime filter
       (naive baseline — does topology add value beyond plain vol?)
  C) Sector Laplacian signal, no filter
       (floor baseline — does any filter help?)

Factor regression: Fama-French 5 factors + UMD (momentum)
  -- if alpha is not significant, the strategy is just loading factors

To run on Google Colab
-----------------------
    !pip install yfinance pandas-datareader matplotlib scipy statsmodels
    from google.colab import files
    files.upload()   # upload this file, then:
    !python risk_report.py

Author: Adam Levine, TDA Trading Strategy Thesis
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import scipy.linalg as la
import statsmodels.api as sm
from pathlib import Path

# ── CONFIG ───────────────────────────────────────────────────────────────────

SECTORS = {
    "Financials": ["JPM", "BAC", "WFC", "GS",  "MS",   "BLK"],
    "Energy":     ["XOM", "CVX", "COP", "SLB",  "MPC",  "VLO"],
    "Technology": ["AAPL","MSFT","NVDA","AMD",  "INTC", "ORCL"],
}
BENCHMARK   = "SPY"
START       = "2019-01-01"
END         = "2024-12-31"

TRAIN_END   = "2021-12-31"   # walk-forward split: train on 2019-2021
TEST_START  = "2022-01-01"   # test on 2022-2024

LOOKBACK    = 60     # days for rolling correlation window
CORR_THRESH = 0.30   # edge threshold for adjacency matrix
ALPHA_DIFF  = 0.50   # graph diffusion strength
T_ITER      = 3      # diffusion iterations
N_LONG      = 2      # assets to long per sector
N_SHORT     = 2      # assets to short per sector
REGIME_PCT  = 75     # percentile threshold for "unstable" regime
TC_BPS      = 5      # transaction costs in basis points

OUTPUT_DIR  = Path(".")

# ── DATA ─────────────────────────────────────────────────────────────────────

def download_prices():
    """Download adjusted close prices. Returns (prices_df, spy_df)."""
    try:
        import yfinance as yf
    except ImportError:
        raise SystemExit("yfinance not installed. Run:  pip install yfinance")

    all_tickers = [t for ts in SECTORS.values() for t in ts] + [BENCHMARK]
    print(f"[1/5] Downloading {len(all_tickers)} tickers from Yahoo Finance …")
    raw = yf.download(all_tickers, start=START, end=END,
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": all_tickers[0]})

    spy  = prices[[BENCHMARK]].dropna()
    sect = prices.drop(columns=[BENCHMARK])
    print(f"    Downloaded {sect.shape[0]} trading days, "
          f"{sect.shape[1]} sector tickers, {spy.shape[0]} SPY days.")
    return sect, spy


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()


# ── SIGNAL ENGINE ─────────────────────────────────────────────────────────────

def build_laplacian(corr_matrix: np.ndarray, threshold: float):
    """Normalised graph Laplacian from a correlation matrix."""
    W = np.where(corr_matrix > threshold, corr_matrix, 0.0)
    np.fill_diagonal(W, 0.0)
    d = W.sum(axis=1)
    # isolated nodes: treat as self-loop so D is invertible
    d = np.where(d == 0, 1.0, d)
    d_inv_sqrt = 1.0 / np.sqrt(d)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L = np.eye(len(d)) - D_inv_sqrt @ W @ D_inv_sqrt
    return L


def diffuse(L: np.ndarray, x: np.ndarray, alpha: float, iters: int) -> np.ndarray:
    """Graph diffusion:  h = (I - alpha*L)^iters @ x"""
    op = np.eye(L.shape[0]) - alpha * L
    h = x.copy()
    for _ in range(iters):
        h = op @ h
    return h


def fiedler_value(L: np.ndarray) -> float:
    """Second smallest eigenvalue of L (algebraic connectivity)."""
    vals = np.sort(la.eigvalsh(L))
    return float(vals[1]) if len(vals) > 1 else 0.0


def run_sector_signals(returns: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """
    For a single sector, compute daily:
      - laplacian_residuals   (raw signal)
      - fiedler               (TDA proxy for regime)
      - realized_vol          (naive regime proxy)

    Returns a DataFrame indexed by date.
    """
    sect_ret = returns[tickers].dropna()
    n = len(tickers)
    results = []

    for i in range(LOOKBACK, len(sect_ret)):
        window  = sect_ret.iloc[i - LOOKBACK : i]
        today   = sect_ret.iloc[i]
        corr    = window.corr().values

        L  = build_laplacian(corr, CORR_THRESH)
        h  = diffuse(L, today.values, ALPHA_DIFF, T_ITER)
        e  = today.values - h          # residuals: high = underperformed peers

        fv = fiedler_value(L)
        rv = window.std().mean() * np.sqrt(252)   # annualised avg vol

        row = {"date": sect_ret.index[i], "fiedler": fv, "realized_vol": rv}
        for j, tk in enumerate(tickers):
            row[f"resid_{tk}"] = e[j]
        results.append(row)

    return pd.DataFrame(results).set_index("date")


# ── REGIME FILTERS ────────────────────────────────────────────────────────────

def fiedler_filter(signal_df: pd.DataFrame, train_end: str) -> pd.Series:
    """
    Unstable = 1 when 30-day rolling std of Fiedler value exceeds
    the training-period 75th percentile.
    Regime filter is calibrated on TRAIN data only (walk-forward).
    """
    fv = signal_df["fiedler"]
    fv_vol = fv.rolling(30).std()
    train_mask = fv_vol.loc[:train_end]
    threshold  = np.nanpercentile(train_mask, REGIME_PCT)
    unstable   = (fv_vol > threshold).astype(int)
    return unstable.rename("unstable_fiedler")


def vol_filter(signal_df: pd.DataFrame, train_end: str) -> pd.Series:
    """
    Unstable = 1 when 20-day realized vol exceeds training 75th pct.
    """
    rv = signal_df["realized_vol"]
    rv_roll = rv.rolling(20).mean()
    threshold = np.nanpercentile(rv_roll.loc[:train_end], REGIME_PCT)
    unstable  = (rv_roll > threshold).astype(int)
    return unstable.rename("unstable_vol")


# ── PORTFOLIO CONSTRUCTION ────────────────────────────────────────────────────

def sector_returns(signal_df: pd.DataFrame, tickers: list,
                   raw_returns: pd.DataFrame,
                   unstable_fiedler: pd.Series,
                   unstable_vol: pd.Series) -> pd.DataFrame:
    """
    Construct three daily return series for one sector.

    MEAN-REVERSION with 1-day lag (no look-ahead bias):
      - Signal on day i:   compute residuals from day i returns
      - Position on day i+1: LONG most-negative residuals (underperformers
        expected to rebound), SHORT most-positive (overperformers expected
        to revert)
      - PnL on day i+1:    from day i+1 actual returns
    """
    resid_cols = [f"resid_{tk}" for tk in tickers]
    resids = signal_df[resid_cols].copy()
    resids.columns = tickers

    dates = resids.index
    prev_pos = {s: np.zeros(len(tickers)) for s in ["A", "B", "C"]}
    rows = []

    for i in range(1, len(dates)):
        signal_date = dates[i - 1]     # yesterday's signal
        trade_date  = dates[i]         # today's execution + PnL

        if trade_date not in raw_returns.index:
            continue

        r   = resids.loc[signal_date].values
        idx = np.argsort(r)       # ascending order

        # MEAN REVERSION: long the underperformers, short the overperformers
        pos = np.zeros(len(tickers))
        pos[idx[:N_LONG]]   =  1.0 / N_LONG     # lowest residuals → long
        pos[idx[-N_SHORT:]] = -1.0 / N_SHORT     # highest residuals → short

        day_rets = raw_returns.loc[trade_date, tickers].values

        ret = {}
        for strat, filter_flag in [("A", unstable_fiedler.get(trade_date, 0)),
                                    ("B", unstable_vol.get(trade_date, 0)),
                                    ("C", 0)]:
            effective_pos = np.zeros(len(tickers)) if filter_flag else pos
            turnover = np.abs(effective_pos - prev_pos[strat]).sum()
            tc = turnover * TC_BPS / 10_000
            ret[strat] = float(effective_pos @ day_rets) - tc
            prev_pos[strat] = effective_pos

        rows.append({"date": trade_date, **ret})

    return pd.DataFrame(rows).set_index("date")


# ── STRATEGY D: BETA-SPREAD + FIEDLER REGIME (Section 7 logic) ────────────────

def beta_spread_strategy(returns: pd.DataFrame, tickers: list,
                         signal_df: pd.DataFrame,
                         train_end: str) -> pd.Series:
    """
    Replicates Phase 2 / Section 7 strategy:
      - Compute betas vs sector-equal-weight from training data
      - Split into top-3 high-beta, bottom-3 low-beta
      - Fiedler < 25th pct (stressed) → long high-beta spread
      - Fiedler ≥ 25th pct (calm)     → short high-beta spread
      - 5-day rebalance, 1-day signal lag, 5 bps TC
    """
    sect_ret = returns[tickers].dropna()
    market   = sect_ret.mean(axis=1)

    # Compute betas from training period only
    train_ret = sect_ret.loc[:train_end]
    train_mkt = market.loc[:train_end]
    betas = {}
    for tk in tickers:
        cov = train_ret[tk].cov(train_mkt)
        var = train_mkt.var()
        betas[tk] = cov / var if var > 0 else 1.0
    betas = pd.Series(betas).sort_values(ascending=False)

    n_pairs   = min(3, len(tickers) // 2)
    high_beta = betas.index[:n_pairs].tolist()
    low_beta  = betas.index[-n_pairs:].tolist()

    # Calibrate Fiedler threshold from training period
    fv_train = signal_df["fiedler"].loc[:train_end]
    fiedler_threshold = np.nanpercentile(fv_train, 25)

    dates = signal_df.index
    prev_position = 0.0
    rows = []

    for i in range(1, len(dates)):
        signal_date = dates[i - 1]
        trade_date  = dates[i]

        if trade_date not in sect_ret.index:
            continue

        fv = signal_df.loc[signal_date, "fiedler"]

        # Regime: low Fiedler = fragmented graph ≈ high H1 loops = stressed
        if fv < fiedler_threshold:
            position = 1.0     # stressed → long high-beta spread
        else:
            position = -1.0    # calm → short high-beta spread

        # Only rebalance every 5 days
        day_idx = (i - 1)
        if day_idx % 5 != 0 and prev_position != 0:
            position = prev_position

        hb_ret = sect_ret.loc[trade_date, high_beta].mean()
        lb_ret = sect_ret.loc[trade_date, low_beta].mean()
        spread = hb_ret - lb_ret

        strat_ret = position * spread

        # TC on position changes
        if position != prev_position:
            strat_ret -= 2 * n_pairs * TC_BPS / 10_000

        prev_position = position
        rows.append({"date": trade_date, "D": strat_ret})

    return pd.DataFrame(rows).set_index("date")["D"]


# ── FACTOR REGRESSION ─────────────────────────────────────────────────────────

def load_ff_factors(start: str, end: str) -> pd.DataFrame:
    """
    Download Fama-French 5 factors + momentum from Ken French's library.
    Falls back to a zeros DataFrame if internet is unavailable.
    """
    try:
        import pandas_datareader.data as web
        ff5 = web.DataReader("F-F_Research_Data_5_Factors_2x3_daily",
                             "famafrench", start, end)[0]
        mom = web.DataReader("F-F_Momentum_Factor_daily",
                             "famafrench", start, end)[0]
        mom.columns = ["UMD"]
        factors = ff5.join(mom, how="inner") / 100.0
        # Ken French data comes back with a PeriodIndex; convert to timestamps.
        idx = factors.index
        factors.index = idx.to_timestamp() if hasattr(idx, "to_timestamp") \
            else pd.to_datetime(idx)
        print("    Fama-French factors downloaded OK.")
        return factors
    except Exception as exc:
        print(f"    [WARN] FF factors unavailable ({exc}). "
              "Regression will show zeros — run with internet access for real alphas.")
        return pd.DataFrame()


def factor_regression(strategy_ret: pd.Series,
                      factors: pd.DataFrame,
                      label: str) -> dict:
    if factors.empty:
        return {"label": label, "alpha_ann": np.nan, "t_alpha": np.nan,
                "r2": np.nan, "beta_mkt": np.nan}

    y = strategy_ret.rename("ret")
    rf = factors["RF"]
    excess = (y - rf).dropna()

    X = factors.drop(columns=["RF"]).reindex(excess.index).dropna()
    excess = excess.reindex(X.index)

    X = sm.add_constant(X)
    model = sm.OLS(excess, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

    alpha_daily = model.params["const"]
    t_alpha     = model.tvalues["const"]
    r2          = model.rsquared
    beta_mkt    = model.params.get("Mkt-RF", np.nan)

    return {
        "label":       label,
        "alpha_ann":   alpha_daily * 252,
        "t_alpha":     t_alpha,
        "r2":          r2,
        "beta_mkt":    beta_mkt,
    }


# ── METRICS ───────────────────────────────────────────────────────────────────

def performance_metrics(ret: pd.Series, label: str) -> dict:
    test = ret.loc[TEST_START:]
    ann  = test.mean() * 252
    vol  = test.std()  * np.sqrt(252)
    sr   = ann / vol if vol > 0 else np.nan

    cum    = (1 + test).cumprod()
    peak   = cum.cummax()
    dd     = (cum - peak) / peak
    mdd    = dd.min()

    monthly = test.resample("ME").sum()
    hit     = (monthly > 0).mean()

    return {"label": label, "ann_ret": ann, "ann_vol": vol,
            "sharpe": sr, "max_dd": mdd, "hit_rate": hit}


# ── VISUALISATION ─────────────────────────────────────────────────────────────

def plot_report(combined: dict, spy_ret: pd.Series,
                factor_results: list, output_path: Path):
    """
    6-panel figure:
      [0] Equity curves A / B / C / SPY (test period)
      [1] Rolling 252-day Sharpe  — Strategy A
      [2] Underwater drawdown     — Strategy A vs SPY
      [3] Monthly returns heatmap — Strategy A
      [4] Factor regression α, β  (bar chart)
      [5] Return distribution     — Strategy A vs Normal
    """
    retA = combined["A"].loc[TEST_START:]
    retB = combined["B"].loc[TEST_START:]
    retC = combined["C"].loc[TEST_START:]
    retD = combined["D"].loc[TEST_START:]
    spy  = spy_ret.loc[TEST_START:]

    fig = plt.figure(figsize=(18, 20))
    fig.suptitle("TDA Sector Strategy — Verifiable Backtest Risk Report\n"
                 "(Test Period: 2022–2024, 5 bps transaction costs)",
                 fontsize=14, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 2, figure=fig,
                           hspace=0.40, wspace=0.30,
                           top=0.94, bottom=0.05)

    colors = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c",
              "D": "#9467bd", "SPY": "#d62728"}

    # ── Panel 0: equity curves ─────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    for label, ret, col in [
            ("A: Laplacian+Fiedler", retA, colors["A"]),
            ("B: Laplacian+Vol",     retB, colors["B"]),
            ("C: Laplacian only",    retC, colors["C"]),
            ("D: Beta-spread (Sec7)",retD, colors["D"]),
            ("SPY",                  spy,  colors["SPY"])]:
        cum = (1 + ret).cumprod()
        ax0.plot(cum.index, cum.values, label=label, color=col, linewidth=1.5)
    ax0.axhline(1.0, color="black", linewidth=0.6, linestyle="--")
    ax0.set_title("Equity Curves (rebased to 1.0)")
    ax0.set_ylabel("Cumulative return")
    ax0.legend(fontsize=8)
    ax0.grid(alpha=0.3)

    # ── Panel 1: rolling Sharpe ────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    roll_sr = retA.rolling(252).apply(
        lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else np.nan)
    ax1.plot(roll_sr.index, roll_sr.values, color=colors["A"], linewidth=1.5)
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax1.fill_between(roll_sr.index, roll_sr.values, 0,
                     where=(roll_sr.values > 0), alpha=0.2, color="green")
    ax1.fill_between(roll_sr.index, roll_sr.values, 0,
                     where=(roll_sr.values < 0), alpha=0.2, color="red")
    ax1.set_title("Rolling 252-Day Sharpe — Strategy A")
    ax1.set_ylabel("Sharpe ratio")
    ax1.grid(alpha=0.3)

    # ── Panel 2: underwater drawdown ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    for label, ret, col in [("Strategy A", retA, colors["A"]),
                              ("SPY",        spy,  colors["SPY"])]:
        cum  = (1 + ret).cumprod()
        peak = cum.cummax()
        dd   = (cum - peak) / peak * 100
        ax2.fill_between(dd.index, dd.values, 0, alpha=0.35, color=col, label=label)
    ax2.set_title("Underwater Drawdown (%)")
    ax2.set_ylabel("Drawdown (%)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # ── Panel 3: monthly returns heatmap ──────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    monthly = retA.resample("ME").sum() * 100
    monthly.index = monthly.index.to_period("M")
    df_heat = monthly.reset_index()
    df_heat.columns = ["period", "ret"]
    df_heat["year"]  = df_heat["period"].dt.year
    df_heat["month"] = df_heat["period"].dt.month
    pivot = df_heat.pivot(index="year", columns="month", values="ret")
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"][:len(pivot.columns)]

    vmax = max(abs(pivot.values[~np.isnan(pivot.values)]).max(), 1)
    cmap = LinearSegmentedColormap.from_list("rg", ["#d62728","white","#2ca02c"])
    im = ax3.imshow(pivot.values, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax3.set_xticks(range(pivot.shape[1]))
    ax3.set_xticklabels(pivot.columns, fontsize=7)
    ax3.set_yticks(range(pivot.shape[0]))
    ax3.set_yticklabels(pivot.index, fontsize=8)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax3.text(j, i, f"{val:.1f}", ha="center", va="center",
                         fontsize=6, color="black")
    plt.colorbar(im, ax=ax3, fraction=0.03)
    ax3.set_title("Monthly Returns (%) — Strategy A")

    # ── Panel 4: factor regression bar chart ──────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    if all(np.isnan(r["alpha_ann"]) for r in factor_results):
        ax4.text(0.5, 0.5, "Factor data unavailable\n(no internet access)",
                 ha="center", va="center", transform=ax4.transAxes, fontsize=11)
        ax4.set_title("Factor-Adjusted Alpha (Fama-French 5 + UMD)")
    else:
        labels  = [r["label"] for r in factor_results]
        alphas  = [r["alpha_ann"] * 100 for r in factor_results]   # % annualised
        t_stats = [r["t_alpha"]          for r in factor_results]
        bar_cols = [colors.get(l[0], "steelblue") for l in labels]
        bars = ax4.bar(labels, alphas, color=bar_cols, alpha=0.8, edgecolor="black")
        for bar, ts in zip(bars, t_stats):
            sig = "***" if abs(ts) > 3.0 else ("**" if abs(ts) > 2.0 else
                  ("*" if abs(ts) > 1.65 else ""))
            ax4.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + (0.05 if bar.get_height() >= 0 else -0.15),
                     f"t={ts:.2f}{sig}", ha="center", fontsize=8)
        ax4.axhline(0, color="black", linewidth=0.8)
        ax4.set_ylabel("Annualised alpha (%)")
        ax4.set_title("Factor-Adjusted Alpha (FF5 + UMD)\n"
                      "* p<0.10  ** p<0.05  *** p<0.01")
        ax4.grid(axis="y", alpha=0.3)

    # ── Panel 5: return distribution ──────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    from scipy.stats import norm
    data = retA.values * 100
    ax5.hist(data, bins=60, density=True, alpha=0.6,
             color=colors["A"], edgecolor="white", linewidth=0.3, label="Strategy A")
    mu, sigma = data.mean(), data.std()
    xs = np.linspace(data.min(), data.max(), 300)
    ax5.plot(xs, norm.pdf(xs, mu, sigma), "k--", linewidth=1.5, label="Normal fit")
    ax5.axvline(0, color="red", linewidth=0.8, linestyle="--")

    from scipy.stats import kurtosis, skew
    k = kurtosis(data, fisher=True)
    s = skew(data)
    ax5.set_title(f"Daily Return Distribution — Strategy A\n"
                  f"μ={mu:.3f}%  σ={sigma:.3f}%  skew={s:.2f}  excess kurt={k:.2f}")
    ax5.set_xlabel("Daily return (%)")
    ax5.set_ylabel("Density")
    ax5.legend(fontsize=8)
    ax5.grid(alpha=0.3)

    out = output_path / "risk_report.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n[5/5] Report saved → {out.resolve()}")
    return out


# ── SUMMARY TABLE ─────────────────────────────────────────────────────────────

def print_summary(metrics: list, factor_results: list):
    print("\n" + "=" * 68)
    print("  STRATEGY PERFORMANCE SUMMARY  (Test: 2022–2024)")
    print("=" * 68)
    header = f"{'Strategy':<28} {'Ann Ret':>8} {'Ann Vol':>8} {'Sharpe':>7} {'MaxDD':>8} {'Hit':>6}"
    print(header)
    print("-" * 68)
    for m in metrics:
        print(f"{m['label']:<28} {m['ann_ret']:>7.1%} {m['ann_vol']:>8.1%} "
              f"{m['sharpe']:>7.2f} {m['max_dd']:>7.1%} {m['hit_rate']:>6.1%}")

    print("\n" + "=" * 68)
    print("  FACTOR REGRESSION  (FF5 + UMD, HAC t-stats, test period)")
    print("=" * 68)
    header2 = f"{'Strategy':<28} {'Alpha Ann%':>10} {'t-stat':>8} {'β Mkt':>7} {'R²':>6}"
    print(header2)
    print("-" * 68)
    for r in factor_results:
        a  = f"{r['alpha_ann']*100:.2f}%" if not np.isnan(r["alpha_ann"]) else "  N/A"
        t  = f"{r['t_alpha']:.2f}"        if not np.isnan(r["t_alpha"])   else "  N/A"
        bm = f"{r['beta_mkt']:.2f}"       if not np.isnan(r["beta_mkt"])  else "  N/A"
        r2 = f"{r['r2']:.3f}"             if not np.isnan(r["r2"])         else "  N/A"
        print(f"{r['label']:<28} {a:>10} {t:>8} {bm:>7} {r2:>6}")

    print("=" * 68)
    print("\nNotes:")
    print("  Strategy A uses Fiedler-value (spectral gap) as TDA proxy.")
    print("    Validated in Section 11: Fiedler↔topology-CV correlation = −0.991.")
    print("  Significance: * p<0.10  ** p<0.05  *** p<0.01 (two-tailed)")
    print("  Alpha is annualised; t-stats use Newey-West HAC (5 lags).")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    # 1. Download data
    prices, spy_prices = download_prices()
    all_ret  = compute_returns(prices)
    spy_ret  = compute_returns(spy_prices)[BENCHMARK]

    # 2. Build sector signals
    print("[2/5] Computing Laplacian signals and Fiedler values …")
    sector_signals = {}
    for sec_name, tickers in SECTORS.items():
        avail = [t for t in tickers if t in all_ret.columns]
        if len(avail) < 4:
            print(f"    [WARN] {sec_name}: only {len(avail)} tickers, skipping.")
            continue
        sig = run_sector_signals(all_ret, avail)
        sector_signals[sec_name] = (sig, avail)
        print(f"    {sec_name}: {len(sig)} signal days, "
              f"mean Fiedler={sig['fiedler'].mean():.3f}")

    # 3. Apply regime filters + build portfolio
    print("[3/5] Applying regime filters and building portfolios …")
    sector_rets_list = {"A": [], "B": [], "C": []}

    sector_d_list = []

    for sec_name, (sig, tickers) in sector_signals.items():
        uf = fiedler_filter(sig, TRAIN_END)
        uv = vol_filter(sig, TRAIN_END)
        sr = sector_returns(sig, tickers, all_ret, uf, uv)
        for s in ["A", "B", "C"]:
            sector_rets_list[s].append(sr[s])

        # Strategy D: beta-spread + Fiedler regime (Section 7 logic)
        d_ret = beta_spread_strategy(all_ret, tickers, sig, TRAIN_END)
        sector_d_list.append(d_ret)
        print(f"    {sec_name} D: high/low-beta spread, "
              f"{(sig['fiedler'].loc[:TRAIN_END] < np.nanpercentile(sig['fiedler'].loc[:TRAIN_END], 25)).mean():.0%} stressed days")

    # Equal-weight across sectors
    combined = {}
    for s in ["A", "B", "C"]:
        stacked = pd.concat(sector_rets_list[s], axis=1).dropna()
        combined[s] = stacked.mean(axis=1)

    stacked_d = pd.concat(sector_d_list, axis=1).dropna()
    combined["D"] = stacked_d.mean(axis=1)

    # 4. Download Fama-French factors
    print("[4/5] Fetching Fama-French factors …")
    factors = load_ff_factors(TEST_START, END)

    # 5. Compute metrics
    all_metrics = [
        performance_metrics(combined["A"], "A: Laplacian+Fiedler (TDA)"),
        performance_metrics(combined["B"], "B: Laplacian+Vol filter"),
        performance_metrics(combined["C"], "C: Laplacian only"),
        performance_metrics(combined["D"], "D: Beta-spread+Fiedler (Sec7)"),
        performance_metrics(spy_ret,       "SPY benchmark"),
    ]

    factor_results = [
        factor_regression(combined["A"].loc[TEST_START:],
                          factors, "A: Laplacian+Fiedler (TDA)"),
        factor_regression(combined["B"].loc[TEST_START:],
                          factors, "B: Laplacian+Vol filter"),
        factor_regression(combined["C"].loc[TEST_START:],
                          factors, "C: Laplacian only"),
        factor_regression(combined["D"].loc[TEST_START:],
                          factors, "D: Beta-spread+Fiedler (Sec7)"),
    ]

    print_summary(all_metrics, factor_results)
    plot_report(combined, spy_ret, factor_results, OUTPUT_DIR)


if __name__ == "__main__":
    main()
