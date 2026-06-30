"""
Torture tests for the attention signal -- try hard to KILL the +0.48 Sharpe.
A weekly long/short on mid-cap consumer names has two obvious ways to be fake:
  A. Trading costs eat it (weekly rebalance, not-huge-cap names).
  B. "Attention" is just last week's PRICE MOVE repackaged (a stock that popped
     gets looked up), i.e. the signal is short-term momentum, not new info.
Plus the usual: is it just the 2020-21 meme era? one or two names? illiquid junk?
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import statsmodels.api as sm
import test_signal as TS

PPY = 52
def sharpe(s): return s.mean()*PPY / (s.std()*np.sqrt(PPY)) if s.std() > 0 else np.nan


def basket_net(panel, cost_bps, n_legs=5, yvar="ret_fwd1"):
    """Long top-n_legs ASVI, short bottom-n_legs, equal weight. Charge cost on
    week-over-week turnover of the (long-short) weight vector."""
    weeks = sorted(panel["week"].unique())
    w_prev = pd.Series(dtype=float)
    rows = []
    for wk in weeks:
        g = panel[(panel["week"] == wk)].dropna(subset=[yvar, "asvi"])
        if len(g) < 2 * n_legs:
            continue
        gs = g.sort_values("asvi")
        w = pd.Series(0.0, index=g["ticker"].values)
        longs = gs.iloc[-n_legs:]["ticker"].values
        shorts = gs.iloc[:n_legs]["ticker"].values
        w[longs] = 1.0 / n_legs
        w[shorts] = -1.0 / n_legs
        gross = float((w.reindex(g["ticker"]).values * g[yvar].values).sum())
        allk = w.index.union(w_prev.index)
        turn = (w.reindex(allk).fillna(0) - w_prev.reindex(allk).fillna(0)).abs().sum()
        rows.append({"week": wk, "ret": gross - turn * cost_bps / 1e4})
        w_prev = w
    return pd.DataFrame(rows).set_index("week")["ret"].dropna()


def build_fwd_returns(panel):
    return panel.pivot_table(index="week", columns="ticker", values="ret")


def hold_h_basket(panel, W, h=1, n_legs=5, cost_bps=10):
    """Form the long/short book every h weeks from ASVI, hold h weeks, charge
    cost only on rebalance weeks. Returns a WEEKLY long-short series (earning
    week = formation week + 1, so strictly no look-ahead)."""
    weeks = sorted(panel["week"].unique())
    asvi_w = panel.pivot_table(index="week", columns="ticker", values="asvi")
    cur = pd.Series(0.0, index=W.columns)
    rows = []
    for i in range(len(weeks) - 1):
        wk, nxt = weeks[i], weeks[i + 1]
        cost = 0.0
        if i % h == 0:
            a = asvi_w.loc[wk].dropna()
            new = pd.Series(0.0, index=W.columns)
            if len(a) >= 2 * n_legs:
                a = a.sort_values()
                new[a.index[-n_legs:]] = 1.0 / n_legs
                new[a.index[:n_legs]] = -1.0 / n_legs
            cost = (new - cur).abs().sum() * cost_bps / 1e4
            cur = new
        r = W.loc[nxt].fillna(0.0)
        rows.append({"week": nxt, "ret": float((cur * r).sum()) - cost})
    return pd.DataFrame(rows).set_index("week")["ret"].dropna()


def main():
    panel = TS.build_panel()
    print("="*82)
    print(f"  ATTENTION TORTURE TESTS  ({panel['week'].min().date()} -> {panel['week'].max().date()}, "
          f"{panel['ticker'].nunique()} tickers)")
    print("="*82)

    # ---- A. transaction costs (the big one) ----
    print("\n[A] TRANSACTION COSTS (weekly rebalance turnover)")
    for bps in [0, 5, 10, 20, 35, 50]:
        s = basket_net(panel, bps)
        print(f"  {bps:>3} bps/side: ann={s.mean()*PPY:+.2%}  Sharpe={sharpe(s):+.2f}")

    # ---- B. momentum confound: control for own recent return ----
    print("\n[B] IS IT JUST PRICE MOMENTUM? (control ret_{t+1} for own return_t)")
    panel["ret_z"] = panel.groupby("week")["ret"].transform(
        lambda x: (x - x.mean())/x.std() if x.std() > 0 else x*0)
    d = panel.dropna(subset=["ret_fwd1", "asvi_z", "ret_z"])
    for xs, lab in [(["asvi_z"], "asvi alone"),
                    (["ret_z"], "own return_t alone"),
                    (["asvi_z", "ret_z"], "asvi + own return_t")]:
        m = sm.OLS(d["ret_fwd1"], sm.add_constant(d[xs])).fit(
            cov_type="cluster", cov_kwds={"groups": d["week"]})
        parts = "  ".join(f"{v}: coef={m.params[v]:+.5f} t={m.tvalues[v]:+.2f}" for v in xs)
        print(f"  {lab:<22} {parts}")

    # also: does asvi survive ORTHOGONALIZING to own return? (residual attention)
    res = sm.OLS(d["asvi_z"], sm.add_constant(d[["ret_z"]])).fit().resid
    d2 = d.assign(asvi_resid=res)
    m = sm.OLS(d2["ret_fwd1"], sm.add_constant(d2[["asvi_resid"]])).fit(
        cov_type="cluster", cov_kwds={"groups": d2["week"]})
    print(f"  attention ORTHOGONAL to own return: coef={m.params['asvi_resid']:+.5f} "
          f"t={m.tvalues['asvi_resid']:+.2f} p={m.pvalues['asvi_resid']:.3f}")

    # ---- C. sub-period stability (is it just 2020-21 meme era?) ----
    print("\n[C] SUB-PERIOD STABILITY (10 bps/side net)")
    s = basket_net(panel, 10)
    for lab, a, b in [("2015-2019", "2015", "2019"), ("2020-2021", "2020", "2021"),
                      ("2022-2024", "2022", "2024")]:
        seg = s.loc[a:b]
        print(f"  {lab}: {len(seg)} wk  ann={seg.mean()*PPY:+.2%}  Sharpe={sharpe(seg):+.2f}")

    # ---- D. drop-one-ticker (net 10 bps) ----
    print("\n[D] DROP-ONE-TICKER (net 10 bps Sharpe; is it one name?)")
    base = sharpe(basket_net(panel, 10))
    res_d = []
    for tk in sorted(panel["ticker"].unique()):
        s2 = basket_net(panel[panel["ticker"] != tk], 10)
        res_d.append((tk, sharpe(s2)))
    for tk, sh in sorted(res_d, key=lambda x: x[1])[:5]:
        print(f"  worst drop: without {tk:<5} Sharpe={sh:+.2f} ({sh-base:+.2f})")
    print(f"  full-universe net Sharpe = {base:+.2f}")

    # ---- E. large-cap-only (liquidity: drop small/illiquid names) ----
    print("\n[E] LARGE-CAP-ONLY SUBSET (drop small/illiquid; costs are real there)")
    big = ["HD","LOW","NKE","SBUX","MCD","TGT","BBY","ULTA","CMG","LULU","DPZ"]
    sub = panel[panel["ticker"].isin(big)]
    for bps in [0, 10, 20]:
        s2 = basket_net(sub, bps)
        print(f"  big-caps only, {bps:>2} bps: ann={s2.mean()*PPY:+.2%}  Sharpe={sharpe(s2):+.2f}  "
              f"({sub['ticker'].nunique()} names)")

    # ---- F. lower turnover: can holding longer beat the cost wall? ----
    print("\n[F] LOWER TURNOVER -- hold the book h weeks, rebalance every h weeks (net 10 bps)")
    fwd = build_fwd_returns(panel)
    for h in [1, 2, 4, 8]:
        s = hold_h_basket(panel, fwd, h=h, cost_bps=10)
        print(f"  hold {h} wk: ann={s.mean()*PPY:+.2%}  Sharpe={sharpe(s):+.2f}  "
              f"(gross {hold_h_basket(panel, fwd, h=h, cost_bps=0).mean()*PPY:+.2%})")


if __name__ == "__main__":
    main()
