"""
FORWARD-TEST TRACKER — the live out-of-sample evidence log.

Usage, once a month (rebalance day):
  1. Append one row to forward_log.csv:  date, IBKR account NAV, SPY close,
     optional note (e.g. "rebalanced: GLD back to HOLD").
  2. python forward_track.py   -> regenerates forward.html

The page indexes both series to 100 at inception and reports the running
stats that matter for the thesis: strategy vs SPY return, realized monthly
beta/vol once enough points exist, and current drawdown. This file is the
artifact that accumulates credibility no backtest can: months of following
a pre-committed rule with real (paper) fills, logged as they happen.

Paper account: IBKR DU5881331, inception 2026-07-06, ~$1.277M.
Strategy spec: multi_asset.py V0 (binary 10m trend, inverse-12m-vol, cash->SGOV).
"""
import csv, json, os
import datetime as dt

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "forward_log.csv")
OUT = os.path.join(HERE, "forward.html")


def load():
    rows = []
    with open(LOG) as f:
        for r in csv.DictReader(f):
            rows.append({"date": r["date"].strip(),
                         "nav": float(r["account_nav"]),
                         "spy": float(r["spy_close"]),
                         "note": (r.get("note") or "").strip()})
    rows.sort(key=lambda r: r["date"])
    return rows


def stats(rows):
    base_n, base_s = rows[0]["nav"], rows[0]["spy"]
    strat = [r["nav"] / base_n for r in rows]
    spy = [r["spy"] / base_s for r in rows]
    out = {"n": len(rows),
           "strat_ret": strat[-1] - 1, "spy_ret": spy[-1] - 1,
           "dd": min(0.0, strat[-1] / max(strat) - 1)}
    if len(rows) >= 4:                       # monthly beta/vol need a few points
        rs = [strat[i] / strat[i - 1] - 1 for i in range(1, len(strat))]
        rm = [spy[i] / spy[i - 1] - 1 for i in range(1, len(spy))]
        n = len(rs)
        mx, my = sum(rs) / n, sum(rm) / n
        varm = sum((y - my) ** 2 for y in rm) / n
        cov = sum((x - mx) * (y - my) for x, y in zip(rs, rm)) / n
        vars = sum((x - mx) ** 2 for x in rs) / n
        out["beta"] = cov / varm if varm > 0 else None
        out["vol"] = (vars * 12) ** 0.5 if vars > 0 else None
    return out


HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RP+Trend — Live Forward Test</title>
<style>
:root{--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);--up:#006300;--dn:#d03b3b;--c1:#2a78d6}
@media (prefers-color-scheme:dark){:root{--surface:#1a1a19;--page:#0d0d0d;--ink:#fff;
--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);
--up:#0ca30c;--dn:#e66767;--c1:#3987e5}}
*{box-sizing:border-box;margin:0}
body{background:var(--page);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;padding:20px}
h1{font-size:19px}.sub{color:var(--ink2);font-size:13px;margin:2px 0 14px}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:14px 16px;margin-bottom:14px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin-bottom:14px}
.tile{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:10px 12px}
.tile .k{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.04em}
.tile .v{font-size:20px;font-weight:700;margin-top:2px}
.pos{color:var(--up)}.neg{color:var(--dn)}
svg{display:block;width:100%}.lbl{fill:var(--ink2);font-size:11px}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th{color:var(--muted);font-size:11px;text-transform:uppercase;text-align:right;padding:6px 8px;border-bottom:1px solid var(--grid)}
th:first-child,td:first-child{text-align:left}
td{padding:6px 8px;text-align:right;border-bottom:1px solid var(--grid);font-size:13px}
.note{color:var(--muted);font-size:12px;margin-top:8px}
</style></head><body>
<h1>RP+Trend — live forward test</h1>
<div class="sub">IBKR paper DU5881331 · inception 2026-07-06 · V0 spec, monthly rebalance ·
log a row each month-end, rerun <code>forward_track.py</code></div>
<div class="tiles" id="tiles"></div>
<div class="card"><b>Indexed to 100 at inception</b><svg id="ch" height="300"></svg></div>
<div class="card"><b>Monthly log</b><table id="tbl"></table>
<div class="note">Beta and vol appear after 4+ logged rows. This log — not the backtest —
is the evidence that accumulates: every row is an out-of-sample month following the
pre-committed rule.</div></div>
<div class="card"><b>What normal looks like (from 214 backtest months / 5,000+ days)</b>
<div class="note" style="font-size:13px;color:var(--ink2)">
44% of all days are red; a typical red day is −0.28% and a −0.25%-or-worse day happens
~4x per month. 38% of months are red (average −1.1%, worst −3.45%), and the longest
historical run was 6 red months in a row. On days SPY falls, the strategy captures only
~18% of the drop on average — so "down, but much less than SPY" is the strategy working,
not failing. None of these are stop conditions; the rule only acts at month-end.</div></div>
<script>
const D=__DATA__,S=__STATS__;
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const pct=(x,d=1)=>(x>=0?'+':'')+(100*x).toFixed(d)+'%';
const t=[['Months logged',S.n,''],['Strategy',pct(S.strat_ret),S.strat_ret],
['SPY',pct(S.spy_ret),S.spy_ret],['vs SPY',pct(S.strat_ret-S.spy_ret),S.strat_ret-S.spy_ret],
['Drawdown',pct(S.dd),S.dd]];
if(S.beta!=null)t.push(['Beta (live)',S.beta.toFixed(2),'']);
if(S.vol!=null)t.push(['Vol ann (live)',(S.vol*100).toFixed(1)+'%','']);
document.getElementById('tiles').innerHTML=t.map(([k,v,s])=>
`<div class="tile"><div class="k">${k}</div><div class="v ${s===''?'':(s>=0?'pos':'neg')}">${v}</div></div>`).join('');
const b0n=D[0].nav,b0s=D[0].spy,W=document.getElementById('ch').clientWidth,H=300,L=44,R=12,T=14,B=26;
const sv=D.map(r=>100*r.nav/b0n),sp=D.map(r=>100*r.spy/b0s);
const lo=Math.min(...sv,...sp,98),hi=Math.max(...sv,...sp,102);
const x=i=>L+(W-L-R)*(D.length>1?i/(D.length-1):0.5),y=v=>T+(H-T-B)*(1-(v-lo)/(hi-lo));
let g='';for(const v of [100,Math.round(hi)]) if(v>lo&&v<hi)
g+=`<line x1="${L}" x2="${W-R}" y1="${y(v)}" y2="${y(v)}" stroke="${css('--grid')}"/>
<text x="${L-6}" y="${y(v)+4}" text-anchor="end" class="lbl">${v}</text>`;
const path=a=>a.map((v,i)=>(i?'L':'M')+x(i).toFixed(1)+' '+y(v).toFixed(1)).join('');
g+=`<path d="${path(sp)}" fill="none" stroke="${css('--muted')}" stroke-width="2" opacity=".75"/>`;
g+=`<path d="${path(sv)}" fill="none" stroke="${css('--c1')}" stroke-width="2.5"/>`;
sv.forEach((v,i)=>{g+=`<circle cx="${x(i)}" cy="${y(v)}" r="4" fill="${css('--c1')}"/>
<circle cx="${x(i)}" cy="${y(sp[i])}" r="3.5" fill="${css('--muted')}"/>`;});
g+=`<text x="${x(sv.length-1)-8}" y="${y(sv[sv.length-1])-9}" text-anchor="end" fill="${css('--c1')}" font-size="12" font-weight="700">Strategy</text>`;
g+=`<text x="${x(sp.length-1)-8}" y="${y(sp[sp.length-1])+16}" text-anchor="end" fill="${css('--muted')}" font-size="12">SPY</text>`;
D.forEach((r,i)=>{g+=`<text x="${x(i)}" y="${H-8}" text-anchor="middle" class="lbl">${r.date.slice(2)}</text>`;});
const ch=document.getElementById('ch');ch.setAttribute('viewBox',`0 0 ${W} ${H}`);ch.innerHTML=g;
document.getElementById('tbl').innerHTML=
'<tr><th>Date</th><th>Account NAV</th><th>SPY</th><th>Strategy idx</th><th>SPY idx</th><th>Note</th></tr>'+
D.map((r,i)=>`<tr><td>${r.date}</td><td>$${r.nav.toLocaleString()}</td><td>${r.spy}</td>
<td>${(100*r.nav/b0n).toFixed(2)}</td><td>${(100*r.spy/b0s).toFixed(2)}</td>
<td style="text-align:left;color:var(--muted)">${r.note}</td></tr>`).join('');
</script></body></html>
"""


def main():
    rows = load()
    s = stats(rows)
    html = (HTML.replace("__DATA__", json.dumps(rows))
                .replace("__STATS__", json.dumps(s)))
    with open(OUT, "w") as f:
        f.write(html)
    print(f"wrote forward.html — {s['n']} rows, strategy {s['strat_ret']:+.2%} "
          f"vs SPY {s['spy_ret']:+.2%} since inception")


if __name__ == "__main__":
    main()
