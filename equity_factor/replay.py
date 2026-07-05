"""
INTERACTIVE BACKTEST REPLAY — generates replay.html, a self-contained page that
plays the 8-ETF RP+Trend backtest month by month: NAV vs SPY growing in real
time, per-sleeve $P&L each month, allocation history filling in, and running
stats (CAGR / Sharpe / MaxDD to date). Also includes the forward-testing order
calculator: type an account size, get the dollar amount to buy of each ETF
(cash sleeve -> SGOV) for the selected month's target weights.

All numbers come straight from multi_asset.run() — same engine, no
re-implementation. $ P&L per sleeve = weight * prior NAV * asset monthly
return; the residual vs the strategy return is trading cost, shown as its own
row so the table reconciles exactly to the NAV path.
"""
import warnings; warnings.filterwarnings("ignore")
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multi_asset as MA

START_CAPITAL = 100_000.0
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "replay.html")


def build_data():
    bt, w = MA.run()
    px = MA.load_prices()
    mret = px.pct_change()
    rf = MA.get_rf_monthly(bt.index)
    assets = list(MA.ASSETS)

    nav, spy = START_CAPITAL, START_CAPITAL
    navs, spys, contribs, cashc, costc = [], [], {a: [] for a in assets}, [], []
    for t in bt.index:
        prev = nav
        r = float(bt.loc[t, "RP+Trend"])
        nav *= 1 + r
        spy *= 1 + float(bt.loc[t, "SPY"])
        tot = 0.0
        for a in assets:
            c = float(w.loc[t, a]) * float(mret.loc[t, a]) * prev
            contribs[a].append(round(c, 2)); tot += c
        cc = float(w.loc[t, "CASH"]) * float(rf.loc[t]) * prev
        cashc.append(round(cc, 2))
        costc.append(round(prev * r - tot - cc, 2))   # trading-cost residual
        navs.append(round(nav, 2)); spys.append(round(spy, 2))

    return {
        "assets": assets,
        "names": {a: n for a, n, _ in MA.ASSET_META},
        "dates": [t.strftime("%Y-%m") for t in bt.index],
        "ret": [round(float(x), 6) for x in bt["RP+Trend"]],
        "spyret": [round(float(x), 6) for x in bt["SPY"]],
        "nav": navs, "spynav": spys,
        "w": {a: [round(float(x), 4) for x in w[a].loc[bt.index]] for a in assets},
        "cashw": [round(float(x), 4) for x in w["CASH"].loc[bt.index]],
        "pnl": contribs, "cashpnl": cashc, "costpnl": costc,
        "start": START_CAPITAL,
    }


HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RP+Trend — Backtest Replay</title>
<style>
  :root{
    --surface:#fcfcfb; --page:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e;
    --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7; --ring:rgba(11,11,11,.10);
    --up:#006300; --dn:#d03b3b;
    --c1:#2a78d6; --c2:#1baf7a; --c3:#eda100; --c4:#008300;
    --c5:#4a3aa7; --c6:#e34948; --c7:#e87ba4; --c8:#eb6834;
  }
  @media (prefers-color-scheme: dark){ :root{
    --surface:#1a1a19; --page:#0d0d0d; --ink:#ffffff; --ink2:#c3c2b7;
    --muted:#898781; --grid:#2c2c2a; --axis:#383835; --ring:rgba(255,255,255,.10);
    --up:#0ca30c; --dn:#e66767;
    --c1:#3987e5; --c2:#199e70; --c3:#c98500; --c4:#008300;
    --c5:#9085e9; --c6:#e66767; --c7:#d55181; --c8:#d95926;
  }}
  *{box-sizing:border-box;margin:0}
  body{background:var(--page);color:var(--ink);
       font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;padding:20px}
  h1{font-size:19px;margin-bottom:2px}
  .sub{color:var(--ink2);font-size:13px;margin-bottom:14px}
  .card{background:var(--surface);border:1px solid var(--ring);border-radius:10px;
        padding:14px 16px;margin-bottom:14px}
  .controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
  button,select,input[type=number]{font:inherit;color:var(--ink);background:var(--surface);
        border:1px solid var(--axis);border-radius:7px;padding:6px 14px;cursor:pointer}
  button.primary{background:var(--c1);color:#fff;border-color:var(--c1);font-weight:600;
        min-width:86px}
  input[type=range]{flex:1;min-width:180px;accent-color:var(--c1)}
  .month{font-weight:700;font-size:16px;min-width:74px;font-variant-numeric:tabular-nums}
  .tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(128px,1fr));gap:10px}
  .tile{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:10px 12px}
  .tile .k{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.04em}
  .tile .v{font-size:20px;font-weight:700;margin-top:2px}
  .tile .d{font-size:12px;color:var(--ink2)}
  .pos{color:var(--up)} .neg{color:var(--dn)}
  .charts{display:grid;grid-template-columns:1fr;gap:14px}
  svg{display:block;width:100%}
  .lbl{fill:var(--ink2);font-size:11px}
  table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
  th{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.04em;
     text-align:right;padding:6px 8px;border-bottom:1px solid var(--grid)}
  th:first-child,td:first-child{text-align:left}
  td{padding:6px 8px;text-align:right;border-bottom:1px solid var(--grid);font-size:13px}
  tr:last-child td{border-bottom:none}
  .chip{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:7px;
        vertical-align:baseline}
  .sec{font-size:15px;font-weight:700;margin-bottom:8px}
  .note{color:var(--muted);font-size:12px;margin-top:8px}
  .calcrow{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:10px}
  .tip{position:fixed;pointer-events:none;background:var(--surface);border:1px solid var(--axis);
       border-radius:7px;padding:7px 10px;font-size:12px;display:none;z-index:9;
       box-shadow:0 2px 8px rgba(0,0,0,.18)}
  @media(min-width:960px){ .charts{grid-template-columns:3fr 2fr} }
</style></head><body>
<h1>RP+Trend — month-by-month replay</h1>
<div class="sub">8-ETF universe, engine output from <code>multi_asset.py</code> · $100,000 start ·
press Play and watch the P&amp;L build (or drag the slider)</div>

<div class="card controls">
  <button id="play" class="primary">&#9654; Play</button>
  <button id="step">+1 mo</button>
  <select id="speed"><option value="220">Normal</option><option value="80">Fast</option>
    <option value="30">Very fast</option></select>
  <input type="range" id="slider" min="0" value="0">
  <span class="month" id="mlabel"></span>
</div>

<div class="tiles" id="tiles"></div>
<div style="height:14px"></div>

<div class="charts">
  <div class="card"><div class="sec">Portfolio value — strategy vs 100% SPY</div>
    <svg id="navsvg" height="300"></svg></div>
  <div class="card"><div class="sec">Allocation history (cash in gray)</div>
    <svg id="allocsvg" height="300"></svg></div>
</div>

<div class="card">
  <div class="sec">This month's book — <span id="tmonth"></span></div>
  <table id="booktbl"></table>
  <div class="note">$P&amp;L = target weight × prior portfolio value × that ETF's monthly return.
  The costs row is the residual vs the engine's net return, so the table sums exactly to the
  month's change in portfolio value.</div>
</div>

<div class="card">
  <div class="sec">Forward-test order calculator</div>
  <div class="calcrow">Account size $
    <input type="number" id="acct" value="10000" min="100" step="100" style="width:120px">
    <span class="note" style="margin:0">uses the <b>selected month's</b> target weights —
    drag the slider to the last month for today's live targets. Cash sleeve &rarr; buy SGOV.</span>
  </div>
  <table id="calctbl"></table>
</div>
<div class="tip" id="tip"></div>

<script>
const D = __DATA__;
const N = D.dates.length, A = D.assets;
const COL = a => getComputedStyle(document.documentElement)
                 .getPropertyValue('--c'+(A.indexOf(a)+1)).trim();
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const $ = id => document.getElementById(id);
const fmt$ = x => (x<0?'-$':'$') + Math.abs(x).toLocaleString(undefined,{maximumFractionDigits:0});
const fmt2 = x => (x<0?'-$':'$') + Math.abs(x).toLocaleString(undefined,{minimumFractionDigits:0,maximumFractionDigits:0});
const pct = (x,d=1) => (100*x).toFixed(d) + '%';
let i = N-1, timer = null;

function stats(k){                       // running stats through month k
  const r = D.ret.slice(0,k+1);
  const mean = r.reduce((s,x)=>s+x,0)/r.length;
  const sd = Math.sqrt(r.reduce((s,x)=>s+(x-mean)*(x-mean),0)/(r.length-1||1));
  const sharpe = sd>0 ? mean/sd*Math.sqrt(12) : 0;
  const yrs = r.length/12;
  const cagr = Math.pow(D.nav[k]/D.start, 1/yrs) - 1;
  let peak = D.start, mdd = 0;
  for(let j=0;j<=k;j++){ peak = Math.max(peak, D.nav[j]); mdd = Math.min(mdd, D.nav[j]/peak-1); }
  return {sharpe, cagr, mdd};
}

function tiles(){
  const s = stats(i), prev = i>0 ? D.nav[i-1] : D.start;
  const mo = D.nav[i]-prev, tot = D.nav[i]-D.start;
  const rows = [
    ['Portfolio value', fmt$(D.nav[i]), 'SPY: '+fmt$(D.spynav[i])],
    ['Total P&L', fmt$(tot), pct(tot/D.start,0)+' since start', tot],
    ['This month', fmt$(mo), pct(D.ret[i]), mo],
    ['CAGR to date', pct(s.cagr), (i+1)+' months'],
    ['Sharpe to date', s.sharpe.toFixed(2), 'annualized'],
    ['Max drawdown', pct(s.mdd), 'to date'],
    ['Cash (SGOV)', pct(D.cashw[i],0), 'target this month'],
  ];
  $('tiles').innerHTML = rows.map(([k,v,d,sign])=>
    `<div class="tile"><div class="k">${k}</div>
     <div class="v ${sign===undefined?'':(sign>=0?'pos':'neg')}">${v}</div>
     <div class="d">${d}</div></div>`).join('');
}

function navChart(){
  const svg = $('navsvg'), W = svg.clientWidth, H = 300, L = 52, R = 12, T = 14, B = 26;
  const hi = Math.max(...D.nav, ...D.spynav), lo = Math.min(D.start, ...D.nav, ...D.spynav);
  const x = j => L + (W-L-R) * j/(N-1);
  const y = v => T + (H-T-B) * (1 - (Math.log(v)-Math.log(lo))/(Math.log(hi)-Math.log(lo)));
  let g = '';
  const ticks = [1e5, 2e5, 4e5, 8e5].filter(v=>v>=lo&&v<=hi);
  for(const v of ticks) g += `<line x1="${L}" x2="${W-R}" y1="${y(v)}" y2="${y(v)}"
      stroke="${css('--grid')}"/><text x="${L-6}" y="${y(v)+4}" text-anchor="end"
      class="lbl">$${v/1000}k</text>`;
  for(let yy=2006; yy<=2024; yy+=3){
    const j = D.dates.findIndex(d=>d.startsWith(yy+'-'));
    if(j>=0) g += `<text x="${x(j)}" y="${H-8}" text-anchor="middle" class="lbl">${yy}</text>`;
  }
  const path = (arr,k) => arr.slice(0,k+1).map((v,j)=>(j?'L':'M')+x(j).toFixed(1)+' '+y(v).toFixed(1)).join('');
  g += `<path d="${path(D.spynav,i)}" fill="none" stroke="${css('--muted')}" stroke-width="2" opacity=".75"/>`;
  g += `<path d="${path(D.nav,i)}" fill="none" stroke="${css('--c1')}" stroke-width="2.5"/>`;
  g += `<circle cx="${x(i)}" cy="${y(D.nav[i])}" r="4" fill="${css('--c1')}"/>`;
  g += `<text x="${x(i)-6}" y="${y(D.nav[i])-9}" text-anchor="end" fill="${css('--c1')}"
        font-size="12" font-weight="700">RP+Trend</text>`;
  g += `<text x="${x(i)-6}" y="${y(D.spynav[i])+16}" text-anchor="end" fill="${css('--muted')}"
        font-size="12">SPY</text>`;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.innerHTML = g;
  svg.onmousemove = e => {
    const r = svg.getBoundingClientRect();
    const j = Math.max(0, Math.min(i, Math.round((e.clientX-r.left-L)/(W-L-R)*(N-1))));
    const t = $('tip');
    t.style.display='block'; t.style.left=(e.clientX+14)+'px'; t.style.top=(e.clientY+10)+'px';
    t.innerHTML = `<b>${D.dates[j]}</b><br>RP+Trend ${fmt$(D.nav[j])}<br>SPY ${fmt$(D.spynav[j])}`;
  };
  svg.onmouseleave = () => $('tip').style.display='none';
}

function allocChart(){
  const svg = $('allocsvg'), W = svg.clientWidth, H = 300, L = 40, R = 8, T = 14, B = 26;
  const bw = (W-L-R)/N;
  let g = '';
  for(const f of [0,.5,1]) g += `<text x="${L-6}" y="${T+(H-T-B)*(1-f)+4}" text-anchor="end"
      class="lbl">${f*100}%</text>`;
  for(let j=0;j<=i;j++){
    let acc = 0;
    for(const a of A){
      const w = D.w[a][j]; if(w<=0) continue;
      const h = w*(H-T-B);
      g += `<rect x="${(L+j*bw).toFixed(2)}" y="${(T+(H-T-B)*(1-acc-w)).toFixed(2)}"
            width="${Math.max(bw,0.8).toFixed(2)}" height="${h.toFixed(2)}" fill="${COL(a)}"/>`;
      acc += w;
    }
    const cw = D.cashw[j];
    if(cw>0) g += `<rect x="${(L+j*bw).toFixed(2)}" y="${T}" width="${Math.max(bw,0.8).toFixed(2)}"
          height="${(cw*(H-T-B)).toFixed(2)}" fill="${css('--muted')}" opacity=".45"/>`;
  }
  for(let yy=2006; yy<=2024; yy+=3){
    const j = D.dates.findIndex(d=>d.startsWith(yy+'-'));
    if(j>=0) g += `<text x="${L+j*bw}" y="${H-8}" text-anchor="middle" class="lbl">${yy}</text>`;
  }
  g += `<line x1="${L+i*bw+bw/2}" x2="${L+i*bw+bw/2}" y1="${T}" y2="${H-B}"
        stroke="${css('--ink')}" stroke-width="1" opacity=".6"/>`;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.innerHTML = g;
}

function book(){
  $('tmonth').textContent = D.dates[i];
  let cum = {}, cashCum = 0, costCum = 0;
  for(const a of A){ cum[a]=0; for(let j=0;j<=i;j++) cum[a]+=D.pnl[a][j]; }
  for(let j=0;j<=i;j++){ cashCum+=D.cashpnl[j]; costCum+=D.costpnl[j]; }
  const cls = x => x>0?'pos':(x<0?'neg':'');
  let h = `<tr><th>Asset</th><th>Target wt</th><th>$P&L this month</th><th>Cumulative $P&L</th></tr>`;
  for(const a of A){
    const w = D.w[a][i], p = D.pnl[a][i];
    h += `<tr><td><span class="chip" style="background:${COL(a)}"></span><b>${a}</b>
          <span style="color:var(--muted)"> ${D.names[a]}</span></td>
          <td>${w>0?pct(w):'<span style="color:var(--muted)">cash</span>'}</td>
          <td class="${cls(p)}">${fmt2(p)}</td><td class="${cls(cum[a])}">${fmt2(cum[a])}</td></tr>`;
  }
  h += `<tr><td><span class="chip" style="background:var(--muted)"></span><b>Cash</b>
        <span style="color:var(--muted)"> T-bills / SGOV</span></td>
        <td>${pct(D.cashw[i])}</td><td class="${cls(D.cashpnl[i])}">${fmt2(D.cashpnl[i])}</td>
        <td class="${cls(cashCum)}">${fmt2(cashCum)}</td></tr>`;
  h += `<tr><td style="color:var(--muted)">Trading costs</td><td></td>
        <td class="${cls(D.costpnl[i])}">${fmt2(D.costpnl[i])}</td>
        <td class="${cls(costCum)}">${fmt2(costCum)}</td></tr>`;
  $('booktbl').innerHTML = h;
}

function calc(){
  const v = Math.max(0, +$('acct').value || 0);
  let h = `<tr><th>Ticker</th><th>Target wt</th><th>Buy ($)</th></tr>`;
  for(const a of A){
    const w = D.w[a][i];
    h += `<tr><td><span class="chip" style="background:${COL(a)}"></span><b>${a}</b></td>
          <td>${w>0?pct(w):'<span style="color:var(--muted)">—</span>'}</td>
          <td>${w>0?fmt2(w*v):'<span style="color:var(--muted)">—</span>'}</td></tr>`;
  }
  h += `<tr><td><b>SGOV</b> <span style="color:var(--muted)">cash sleeve</span></td>
        <td>${pct(D.cashw[i])}</td><td>${fmt2(D.cashw[i]*v)}</td></tr>`;
  $('calctbl').innerHTML = h;
}

function render(){
  $('slider').value = i;
  $('mlabel').textContent = D.dates[i];
  tiles(); navChart(); allocChart(); book(); calc();
}
function tick(){ if(i>=N-1){ pause(); return; } i++; render(); }
function play(){ if(timer) return; if(i>=N-1) i=0;
  timer = setInterval(tick, +$('speed').value);
  $('play').innerHTML = '&#10073;&#10073; Pause'; }
function pause(){ clearInterval(timer); timer=null; $('play').innerHTML='&#9654; Play'; }

$('play').onclick = () => timer ? pause() : play();
$('step').onclick = () => { pause(); if(i<N-1){ i++; render(); } };
$('slider').max = N-1;
$('slider').oninput = e => { pause(); i = +e.target.value; render(); };
$('speed').onchange = () => { if(timer){ pause(); play(); } };
$('acct').oninput = calc;
window.onresize = render;
render();
</script></body></html>
"""


def main():
    data = build_data()
    html = HTML.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    with open(OUT, "w") as f:
        f.write(html)
    print(f"wrote {OUT}  ({len(data['dates'])} months, "
          f"final NAV ${data['nav'][-1]:,.0f} vs SPY ${data['spynav'][-1]:,.0f})")


if __name__ == "__main__":
    main()
