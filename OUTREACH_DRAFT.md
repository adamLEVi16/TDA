# Outreach draft — for Alan

Not sent. Edit freely — this is a starting point in your voice, not a final copy.
Three things exist; send in this order, spaced out rather than all at once.

---

## Message 1 — the opener (send with the squeeze monitor demo)

> Hey Alan — built something I think is actually useful for your book, not just
> a school project. It's a free early-warning screen for crowded shorts about
> to squeeze — combines short-interest data with a spike in public search/attention
> on a name. Backtested on your side of the market (retail/consumer names):
> when it flags, a stock is roughly 2x more likely to rip 25%+ in the next month.
> It caught CVNA and ETSY well before their squeezes. Take a look when you have
> five minutes — [demo.html link]. Would love your read on whether this is
> useful or if I'm missing something obvious from being on the outside.

**Why this framing:** leads with his P&L, not your methodology. "Take a look
when you have five minutes" — low pressure, no ask yet. "Would love your read"
invites him to engage as a mentor, which is the actual goal of this message,
not the pitch itself.

---

## Message 2 — if he responds positively (send the tearsheet)

> Glad it was useful. Since you asked how I built it — I also put together a
> longer systematic strategy as more of a research exercise (proving out risk
> management for a diversified book, not something for your world specifically).
> 37 years of data, real stats, nothing dressed up — [tearsheet.html link] if
> you're curious. Mostly wanted to show you I take the rigor seriously, not
> just chase backtests that look good.

**Why this framing:** explicitly scoped as "not for your world" — preempts the
mismatch objection before he raises it. Positions it as evidence of how you
work, not a second pitch.

---

## Message 3 — the actual ask (only after he's engaged with both)

> One thing I keep coming back to — everything I built here is on free data.
> The real version of this idea uses the kind of stuff your desk already pays
> for (transaction panels, foot traffic, whatever you're reading). I'd love to
> learn how you and the team actually use that data day to day — even just
> watching how you read it would teach me more than anything else I could do
> right now. And if there's ever an opening to help out, formally or not,
> I'm in.

**Why this framing:** this is the actual goal — a foot in the door — but it's
phrased as wanting to *learn*, not asking for a job outright. "Formally or
not" leaves room for an internship, a call, a coffee, whatever he's able to
offer. No pressure on him to have an opening; the ask is for time and access,
which costs him little and shows your seriousness.

---

## Sequencing notes

- **Don't send all three at once.** Let each one land and get a real reaction
  before sending the next — this is a conversation, not a packet.
- **If he forwards you to a technical colleague:** send the RESEARCH.md /
  GitHub link at that point, not before. That's the one built for a peer
  reviewer, not for Alan himself.
- **If he pushes back or seems unimpressed:** don't argue the stats. Ask what
  would actually be useful to him — that question alone demonstrates more
  maturity than any backtest.
- **Timing:** you said yourself this shouldn't be rushed. There's no reason
  Message 1 can't go out this week — it's low-risk, genuinely useful, and asks
  nothing of him. Messages 2 and 3 should wait for his actual response, not a
  fixed calendar.

---

## PM Q&A crib sheet — the five questions a professional will actually ask

(A reviewer correctly predicted the conversation won't be about the code — it
will be these. Every answer below is backed by a specific documented result;
no improvising needed.)

**"Why 10 months?"** — It isn't a tuned number. Sharpe is flat at 1.15–1.17
across every window from 6 to 14 months, and the certified refinement (V2)
removes the single-window dependence entirely by averaging {3,6,9,12}-month
signals. The parameter genuinely doesn't matter — that's the answer, and it's
the strongest possible one.

**"Why these eight assets?"** — One liquid proxy per major return driver
(US/intl/EM equity, long/mid Treasuries, gold, commodities, REITs), chosen for
economic role, not backtest fit. Evidence it's the mechanism and not the
tickers: the identical engine on a *different* instrument set (5 mutual funds,
1987+) produces the same result, and dropping any single asset keeps Sharpe
at 0.96–1.37 — all above equity's 0.74.

**"Why inverse volatility?"** — It's the risk-shaping layer, not the return
engine, and we proved which is which: the same weights *without* the trend
filter earn Sharpe 0.58; adding trend timing takes it to 1.16. A spanning
regression rejects any static mix of the same assets as an explanation
(p=0.035 in the V2 spec). The timing is the product.

**"Why should this keep working?"** — Trend persistence is documented across a
century of markets; the premium survives because it's behaviorally brutal to
hold: it lags buy-and-hold in 69% of rolling 3-year windows, once for 13
straight years. Institutions judged on 3-year numbers structurally can't
harvest it — which is consistent with the alpha *not* decaying in the decades
after Faber published the rule. Honest counter-hypothesis, stated on the
tearsheet: part of the premium may have been the 1982–2021 falling-rate era.

**"What would break it?"** — Three known failure modes, all quantified: fast
crashes (Oct-1987 type; monthly signals can't dodge a one-week collapse),
V-shaped recoveries (2020 — it re-enters late), and whipsaw markets (66% of
its cash-flights are false alarms costing ~4.5pts each; the edge lives in the
one-third that catch real bears). It is crash *insurance* with a measurable
premium, not a market-timing oracle — and saying exactly that is what keeps
this credible.
