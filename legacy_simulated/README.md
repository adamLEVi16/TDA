# Quarantine: simulated / fabricated code — NOT real validation

Everything in this folder produces **synthetic or fabricated numbers**, or contains
fabricating fallbacks that emit hypothesis-confirming results regardless of input. None
of it is evidence for any claim. It is kept here (rather than deleted) for provenance —
see `git log` for full history. The trustworthy, real-data pipeline lives at the repo
root (`risk_report.py`, `tda_sector_backtest.py`, `strategy_variants_real.py`,
`ml_integration_real.py`) and its reproduced output is in `../HONEST_RESULTS.md`.

| File | Why it is quarantined |
|---|---|
| `generate_all_thesis_figures.py` | **Generates every thesis figure from `np.random`.** The headline relationships are hardcoded into the data generators (e.g. `cv_obs = -1.2*rho + 1.1 + noise`, `gap = 2.5 - 3.0*cv + noise`), then "validated" by the very figure that plots them. All 22 embedded PDFs/PNGs in `../thesis_latex/figures/` come from this. |
| `validation_tests.py` | Test 1 (breakpoint) builds the answer `ρc=0.50` into engineered synthetic data and then "discovers" it. Test 2 (international) silently skips every international market (5 < 10-ticker gate) and uses a rigged `cv = 0.65 - 0.4*mean_corr` fallback. Only Test 3 used real data — salvageable by deleting the non-ripser fallbacks. |
| `null_model_test.py` | Non-ripser fallback fabricates `cv = 0.65 - 0.4*mean_corr`, which guarantees "signal confirmed" on any input, even noise. Salvageable: delete the fallback so it raises when `ripser` is absent, and seed the RNG. |
| `visualizations_3d.py` | Illustrative 3D plots built on synthetic surfaces (`create_rho_cv_surface` hardcodes the ρ–CV slope and a ρc=0.50 plane) and random fallbacks. A presentation aid, not validation. |
| `thesis_expansion/` | Legacy phase code. Per the original README, phase 2/3 are look-ahead biased, phase 4 is explicitly simulated (`PHASE4_SIMULATED.py`), phase 5 is synthetic-data ML, phase 6 is theoretical bounds on simulated correlation matrices. Superseded by the real-data scripts at the repo root. |

**Note:** `../thesis_latex/` still cites `generate_all_thesis_figures.py` by name as its
data source (e.g. `thesis_main.tex:261`, sec09 cross-market). Those figures/tables remain
synthetic; the thesis already carries disclosure banners in sec07/sec08/sec09/sec10
acknowledging this.
