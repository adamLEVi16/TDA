"""
Universe for the squeeze monitor: the attention-study consumer names PLUS the
known squeeze cohort (still-listed). Committed before evaluation.

HONEST LIMITATION (survivorship): several canonical squeeze names are delisted
(BBBY, EXPR) and Yahoo no longer serves their history, so they cannot be
included. This biases the evaluation TOWARD names that survived; the episode
table therefore leans on GME/AMC (still listed) as the canonical events.
"""
import os, importlib.util

_spec = importlib.util.spec_from_file_location("aa_universe", os.path.join(
    os.path.dirname(__file__), "..", "attention_alpha", "universe.py"))
_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
_CONSUMER = _mod.TICKER_ARTICLE

SQUEEZE_COHORT = {
    "GME":  "GameStop",
    "AMC":  "AMC_Theatres",
    "M":    "Macy's,_Inc.",
    "CVNA": "Carvana",
    "BYND": "Beyond_Meat",
    "KSS":  "Kohl's",
}

TICKER_ARTICLE = {**_CONSUMER, **SQUEEZE_COHORT}
