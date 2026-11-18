"""Datasets: ProPublica COMPAS (downloaded on demand) and a synthetic lending set."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

COMPAS_URL = ("https://raw.githubusercontent.com/propublica/compas-analysis/master/"
              "compas-scores-two-years.csv")


def _cache_dir() -> Path:
    d = Path(os.environ.get("FAIRLENS_CACHE", Path.home() / ".cache" / "fairlens"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_compas(path: str | Path | None = None, apply_propublica_filters: bool = True) -> pd.DataFrame:
    """ProPublica's COMPAS two-year recidivism data (Broward County, FL, 2013-14).

    Downloads the CSV from ProPublica's public GitHub repo on first use and
    caches it (set ``FAIRLENS_CACHE`` to change where). With
    ``apply_propublica_filters`` the same row filters ProPublica used in
    "Machine Bias" (2016) are applied, giving 6,172 rows.

    Returns columns ``score`` (1 if COMPAS decile ≥ 5, i.e. "Medium/High"),
    ``decile_score``, ``label_value`` (re-offended within two years), ``race``,
    ``sex``, ``age_cat``.
    """
    if path is None:
        path = _cache_dir() / "compas-scores-two-years.csv"
        if not path.exists():
            urllib.request.urlretrieve(COMPAS_URL, path)  # noqa: S310 (fixed https URL)
    df = pd.read_csv(path)
    if apply_propublica_filters:
        df = df[(df["days_b_screening_arrest"] <= 30) & (df["days_b_screening_arrest"] >= -30)
                & (df["is_recid"] != -1) & (df["c_charge_degree"] != "O") & (df["score_text"] != "N/A")]
    out = pd.DataFrame({
        "id": df["id"],
        "decile_score": df["decile_score"],
        "score": (df["decile_score"] >= 5).astype(int),
        "label_value": df["two_year_recid"].astype(int),
        "race": df["race"],
        "sex": df["sex"],
        "age_cat": df["age_cat"],
    })
    return out.reset_index(drop=True)


def make_lending_data(n: int = 5000, bias: float = 0.6, seed: int = 0) -> pd.DataFrame:
    """Synthetic loan-approval data with a controllable amount of model bias.

    True repayment depends on income and credit history only. The "model"
    score adds a penalty for group B scaled by ``bias`` (0 = unbiased).
    ``label_value`` = 1 means the applicant actually repaid; ``score`` is the
    model's probability of repayment.
    """
    rng = np.random.default_rng(seed)
    group = rng.choice(["A", "B"], size=n, p=[0.7, 0.3])
    sex = rng.choice(["F", "M"], size=n)
    income = rng.normal(0, 1, n) - 0.3 * (group == "B")
    history = rng.normal(0, 1, n)
    logit = 0.8 + 1.1 * income + 0.9 * history
    repaid = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    model_logit = 0.8 + 1.0 * income + 0.8 * history - bias * (group == "B") + rng.normal(0, 0.4, n)
    return pd.DataFrame({"score": 1 / (1 + np.exp(-model_logit)), "label_value": repaid,
                         "group": group, "sex": sex})
