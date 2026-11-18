"""Disparities between groups, significance tests and fairness determinations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .metrics import ALL_METRICS, METRICS


def choose_reference_groups(gm: pd.DataFrame, ref: str | dict = "majority",
                            metric: str | None = None) -> dict[str, str]:
    """Pick one reference group per attribute.

    ``ref`` can be ``"majority"`` (largest group), ``"min_metric"`` (group
    with the lowest value of ``metric``, i.e. the best-treated group for an
    error rate), or a dict ``{attribute: group}``.
    """
    out = {}
    for attr, sub in gm.groupby("attribute", sort=False):
        if isinstance(ref, dict) and attr in ref:
            if str(ref[attr]) not in set(sub["group"]):
                raise ValueError(f"Reference group '{ref[attr]}' not found for '{attr}'")
            out[attr] = str(ref[attr])
        elif ref == "min_metric":
            if metric is None:
                raise ValueError("min_metric needs metric=")
            out[attr] = sub.loc[sub[metric].idxmin(), "group"]
        else:
            out[attr] = sub.loc[sub["n"].idxmax(), "group"]
    return out


def _two_proportion_p(x1: int, n1: int, x2: int, n2: int) -> float:
    if min(n1, n2) == 0:
        return np.nan
    table = np.array([[x1, n1 - x1], [x2, n2 - x2]])
    if (table < 0).any():
        return np.nan
    if table.min() < 5:
        return float(stats.fisher_exact(table)[1])
    return float(stats.chi2_contingency(table, correction=False)[1])


def disparities(gm: pd.DataFrame, ref_groups: dict[str, str], alpha: float = 0.05,
                n_tests: int | None = None) -> pd.DataFrame:
    """Add ``<metric>_disparity`` (group ÷ reference) and ``<metric>_pvalue`` columns.

    p-values come from a chi-squared test (Fisher's exact for small cells)
    comparing the group's proportion with the reference group's. With
    ``n_tests`` a Bonferroni-adjusted ``<metric>_significant`` flag is added.
    """
    out = gm.copy()
    out["reference_group"] = out["attribute"].map(ref_groups)
    out["is_reference"] = out["group"] == out["reference_group"]
    for m in ALL_METRICS:
        ref_vals = out[out["is_reference"]].set_index("attribute")[m]
        base = out["attribute"].map(ref_vals)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"{m}_disparity"] = np.where(base > 0, out[m] / base, np.nan)
    ref_rows = out[out["is_reference"]].set_index("attribute")
    for m, (num, den, _) in METRICS.items():
        pv = []
        for _, r in out.iterrows():
            rr = ref_rows.loc[r["attribute"]]
            pv.append(np.nan if r["is_reference"] else _two_proportion_p(r[num], r[den], rr[num], rr[den]))
        out[f"{m}_pvalue"] = pv
        adj = alpha / (n_tests or 1)
        out[f"{m}_significant"] = out[f"{m}_pvalue"] < adj
    return out


def fairness(disp: pd.DataFrame, tau: float = 0.8,
             metrics: tuple[str, ...] = ("pprev", "ppr", "fpr", "fnr", "precision", "fdr", "for"),
             min_group_size: int = 0) -> pd.DataFrame:
    """Flag each ``(group, metric)`` as fair when ``tau <= disparity <= 1/tau``.

    ``tau=0.8`` mirrors the "four-fifths rule" used in U.S. employment
    selection guidance. Undefined disparities (empty denominators) are NaN.
    """
    if not 0 < tau <= 1:
        raise ValueError("tau must be in (0, 1]")
    out = disp[["attribute", "group", "n", "reference_group", "is_reference"]].copy()
    for m in metrics:
        d = disp[f"{m}_disparity"]
        verdict = np.where(d.isna(), np.nan, ((d >= tau) & (d <= 1 / tau)).astype(float))
        out[f"{m}_fair"] = np.where(out["n"] < min_group_size, np.nan, verdict)
    fair_cols = [f"{m}_fair" for m in metrics]
    out["all_fair"] = out[fair_cols].min(axis=1, skipna=True)
    return out


FAIRNESS_DEFINITIONS = {
    "pprev": ("Demographic / statistical parity", "Each group is flagged at a similar rate"),
    "ppr": ("Proportional parity", "Each group's share of all flags is similar"),
    "fpr": ("False positive rate parity", "Innocent / negative people are wrongly flagged at similar rates"),
    "fnr": ("False negative rate parity", "People who need help / are positive are missed at similar rates"),
    "precision": ("Predictive parity", "A flag means the same thing for every group"),
    "fdr": ("False discovery rate parity", "Flags are wrong at similar rates"),
    "for": ("False omission rate parity", "Non-flags are wrong at similar rates"),
}
