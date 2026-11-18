"""Bias mitigation helpers.

* :func:`reweighing` — pre-processing (Kamiran & Calders, 2012): sample
  weights that make the label statistically independent of the attribute.
* :func:`group_thresholds` — post-processing: a decision threshold per group
  chosen so a chosen error rate matches the reference group's (in the spirit
  of Hardt et al., 2016, equalised odds).

Using a protected attribute to set group-specific thresholds can itself be
unlawful in some settings (e.g. U.S. employment and credit decisions). Treat
these tools as diagnostics and talk to counsel before deploying them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import confusion_counts


def reweighing(df: pd.DataFrame, attribute: str, label_col: str) -> np.ndarray:
    """Weights ``P(A=a) P(Y=y) / P(A=a, Y=y)`` for each row."""
    n = len(df)
    pa = df[attribute].map(df[attribute].value_counts() / n)
    py = df[label_col].map(df[label_col].value_counts() / n)
    joint = df.groupby([attribute, label_col], observed=True).size() / n
    pay = np.array([joint[(a, y)] for a, y in zip(df[attribute], df[label_col])])
    return (pa.to_numpy() * py.to_numpy() / pay).astype(float)


def _rate(y, pred, metric):
    c = confusion_counts(y, pred)
    return {"fpr": c["fp"] / c["neg"] if c["neg"] else np.nan,
            "tpr": c["tp"] / c["pos"] if c["pos"] else np.nan,
            "fnr": c["fn"] / c["pos"] if c["pos"] else np.nan,
            "pprev": c["pp"] / c["n"] if c["n"] else np.nan}[metric]


def group_thresholds(scores, labels, groups, metric: str = "fpr", reference: str | None = None,
                     reference_threshold: float = 0.5) -> dict[str, float]:
    """Per-group thresholds that bring ``metric`` close to the reference group's value.

    The reference group keeps ``reference_threshold``; for every other group
    the candidate threshold (among its observed scores) with the closest
    ``metric`` is chosen.
    """
    scores, labels, groups = np.asarray(scores, float), np.asarray(labels, int), np.asarray(groups).astype(str)
    if metric not in {"fpr", "tpr", "fnr", "pprev"}:
        raise ValueError("metric must be fpr, tpr, fnr or pprev")
    uniq, counts = np.unique(groups, return_counts=True)
    reference = reference or uniq[np.argmax(counts)]
    m_ref = reference == groups
    target = _rate(labels[m_ref], scores[m_ref] >= reference_threshold, metric)
    out = {str(reference): reference_threshold}
    for g in uniq:
        if g == reference:
            continue
        m = groups == g
        cands = np.unique(np.concatenate([scores[m], [scores[m].max() + 1e-9]]))
        gaps = [abs(_rate(labels[m], scores[m] >= t, metric) - target) for t in cands]
        out[str(g)] = float(cands[int(np.nanargmin(gaps))])
    return out


def apply_thresholds(scores, groups, thresholds: dict[str, float]) -> np.ndarray:
    scores, groups = np.asarray(scores, float), np.asarray(groups).astype(str)
    t = np.array([thresholds[g] for g in groups])
    return (scores >= t).astype(int)
