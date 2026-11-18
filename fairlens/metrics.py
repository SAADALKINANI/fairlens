"""Group-level confusion-matrix metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

#: metric name → (numerator count, denominator count, description)
METRICS: dict[str, tuple[str, str, str]] = {
    "pprev": ("pp", "n", "Predicted prevalence (selection rate): share of the group flagged positive"),
    "prev": ("pos", "n", "Prevalence: share of the group whose true label is positive"),
    "tpr": ("tp", "pos", "True positive rate (recall)"),
    "tnr": ("tn", "neg", "True negative rate (specificity)"),
    "fpr": ("fp", "neg", "False positive rate: negatives wrongly flagged"),
    "fnr": ("fn", "pos", "False negative rate: positives missed"),
    "precision": ("tp", "pp", "Precision / positive predictive value"),
    "npv": ("tn", "pn", "Negative predictive value"),
    "fdr": ("fp", "pp", "False discovery rate: flagged people who were actually negative"),
    "for": ("fn", "pn", "False omission rate: unflagged people who were actually positive"),
    "accuracy": ("correct", "n", "Accuracy"),
}
#: ``ppr`` (share of all positive predictions that go to the group) is not a
#: within-group proportion, so it is computed separately.
ALL_METRICS = list(METRICS) + ["ppr"]


def binarize(scores, threshold: float | None = None, top_k: int | None = None) -> np.ndarray:
    """Turn scores into 0/1 decisions.

    * already binary → returned as-is
    * ``threshold`` → ``score >= threshold``
    * ``top_k`` → the ``k`` highest scores are flagged
    """
    s = np.asarray(scores, dtype=float)
    if top_k is not None:
        out = np.zeros(len(s), dtype=int)
        out[np.argsort(-s, kind="stable")[:top_k]] = 1
        return out
    if threshold is not None:
        return (s >= threshold).astype(int)
    if set(np.unique(s)) <= {0.0, 1.0}:
        return s.astype(int)
    raise ValueError("Scores are not binary: pass threshold= or top_k=")


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return {"n": tp + fp + tn + fn, "pp": tp + fp, "pn": tn + fn, "pos": tp + fn, "neg": tn + fp,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn, "correct": tp + tn}


def group_metrics(df: pd.DataFrame, attributes: list[str], label_col: str = "label_value",
                  pred_col: str = "score") -> pd.DataFrame:
    """One row per ``(attribute, group)`` with counts and every metric in ``ALL_METRICS``."""
    missing = [c for c in [label_col, pred_col, *attributes] if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}")
    total_pp = int(df[pred_col].sum())
    rows = []
    for attr in attributes:
        for group, sub in df.groupby(attr, dropna=False, observed=True):
            c = confusion_counts(sub[label_col], sub[pred_col])
            row = {"attribute": attr, "group": str(group), **c}
            for m, (num, den, _) in METRICS.items():
                row[m] = c[num] / c[den] if c[den] else np.nan
            row["ppr"] = c["pp"] / total_pp if total_pp else np.nan
            rows.append(row)
    return pd.DataFrame(rows)
