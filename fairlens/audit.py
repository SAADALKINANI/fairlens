"""``Audit``: the one-call bias audit."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .disparity import FAIRNESS_DEFINITIONS, choose_reference_groups, disparities, fairness
from .metrics import METRICS, binarize, group_metrics


@dataclass
class Audit:
    """Bias audit of binary decisions across one or more protected attributes.

    Args:
        df: one row per person with a score/decision, a true label and attributes.
        attributes: protected-attribute columns, e.g. ``["race", "sex"]``.
        score_col / label_col: decision (or score) and ground-truth columns.
        threshold / top_k: how to binarise a continuous score.
        reference: ``"majority"``, ``"min_metric"`` or ``{attribute: group}``.
        tau: fairness tolerance; a disparity in ``[tau, 1/tau]`` counts as fair.
        min_group_size: smaller groups are reported but excluded from verdicts.

    >>> audit = Audit(df, ["race"], reference={"race": "Caucasian"}).run()
    >>> audit.summary()
    """

    df: pd.DataFrame
    attributes: list[str]
    score_col: str = "score"
    label_col: str = "label_value"
    threshold: float | None = None
    top_k: int | None = None
    reference: str | dict = "majority"
    reference_metric: str | None = None
    tau: float = 0.8
    alpha: float = 0.05
    min_group_size: int = 30
    fairness_metrics: tuple[str, ...] = ("pprev", "fpr", "fnr", "precision", "fdr", "for")
    group_metrics_: pd.DataFrame | None = field(default=None, init=False)
    disparities_: pd.DataFrame | None = field(default=None, init=False)
    fairness_: pd.DataFrame | None = field(default=None, init=False)

    def run(self) -> "Audit":
        data = self.df.copy()
        data["_pred"] = binarize(data[self.score_col], self.threshold, self.top_k)
        data["_label"] = binarize(data[self.label_col])
        gm = group_metrics(data, self.attributes, label_col="_label", pred_col="_pred")
        refs = choose_reference_groups(gm, self.reference, self.reference_metric)
        n_tests = max(1, int((gm.groupby("attribute").size() - 1).sum()))
        self.group_metrics_ = gm
        self.disparities_ = disparities(gm, refs, alpha=self.alpha, n_tests=n_tests)
        self.fairness_ = fairness(self.disparities_, self.tau, self.fairness_metrics, self.min_group_size)
        return self

    # ----------------------------------------------------------- reporting
    def _check(self):
        if self.disparities_ is None:
            raise RuntimeError("Call run() first")

    def summary(self) -> pd.DataFrame:
        """Per attribute × metric: is every (large enough) group within tolerance?"""
        self._check()
        rows = []
        for attr, sub in self.fairness_.groupby("attribute", sort=False):
            d = self.disparities_[(self.disparities_["attribute"] == attr)
                                  & (self.disparities_["n"] >= self.min_group_size)]
            for m in self.fairness_metrics:
                vals = sub[f"{m}_fair"].dropna()
                worst = d.loc[d[f"{m}_disparity"].sub(1).abs().idxmax()] if d[f"{m}_disparity"].notna().any() else None
                rows.append({
                    "attribute": attr, "metric": m, "fairness_criterion": FAIRNESS_DEFINITIONS[m][0],
                    "verdict": "n/a" if vals.empty else ("FAIR" if vals.all() else "UNFAIR"),
                    "reference": sub["reference_group"].iloc[0],
                    "worst_group": None if worst is None else worst["group"],
                    "worst_disparity": None if worst is None else round(float(worst[f"{m}_disparity"]), 3),
                })
        return pd.DataFrame(rows)

    def table(self, metrics: tuple[str, ...] | None = None) -> pd.DataFrame:
        """Compact table of group metrics and disparities."""
        self._check()
        metrics = metrics or self.fairness_metrics
        cols = ["attribute", "group", "n"]
        for m in metrics:
            cols += [m, f"{m}_disparity"]
        return self.disparities_[cols]

    def plot_disparities(self, metric: str = "fpr", ax=None):
        import matplotlib.pyplot as plt

        self._check()
        d = self.disparities_[self.disparities_["n"] >= self.min_group_size]
        d = d[d[f"{metric}_disparity"].notna()]
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 0.45 * len(d) + 1.2))
        labels = d["attribute"] + ": " + d["group"]
        vals = d[f"{metric}_disparity"].to_numpy()
        colors = ["#888888" if r else ("#2e8b57" if self.tau <= v <= 1 / self.tau else "#c0392b")
                  for v, r in zip(vals, d["is_reference"])]
        ax.barh(labels, vals, color=colors)
        ax.axvline(1, color="black", lw=0.8)
        for x in (self.tau, 1 / self.tau):
            ax.axvline(x, color="#555555", lw=0.8, ls="--")
        ax.invert_yaxis()
        ax.set_xlabel(f"{metric.upper()} disparity vs. reference (grey)")
        ax.set_title(f"{FAIRNESS_DEFINITIONS.get(metric, (metric,))[0]}  (fair band {self.tau:g}–{1 / self.tau:.2f})")
        return ax

    def to_html(self, path: str, title: str = "Bias audit report", notes: str = "") -> str:
        """Write a self-contained HTML report (tables + embedded charts)."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from jinja2 import Template

        self._check()
        charts = []
        for m in self.fairness_metrics:
            fig, ax = plt.subplots(figsize=(7, 0.45 * len(self.disparities_) + 1.2))
            self.plot_disparities(m, ax=ax)
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=110)
            plt.close(fig)
            charts.append((m, base64.b64encode(buf.getvalue()).decode()))
        fmt = {c: "{:.3f}" for c in self.table().columns if c not in ("attribute", "group", "n")}
        html = Template(_TEMPLATE).render(
            title=title, notes=notes, tau=self.tau, n=len(self.df), attributes=", ".join(self.attributes),
            summary=self.summary().to_html(index=False, classes="t", na_rep="–"),
            table=self.table().style.format(fmt, na_rep="–").hide(axis="index").to_html(),
            charts=charts, defs=FAIRNESS_DEFINITIONS, metric_defs={k: v[2] for k, v in METRICS.items()},
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        return path


_TEMPLATE = """<!doctype html><html><head><meta charset="utf-8"><title>{{title}}</title>
<style>
body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#222}
h1{margin-bottom:.2rem} .meta{color:#666} table{border-collapse:collapse;font-size:.85rem;margin:1rem 0}
td,th{border:1px solid #ddd;padding:4px 8px;text-align:right} th{background:#f4f4f4}
td:first-child,td:nth-child(2),td:nth-child(3){text-align:left}
img{max-width:100%;border:1px solid #eee;margin:.5rem 0} .note{background:#fff8e1;padding:.8rem;border-left:4px solid #f0b400}
</style></head><body>
<h1>{{title}}</h1>
<p class="meta">{{n}} records · attributes: {{attributes}} · fairness tolerance τ = {{tau}} (fair if disparity in [{{tau}}, {{'%.2f'|format(1/tau)}}])</p>
{% if notes %}<p class="note">{{notes}}</p>{% endif %}
<h2>Verdicts</h2>{{summary}}
<h2>Group metrics and disparities</h2>{{table}}
<h2>Disparity charts</h2>
{% for m, img in charts %}<img src="data:image/png;base64,{{img}}" alt="{{m}} disparity">{% endfor %}
<h2>What the criteria mean</h2><ul>
{% for k,(name,desc) in defs.items() %}<li><b>{{name}}</b> ({{k}}): {{desc}}.</li>{% endfor %}</ul>
<p class="meta">Generated by fairlens. Fairness criteria can conflict mathematically when base rates differ between groups; choose the ones that match the harm you care about.</p>
</body></html>"""
