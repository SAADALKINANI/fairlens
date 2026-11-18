"""Bias audit of the COMPAS recidivism risk score (ProPublica data).

COMPAS assigns defendants a risk decile from 1 to 10. Following ProPublica's
"Machine Bias" analysis, deciles 5-10 ("Medium" or "High") are treated as
a *high-risk flag*, and the outcome is whether the person was re-arrested
within two years.

Run:  python examples/compas_audit.py     (downloads ~2.5 MB on first run)
Writes output/compas_report.html and output/compas_fpr.png.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from fairlens import Audit, apply_thresholds, group_thresholds, load_compas  # noqa: E402

pd.set_option("display.width", 160)
out = Path("output")
out.mkdir(exist_ok=True)

df = load_compas()
print(f"{len(df)} defendants after ProPublica's filters\n")

audit = Audit(df, attributes=["race", "sex", "age_cat"],
              reference={"race": "Caucasian", "sex": "Male", "age_cat": "25 - 45"},
              tau=0.8, min_group_size=50).run()

print("=== Fairness verdicts (τ = 0.8) ===")
print(audit.summary().to_string(index=False))

race = audit.disparities_.query("attribute == 'race' and n >= 50")
print("\n=== Error rates by race ===")
print(race[["group", "n", "pprev", "fpr", "fnr", "precision", "fpr_pvalue"]].round(3).to_string(index=False))

aa = race.set_index("group").loc["African-American"]
wh = race.set_index("group").loc["Caucasian"]
print(f"\nBlack defendants who did NOT re-offend were flagged high-risk {aa.fpr:.0%} of the time "
      f"vs {wh.fpr:.0%} for white defendants ({aa.fpr_disparity:.2f}x, p={aa.fpr_pvalue:.1e}).")
print(f"White defendants who DID re-offend were labelled low-risk {wh.fnr:.0%} of the time "
      f"vs {aa.fnr:.0%} for Black defendants.")
print(f"Yet precision is similar ({aa.precision:.0%} vs {wh.precision:.0%}): the score is 'calibrated' "
      "in Northpointe's sense while failing error-rate parity. With different base rates, both cannot hold at once.")

fig, ax = plt.subplots(figsize=(7, 4.5))
audit.plot_disparities("fpr", ax=ax)
fig.tight_layout()
fig.savefig(out / "compas_fpr.png", dpi=120)

# ---- What would it take to equalise false positive rates? (diagnostic only)
sub = df[df.race.isin(["African-American", "Caucasian", "Hispanic", "Other"])].copy()
th = group_thresholds(sub.decile_score, sub.label_value, sub.race, metric="fpr",
                      reference="Caucasian", reference_threshold=5)
sub["score_eq"] = apply_thresholds(sub.decile_score, sub.race, th)
after = Audit(sub, ["race"], score_col="score_eq", reference={"race": "Caucasian"}).run()
print("\n=== Diagnostic: group-specific decile cut-offs that equalise FPR ===")
print({k: round(v, 1) for k, v in th.items()})
print(after.table(("fpr", "fnr", "precision")).round(3).to_string(index=False))

audit.to_html(out / "compas_report.html", title="COMPAS bias audit",
              notes="Positive prediction = COMPAS decile ≥ 5. Label = re-arrest within two years. "
                    "Reference groups: Caucasian, Male, 25–45. Groups under 50 people get no verdict.")
print(f"\nReport → {out / 'compas_report.html'}")
