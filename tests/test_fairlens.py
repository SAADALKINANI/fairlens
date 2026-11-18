import numpy as np
import pandas as pd
import pytest

from fairlens import (Audit, apply_thresholds, binarize, confusion_counts, group_metrics, group_thresholds,
                      make_lending_data, reweighing)
from fairlens.cli import main as cli


def toy():
    # group a: 4 people, group b: 4 people
    return pd.DataFrame({
        "score":       [1, 1, 0, 0,  1, 1, 1, 0],
        "label_value": [1, 0, 0, 1,  1, 0, 0, 0],
        "g":           ["a"] * 4 + ["b"] * 4,
    })


def test_confusion_and_metrics_by_hand():
    c = confusion_counts([1, 0, 0, 1], [1, 1, 0, 0])
    assert (c["tp"], c["fp"], c["tn"], c["fn"]) == (1, 1, 1, 1)
    gm = group_metrics(toy(), ["g"]).set_index("group")
    assert gm.loc["a", "fpr"] == 0.5 and gm.loc["b", "fpr"] == pytest.approx(2 / 3)
    assert gm.loc["a", "fnr"] == 0.5 and gm.loc["b", "fnr"] == 0.0
    assert gm.loc["b", "ppr"] == pytest.approx(3 / 5)


def test_binarize():
    assert binarize([0.2, 0.7, 0.5], threshold=0.5).tolist() == [0, 1, 1]
    assert binarize([0.2, 0.7, 0.5], top_k=1).tolist() == [0, 1, 0]
    with pytest.raises(ValueError):
        binarize([0.2, 0.7])


def test_disparity_and_verdict():
    a = Audit(toy(), ["g"], reference={"g": "a"}, min_group_size=1).run()
    d = a.disparities_.set_index("group")
    assert d.loc["b", "fpr_disparity"] == pytest.approx((2 / 3) / 0.5)
    s = a.summary().set_index("metric")
    assert s.loc["fpr", "verdict"] == "UNFAIR" and s.loc["fpr", "worst_group"] == "b"


def test_unbiased_vs_biased_lending():
    fair = Audit(make_lending_data(bias=0.0, n=20000), ["group"], threshold=0.5).run().summary()
    biased = Audit(make_lending_data(bias=1.2, n=20000), ["group"], threshold=0.5).run().summary()
    f, b = fair.set_index("metric"), biased.set_index("metric")
    assert b.loc["fnr", "worst_disparity"] > f.loc["fnr", "worst_disparity"]
    assert b.loc["fnr", "verdict"] == "UNFAIR"


def test_small_groups_get_no_verdict():
    df = make_lending_data(n=2000)
    df.loc[:5, "group"] = "tiny"
    a = Audit(df, ["group"], threshold=0.5, min_group_size=30).run()
    assert a.fairness_.set_index("group").loc["tiny"].filter(like="_fair").isna().all()


def test_reweighing_makes_label_independent():
    df = make_lending_data(n=5000, seed=1)
    w = reweighing(df, "group", "label_value")
    rates = {g: np.average(df.label_value[df.group == g], weights=w[df.group == g]) for g in ["A", "B"]}
    assert rates["A"] == pytest.approx(rates["B"], abs=1e-9)


def test_group_thresholds_reduce_gap():
    df = make_lending_data(bias=1.2, n=20000, seed=2)
    before = Audit(df, ["group"], threshold=0.5).run().disparities_.set_index("group").loc["B", "fnr_disparity"]
    th = group_thresholds(df.score, df.label_value, df.group, metric="fnr")
    df["score_adj"] = apply_thresholds(df.score, df.group, th)
    after = Audit(df, ["group"], score_col="score_adj").run().disparities_.set_index("group").loc["B", "fnr_disparity"]
    assert abs(after - 1) < abs(before - 1) and abs(after - 1) < 0.1


def test_html_and_cli(tmp_path, capsys):
    df = make_lending_data(n=3000)
    path = tmp_path / "d.csv"
    df.to_csv(path, index=False)
    out = tmp_path / "r.html"
    cli(["audit", str(path), "--attrs", "group", "sex", "--threshold", "0.5", "--ref", "group=A", "--out", str(out)])
    html = out.read_text()
    assert "Verdicts" in html and "data:image/png;base64" in html
    assert "UNFAIR" in capsys.readouterr().out
