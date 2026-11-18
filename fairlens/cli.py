"""``fairlens audit data.csv --attrs race sex --ref race=Caucasian --out report.html``"""

from __future__ import annotations

import argparse

import pandas as pd

from . import __version__
from .audit import Audit


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="fairlens", description="Bias audit for binary decisions")
    p.add_argument("--version", action="version", version=f"fairlens {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audit", help="audit a CSV file")
    a.add_argument("csv")
    a.add_argument("--attrs", nargs="+", required=True, help="protected attribute columns")
    a.add_argument("--score", default="score")
    a.add_argument("--label", default="label_value")
    a.add_argument("--threshold", type=float)
    a.add_argument("--top-k", type=int)
    a.add_argument("--ref", nargs="*", default=[], help="reference groups as attr=group")
    a.add_argument("--tau", type=float, default=0.8)
    a.add_argument("--min-group-size", type=int, default=30)
    a.add_argument("--out", help="write an HTML report here")
    args = p.parse_args(argv)

    df = pd.read_csv(args.csv)
    ref = dict(r.split("=", 1) for r in args.ref) if args.ref else "majority"
    audit = Audit(df, args.attrs, score_col=args.score, label_col=args.label, threshold=args.threshold,
                  top_k=args.top_k, reference=ref, tau=args.tau, min_group_size=args.min_group_size).run()
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(audit.summary().to_string(index=False))
    if args.out:
        audit.to_html(args.out)
        print(f"\nreport → {args.out}")


if __name__ == "__main__":
    main()
