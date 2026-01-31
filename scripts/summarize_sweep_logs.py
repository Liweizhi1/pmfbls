#!/usr/bin/env python3
"""Summarize sweep logs produced by run_graph_sweep_rl5_rl6_iter100.sh.

Extracts Acc@1 mean/std and s/it from each live.log, sorts by Acc@1.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ACC1_RE = re.compile(r"^\*\s*Acc@1\s+([0-9.]+)\b", re.MULTILINE)
DOMAIN_RE = re.compile(r"^ISIC\s+([0-9.]+)\s+\+[-/]\s*([0-9.]+)\b", re.MULTILINE)
SIT_RE = re.compile(r"\(([^\s]+)\s*s\s*/\s*it\)")


def parse_log(log_path: Path):
    text = log_path.read_text(errors="ignore")

    # Prefer domain summary row (has mean +/- std); fall back to '* Acc@1' (mean only).
    domain_match = None
    for m in DOMAIN_RE.finditer(text):
        domain_match = m

    acc1_match = None
    for m in ACC1_RE.finditer(text):
        acc1_match = m

    if domain_match is None and acc1_match is None:
        return None

    if domain_match is not None:
        acc1 = float(domain_match.group(1))
        acc1_std = float(domain_match.group(2))
    else:
        acc1 = float(acc1_match.group(1))
        acc1_std = float("nan")

    s_it_match = None
    for m in SIT_RE.finditer(text):
        s_it_match = m

    s_it = None
    if s_it_match:
        try:
            s_it = float(s_it_match.group(1))
        except ValueError:
            s_it = None

    return {
        "acc1": acc1,
        "acc1_std": acc1_std,
        "s_it": s_it,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=str, help="sweep root directory")
    args = ap.parse_args()

    root = Path(args.root)
    logs = sorted(root.glob("**/live.log"))

    rows = []
    for log in logs:
        parsed = parse_log(log)
        if not parsed:
            continue
        rel = log.relative_to(root)
        rows.append((parsed["acc1"], parsed["acc1_std"], parsed["s_it"], str(rel.parent)))

    rows.sort(key=lambda x: x[0], reverse=True)

    print("acc1\tstd\ts/it\trun")
    for acc1, std, s_it, run in rows:
        s_it_str = "-" if s_it is None else f"{s_it:.4f}"
        print(f"{acc1:.3f}\t{std:.3f}\t{s_it_str}\t{run}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
