#!/usr/bin/env python3
"""Deterministically split policy JSONL by operator group without leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


VERSION_RE = re.compile(r"_[01]$")


def bucket(operator: str, seed: str) -> float:
    digest = hashlib.sha256(f"{seed}\0{operator}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def write(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("datasets/kernel_policy_data.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--seed", default="kernel-policy-v1")
    args = parser.parse_args()
    if not 0 < args.eval_ratio < 1:
        raise SystemExit("--eval-ratio 必须在 (0,1) 内")
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if list(row) != ["ops", "input", "output"]:
            raise SystemExit("输入必须严格为 ops/input/output 三字段")
        group = VERSION_RE.sub("", row["ops"])
        groups.setdefault(group, []).append(row)
    evaluation_groups = {group for group in groups if bucket(group, args.seed) < args.eval_ratio}
    if not evaluation_groups or evaluation_groups == set(groups):
        ordered = sorted(groups, key=lambda group: bucket(group, args.seed))
        count = min(len(ordered) - 1, max(1, round(len(ordered) * args.eval_ratio)))
        evaluation_groups = set(ordered[:count])
    train = [row for row in rows if VERSION_RE.sub("", row["ops"]) not in evaluation_groups]
    evaluation = [row for row in rows if VERSION_RE.sub("", row["ops"]) in evaluation_groups]
    write(args.output_dir / "train.jsonl", train)
    write(args.output_dir / "eval.jsonl", evaluation)
    (args.output_dir / "split.audit.json").write_text(json.dumps({
        "input": str(args.input.resolve()),
        "seed": args.seed,
        "eval_ratio": args.eval_ratio,
        "train_samples": len(train),
        "eval_samples": len(evaluation),
        "train_operator_groups": len(groups) - len(evaluation_groups),
        "eval_operator_groups": len(evaluation_groups),
        "eval_groups": sorted(evaluation_groups),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"train={len(train)} eval={len(evaluation)} output={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
