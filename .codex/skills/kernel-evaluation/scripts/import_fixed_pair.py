#!/usr/bin/env python3
"""Import one fixed train-probe op and one fixed eval op for every saved step."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_policy_model import EvaluationError, build_report, read_jsonl, write_jsonl  # noqa: E402


EVAL_RE = re.compile(r"^(?P<epoch>\d+)_(?P<step>\d+)\.jsonl$")


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def unique(rows: list[dict[str, Any]], ops: str, source: Path) -> dict[str, Any]:
    matches = [row for row in rows if row.get("ops") == ops]
    if len(matches) != 1:
        raise SystemExit(f"{source} 中 {ops} 数量不是 1：{len(matches)}")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--train-ops", required=True)
    parser.add_argument("--eval-ops", required=True)
    parser.add_argument("--gold-dataset", type=Path, default=Path("datasets/kernel_policy_data.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"output-dir 已存在，禁止覆盖：{args.output_dir}")

    gold_rows = read_jsonl(args.gold_dataset)
    gold = {row.get("ops"): row for row in gold_rows}
    for ops in (args.train_ops, args.eval_ops):
        if ops not in gold:
            raise SystemExit(f"gold dataset 缺少：{ops}")

    snapshots: list[dict[str, Any]] = []
    for eval_path in (args.training_output / "eval_samples").glob("*.jsonl"):
        match = EVAL_RE.fullmatch(eval_path.name)
        if match is None:
            continue
        epoch, step = int(match.group("epoch")), int(match.group("step"))
        train_path = args.training_output / "train_probe_samples" / f"{step}.jsonl"
        if not train_path.is_file():
            raise SystemExit(f"缺少对应 train probe：{train_path}")
        train_row = unique(read_jsonl(train_path), args.train_ops, train_path)
        eval_row = unique(read_jsonl(eval_path), args.eval_ops, eval_path)
        selected = [("train", train_row), ("eval", eval_row)]
        dataset: list[dict[str, str]] = []
        predictions: list[dict[str, str]] = []
        roles: dict[str, str] = {}
        for role, row in selected:
            ops, input_text, output_text = row.get("ops"), row.get("input"), row.get("output")
            if not all(isinstance(value, str) for value in (ops, input_text, output_text)):
                raise SystemExit(f"{role} step={step} 缺少字符串 ops/input/output")
            dataset.append({"ops": ops, "input": input_text, "output": gold[ops]["output"]})
            predictions.append({"ops": ops, "output": output_text})
            roles[ops] = role
        name = f"{epoch}_{step}"
        snapshot_dir = args.output_dir / "snapshots" / name
        write_jsonl(snapshot_dir / "dataset.jsonl", dataset)
        write_jsonl(snapshot_dir / "predictions.jsonl", predictions)
        dump(snapshot_dir / "roles.json", roles)
        report = build_report(dataset, predictions)
        dump(snapshot_dir / "offline_report.json", report)
        metrics = report["sample_micro"]["metrics"]
        snapshots.append({
            "snapshot": name, "epoch": epoch, "step": step,
            "dataset": str((snapshot_dir / "dataset.jsonl").resolve()),
            "predictions": str((snapshot_dir / "predictions.jsonl").resolve()),
            "offline_report": str((snapshot_dir / "offline_report.json").resolve()),
            "roles": roles,
            "json_valid": metrics["json_valid"],
            "bottleneck_key_exact": metrics["bottleneck_key_exact"],
            "strategy_key_exact": metrics["strategy_key_exact"],
        })
    snapshots.sort(key=lambda item: (item["step"], item["epoch"]))
    if not snapshots:
        raise SystemExit("没有可导入的 eval snapshot")
    summary = {
        "training_output": str(args.training_output.resolve()),
        "gold_dataset": str(args.gold_dataset.resolve()),
        "evaluation_rule": "every step, fixed train op then fixed eval op",
        "train_ops": args.train_ops, "eval_ops": args.eval_ops,
        "snapshots": snapshots,
    }
    dump(args.output_dir / "training_output_eval.json", summary)
    print(f"output={args.output_dir} snapshots={len(snapshots)} cases={len(snapshots) * 2}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EvaluationError as error:
        raise SystemExit(f"evaluation error: {error}") from error
