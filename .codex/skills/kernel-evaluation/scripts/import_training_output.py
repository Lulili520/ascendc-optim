#!/usr/bin/env python3
"""Import saved training eval snapshots into kernel-evaluation artifacts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_policy_model import EvaluationError, build_report, read_jsonl, write_jsonl  # noqa: E402


SNAPSHOT_RE = re.compile(r"^(?P<epoch>\d+)_(?P<step>\d+)\.jsonl$")


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--gold-dataset", type=Path, default=Path("datasets/kernel_policy_data.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    split_path = args.training_output / "dataset_split_ops.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    eval_ops = split.get("eval_ops")
    if not isinstance(eval_ops, list) or not eval_ops or not all(isinstance(x, str) for x in eval_ops):
        raise SystemExit("dataset_split_ops.json 缺少有效 eval_ops")
    if set(split.get("train_ops", [])) & set(eval_ops):
        raise SystemExit("训练集与评测集 ops 泄漏")

    gold_rows = read_jsonl(args.gold_dataset)
    gold_map = {row.get("ops"): row for row in gold_rows}
    missing_gold = sorted(set(eval_ops) - set(gold_map))
    if missing_gold:
        raise SystemExit("gold dataset 缺少 eval ops：" + ", ".join(missing_gold))

    snapshots: list[dict[str, Any]] = []
    for path in sorted((args.training_output / "eval_samples").glob("*.jsonl")):
        match = SNAPSHOT_RE.fullmatch(path.name)
        if match is None:
            continue
        generated = read_jsonl(path)
        generated_map = {row.get("ops"): row for row in generated}
        if len(generated_map) != len(generated):
            raise SystemExit(f"{path} 含重复或非法 ops")
        if set(generated_map) != set(eval_ops):
            raise SystemExit(f"{path} 的 ops 与独立 eval split 不一致")
        dataset = []
        predictions = []
        for ops in eval_ops:
            row = generated_map[ops]
            input_text, output_text = row.get("input"), row.get("output")
            if not isinstance(input_text, str) or not isinstance(output_text, str):
                raise SystemExit(f"{path}:{ops} 缺少字符串 input/output")
            dataset.append({"ops": ops, "input": input_text, "output": gold_map[ops]["output"]})
            predictions.append({"ops": ops, "output": output_text})
        snapshot = path.stem
        snapshot_dir = args.output_dir / "snapshots" / snapshot
        write_jsonl(snapshot_dir / "dataset.jsonl", dataset)
        write_jsonl(snapshot_dir / "predictions.jsonl", predictions)
        report = build_report(dataset, predictions)
        dump(snapshot_dir / "offline_report.json", report)
        metrics = report["sample_micro"]["metrics"]
        snapshots.append({
            "snapshot": snapshot,
            "epoch": int(match.group("epoch")),
            "step": int(match.group("step")),
            "dataset": str((snapshot_dir / "dataset.jsonl").resolve()),
            "predictions": str((snapshot_dir / "predictions.jsonl").resolve()),
            "offline_report": str((snapshot_dir / "offline_report.json").resolve()),
            "json_valid": metrics["json_valid"],
            "bottleneck_key_exact": metrics["bottleneck_key_exact"],
            "strategy_key_exact": metrics["strategy_key_exact"],
            "change_keys_ordered_exact": metrics["change_keys_ordered_exact"],
            "policy_contract_exact": metrics["policy_contract_exact"],
        })
    if not snapshots:
        raise SystemExit("没有找到 <epoch>_<step>.jsonl eval snapshot")
    summary = {
        "training_output": str(args.training_output.resolve()),
        "gold_dataset": str(args.gold_dataset.resolve()),
        "evaluation_rule": "evaluate every saved snapshot independently; no snapshot selection",
        "eval_ops": eval_ops,
        "snapshots": sorted(snapshots, key=lambda item: (item["step"], item["epoch"])),
    }
    dump(args.output_dir / "training_output_eval.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EvaluationError as error:
        raise SystemExit(f"evaluation error: {error}") from error
