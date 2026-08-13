#!/usr/bin/env python3
"""Run or evaluate a trained model on KernelBench policy JSONL data."""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SECTION_RE = re.compile(r"\[(BOTTLENECK_JSON|STRATEGY_JSON)\]")
VERSION_RE = re.compile(r"_[01]$")


class EvaluationError(ValueError):
    pass


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise EvaluationError(f"{path}:{line_number}: 非法 JSON：{error}") from error
        if not isinstance(value, dict):
            raise EvaluationError(f"{path}:{line_number}: 顶层必须是对象")
        records.append(value)
    return records


def parse_tagged_json(text: str, tag: str) -> dict[str, Any]:
    marker = f"[{tag}]"
    start = text.find(marker)
    if start < 0:
        raise EvaluationError(f"缺少 {marker}")
    start += len(marker)
    while start < len(text) and text[start].isspace():
        start += 1
    if text.startswith("```", start):
        newline = text.find("\n", start)
        if newline < 0:
            raise EvaluationError(f"{marker} 后代码围栏不完整")
        start = newline + 1
    try:
        value, _ = json.JSONDecoder().raw_decode(text, start)
    except json.JSONDecodeError as error:
        raise EvaluationError(f"{marker} 后不是合法 JSON：{error}") from error
    if not isinstance(value, dict):
        raise EvaluationError(f"{marker} JSON 顶层必须是对象")
    return value


def parse_policy_output(text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(text, str):
        raise EvaluationError("output 必须是字符串")
    return (
        parse_tagged_json(text, "BOTTLENECK_JSON"),
        parse_tagged_json(text, "STRATEGY_JSON"),
    )


def parse_agent_policy_output(text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Parse model policy for Agent execution, tolerating common wrapper omissions."""
    try:
        return parse_policy_output(text)
    except EvaluationError as tagged_error:
        decoder = json.JSONDecoder()
        values: list[dict[str, Any]] = []
        position = 0
        while True:
            position = text.find("{", position)
            if position < 0:
                break
            try:
                value, end = decoder.raw_decode(text, position)
            except json.JSONDecodeError:
                position += 1
                continue
            if isinstance(value, dict):
                values.append(value)
            position = end
        for value in values:
            bottleneck = value.get("bottleneck_json")
            strategy = value.get("strategy_json")
            if isinstance(bottleneck, dict) and isinstance(strategy, dict):
                return bottleneck, strategy
        bottlenecks = [value for value in values if isinstance(value.get("bottleneck"), dict)]
        strategies = [value for value in values if isinstance(value.get("strategy"), dict)]
        if bottlenecks and strategies:
            return bottlenecks[0], strategies[0]
        raise EvaluationError(f"无法采集 Agent 策略：{tagged_error}") from tagged_error


def parse_operator_level(input_text: str) -> str:
    marker = "[OPERATOR_JSON]"
    start = input_text.find(marker)
    if start < 0:
        return "unknown"
    start += len(marker)
    while start < len(input_text) and input_text[start].isspace():
        start += 1
    try:
        value, _ = json.JSONDecoder().raw_decode(input_text, start)
    except json.JSONDecodeError:
        return "unknown"
    level = value.get("level") if isinstance(value, dict) else None
    return level if isinstance(level, str) else "unknown"


def require_object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvaluationError(f"{name} 必须是对象")
    return value


def require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise EvaluationError(f"{name} 必须是非空字符串")
    return value


@dataclass(frozen=True)
class PolicyView:
    bottleneck_key: str
    bottleneck_description: str
    evidence_keys: tuple[str, ...]
    strategy_key: str
    strategy_description: str
    change_keys: tuple[str, ...]
    targets: tuple[str, ...]
    actions: tuple[str, ...]


def policy_view(
    bottleneck_report: dict[str, Any], strategy_report: dict[str, Any]
) -> PolicyView:
    bottleneck = require_object(bottleneck_report.get("bottleneck"), "bottleneck")
    strategy = require_object(strategy_report.get("strategy"), "strategy")
    evidence_keys = bottleneck.get("evidence_keys")
    changes = strategy.get("changes")
    if not isinstance(evidence_keys, list) or not all(isinstance(x, str) for x in evidence_keys):
        raise EvaluationError("bottleneck.evidence_keys 必须是字符串数组")
    if not isinstance(changes, list) or not changes:
        raise EvaluationError("strategy.changes 必须是非空数组")
    normalized_changes = [require_object(x, "strategy.changes[]") for x in changes]
    return PolicyView(
        bottleneck_key=require_string(bottleneck.get("bottleneck_key"), "bottleneck_key"),
        bottleneck_description=require_string(bottleneck.get("description"), "bottleneck.description"),
        evidence_keys=tuple(evidence_keys),
        strategy_key=require_string(strategy.get("strategy_key"), "strategy_key"),
        strategy_description=require_string(strategy.get("description"), "strategy.description"),
        change_keys=tuple(require_string(x.get("change_key"), "change_key") for x in normalized_changes),
        targets=tuple(require_string(x.get("target"), "target") for x in normalized_changes),
        actions=tuple(require_string(x.get("action"), "action") for x in normalized_changes),
    )


def set_counts(expected: Iterable[str], predicted: Iterable[str]) -> tuple[int, int, int]:
    expected_set, predicted_set = set(expected), set(predicted)
    return (
        len(expected_set & predicted_set),
        len(predicted_set - expected_set),
        len(expected_set - predicted_set),
    )


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def prf(counts: tuple[int, int, int]) -> dict[str, float]:
    tp, fp, fn = counts
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": safe_div(2 * precision * recall, precision + recall),
    }


BOOLEAN_METRICS = (
    "json_valid",
    "bottleneck_key_exact",
    "bottleneck_description_exact",
    "evidence_keys_exact",
    "strategy_key_exact",
    "strategy_description_exact",
    "change_keys_ordered_exact",
    "change_keys_set_exact",
    "targets_ordered_exact",
    "actions_exact",
    "policy_contract_exact",
    "full_output_exact",
)


def score_one(
    ops: str, level: str, expected_text: str, predicted_text: str | None
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ops": ops,
        "operator_group": VERSION_RE.sub("", ops),
        "level": level,
        **{key: False for key in BOOLEAN_METRICS},
        "evidence_counts": [0, 0, 0],
        "change_counts": [0, 0, 0],
        "error": None,
    }
    if predicted_text is None:
        result["error"] = "missing_prediction"
        return result
    try:
        expected_bn, expected_strategy = parse_policy_output(expected_text)
        predicted_bn, predicted_strategy = parse_policy_output(predicted_text)
        expected = policy_view(expected_bn, expected_strategy)
        predicted = policy_view(predicted_bn, predicted_strategy)
    except EvaluationError as error:
        result["error"] = str(error)
        return result
    result["json_valid"] = True
    result["bottleneck_key_exact"] = predicted.bottleneck_key == expected.bottleneck_key
    result["bottleneck_description_exact"] = (
        predicted.bottleneck_description == expected.bottleneck_description
    )
    result["evidence_keys_exact"] = set(predicted.evidence_keys) == set(expected.evidence_keys)
    result["strategy_key_exact"] = predicted.strategy_key == expected.strategy_key
    result["strategy_description_exact"] = (
        predicted.strategy_description == expected.strategy_description
    )
    result["change_keys_ordered_exact"] = predicted.change_keys == expected.change_keys
    result["change_keys_set_exact"] = set(predicted.change_keys) == set(expected.change_keys)
    result["targets_ordered_exact"] = predicted.targets == expected.targets
    result["actions_exact"] = predicted.actions == expected.actions
    result["policy_contract_exact"] = all(result[key] for key in (
        "bottleneck_key_exact", "evidence_keys_exact", "strategy_key_exact",
        "change_keys_ordered_exact", "targets_ordered_exact",
    ))
    result["full_output_exact"] = predicted_bn == expected_bn and predicted_strategy == expected_strategy
    result["evidence_counts"] = list(set_counts(expected.evidence_keys, predicted.evidence_keys))
    result["change_counts"] = list(set_counts(expected.change_keys, predicted.change_keys))
    return result


def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(items)
    metrics = {
        key: safe_div(sum(bool(item[key]) for item in items), count)
        for key in BOOLEAN_METRICS
    }
    evidence = tuple(sum(item["evidence_counts"][i] for item in items) for i in range(3))
    changes = tuple(sum(item["change_counts"][i] for item in items) for i in range(3))
    metrics["evidence_keys_micro"] = prf(evidence)
    metrics["change_keys_micro"] = prf(changes)
    return {"samples": count, "metrics": metrics}


def build_report(
    dataset: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> dict[str, Any]:
    expected_ops: list[str] = []
    for item in dataset:
        ops = item.get("ops")
        if not isinstance(ops, str) or not isinstance(item.get("input"), str) or not isinstance(item.get("output"), str):
            raise EvaluationError("dataset 每条必须包含字符串 ops/input/output")
        expected_ops.append(ops)
    if len(set(expected_ops)) != len(expected_ops):
        raise EvaluationError("dataset 包含重复 ops")
    prediction_map: dict[str, str] = {}
    for item in predictions:
        ops = item.get("ops")
        output = item.get("output", item.get("prediction"))
        if not isinstance(ops, str) or not isinstance(output, str):
            raise EvaluationError("prediction 每条必须包含字符串 ops 和 output/prediction")
        if ops in prediction_map:
            raise EvaluationError(f"predictions 包含重复 ops：{ops}")
        prediction_map[ops] = output
    extras = sorted(set(prediction_map) - set(expected_ops))
    if extras:
        raise EvaluationError("predictions 包含数据集外 ops：" + ", ".join(extras[:5]))
    details = [
        score_one(
            item["ops"], parse_operator_level(item["input"]), item["output"],
            prediction_map.get(item["ops"]),
        )
        for item in dataset
    ]
    by_level = {
        level: aggregate([item for item in details if item["level"] == level])
        for level in sorted({item["level"] for item in details})
    }
    by_operator = collections.defaultdict(list)
    for item in details:
        by_operator[item["operator_group"]].append(item)
    operator_metrics: dict[str, float] = {}
    for key in BOOLEAN_METRICS:
        group_values = [
            safe_div(sum(bool(item[key]) for item in group), len(group))
            for group in by_operator.values()
        ]
        operator_metrics[key] = safe_div(sum(group_values), len(group_values))
    errors = collections.Counter(item["error"] for item in details if item["error"])
    return {
        "dataset_samples": len(dataset),
        "prediction_samples": len(predictions),
        "missing_predictions": sorted(set(expected_ops) - set(prediction_map)),
        "sample_micro": aggregate(details),
        "operator_group_macro": {
            "operator_groups": len(by_operator),
            "metrics": operator_metrics,
        },
        "by_level": by_level,
        "errors": dict(sorted(errors.items())),
        "details": details,
    }


def resolve_dtype(torch: Any, name: str) -> Any:
    if name == "auto":
        return "auto"
    value = getattr(torch, name, None)
    if value is None:
        raise EvaluationError(f"torch 不支持 dtype：{name}")
    return value


def run_transformers(
    dataset: list[dict[str, Any]], model_path: str, max_new_tokens: int,
    device_map: str, torch_dtype: str, trust_remote_code: bool,
) -> list[dict[str, str]]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise EvaluationError("--model 需要安装 torch 和 transformers") from error
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        torch_dtype=resolve_dtype(torch, torch_dtype),
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    predictions: list[dict[str, str]] = []
    model.eval()
    for index, item in enumerate(dataset, 1):
        encoded = tokenizer(item["input"], return_tensors="pt")
        model_device = next(model.parameters()).device
        encoded = {key: value.to(model_device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_length = encoded["input_ids"].shape[1]
        output = tokenizer.decode(generated[0, prompt_length:], skip_special_tokens=True)
        predictions.append({"ops": item["ops"], "output": output})
        print(f"generated={index}/{len(dataset)} ops={item['ops']}", file=sys.stderr)
    return predictions


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/kernel_policy_data.jsonl"))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--predictions", type=Path)
    source.add_argument("--model")
    parser.add_argument("--save-predictions", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens 必须为正数")
    try:
        dataset = read_jsonl(args.dataset)
        if args.predictions:
            predictions = read_jsonl(args.predictions)
        else:
            predictions = run_transformers(
                dataset, args.model, args.max_new_tokens, args.device_map,
                args.torch_dtype, args.trust_remote_code,
            )
            if args.save_predictions is None:
                raise EvaluationError("--model 模式必须指定 --save-predictions")
            write_jsonl(args.save_predictions, predictions)
        report = build_report(dataset, predictions)
    except EvaluationError as error:
        raise SystemExit(f"evaluation error: {error}") from error
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
        print(f"report={args.report}")
    else:
        sys.stdout.write(rendered)
    summary = report["sample_micro"]["metrics"]
    print(
        "samples={dataset_samples} predictions={prediction_samples} "
        "json_valid={json_valid:.4f} bottleneck_key={bottleneck_key_exact:.4f} "
        "strategy_key={strategy_key_exact:.4f} changes={change_keys_ordered_exact:.4f} "
        "contract={policy_contract_exact:.4f}".format(
            dataset_samples=report["dataset_samples"],
            prediction_samples=report["prediction_samples"], **summary,
        ),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
