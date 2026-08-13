#!/usr/bin/env python3
"""Manage isolated, agent-executed policy effect evaluation runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_policy_model import (  # noqa: E402
    EvaluationError, parse_agent_policy_output, parse_operator_level, policy_view, read_jsonl,
)


OPS_RE = re.compile(r"^(?P<operator>.+)_(?P<version>[01])$")
FILE_RE = re.compile(r"^--- FILE: (?P<path>[^\n]+) ---\n", re.MULTILINE)
SNAPSHOT_RE = re.compile(r"^(?P<epoch>\d+)_(?P<step>\d+)$")
TERMINAL = {"COMPLETED", "FAILED_PRECISION", "FAILED_PERFORMANCE", "PREDICTION_INVALID", "STOPPED_IMPLEMENTATION"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fingerprint(project: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        path for folder in ("op_host", "op_kernel")
        for path in (project / folder).rglob("*") if path.is_file()
    )
    for path in files:
        digest.update(path.relative_to(project).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def queue_path(run_dir: Path) -> Path:
    return run_dir / "queue.json"


def load_queue(run_dir: Path) -> dict[str, Any]:
    return json.loads(queue_path(run_dir).read_text(encoding="utf-8"))


def parse_input_section(text: str, tag: str, next_tag: str) -> str:
    start_marker, end_marker = f"[{tag}]", f"[{next_tag}]"
    start, end = text.find(start_marker), text.find(end_marker)
    if start < 0 or end < 0 or end <= start:
        raise EvaluationError(f"输入缺少有序段落 {start_marker} -> {end_marker}")
    return text[start + len(start_marker):end].strip("\n")


def parse_input_json(text: str, tag: str, next_tag: str) -> dict[str, Any]:
    body = parse_input_section(text, tag, next_tag)
    try:
        value = json.loads(body)
    except json.JSONDecodeError as error:
        raise EvaluationError(f"{tag} 不是合法 JSON：{error}") from error
    if not isinstance(value, dict):
        raise EvaluationError(f"{tag} 顶层必须是对象")
    return value


def parse_sources(text: str, tag: str, next_tag: str, prefix: str) -> dict[str, str]:
    body = parse_input_section(text, tag, next_tag)
    matches = list(FILE_RE.finditer(body))
    if not matches:
        raise EvaluationError(f"{tag} 不含 FILE 标记")
    files: dict[str, str] = {}
    for index, match in enumerate(matches):
        relative = match.group("path")
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or not relative.startswith(prefix + "/"):
            raise EvaluationError(f"{tag} 含不安全路径：{relative}")
        if relative in files:
            raise EvaluationError(f"{tag} 含重复路径：{relative}")
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        files[relative] = body[match.end():end].strip("\n") + "\n"
    return files


def write_sources(project: Path, files: dict[str, str]) -> None:
    for relative, content in files.items():
        destination = project / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")


def copy_build_scaffold(template: Path, project: Path) -> None:
    for name in ("CMakeLists.txt", "CMakePresets.json", "build.sh"):
        source = template / name
        if not source.is_file():
            raise EvaluationError(f"构建脚手架缺少：{source}")
        shutil.copy2(source, project / name)
    for folder in ("op_host", "op_kernel"):
        cmake_files = sorted((template / folder).rglob("CMakeLists.txt"))
        if not cmake_files:
            raise EvaluationError(f"构建脚手架缺少 {folder}/CMakeLists.txt")
        for source in cmake_files:
            destination = project / source.relative_to(template)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)


def restore_source_project(
    source_project: Path, template: Path, input_text: str,
    operator: str, parent_version: int, vendor: str,
) -> None:
    host_sources = parse_sources(input_text, "OP_HOST_SOURCE", "OP_KERNEL_SOURCE", "op_host")
    kernel_sources = parse_sources(input_text, "OP_KERNEL_SOURCE", "OUTPUT_FORMAT", "op_kernel")
    if source_project.exists():
        for relative, content in (host_sources | kernel_sources).items():
            path = source_project / relative
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                raise SystemExit(f"source 与当前 step 输入源码不一致：{path}")
        return
    source_project.mkdir(parents=True)
    copy_build_scaffold(template, source_project)
    write_sources(source_project, host_sources)
    write_sources(source_project, kernel_sources)
    shutil.copytree(
        template / "CppExtension", source_project / "CppExtension", symlinks=True,
        ignore=shutil.ignore_patterns("build", "*.o", "*.d", "__pycache__"),
    )
    dump(source_project / "workspace.json", {
        "operator": operator, "version": "source", "source_project": "training_record.input",
        "parent_version": parent_version, "vendor": vendor, "status": "PREPARED",
        "source_fingerprint": fingerprint(source_project), "precision": None,
        "performance": None, "bottleneck": None, "strategy": None,
        "implementation": None, "updated_at": now(),
    })


def copy_source_to_candidate(source_project: Path, candidate: Path) -> None:
    candidate.mkdir(parents=True)
    for name in ("CMakeLists.txt", "CMakePresets.json", "build.sh"):
        shutil.copy2(source_project / name, candidate / name)
    shutil.copytree(source_project / "op_host", candidate / "op_host", symlinks=True)
    shutil.copytree(source_project / "op_kernel", candidate / "op_kernel", symlinks=True)
    shutil.copytree(
        source_project / "CppExtension", candidate / "CppExtension", symlinks=True,
        ignore=shutil.ignore_patterns("build", "*.o", "*.d", "__pycache__", "*.so"),
    )


def positive_latency(path: Path) -> float:
    value = json.loads(path.read_text(encoding="utf-8")).get("kernel_latency_us")
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise EvaluationError(f"无有效 latency：{path}")
    return float(value)


def init_run(args: argparse.Namespace) -> int:
    run_dir: Path = args.run_dir.resolve()
    if run_dir.exists():
        raise SystemExit(f"run-dir 已存在，禁止覆盖：{run_dir}")
    dataset = read_jsonl(args.dataset)
    predictions = read_jsonl(args.predictions)
    prediction_map: dict[str, str] = {}
    for item in predictions:
        ops, output = item.get("ops"), item.get("output", item.get("prediction"))
        if not isinstance(ops, str) or not isinstance(output, str):
            raise SystemExit("prediction 必须包含字符串 ops 和 output/prediction")
        if ops in prediction_map:
            raise SystemExit(f"重复 prediction：{ops}")
        prediction_map[ops] = output
    items: list[dict[str, Any]] = []
    gold_dir = run_dir / ".gold"
    for position, sample in enumerate(dataset, 1):
        ops = sample.get("ops")
        if not isinstance(ops, str) or OPS_RE.fullmatch(ops) is None:
            raise SystemExit(f"非法 ops：{ops}")
        input_text = sample.get("input")
        if not isinstance(input_text, str) or not isinstance(sample.get("output"), str):
            raise SystemExit(f"{ops} 的 input/output 必须是字符串")
        level = parse_operator_level(input_text)
        operator_meta = parse_input_json(input_text, "OPERATOR_JSON", "PERFORMANCE_JSON")
        performance = parse_input_json(input_text, "PERFORMANCE_JSON", "OP_HOST_SOURCE")
        baseline_latency = performance.get("kernel_latency_us")
        if not isinstance(baseline_latency, (int, float)) or not math.isfinite(baseline_latency) or baseline_latency <= 0:
            raise SystemExit(f"{ops} 的训练输入没有有效 baseline latency")
        host_sources = parse_sources(input_text, "OP_HOST_SOURCE", "OP_KERNEL_SOURCE", "op_host")
        kernel_sources = parse_sources(input_text, "OP_KERNEL_SOURCE", "OUTPUT_FORMAT", "op_kernel")
        if operator_meta.get("operator") != OPS_RE.fullmatch(ops).group("operator"):
            raise SystemExit(f"{ops} 与 OPERATOR_JSON.operator 不一致")
        template = args.template_root.resolve() / level / operator_meta.get("operator", "")
        if not template.is_dir():
            raise SystemExit(f"构建脚手架不存在：{template}")
        case_dir = run_dir / "cases" / ops
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "input.txt").write_text(sample["input"], encoding="utf-8")
        dump(case_dir / "source_manifest.json", {
            "operator": operator_meta, "template_project": str(template),
            "op_host_files": sorted(host_sources), "op_kernel_files": sorted(kernel_sources),
        })
        dump(gold_dir / f"{ops}.json", {"ops": ops, "output": sample["output"]})
        prediction = prediction_map.get(ops)
        status, reason = "PREDICTED", None
        if prediction is None:
            status, reason = "PREDICTION_INVALID", "missing_prediction"
        else:
            (case_dir / "prediction.txt").write_text(prediction, encoding="utf-8")
            try:
                bottleneck, strategy = parse_agent_policy_output(prediction)
                policy_view(bottleneck, strategy)
                dump(case_dir / "policy/bottleneck.json", bottleneck)
                dump(case_dir / "policy/strategy.json", strategy)
            except EvaluationError as error:
                status, reason = "PREDICTION_INVALID", str(error)
        items.append({
            "position": position,
            "ops": ops,
            "operator": OPS_RE.fullmatch(ops).group("operator"),
            "parent_version": int(OPS_RE.fullmatch(ops).group("version")),
            "level": level,
            "template_project": str(template),
            "baseline_latency_us": baseline_latency,
            "status": status,
            "failure_stage": "prediction" if reason else None,
            "reason": reason,
            "candidate_project": None,
            "result": None,
        })
    extras = sorted(set(prediction_map) - {item["ops"] for item in items})
    if extras:
        raise SystemExit("predictions 含数据集外 ops：" + ", ".join(extras[:5]))
    dump(queue_path(run_dir), {
        "created_at": now(),
        "dataset": str(args.dataset.resolve()),
        "predictions": str(args.predictions.resolve()),
        "evaluation_workspace": str(args.evaluation_workspace.resolve()),
        "template_root": str(args.template_root.resolve()),
        "items": items,
    })
    print(f"run={run_dir} cases={len(items)}")
    return 0


def locate_item(queue: dict[str, Any], ops: str) -> dict[str, Any]:
    matches = [item for item in queue["items"] if item["ops"] == ops]
    if len(matches) != 1:
        raise SystemExit(f"queue 中没有唯一 case：{ops}")
    return matches[0]


def materialize(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    queue = load_queue(run_dir)
    item = locate_item(queue, args.ops)
    earlier = [x for x in queue["items"] if x["position"] < item["position"]]
    unfinished = [x["ops"] for x in earlier if x["status"] not in TERMINAL]
    if unfinished:
        raise SystemExit(f"前序 case 尚未终止：{unfinished[0]}")
    if item["status"] != "PREDICTED":
        raise SystemExit(f"case 状态不是 PREDICTED：{item['status']}")
    template = Path(item["template_project"])
    snapshot = SNAPSHOT_RE.fullmatch(run_dir.name)
    if snapshot is None:
        raise SystemExit(f"run-dir 名不是 <epoch>_<step>：{run_dir.name}")
    step = int(snapshot.group("step"))
    operator_root = args.evaluation_workspace.resolve() / item["level"] / item["ops"]
    source_project = operator_root / "source"
    candidate = operator_root / f"eval_step{step}"
    if candidate.exists():
        raise SystemExit(f"candidate 已存在，禁止覆盖：{candidate}")
    input_text = (run_dir / "cases" / item["ops"] / "input.txt").read_text(encoding="utf-8")
    template_root = Path(queue["template_root"])
    manifest = json.loads((template_root / "manifest.json").read_text(encoding="utf-8"))
    manifest_item = manifest[item["operator"]]
    restore_source_project(
        source_project, template, input_text, item["operator"],
        item["parent_version"], manifest_item["vendor"],
    )
    copy_source_to_candidate(source_project, candidate)
    (candidate / "strategy").mkdir()
    shutil.copy2(run_dir / "cases" / item["ops"] / "policy/strategy.json", candidate / "strategy/strategy.json")
    dump(candidate / "workspace.json", {
        "operator": item["operator"],
        "version": "effect_candidate",
        "source_project": "training_record.input",
        "parent_version": item["parent_version"],
        "vendor": manifest_item["vendor"],
        "status": "PREPARED",
        "source_fingerprint": fingerprint(candidate),
        "precision": None,
        "performance": None,
        "bottleneck": None,
        "strategy": {"status": "INPUT", "result": "strategy/strategy.json"},
        "implementation": None,
        "updated_at": now(),
    })
    item["source_project"] = str(source_project)
    item["step"] = step
    item["candidate_project"] = str(candidate)
    item["status"] = "MATERIALIZED"
    item["failure_stage"] = None
    item["reason"] = None
    dump(queue_path(run_dir), queue)
    task = run_dir / "cases" / item["ops"] / "AGENT_TASK.md"
    task.write_text(
        "\n".join([
            f"# Policy effect case: {item['ops']}", "",
            "只读取预测 policy，不读取 `.gold/`。把预测 strategy 作为被测输入，按其修改意图实施。",
            "不运行固定策略推导校验，不因与 gold/固定 change 顺序不一致而拒绝；只在 target 不存在、API/dtype/容量或数学语义明确不可实施时停止。",
            "- baseline 与待优化源码均来自该 case 的训练格式 input；禁止改用历史工作版本源码。",
            f"- bottleneck: `{run_dir / 'cases' / item['ops'] / 'policy/bottleneck.json'}`",
            f"- strategy: `{run_dir / 'cases' / item['ops'] / 'policy/strategy.json'}`",
            f"- candidate: `{candidate}`",
            f"- precision: `python .codex/skills/kernel-precision/scripts/validate_precision.py {item['operator']} --project-dir {candidate}`",
            f"- performance: `python .codex/skills/kernel-performance/scripts/collect_performance.py {item['operator']} --device 0 --project-dir {candidate}`",
            "精度失败不得采性能；最多三次原 change 范围修复；完成后运行 finalize。", "",
        ]), encoding="utf-8",
    )
    print(f"candidate={candidate}\ntask={task}")
    return 0


def finalize(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    queue = load_queue(run_dir)
    item = locate_item(queue, args.ops)
    if item["status"] != "MATERIALIZED":
        raise SystemExit(f"case 状态不是 MATERIALIZED：{item['status']}")
    candidate = Path(item["candidate_project"])
    precision_path = candidate / "precision/precision.json"
    if not precision_path.is_file():
        raise SystemExit("candidate 尚无 precision.json")
    precision = json.loads(precision_path.read_text(encoding="utf-8"))
    if precision.get("status") != "PASS" or precision.get("exit_code") != 0:
        item.update({
            "status": "FAILED_PRECISION", "failure_stage": precision.get("stage", "precision"),
            "reason": precision.get("reason", "precision failed"),
            "result": {"precision_pass": False, "candidate_latency_us": None, "reduction_percent": None},
        })
    else:
        performance_path = candidate / "performance/performance.json"
        if not performance_path.is_file():
            raise SystemExit("精度 PASS，但 candidate 尚无 performance.json")
        candidate_latency = positive_latency(performance_path)
        reduction = (item["baseline_latency_us"] - candidate_latency) / item["baseline_latency_us"] * 100.0
        item.update({
            "status": "COMPLETED", "failure_stage": None, "reason": None,
            "result": {
                "precision_pass": True,
                "candidate_latency_us": candidate_latency,
                "reduction_percent": reduction,
                "improved_over_1_percent": reduction > 1.0,
            },
        })
    dump(queue_path(run_dir), queue)
    print(json.dumps(item, ensure_ascii=False, indent=2))
    return 0


def mark_failed(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    queue = load_queue(run_dir)
    item = locate_item(queue, args.ops)
    if item["status"] in TERMINAL:
        raise SystemExit(f"case 已终止：{item['status']}")
    item.update({
        "status": args.status,
        "failure_stage": args.stage,
        "reason": args.reason,
        "result": {"precision_pass": False, "candidate_latency_us": None, "reduction_percent": None},
    })
    dump(queue_path(run_dir), queue)
    print(json.dumps(item, ensure_ascii=False, indent=2))
    return 0


def report(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    queue = load_queue(run_dir)
    items = queue["items"]
    completed = [x for x in items if x["status"] == "COMPLETED"]
    passing = [x for x in completed if x["result"]["improved_over_1_percent"]]
    reductions = [x["result"]["reduction_percent"] for x in completed]
    prediction_valid = [x for x in items if x["status"] != "PREDICTION_INVALID"]
    precision_attempted = [
        x for x in items
        if x["status"] in {"FAILED_PRECISION", "FAILED_PERFORMANCE", "COMPLETED"}
    ]
    precision_passing = [x for x in precision_attempted if x.get("result", {}).get("precision_pass") is True]
    summary = {
        "cases": len(items),
        "status_counts": dict(sorted(__import__("collections").Counter(x["status"] for x in items).items())),
        "prediction_parse_valid_cases": len(prediction_valid),
        "prediction_parse_valid_rate": len(prediction_valid) / len(items) if items else 0.0,
        "precision_attempted_cases": len(precision_attempted),
        "precision_pass_rate_among_attempted": len(precision_passing) / len(precision_attempted) if precision_attempted else None,
        "end_to_end_precision_pass_rate": len(precision_passing) / len(items) if items else 0.0,
        "effect_evaluable_cases": len(completed),
        "improved_over_1_percent": len(passing),
        "improved_over_1_percent_rate": len(passing) / len(completed) if completed else 0.0,
        "mean_reduction_percent": sum(reductions) / len(reductions) if reductions else None,
        "items": items,
    }
    output = args.output or (run_dir / "effect_report.json")
    dump(output, summary)
    print(f"report={output}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--dataset", type=Path, required=True)
    init.add_argument("--predictions", type=Path, required=True)
    init.add_argument("--run-dir", type=Path, required=True)
    init.add_argument("--evaluation-workspace", type=Path, default=ROOT / "kernel_workspace_eval/KernelBench910B")
    init.add_argument("--template-root", type=Path, default=ROOT / "kernel/KernelBench910B")
    for name in ("materialize", "finalize"):
        command = sub.add_parser(name)
        command.add_argument("--run-dir", type=Path, required=True)
        command.add_argument("--ops", required=True)
        if name == "materialize":
            command.add_argument("--evaluation-workspace", type=Path, default=ROOT / "kernel_workspace_eval/KernelBench910B")
    summary = sub.add_parser("report")
    summary.add_argument("--run-dir", type=Path, required=True)
    summary.add_argument("--output", type=Path)
    failed = sub.add_parser("mark-failed")
    failed.add_argument("--run-dir", type=Path, required=True)
    failed.add_argument("--ops", required=True)
    failed.add_argument("--status", choices=["STOPPED_IMPLEMENTATION", "FAILED_PERFORMANCE"], required=True)
    failed.add_argument("--stage", required=True)
    failed.add_argument("--reason", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return {"init": init_run, "materialize": materialize, "finalize": finalize, "report": report, "mark-failed": mark_failed}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
