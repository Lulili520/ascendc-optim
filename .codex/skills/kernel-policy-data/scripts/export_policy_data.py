#!/usr/bin/env python3
"""Export validated, performance-positive KernelBench policy JSONL."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp"}
VERSION_RE = re.compile(r"^(?P<operator>.+)_(?P<version>\d+)$")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def skill_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层不是对象：{path}")
    return value


def pretty_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def compact_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def compact_performance(report: dict[str, Any]) -> dict[str, Any]:
    """Keep fields needed for diagnosis and source-grounded action planning."""
    task = report.get("task", {})
    pipeline = report.get("pipeline", {})
    memory = report.get("memory", {})
    l2 = report.get("l2_cache", {})
    conflict = report.get("resource_conflict", {})
    hardware = report.get("hardware", {})
    return {
        "task": {key: task.get(key) for key in (
            "Task Type", "Block Num", "Input Shapes", "Input Data Types",
            "Output Shapes", "Output Data Types", "head_overhead_ratio",
        )},
        "pipeline": {key: pipeline.get(key) for key in (
            "aic_mac_ratio", "aic_scalar_ratio", "aic_mte2_ratio", "aic_fixpipe_ratio",
            "aiv_vec_ratio", "aiv_scalar_ratio", "aiv_mte2_ratio", "aiv_mte3_ratio",
            "aic_icache_miss_rate", "aiv_icache_miss_rate",
        )},
        "memory": {"gm_peak_utilization_percent": memory.get("gm_peak_utilization_percent")},
        "l2_cache": {"aiv_derived_hit_rate_percent": l2.get("aiv_derived_hit_rate_percent")},
        "resource_conflict": {key: conflict.get(key) for key in (
            "aiv_vec_bankgroup_cflt_ratio", "aiv_vec_bank_cflt_ratio", "aiv_vec_resc_cflt_ratio",
        )},
        "per_core": {key: (report.get("per_core") or {}).get(key) for key in (
            "active_cores", "imbalance_percent",
        )},
        "hardware": {key: hardware.get(key) for key in (
            "aic_core_count", "aiv_core_count", "ub_bytes_per_core", "l1_bytes_per_core",
            "l0a_bytes_per_core", "l0b_bytes_per_core", "l0c_bytes_per_core", "l2_bytes",
        )},
    }


def compact_evidence(items: list[dict[str, Any]]) -> list[str]:
    return [f'{item["source"]}|{item["observation"]}' for item in items]


def project_policy(bottleneck: dict[str, Any], strategy: dict[str, Any]) -> dict[str, Any]:
    """Merge workspace reports into one compact causal training answer."""
    issues = bottleneck.get("issues")
    strategies = strategy.get("strategies")
    b_reasoning = bottleneck.get("reasoning")
    s_reasoning = strategy.get("reasoning")
    if not all(isinstance(value, list) for value in (issues, strategies, b_reasoning, s_reasoning)):
        raise ValueError("bottleneck/strategy 缺少有序列表")
    if not issues or len(b_reasoning) != len(issues) or len(strategies) != len(issues) or len(s_reasoning) != len(strategies):
        raise ValueError("必须按顺序完整导出全部 issues 及其 strategies")
    policies = []
    for issue, item, b_reason, s_reason in zip(issues, strategies, b_reasoning, s_reasoning):
        bottleneck_item = issue["bottleneck"]
        changes = [{
            "target": action["target"],
            "edits": action["edits"],
            "constraints": action["constraints"],
        } for action in item["actions"]]
        policies.append({
            "bottleneck": bottleneck_item["bottleneck_key"],
            "cause": bottleneck_item["cause_key"],
            "evidence": compact_evidence(issue["evidence"]),
            "strategy": item["strategy_key"],
            "reasoning": f"{b_reason}；{s_reason}",
            "changes": changes,
        })
    return {"policy": policies}


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"无法加载模块：{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_policy_knowledge(knowledge: dict[str, Any]) -> None:
    root = repo_root()
    strategy = load_module(
        "kernel_strategy_contract",
        root / ".codex/skills/kernel-strategy/scripts/derive_strategy.py",
    )
    if knowledge.get("decision_order") != [
        "read_complete_source", "identify_concrete_causes",
        "deduplicate_and_rank", "derive_deterministic_strategies",
    ]:
        raise ValueError("policy knowledge 的全问题决策顺序不一致")
    if knowledge.get("cause_strategies") != strategy.RULES:
        raise ValueError("policy knowledge 的 cause→strategy 映射与官方推导器不一致")
    if knowledge.get("strategy_operations") != strategy.OPERATIONS:
        raise ValueError("policy knowledge 的 strategy→operation 映射与官方推导器不一致")
    if knowledge.get("selection_rules") != {
        "source_first": "直接读取当前 shape 的完整 Host/Kernel 执行路径并输出确定问题",
        "concrete_cause": "cause 必须具体、排他、有直接源码证据并唯一决定 strategy",
        "performance_role": "性能原值只补充影响、排序和因果一致性，不以固定阈值产生 cause",
        "deterministic_strategy": "strategy 不重新诊断或选择候选方向",
    }:
        raise ValueError("policy knowledge 的确定 cause/strategy 规则不一致")
    action_fields = knowledge.get("action_field_meanings", {})
    if set(action_fields) != {"target", "operation", "edits", "constraints"}:
        raise ValueError("policy knowledge 缺少 action 字段语义")
    slots = load_json(root / ".codex/skills/kernel-strategy/references/operation-slots.json")
    if set(slots) != set(strategy.OPERATIONS.values()) or any(
        not isinstance(value, dict) or set(value) != {"required", "optional"}
        or not isinstance(value["required"], list) or not 1 <= len(value["required"]) <= 4
        or not isinstance(value["optional"], list)
        or len(value["required"]) + len(value["optional"]) > 4
        or any(not isinstance(item, str) or not item for item in value["required"] + value["optional"])
        for value in slots.values()
    ):
        raise ValueError("operation slots 未完整覆盖固定 operation")


def validate_parent_reports(parent: Path) -> None:
    root = repo_root()
    bottleneck_scripts = root / ".codex/skills/kernel-bottleneck/scripts"
    strategy_scripts = root / ".codex/skills/kernel-strategy/scripts"
    for directory in (str(bottleneck_scripts), str(strategy_scripts)):
        if directory not in sys.path:
            sys.path.insert(0, directory)
    bottleneck_validator = load_module(
        "kernel_policy_bottleneck_validator",
        bottleneck_scripts / "validate_report.py",
    )
    strategy_validator = load_module(
        "kernel_policy_strategy_validator",
        strategy_scripts / "validate_strategy.py",
    )
    bottleneck_path = parent / "bottleneck/bottleneck.json"
    strategy_path = parent / "strategy/strategy.json"
    bottleneck_validator.validate(bottleneck_path)
    strategy_validator.validate(bottleneck_path, strategy_path)


def validate_completed_version(project: Path, label: str) -> dict[str, Any]:
    workspace = load_json(project / "workspace.json")
    precision = load_json(project / "precision/precision.json")
    fingerprint = source_fingerprint(project)
    if workspace.get("status") != "PERFORMANCE_DONE":
        raise ValueError(f"{label} workspace 不是 PERFORMANCE_DONE")
    if workspace.get("source_fingerprint") != fingerprint:
        raise ValueError(f"{label} workspace 源码指纹不匹配")
    if precision.get("status") != "PASS" or precision.get("exit_code") != 0:
        raise ValueError(f"{label} precision 未 PASS")
    if precision.get("source_fingerprint") != fingerprint:
        raise ValueError(f"{label} precision 源码指纹不匹配")
    return workspace


def source_file_bytes(project: Path) -> dict[str, bytes]:
    return {
        path.relative_to(project).as_posix(): path.read_bytes()
        for folder in ("op_host", "op_kernel")
        for path in (project / folder).rglob("*") if path.is_file()
    }


def validate_implementation_link(parent: Path, parent_strategy: dict[str, Any], child: Path) -> None:
    implementation = load_json(child / "strategy/implementation.json")
    strategies = parent_strategy.get("strategies")
    if not isinstance(strategies, list) or not strategies:
        raise ValueError("父版本 strategies 为空")
    if implementation.get("strategy_keys") != [item.get("strategy_key") for item in strategies]:
        raise ValueError("子版本 implementation.strategy_keys 与父版本策略不一致")
    modified = implementation.get("modified_files")
    if not isinstance(modified, list) or not modified:
        raise ValueError("子版本 implementation 缺少 modified_files")
    planned_actions = [action for strategy in strategies for action in strategy.get("actions", [])]
    target_files = {action["target"].split("::", 1)[0] for action in planned_actions}
    parent_files = source_file_bytes(parent)
    child_files = source_file_bytes(child)
    actual_modified = {
        path for path in set(parent_files) | set(child_files)
        if parent_files.get(path) != child_files.get(path)
    }
    if set(modified) != actual_modified:
        raise ValueError("implementation.modified_files 与父子源码真实 diff 不一致")
    if not target_files <= actual_modified:
        raise ValueError("真实源码 diff 未覆盖全部 action target")
    actions = implementation.get("actions")
    expected_indices = list(range(1, len(planned_actions) + 1))
    if not isinstance(actions, list) or [item.get("action_index") for item in actions if isinstance(item, dict)] != expected_indices:
        raise ValueError("implementation.actions 未按 action_index 完整关联父策略")
    effect = load_module(
        "kernel_policy_source_effect",
        repo_root() / ".codex/skills/kernel-implementation/scripts/validate_source_effect.py",
    )
    effect.validate(parent, child)


def source_fingerprint(project: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        path
        for folder in ("op_host", "op_kernel")
        for path in (project / folder).rglob("*")
        if path.is_file()
    )
    if not files:
        raise ValueError("op_host/op_kernel 中没有文件")
    for path in files:
        digest.update(path.relative_to(project).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def render_sources(project: Path, folder: str) -> str:
    paths = sorted(
        path for path in (project / folder).rglob("*")
        if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES
    )
    if not paths:
        raise ValueError(f"{folder} 中没有可导出的源码")
    chunks = []
    for path in paths:
        relative = path.relative_to(project).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace").rstrip()
        chunks.append(f"--- FILE: {relative} ---\n{text}")
    return "\n\n".join(chunks)


def latency(performance: dict[str, Any]) -> float:
    report_value = performance.get("kernel_latency_us")
    if not isinstance(report_value, (int, float)) or not math.isfinite(report_value) or report_value <= 0:
        raise ValueError("performance kernel_latency_us 不是有限正数")
    return float(report_value)


def evaluate_candidate(
    parent: Path,
    manifest: dict[str, Any],
    knowledge: dict[str, Any],
    min_reduction: float,
) -> tuple[dict[str, str] | None, dict[str, Any]]:
    match = VERSION_RE.fullmatch(parent.name)
    if match is None:
        raise ValueError("目录名不是 <OperatorName>_<version>")
    operator = match.group("operator")
    version = int(match.group("version"))
    child = parent.with_name(f"{operator}_{version + 1}")
    audit: dict[str, Any] = {"ops": parent.name, "child_ops": child.name, "eligible": False}
    if operator not in manifest:
        raise ValueError(f"manifest 中不存在算子 {operator}")
    if not child.is_dir():
        raise ValueError("对应子版本不存在")

    validate_completed_version(parent, "父版本")
    validate_completed_version(child, "子版本")

    parent_perf = load_json(parent / "performance/performance.json")
    child_perf = load_json(child / "performance/performance.json")
    parent_latency = latency(parent_perf)
    child_latency = latency(child_perf)
    reduction = (parent_latency - child_latency) / parent_latency * 100.0
    audit.update({
        "parent_latency_us": parent_latency,
        "child_latency_us": child_latency,
        "reduction_percent": reduction,
        "threshold_percent": min_reduction,
        "result_kind": "verified_policy" if reduction > min_reduction else "verified_execution_not_training",
    })

    bottleneck = load_json(parent / "bottleneck/bottleneck.json")
    strategy = load_json(parent / "strategy/strategy.json")
    if not bottleneck.get("issues") or not strategy.get("strategies"):
        raise ValueError("父版本 issues 或 strategies 为空")
    validate_parent_reports(parent)
    validate_implementation_link(parent, strategy, child)
    if not reduction > min_reduction:
        audit["exclusion_reason"] = f"latency 下降 {reduction:.12g}% 未严格大于 {min_reduction}%"
        return None, audit

    input_text = "\n\n".join([
        "[PERF]\n" + compact_json(compact_performance(parent_perf)),
        "[HOST]\n" + render_sources(parent, "op_host"),
        "[KERNEL]\n" + render_sources(parent, "op_kernel"),
    ])
    output_text = compact_json(project_policy(bottleneck, strategy))
    sample = {"ops": parent.name, "input": input_text, "output": output_text}
    audit.update({
        "eligible": True,
        "policy_knowledge_sha256": hashlib.sha256(pretty_json(knowledge).encode()).hexdigest(),
        "input_chars": len(input_text),
        "output_chars": len(output_text),
    })
    return sample, audit


def discover(root: Path, requested_ops: set[str], levels: set[str]) -> list[Path]:
    projects = [
        path for path in root.glob("level*/*")
        if path.is_dir() and VERSION_RE.fullmatch(path.name)
        and (not levels or path.parent.name in levels)
    ]
    if requested_ops:
        projects = [path for path in projects if path.name in requested_ops]
        missing = requested_ops - {path.name for path in projects}
        if missing:
            raise SystemExit("未找到 ops：" + ", ".join(sorted(missing)))
    return sorted(projects, key=lambda path: (path.parent.name, path.name))


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", type=Path, default=root / "kernel_workspace/KernelBench910B")
    parser.add_argument("--manifest", type=Path, default=root / "kernel/KernelBench910B/manifest.json")
    parser.add_argument("--output", type=Path, default=root / "datasets/kernel_policy_data.jsonl")
    parser.add_argument("--ops", action="append", default=[])
    parser.add_argument(
        "--level", action="append", choices=("level1", "level2", "level3"), default=[],
        help="仅导出指定 level；可重复传入",
    )
    parser.add_argument("--min-reduction-percent", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not math.isfinite(args.min_reduction_percent) or args.min_reduction_percent < 0:
        raise SystemExit("--min-reduction-percent 必须是有限非负数")
    workspace_root = args.workspace_root.resolve()
    manifest = load_json(args.manifest.resolve())
    knowledge = load_json(skill_root() / "references/policy-knowledge.json")
    validate_policy_knowledge(knowledge)
    projects = discover(workspace_root, set(args.ops), set(args.level))
    samples: list[dict[str, str]] = []
    audit_items: list[dict[str, Any]] = []

    for project in projects:
        try:
            sample, audit = evaluate_candidate(
                project, manifest, knowledge,
                min_reduction=args.min_reduction_percent,
            )
            audit_items.append(audit)
            if sample is None:
                continue
            samples.append(sample)
        except Exception as error:
            audit_items.append({
                "ops": project.name,
                "eligible": False,
                "exclusion_reason": str(error),
            })

    if len({sample["ops"] for sample in samples}) != len(samples):
        raise ValueError("导出结果包含重复 ops")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    audit_path = output.with_suffix(output.suffix + ".audit.json")
    audit_path.write_text(pretty_json({
        "workspace_root": str(workspace_root),
        "output": str(output),
        "minimum_reduction_percent_exclusive": args.min_reduction_percent,
        "candidates": len(projects),
        "exported": len(samples),
        "excluded": len(projects) - len(samples),
        "items": audit_items,
    }), encoding="utf-8")
    print(f"exported={len(samples)} excluded={len(projects) - len(samples)} output={output}")
    print(f"audit={audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
