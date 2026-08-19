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
VERSION_RE = re.compile(r"^(?P<operator>.+)_(?P<version>[0-3])$")


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


def canonical_policy_contract(knowledge: dict[str, Any]) -> dict[str, Any]:
    """Build the shared prompt contract from canonical bottleneck/strategy sources."""
    root = repo_root()
    bottleneck = load_module(
        "kernel_bottleneck_training_contract",
        root / ".codex/skills/kernel-bottleneck/scripts/validate_report.py",
    )
    strategy = load_module(
        "kernel_strategy_training_contract",
        root / ".codex/skills/kernel-strategy/scripts/derive_strategy.py",
    )
    slots = load_json(root / ".codex/skills/kernel-strategy/references/operation-slots.json")
    metric_sources = {key: sources for key, sources in bottleneck.METRIC_SOURCES.items()}
    cause_rules = [
        [
            cause, parent, evidence, strategy.RULES[cause],
            strategy.OPERATIONS[strategy.RULES[cause]],
        ]
        for cause, (parent, evidence) in bottleneck.CAUSES.items()
    ]
    operation_slots = [
        [operation, fields["required"], fields["optional"]]
        for operation, fields in slots.items()
    ]
    return {
        "decision_order": knowledge["decision_order"],
        "metric_evidence_sources": metric_sources,
        "cause_rule_fields": [
            "cause_key", "bottleneck_key", "required_evidence_key", "strategy_key", "operation",
        ],
        "cause_rules": cause_rules,
        "operation_slot_fields": ["operation", "required", "optional"],
        "operation_slots": operation_slots,
        "action_fields": knowledge["action_fields"],
    }


def render_system_prompt(contract: dict[str, Any]) -> str:
    return "\n".join([
        "你是 AscendC Kernel 性能策略教师。根据输入中的 OPERATOR、正式 PERF 与完整 HOST/KERNEL 源码，直接找出全部确定的源码问题并生成可实施策略。",
        "按可消除热路径成本排序，最多输出 3 个互不重复的问题；每个 cause 必须由直接源码证据支持，性能数据只补充影响。",
        "严格按 cause_rules 完成 cause→strategy→operation 映射；actions 必须满足 operation_slots，最多 6 个，并共享无冲突的任务、tiling、容量、地址、dtype、对齐、tail、同步和 ABI 设计。",
        "只输出一个 JSON 对象，形状为 {\"strategies\":[{\"bottleneck_key\":string,\"cause_key\":string,\"evidence\":array,\"strategy_key\":string,\"reasoning\":array,\"actions\":array}]}。不要输出 Markdown、候选方案、额外字段或管理元数据。",
        "契约中的 cause_rules 与 operation_slots 按各自 fields 表头解释。以下 JSON 是唯一固定契约：",
        compact_json(contract),
        "",
    ])


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"无法加载模块：{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_policy_knowledge(knowledge: dict[str, Any]) -> None:
    root = repo_root()
    bottleneck = load_module(
        "kernel_bottleneck_contract",
        root / ".codex/skills/kernel-bottleneck/scripts/validate_report.py",
    )
    strategy = load_module(
        "kernel_strategy_contract",
        root / ".codex/skills/kernel-strategy/scripts/derive_strategy.py",
    )
    if set(knowledge) != {"decision_order", "action_fields"}:
        raise ValueError("policy knowledge 只保留 decision_order 和 action_fields")
    decision_order = knowledge.get("decision_order")
    if not isinstance(decision_order, list) or not decision_order or any(
        not isinstance(value, str) or not value.strip() for value in decision_order
    ):
        raise ValueError("policy knowledge 的 decision_order 非法")
    if knowledge.get("action_fields") != ["target", "operation", "edits", "constraints"]:
        raise ValueError("policy knowledge 的 action_fields 非法")
    if set(bottleneck.CAUSES) != set(strategy.RULES):
        raise ValueError("bottleneck cause 与 strategy 映射未完整对齐")
    if set(strategy.RULES.values()) != set(strategy.OPERATIONS):
        raise ValueError("strategy 与 operation 映射未完整对齐")
    slots = load_json(root / ".codex/skills/kernel-strategy/references/operation-slots.json")
    if set(slots) != set(strategy.OPERATIONS.values()) or any(
        not isinstance(value, dict) or set(value) != {"required", "optional"}
        or not isinstance(value["required"], list) or not 2 <= len(value["required"]) <= 4
        or not isinstance(value["optional"], list)
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


def validate_completed_version(
    project: Path, label: str, operator: str, version: int
) -> dict[str, Any]:
    workspace = load_json(project / "workspace.json")
    precision = load_json(project / "precision/precision.json")
    fingerprint = source_fingerprint(project)
    if workspace.get("operator") != operator or workspace.get("version") != version:
        raise ValueError(f"{label} workspace 算子或版本身份不匹配")
    if workspace.get("status") != "PERFORMANCE_DONE":
        raise ValueError(f"{label} workspace 不是 PERFORMANCE_DONE")
    if workspace.get("source_fingerprint") != fingerprint:
        raise ValueError(f"{label} workspace 源码指纹不匹配")
    if precision.get("status") != "PASS" or precision.get("exit_code") != 0:
        raise ValueError(f"{label} precision 未 PASS")
    if precision.get("operator") != operator:
        raise ValueError(f"{label} precision 算子身份不匹配")
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
    strategy_keys = [item.get("strategy_key") for item in strategies]
    if implementation.get("strategy_keys") != strategy_keys:
        raise ValueError("子版本 implementation.strategy_keys 与父版本策略不一致")
    strategy_actions = [action for item in strategies for action in item.get("actions", [])]
    modified = implementation.get("modified_files")
    if not isinstance(modified, list) or not modified:
        raise ValueError("子版本 implementation 缺少 modified_files")
    target_files = {action["target"].split("::", 1)[0] for action in strategy_actions}
    parent_files = source_file_bytes(parent)
    child_files = source_file_bytes(child)
    actual_modified = {
        path for path in set(parent_files) | set(child_files)
        if parent_files.get(path) != child_files.get(path)
    }
    if set(modified) != actual_modified:
        raise ValueError("implementation.modified_files 与父子源码真实 diff 不一致")
    if target_files != actual_modified:
        raise ValueError("真实源码 diff 必须恰好覆盖全部 action target，且不得混入未声明文件")
    actions = implementation.get("actions")
    expected_indices = list(range(1, len(strategy_actions) + 1))
    if not isinstance(actions, list) or [item.get("action_index") for item in actions if isinstance(item, dict)] != expected_indices:
        raise ValueError("implementation.actions 未按 action_index 完整关联父策略")


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


def operator_json(operator: str, entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "operator": operator,
        "level": entry.get("level"),
        "function": entry.get("function"),
        "parameters": entry.get("parameters"),
    }


def merge_policy(bottleneck: dict[str, Any], strategy: dict[str, Any]) -> dict[str, Any]:
    issues = bottleneck.get("issues")
    strategies = strategy.get("strategies")
    bottleneck_reasoning = bottleneck.get("reasoning")
    strategy_reasoning = strategy.get("reasoning")
    if not all(isinstance(value, list) for value in (
        issues, strategies, bottleneck_reasoning, strategy_reasoning,
    )) or not (len(issues) == len(strategies) == len(bottleneck_reasoning) == len(strategy_reasoning)):
        raise ValueError("bottleneck/strategy 条目与 reasoning 无法一一合并")
    merged = []
    for index, (issue, planned) in enumerate(zip(issues, strategies)):
        cause_key = issue.get("bottleneck", {}).get("cause_key")
        if cause_key != planned.get("cause_key"):
            raise ValueError(f"第 {index + 1} 项 cause_key 未对齐")
        merged.append({
            "bottleneck_key": issue["bottleneck"]["bottleneck_key"],
            "cause_key": cause_key,
            "evidence": issue["evidence"],
            "strategy_key": planned["strategy_key"],
            "reasoning": [bottleneck_reasoning[index], strategy_reasoning[index]],
            "actions": planned["actions"],
        })
    return {"strategies": merged}


def evaluate_candidate(
    parent: Path,
    manifest: dict[str, Any],
    system_prompt: str,
    min_reduction: float,
) -> tuple[dict[str, str] | None, dict[str, Any]]:
    match = VERSION_RE.fullmatch(parent.name)
    if match is None:
        raise ValueError("目录名不是可导出的 <OperatorName>_[0-3]")
    operator = match.group("operator")
    version = int(match.group("version"))
    child = parent.with_name(f"{operator}_{version + 1}")
    audit: dict[str, Any] = {"ops": parent.name, "child_ops": child.name, "eligible": False}
    if operator not in manifest:
        raise ValueError(f"manifest 中不存在算子 {operator}")
    if not child.is_dir():
        raise ValueError("对应子版本不存在")

    parent_workspace = validate_completed_version(parent, "父版本", operator, version)
    child_workspace = validate_completed_version(child, "子版本", operator, version + 1)
    if child_workspace.get("parent_version") != version:
        raise ValueError("子版本 parent_version 未指向紧邻父版本")
    if child_workspace.get("source_project") != parent_workspace.get("source_project"):
        raise ValueError("父子版本 source_project 不一致")
    if child_workspace.get("vendor") != parent_workspace.get("vendor"):
        raise ValueError("父子版本 vendor 不一致")

    parent_perf = load_json(parent / "performance/performance.json")
    child_perf = load_json(child / "performance/performance.json")
    if parent_perf.get("operator") != operator or child_perf.get("operator") != operator:
        raise ValueError("父子版本 performance 算子身份不匹配")
    parent_latency = latency(parent_perf)
    child_latency = latency(child_perf)
    reduction = (parent_latency - child_latency) / parent_latency * 100.0
    audit.update({
        "parent_latency_us": parent_latency,
        "child_latency_us": child_latency,
        "reduction_percent": reduction,
        "threshold_percent": min_reduction,
    })

    bottleneck = load_json(parent / "bottleneck/bottleneck.json")
    strategy = load_json(parent / "strategy/strategy.json")
    if not bottleneck.get("issues") or not strategy.get("strategies"):
        raise ValueError("父版本 bottleneck 或 strategy 为空")
    validate_parent_reports(parent)
    validate_implementation_link(parent, strategy, child)
    if not reduction > min_reduction:
        audit["exclusion_reason"] = f"latency 下降 {reduction:.12g}% 未严格大于 {min_reduction}%"
        return None, audit

    input_text = "\n\n".join([
        "[OPERATOR]\n" + compact_json(operator_json(operator, manifest[operator])),
        "[PERF]\n" + compact_json(compact_performance(parent_perf)),
        "[HOST]\n" + render_sources(parent, "op_host"),
        "[KERNEL]\n" + render_sources(parent, "op_kernel"),
    ])
    output_text = compact_json(merge_policy(bottleneck, strategy))
    sample = {
        "system_prompt": system_prompt,
        "input": input_text,
        "output": output_text,
        "ops": parent.name,
    }
    audit.update({
        "eligible": True,
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
    parser.add_argument("--system-prompt", type=Path)
    parser.add_argument("--ops", action="append", default=[])
    parser.add_argument(
        "--level", action="append", choices=("level1", "level2", "level3"), default=[],
        help="仅导出指定 level；可重复传入",
    )
    parser.add_argument("--min-reduction-percent", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not math.isfinite(args.min_reduction_percent) or args.min_reduction_percent < 1.0:
        raise SystemExit("--min-reduction-percent 必须是大于等于 1.0 的有限数")
    workspace_root = args.workspace_root.resolve()
    manifest = load_json(args.manifest.resolve())
    knowledge = load_json(skill_root() / "references/policy-knowledge.json")
    validate_policy_knowledge(knowledge)
    contract = canonical_policy_contract(knowledge)
    system_prompt = render_system_prompt(contract)
    projects = discover(workspace_root, set(args.ops), set(args.level))
    samples: list[dict[str, str]] = []
    audit_items: list[dict[str, Any]] = []

    for project in projects:
        try:
            sample, audit = evaluate_candidate(
                project, manifest, system_prompt,
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
    system_prompt_path = (
        args.system_prompt.resolve() if args.system_prompt
        else output.parent / "kernel_policy_system_prompt.txt"
    )
    system_prompt_path.parent.mkdir(parents=True, exist_ok=True)
    system_prompt_path.write_text(system_prompt, encoding="utf-8")
    with output.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    audit_path = output.with_suffix(output.suffix + ".audit.json")
    root = repo_root()
    audit_report = {
        "workspace_root": str(workspace_root),
        "output": str(output),
        "system_prompt": str(system_prompt_path),
        "dataset_sha256": sha256_file(output),
        "system_prompt_sha256": sha256_file(system_prompt_path),
        "canonical_sources_sha256": {
            "cause_taxonomy": sha256_file(root / ".codex/skills/kernel-bottleneck/references/cause-taxonomy.json"),
            "cause_strategy_mapping": sha256_file(root / ".codex/skills/kernel-strategy/scripts/derive_strategy.py"),
            "operation_slots": sha256_file(root / ".codex/skills/kernel-strategy/references/operation-slots.json"),
        },
        "minimum_reduction_percent_exclusive": args.min_reduction_percent,
        "candidates": len(projects),
        "exported": len(samples),
        "excluded": len(projects) - len(samples),
        "items": audit_items,
    }
    audit_path.write_text(pretty_json(audit_report), encoding="utf-8")
    print(f"exported={len(samples)} excluded={len(projects) - len(samples)} output={output}")
    print(f"system_prompt={system_prompt_path}")
    print(f"audit={audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
