#!/usr/bin/env python3
"""Run the Level1 optimization queue with one fresh Codex context per operator."""

from __future__ import annotations

import json
import hashlib
import fcntl
import math
import os
import shutil
import subprocess
import sys
import re
import shlex
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/data/lu/ascendc-optim")
QUEUE = ROOT / "kernel_workspace/KernelBench910B/level1_regeneration_queue.json"
RUNS = ROOT / "kernel_workspace/KernelBench910B/level1_regeneration_runs"
LOG = ROOT / "kernel_workspace/KernelBench910B/level1_regeneration_progress.tsv"
LEVEL = ROOT / "kernel_workspace/KernelBench910B/level1"
ARCHIVE_ROOT = ROOT / "train_result/level1_regeneration_archives"
ASCENDC_TMPDIR = Path("/data/lu/ascendc-tmp")
LOCK = ROOT / "kernel_workspace/KernelBench910B/level1_regeneration.lock"
MAX_AGENT_ATTEMPTS = 2
MAX_REPAIRS = 3
TERMINAL_STATES = {
    "completed_two_rounds", "stopped_no_issue", "build_failed",
    "precision_failed", "performance_failed", "strategy_blocked",
    "implementation_blocked", "prepare_failed", "agent_failed", "incomplete",
}


def save(queue: dict) -> None:
    temporary = QUEUE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(queue, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, QUEUE)


def load_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def source_fingerprint(project: Path) -> str | None:
    files = sorted(
        path for folder in ("op_host", "op_kernel")
        for path in (project / folder).rglob("*") if path.is_file()
    )
    if not files:
        return None
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(project).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def precision_gate(project: Path, workspace: dict | None) -> tuple[bool, str]:
    if not workspace or workspace.get("status") not in {"PRECISION_PASS", "PERFORMANCE_DONE"}:
        return False, "workspace status has not passed precision"
    fingerprint = source_fingerprint(project)
    if not fingerprint or workspace.get("source_fingerprint") != fingerprint:
        return False, "workspace source fingerprint mismatch"
    precision = load_json(project / "precision/precision.json")
    if not precision or precision.get("status") != "PASS" or precision.get("exit_code") != 0:
        return False, "precision gate is not PASS"
    if precision.get("source_fingerprint") != fingerprint:
        return False, "precision source fingerprint mismatch"
    return True, "precision durable gate passed"


def performance_gate(project: Path, workspace: dict | None) -> tuple[bool, str]:
    if not workspace or workspace.get("status") != "PERFORMANCE_DONE":
        return False, "workspace status is not PERFORMANCE_DONE"
    passed, reason = precision_gate(project, workspace)
    if not passed:
        return False, reason
    performance = load_json(project / "performance/performance.json")
    required_sections = {
        "task", "pipeline", "arithmetic", "memory", "memory_l0", "memory_ub",
        "l2_cache", "resource_conflict", "per_core", "hardware",
    }
    if not performance or not required_sections <= set(performance):
        return False, "performance report is incomplete"
    required_files = {
        "op_summary_PipeUtilization.csv", "op_summary_ArithmeticUtilization.csv",
        "op_summary_Memory.csv", "op_summary_MemoryL0.csv", "op_summary_MemoryUB.csv",
        "op_summary_L2Cache.csv", "op_summary_ResourceConflictRatio.csv", "per_core_cycles.csv",
    }
    missing = sorted(name for name in required_files if not (project / "performance" / name).is_file())
    if missing:
        return False, "performance artifacts missing: " + ",".join(missing)
    latency = performance.get("kernel_latency_us") if performance else None
    if not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency <= 0:
        return False, "performance latency is not a finite positive number"
    return True, "all durable gates passed"


def bottleneck_gate(project: Path) -> tuple[bool, str]:
    report = load_json(project / "bottleneck/bottleneck.json")
    if not report or set(report) != {"reasoning", "issues"}:
        return False, "bottleneck report is missing or has stale top-level fields"
    if not isinstance(report["reasoning"], list) or not isinstance(report["issues"], list):
        return False, "bottleneck reasoning/issues are not lists"
    if len(report["reasoning"]) != len(report["issues"]):
        return False, "bottleneck reasoning/issues length mismatch"
    validation = subprocess.run(
        [sys.executable, str(ROOT / ".codex/skills/kernel-bottleneck/scripts/validate_report.py"),
         str(project / "bottleneck/bottleneck.json")],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    if validation.returncode != 0:
        return False, "bottleneck validator failed: " + validation.stdout.strip()[-500:]
    return True, "bottleneck durable gate passed"


def completed_round_gate(parent: Path, child: Path) -> tuple[bool, str]:
    parent_workspace = load_json(parent / "workspace.json")
    child_workspace = load_json(child / "workspace.json")
    if not performance_gate(parent, parent_workspace)[0]:
        return False, "parent performance gate failed"
    if not performance_gate(child, child_workspace)[0]:
        return False, "child performance gate failed"
    if not bottleneck_gate(child)[0]:
        return False, "child bottleneck gate failed"
    strategy = parent / "strategy/strategy.json"
    implementation = child / "strategy/implementation.json"
    commands = [
        [sys.executable, str(ROOT / ".codex/skills/kernel-strategy/scripts/validate_strategy.py"),
         "--bottleneck", str(parent / "bottleneck/bottleneck.json"), "--strategy", str(strategy)],
        [sys.executable, str(ROOT / ".codex/skills/kernel-implementation/scripts/validate_implementation.py"),
         "--bottleneck", str(parent / "bottleneck/bottleneck.json"),
         "--strategy", str(strategy), "--implementation", str(implementation),
         "--parent", str(parent), "--project-dir", str(child), "--require-attempts"],
    ]
    for command in commands:
        validation = subprocess.run(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        if validation.returncode != 0:
            return False, "round validator failed: " + validation.stdout.strip()[-500:]
    return True, "round durable gates passed"


def strategy_gate(project: Path) -> tuple[bool, str]:
    command = [
        sys.executable, str(ROOT / ".codex/skills/kernel-strategy/scripts/validate_strategy.py"),
        "--bottleneck", str(project / "bottleneck/bottleneck.json"),
        "--strategy", str(project / "strategy/strategy.json"),
    ]
    result = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    return result.returncode == 0, result.stdout.strip()[-500:]


def implementation_gate(parent: Path, child: Path) -> tuple[bool, str]:
    command = [
        sys.executable,
        str(ROOT / ".codex/skills/kernel-implementation/scripts/validate_implementation.py"),
        "--bottleneck", str(parent / "bottleneck/bottleneck.json"),
        "--strategy", str(parent / "strategy/strategy.json"),
        "--implementation", str(child / "strategy/implementation.json"),
        "--parent", str(parent), "--project-dir", str(child), "--require-attempts",
    ]
    result = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    return result.returncode == 0, result.stdout.strip()[-500:]


def operator_projects(operator: str) -> list[tuple[int, Path]]:
    pattern = re.compile(rf"^{re.escape(operator)}_(\d+)$")
    projects = []
    for path in LEVEL.glob(f"{operator}_*"):
        match = pattern.fullmatch(path.name)
        if path.is_dir() and match:
            projects.append((int(match.group(1)), path))
    return sorted(projects)


def resume_summary(operator: str) -> str:
    """Build the compact durable-state snapshot passed to a content Agent."""
    lines = []
    for version, project in operator_projects(operator):
        workspace = load_json(project / "workspace.json") or {}
        precision_ok = precision_gate(project, workspace)[0]
        performance_ok = performance_gate(project, workspace)[0]
        bottleneck_ok = bottleneck_gate(project)[0] if performance_ok else False
        bottleneck = load_json(project / "bottleneck/bottleneck.json") if bottleneck_ok else None
        issue_count = len(bottleneck.get("issues", [])) if bottleneck else None
        lines.append(
            f"_{version}:status={workspace.get('status', 'MISSING')},"
            f"precision={'PASS' if precision_ok else 'NO'},"
            f"performance={'DONE' if performance_ok else 'NO'},"
            f"bottleneck={'DONE' if bottleneck_ok else 'NO'},"
            f"issues={issue_count if issue_count is not None else 'NA'}"
        )
    return "; ".join(lines) if lines else "no workspace version"


def next_durable_stage(operator: str) -> str:
    """Identify the first stage that can change durable state."""
    projects = operator_projects(operator)
    if not projects:
        return "prepare _0"
    for version, project in projects:
        workspace = load_json(project / "workspace.json") or {}
        if not precision_gate(project, workspace)[0]:
            return f"validate or repair _{version} precision"
        if not performance_gate(project, workspace)[0]:
            return f"collect _{version} formal performance"
        if not bottleneck_gate(project)[0]:
            return f"analyze _{version} bottleneck"
        bottleneck = load_json(project / "bottleneck/bottleneck.json") or {}
        if not bottleneck.get("issues"):
            return f"stop at _{version}: issues=[]"
        if version >= 2:
            return "stop at _2: version limit reached"
        strategy = load_json(project / "strategy/strategy.json")
        if not strategy:
            return f"derive _{version} strategy"
        child = LEVEL / f"{operator}_{version + 1}"
        if not child.exists():
            return f"implement _{version} strategy into _{version + 1}"
    latest = projects[-1][0]
    return f"resume latest _{latest} incomplete transition"


def repair_count(project: Path) -> int:
    implementation = load_json(project / "strategy/implementation.json") or {}
    return sum(1 for attempt in implementation.get("attempts", [])
               if isinstance(attempt, dict) and attempt.get("kind") == "repair")


def failure_summary(project: Path) -> dict:
    """Return a bounded failure record; never inject a complete build log."""
    precision = load_json(project / "precision/precision.json") or {}
    summary = {
        "failure_stage": precision.get("failure_stage"),
        "stage": precision.get("stage"),
        "reason": precision.get("reason") or precision.get("message"),
        "exit_code": precision.get("exit_code"),
    }
    log_value = precision.get("log_path") or precision.get("log")
    if isinstance(log_value, str):
        log_path = Path(log_value)
        if not log_path.is_absolute():
            log_path = project / log_path
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            summary["log_tail"] = lines[-80:]
        except OSError:
            summary["log_tail"] = []
    return {key: value for key, value in summary.items() if value not in (None, [], "")}


def durable_action(operator: str) -> dict:
    """Choose exactly one state-changing action from durable artifacts."""
    projects = operator_projects(operator)
    if not projects:
        return {"kind": "terminal", "state": "prepare_failed", "reason": "missing _0 workspace"}
    version, project = projects[-1]
    workspace = load_json(project / "workspace.json") or {}
    if version > 0:
        parent = LEVEL / f"{operator}_{version - 1}"
        blocking = load_json(project / "strategy/implementation_blocking.json")
        if blocking and blocking.get("reason"):
            return {"kind": "terminal", "state": "implementation_blocked",
                    "reason": f"_{version}: {blocking['reason']}"}
        if not implementation_gate(parent, project)[0]:
            return {"kind": "agent", "stage": "implementation_fix", "version": version,
                    "project": project, "parent": parent}
    if not precision_gate(project, workspace)[0]:
        status = workspace.get("status")
        if status in {"BUILD_FAILED", "PRECISION_FAILED"}:
            precision = load_json(project / "precision/precision.json") or {}
            # A repair changed source after the recorded failure.  The old
            # failure is stale, so rebuild/validate before considering another
            # repair attempt.
            if precision.get("source_fingerprint") != source_fingerprint(project):
                return {"kind": "command", "stage": "precision", "version": version,
                        "project": project}
            repairs = repair_count(project)
            implementation = project / "strategy/implementation.json"
            parent = LEVEL / f"{operator}_{version - 1}"
            if version > 0 and implementation.is_file() and parent.is_dir() and repairs < MAX_REPAIRS:
                return {"kind": "agent", "stage": "repair", "version": version,
                        "project": project, "parent": parent, "repair": repairs + 1}
            return {"kind": "terminal", "state": status.lower(),
                    "reason": f"_{version}={status}; repairs={repairs}/{MAX_REPAIRS}"}
        return {"kind": "command", "stage": "precision", "version": version, "project": project}
    if not performance_gate(project, workspace)[0]:
        if workspace.get("status") == "PERFORMANCE_FAILED":
            return {"kind": "terminal", "state": "performance_failed",
                    "reason": f"_{version}=PERFORMANCE_FAILED"}
        return {"kind": "command", "stage": "performance", "version": version, "project": project}
    if not bottleneck_gate(project)[0]:
        return {"kind": "agent", "stage": "planning", "version": version, "project": project}
    bottleneck = load_json(project / "bottleneck/bottleneck.json") or {}
    if bottleneck.get("issues") == []:
        return {"kind": "terminal", "state": "stopped_no_issue",
                "reason": f"_{version}=PERFORMANCE_DONE,issues=[]"}
    if version >= 2:
        zero = LEVEL / f"{operator}_0"
        one = LEVEL / f"{operator}_1"
        if one.is_dir() and completed_round_gate(zero, one)[0] and completed_round_gate(one, project)[0]:
            return {"kind": "terminal", "state": "completed_two_rounds",
                    "reason": "_2=PERFORMANCE_DONE,bottleneck_reanalyzed"}
        return {"kind": "terminal", "state": "incomplete", "reason": "_2 round validation failed"}
    blocking = load_json(project / "strategy/blocking.json")
    if blocking and blocking.get("reason"):
        return {"kind": "terminal", "state": "strategy_blocked",
                "reason": f"_{version}: {blocking['reason']}"}
    strategy = project / "strategy/strategy.json"
    if not strategy.is_file() or not strategy_gate(project)[0]:
        return {"kind": "agent", "stage": "planning", "version": version, "project": project}
    child = LEVEL / f"{operator}_{version + 1}"
    if child.exists():
        return {"kind": "terminal", "state": "incomplete",
                "reason": f"_{version + 1} exists but was not selected as latest workspace"}
    return {"kind": "agent", "stage": "implementation", "version": version,
            "project": project, "child": child}


def classify_result(operator: str, returncode: int) -> tuple[str, str]:
    """Classify by durable workspace gates, never by agent exit code alone."""
    projects = operator_projects(operator)
    reports = [(version, project, load_json(project / "workspace.json")) for version, project in projects]
    version_two = next(((project, report) for version, project, report in reports if version == 2), None)
    if version_two is not None:
        project, report = version_two
        version_one = next((path for version, path in projects if version == 1), None)
        version_zero = next((path for version, path in projects if version == 0), None)
        if (version_zero is not None and version_one is not None
                and completed_round_gate(version_zero, version_one)[0]
                and completed_round_gate(version_one, project)[0]):
            return "completed_two_rounds", "_2=PERFORMANCE_DONE,bottleneck_reanalyzed"
    for version, project, workspace in reversed(reports):
        if not performance_gate(project, workspace)[0]:
            continue
        bottleneck = load_json(project / "bottleneck/bottleneck.json")
        parent = next((path for candidate, path in projects if candidate == version - 1), None)
        round_ok = version == 0 or (parent is not None and completed_round_gate(parent, project)[0])
        if round_ok and bottleneck_gate(project)[0] and bottleneck.get("issues") == []:
            return "stopped_no_issue", f"_{version}=PERFORMANCE_DONE,issues=[]"
    latest_existing = projects[-1] if projects else None
    for version, project in ([latest_existing] if latest_existing else []):
        blocking = load_json(project / "strategy/blocking.json")
        if blocking and blocking.get("stage") == "strategy" and blocking.get("cause_key") and blocking.get("reason"):
            return "strategy_blocked", f"_{version}: {blocking['cause_key']}: {blocking['reason']}"
        blocking = load_json(project / "strategy/implementation_blocking.json")
        if blocking and blocking.get("stage") == "implementation" and blocking.get("cause_key") and blocking.get("reason"):
            return "implementation_blocked", f"_{version}: {blocking['cause_key']}: {blocking['reason']}"
    latest = next(((version, report) for version, _, report in reversed(reports) if report), None)
    if latest:
        version, workspace = latest
        status = workspace.get("status")
        if status in {"BUILD_FAILED", "PRECISION_FAILED", "PERFORMANCE_FAILED"}:
            return status.lower(), f"_{version}={status}"
    if returncode != 0:
        return "agent_failed", f"codex_returncode={returncode}"
    return "incomplete", "agent exited without a terminal workspace result"


def initialize() -> None:
    """Archive derived decisions/versions, preserve every _0 source and valid baseline report."""
    archive = ARCHIVE_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if archive.exists():
        raise SystemExit(f"archive already exists: {archive}")
    archive.mkdir(parents=True)
    base_projects = sorted(LEVEL.glob("*_0"), key=lambda path: path.name)
    if not base_projects:
        raise SystemExit("no level1 _0 projects")
    for base in base_projects:
        operator = base.name[:-2]
        operator_archive = archive / operator
        moved = False
        for version, derived in operator_projects(operator):
            if version == 0:
                continue
            if derived.exists():
                operator_archive.mkdir(parents=True, exist_ok=True)
                shutil.move(str(derived), str(operator_archive / derived.name))
                moved = True
        for folder in ("bottleneck", "strategy"):
            stale = base / folder
            if stale.exists():
                operator_archive.mkdir(parents=True, exist_ok=True)
                shutil.move(str(stale), str(operator_archive / f"{base.name}_{folder}"))
                moved = True
        if not moved and operator_archive.exists():
            operator_archive.rmdir()
    items = []
    for position, base in enumerate(base_projects, 1):
        workspace = load_json(base / "workspace.json")
        performance_ok, _ = performance_gate(base, workspace)
        precision_ok, _ = precision_gate(base, workspace)
        items.append({
            "position": position,
            "operator": base.name[:-2],
            "state": "pending",
            "baseline": "reuse_performance" if performance_ok else (
                "reuse_precision_collect_performance" if precision_ok else "revalidate_precision_and_performance"
            ),
            "stage_attempts": {},
        })
    queue = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "controller_state_machine_ephemeral_stage_agents_max_two_rounds",
        "archive": str(archive),
        "max_agent_attempts_per_stage": MAX_AGENT_ATTEMPTS,
        "max_repairs_per_version": MAX_REPAIRS,
        "items": items,
    }
    save(queue)
    LOG.write_text("timestamp\tposition\toperator\tstate\treturncode\n")
    print(f"initialized={len(items)} queue={QUEUE} archive={archive}")


def rebuild() -> None:
    """Rebuild scheduling state without moving or overwriting any workspace version."""
    base_projects = sorted(LEVEL.glob("*_0"), key=lambda path: path.name)
    if not base_projects:
        raise SystemExit("no level1 _0 projects")
    items = []
    for position, base in enumerate(base_projects, 1):
        workspace = load_json(base / "workspace.json")
        performance_ok, _ = performance_gate(base, workspace)
        precision_ok, _ = precision_gate(base, workspace)
        versions = [version for version, _ in operator_projects(base.name[:-2])]
        items.append({
            "position": position,
            "operator": base.name[:-2],
            "state": "pending",
            "baseline": "reuse_performance" if performance_ok else (
                "reuse_precision_collect_performance" if precision_ok else "revalidate_precision_and_performance"
            ),
            "existing_versions": versions,
            "stage_attempts": {},
        })
    queue = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "controller_state_machine_ephemeral_stage_agents_max_two_rounds",
        "preserve_existing_versions": True,
        "max_agent_attempts_per_stage": MAX_AGENT_ATTEMPTS,
        "max_repairs_per_version": MAX_REPAIRS,
        "items": items,
    }
    save(queue)
    LOG.write_text("timestamp\tposition\toperator\tstate\treturncode\n")
    print(f"rebuilt={len(items)} queue={QUEUE} existing_versions_preserved=true")


def reconcile() -> None:
    """Reconcile stopped queue bookkeeping from durable artifacts only."""
    queue = json.loads(QUEUE.read_text())
    counts: dict[str, int] = {}
    for item in queue["items"]:
        previous_state = item.get("state")
        state, reason = classify_result(item["operator"], 0)
        if state in {"incomplete", "agent_failed"}:
            if previous_state == "running":
                item["state"] = "pending"
                current_stage = item.get("current_stage")
                if current_stage and current_stage in item.get("stage_attempts", {}):
                    item["stage_attempts"][current_stage] = max(
                        0, item["stage_attempts"][current_stage] - 1
                    )
                elif "attempts" in item:  # migrate queues created by the old runner
                    item["attempts"] = max(0, item.get("attempts", 0) - 1)
                item["result_reason"] = "controller stopped; resume from durable stage"
                item["interrupted_at"] = datetime.now(timezone.utc).isoformat()
        else:
            item["state"] = state
            item["result_reason"] = reason
            item["recovered_at"] = datetime.now(timezone.utc).isoformat()
        counts[item["state"]] = counts.get(item["state"], 0) + 1
    save(queue)
    print("reconciled=" + json.dumps(counts, ensure_ascii=False, sort_keys=True))


def stage_input(operator: str, action: dict, attempt: int) -> Path:
    """Persist the small hand-off contract for one ephemeral Agent."""
    project = action["project"]
    payload = {
        "operator": operator,
        "stage": action["stage"],
        "version": action["version"],
        "project": str(project),
        "host_files": [str(path) for path in sorted((project / "op_host").rglob("*")) if path.is_file()],
        "kernel_files": [str(path) for path in sorted((project / "op_kernel").rglob("*")) if path.is_file()],
    }
    if action["stage"] == "implementation":
        payload.update({
            "parent": str(project),
            "child": str(action["child"]),
            "bottleneck": str(project / "bottleneck/bottleneck.json"),
            "strategy": str(project / "strategy/strategy.json"),
        })
    elif action["stage"] in {"repair", "implementation_fix"}:
        payload.update({
            "parent": str(action["parent"]),
            "bottleneck": str(action["parent"] / "bottleneck/bottleneck.json"),
            "strategy": str(action["parent"] / "strategy/strategy.json"),
            "implementation": str(project / "strategy/implementation.json"),
        })
        if action["stage"] == "repair":
            payload.update({
                "repair": action["repair"],
                "repairs_remaining_after_this": MAX_REPAIRS - action["repair"],
                "failure": failure_summary(project),
            })
    directory = RUNS / f"{operator}.stages"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"v{action['version']}.{action['stage']}.attempt{attempt}.input.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return path


def agent_prompt(input_path: Path, action: dict) -> str:
    stage = action["stage"]
    common = f"""只处理阶段输入 {input_path}。仓库为 {ROOT}，严格遵守 AGENTS.md。
阶段输入是控制器生成的权威最小状态；只读取其中列出的当前源码与报告，不遍历其他算子、历史版本、完整性能 CSV 或构建日志。不要运行 precision、performance、msprof 或 reference；控制器负责全部门禁。不要读取 validator 源码。
"""
    if stage == "planning":
        return common + """这是 planning 短上下文。依次使用 kernel-bottleneck 和 kernel-strategy：完整读取这两个 SKILL.md；源码只读取一次。先从当前 Host/Kernel 源码生成并校验 bottleneck/bottleneck.json；若 issues=[] 立即结束。否则按已确定 cause 生成并校验 strategy/strategy.json。禁止修改源码，禁止生成 coverage.json 或 source_model.json。完成文件落盘后立即结束。"""
    if stage == "implementation":
        return common + """这是 implementation 短上下文。只使用 kernel-implementation，读取其 SKILL.md 和 actions 实际涉及的知识章节。创建阶段输入指定的 child，实施全部冻结 actions，只修改 child/op_host 与 child/op_kernel，写入并校验 implementation.json 和源码效果。禁止运行精度、性能或重新分析瓶颈；完成源码实施记录后立即结束。"""
    if stage == "implementation_fix":
        return common + """这是 implementation 产物修正短上下文。只使用 kernel-implementation；子版本已存在，不得创建或覆盖版本。运行 implementation validator，根据最后一条错误修正 implementation.json 或补全尚未实施的冻结 action，再运行源码效果校验。不得新增优化、运行精度或性能。校验通过后立即结束。"""
    return common + """这是 repair 短上下文。只使用 kernel-implementation；根据阶段输入中的 bounded failure、当前源码 diff 和冻结 actions 做一次最小修复。不得改变或扩展 strategy，不得读取完整日志。更新 implementation.json，新增且仅新增本次 repair attempt，校验 implementation 与源码效果；禁止运行精度、性能或继续下一次修复。完成后立即结束。"""


def run_agent_stage(item: dict, action: dict) -> int:
    stage_attempts = item.setdefault("stage_attempts", {})
    key = f"v{action['version']}:{action['stage']}"
    attempt = stage_attempts.get(key, 0) + 1
    stage_attempts[key] = attempt
    input_path = stage_input(item["operator"], action, attempt)
    stem = f"{item['position']:03d}_{item['operator']}.v{action['version']}.{action['stage']}.attempt{attempt}"
    run_log = RUNS / f"{stem}.jsonl"
    final_message = RUNS / f"{stem}.final.txt"
    with run_log.open("w") as output:
        child_env = os.environ.copy()
        child_env["ASCENDC_TMPDIR"] = str(ASCENDC_TMPDIR)
        child_env["TMPDIR"] = str(ASCENDC_TMPDIR)
        result = subprocess.run([
            "codex", "exec", "--json", "--ephemeral",
            "--dangerously-bypass-approvals-and-sandbox",
            "-C", str(ROOT), "-o", str(final_message), agent_prompt(input_path, action),
        ], cwd=ROOT, env=child_env, stdout=output, stderr=subprocess.STDOUT, text=True)
    return result.returncode


def run_gate(item: dict, action: dict) -> int:
    operator = item["operator"]
    project = action["project"]
    stage = action["stage"]
    script = ROOT / f".codex/skills/kernel-{stage}/scripts" / (
        "validate_precision.py" if stage == "precision" else "collect_performance.py"
    )
    command = (
        "source /data/lu/activate_evokernel.sh && "
        f"python {shlex.quote(str(script))} {shlex.quote(operator)} "
        f"--project-dir {shlex.quote(str(project))}"
    )
    if stage == "performance":
        command += " --device 0"
    stem = f"{item['position']:03d}_{operator}.v{action['version']}.{stage}"
    log_path = RUNS / f"{stem}.log"
    with log_path.open("w") as output:
        result = subprocess.run(["bash", "-lc", command], cwd=ROOT,
                                stdout=output, stderr=subprocess.STDOUT, text=True)
    return result.returncode


def run_state_machine(queue: dict) -> None:
    """Advance each operator one durable stage at a time."""
    queue.pop("acceptance", None)
    queue["mode"] = "controller_state_machine_ephemeral_stage_agents_max_two_rounds"
    queue["max_agent_attempts_per_stage"] = MAX_AGENT_ATTEMPTS
    queue["max_repairs_per_version"] = MAX_REPAIRS
    index = 0
    while index < len(queue["items"]):
        item = queue["items"][index]
        if item["state"] in TERMINAL_STATES:
            index += 1
            continue
        action = durable_action(item["operator"])
        if action["kind"] == "terminal":
            item["state"] = action["state"]
            item["result_reason"] = action["reason"]
            item["finished_at"] = datetime.now(timezone.utc).isoformat()
            save(queue)
            index += 1
            continue
        item["state"] = "running"
        item["current_stage"] = f"v{action['version']}:{action['stage']}"
        item["started_at"] = datetime.now(timezone.utc).isoformat()
        save(queue)
        if action["kind"] == "command":
            returncode = run_gate(item, action)
        else:
            key = item["current_stage"]
            if item.get("stage_attempts", {}).get(key, 0) >= MAX_AGENT_ATTEMPTS:
                item["state"] = "agent_failed"
                item["result_reason"] = f"{key} Agent attempt limit exhausted"
                save(queue)
                index += 1
                continue
            returncode = run_agent_stage(item, action)
        item["returncode"] = returncode
        item["last_stage_finished_at"] = datetime.now(timezone.utc).isoformat()
        item["state"] = "pending"
        if returncode != 0:
            if action["kind"] == "agent":
                key = item["current_stage"]
                if item.get("stage_attempts", {}).get(key, 0) >= MAX_AGENT_ATTEMPTS:
                    item["state"] = "agent_failed"
                    item["result_reason"] = f"{key} failed with returncode={returncode}"
            else:
                item["result_reason"] = f"{item['current_stage']} gate returncode={returncode}"
        save(queue)
        with LOG.open("a") as progress:
            progress.write(
                f"{item['last_stage_finished_at']}\t{item['position']}\t{item['operator']}\t"
                f"{item['current_stage']}:{item['state']}\t{returncode}\n"
            )
        if item["state"] in TERMINAL_STATES:
            index += 1


def main() -> None:
    if "--initialize" in sys.argv:
        initialize()
        return
    if "--rebuild" in sys.argv:
        rebuild()
        return
    if "--reconcile" in sys.argv:
        reconcile()
        return
    if "--show-stages" in sys.argv:
        queue = json.loads(QUEUE.read_text())
        for item in queue["items"]:
            action = durable_action(item["operator"])
            compact = {key: str(value) if isinstance(value, Path) else value
                       for key, value in action.items()}
            print(json.dumps({"operator": item["operator"], **compact}, ensure_ascii=False))
        return
    RUNS.mkdir(parents=True, exist_ok=True)
    ASCENDC_TMPDIR.mkdir(parents=True, exist_ok=True)
    if not os.access(ASCENDC_TMPDIR, os.W_OK):
        raise SystemExit(f"temporary directory is not writable: {ASCENDC_TMPDIR}")
    lock_handle = LOCK.open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"level1 queue is already running: {LOCK}")
    queue = json.loads(QUEUE.read_text())
    run_state_machine(queue)


if __name__ == "__main__":
    main()
