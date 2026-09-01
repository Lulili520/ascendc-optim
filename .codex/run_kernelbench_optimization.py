#!/usr/bin/env python3
"""Serial KernelBench optimizer: source strategy -> implementation -> precision -> latency."""

from __future__ import annotations

import ast
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import shlex
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/data/lu/ascendc-optim")
sys.path.insert(0, str(ROOT / ".codex"))
from suite_config import SUITES, load_suite_manifest, suite_paths


def consume_suite_argument() -> str:
    if "--suite" not in sys.argv:
        return "KernelBench910B"
    index = sys.argv.index("--suite")
    if index + 1 >= len(sys.argv):
        raise SystemExit("--suite requires a value")
    suite = sys.argv[index + 1]
    del sys.argv[index:index + 2]
    if suite not in SUITES:
        raise SystemExit(f"unsupported suite: {suite}")
    return suite


SUITE = consume_suite_argument()
SOURCE_PROJECTS, WORKSPACE, REFERENCES = suite_paths(ROOT, SUITE)
QUEUE = WORKSPACE / "optimization_queue.json"
RUNS = WORKSPACE / "optimization_runs"
LOG = WORKSPACE / "optimization_progress.tsv"
LOCK = WORKSPACE / "optimization.lock"
DEVICE_VALIDATION_LOCK = ROOT / "kernel_workspace/device_validation.lock"
HARDWARE_ROOT = ROOT / "kernel_workspace/KernelBench910B/hardware"
ARCHIVE_ROOT = ROOT / f"train_result/{SUITE.lower()}_optimization_archives"
REJECTION_ROOT = ROOT / "train_result/kernelbench_strategy_rejections"
ASCENDC_TMPDIR = Path("/data/lu/ascendc-tmp")
VENDORS_ROOT = Path("/usr/local/Ascend/cann-9.0.0/opp/vendors")
SDK_INCLUDE_ROOTS = (
    Path("/usr/local/Ascend/cann-9.0.0/x86_64-linux/asc/include"),
    Path("/usr/local/Ascend/cann-9.0.0/x86_64-linux/asc/impl"),
)
MAX_ROUNDS = 4
MAX_CONCURRENCY = 4
MAX_AGENT_CONCURRENCY = 2
MAX_AGENT_ATTEMPTS = 3
MAX_IMPLEMENTATION_FIXES = 1
MAX_AGENT_TRANSPORT_RETRIES = 8
MAX_IMPLEMENTATION_REPAIRS = 3
MAX_COMMAND_ATTEMPTS = 2
MAX_REPLANS = 3
MIN_IMPROVEMENT = 5.0
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp"}
TERMINAL = {
    "completed_max_rounds", "stopped_no_strategy", "stopped_no_improvement",
    "build_failed", "precision_failed", "performance_failed", "agent_failed",
    "adapter_missing",
    "implementation_blocked", "prepare_failed", "environment_failed", "stopped_by_user",
}
ENVIRONMENT_FAILURE_TOKENS = (
    "no space left on device", "not enough space left", "device or resource busy", "resource temporarily unavailable",
    "npu-smi: command not found", "msprof: command not found", "no npu device",
    # Legacy precision runs treated best-effort reference-cache write failures
    # as numerical failures.  Keep these exact serialization signatures
    # recoverable so their real comparison can be rerun with the fixed writer.
    "basic_ios::clear: iostream error", "unexpected pos ",
)
AGENT_TRANSPORT_TOKENS = (
    "stream disconnected before completion",
    "request timed out",
    "failed to refresh available models",
    "error sending request for url",
    "transport channel closed",
)
AGENT_TRANSPORT_CODE = 75
STOP_REQUESTED = threading.Event()
PROCESS_LOCK = threading.Lock()
ACTIVE_PROCESSES: set[subprocess.Popen] = set()


def managed_run(args: list[str], *, cwd: Path, env: dict | None = None,
                stdout=None, stderr=None, text: bool = True) -> int:
    process = subprocess.Popen(args, cwd=cwd, env=env, stdout=stdout, stderr=stderr,
                               text=text, start_new_session=True)
    with PROCESS_LOCK:
        ACTIVE_PROCESSES.add(process)
    try:
        return process.wait()
    finally:
        with PROCESS_LOCK:
            ACTIVE_PROCESSES.discard(process)


def request_stop(signum=None, frame=None) -> None:
    STOP_REQUESTED.set()
    with PROCESS_LOCK:
        processes = list(ACTIVE_PROCESSES)
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    def force_kill() -> None:
        time.sleep(5)
        with PROCESS_LOCK:
            remaining = list(ACTIVE_PROCESSES)
        for process in remaining:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
    threading.Thread(target=force_kill, daemon=True).start()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def save(queue: dict) -> None:
    temporary = QUEUE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(queue, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, QUEUE)


def fingerprint(project: Path) -> str | None:
    files = sorted(path for folder in ("op_host", "op_kernel")
                   for path in (project / folder).rglob("*") if path.is_file())
    if not files:
        return None
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(project).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def source_files(project: Path, folder: str) -> list[Path]:
    return sorted(path for path in (project / folder).rglob("*")
                  if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES)


def static_value(node: ast.AST, values: dict[str, object]) -> object:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in values:
        return values[node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        return [static_value(item, values) for item in node.elts]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = static_value(node.operand, values)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp):
        left, right = static_value(node.left, values), static_value(node.right, values)
        operations = {ast.Add: lambda: left + right, ast.Sub: lambda: left - right,
                      ast.Mult: lambda: left * right, ast.FloorDiv: lambda: left // right,
                      ast.Pow: lambda: left ** right}
        operation = operations.get(type(node.op))
        if operation:
            return operation()
    raise ValueError(f"unsupported static expression: {ast.dump(node, include_attributes=False)}")


def tensor_metadata(node: ast.AST, values: dict[str, object]) -> dict | None:
    if isinstance(node, ast.Name) and isinstance(values.get(node.id), dict):
        return values[node.id]
    if isinstance(node, ast.BinOp):
        candidates = [value for value in (tensor_metadata(node.left, values),
                                           tensor_metadata(node.right, values)) if value is not None]
        return max(candidates, key=lambda value: len(value["shape"]), default=None)
    if not isinstance(node, ast.Call):
        return None
    function = node.func.attr if isinstance(node.func, ast.Attribute) else None
    if function in {"triu", "tril", "transpose", "contiguous", "clone"} and node.args:
        return tensor_metadata(node.args[0], values)
    if function not in {"rand", "randn", "zeros", "ones", "empty", "full", "randint"}:
        return None
    arguments = []
    for argument in node.args:
        value = static_value(argument.value, values) if isinstance(argument, ast.Starred) else static_value(argument, values)
        arguments.extend(value if isinstance(argument, ast.Starred) else [value])
    keywords = {keyword.arg: static_value(keyword.value, values) for keyword in node.keywords
                if keyword.arg and keyword.arg != "dtype"}
    if function == "randint":
        shape = keywords.get("size", arguments[-1] if arguments else None)
        dtype = "torch.int64"
    elif function == "full":
        shape = arguments[0] if arguments else keywords.get("size")
        dtype = "torch.float32"
    else:
        shape = arguments[0] if len(arguments) == 1 and isinstance(arguments[0], list) else arguments
        dtype = "torch.float32"
    dtype_node = next((keyword.value for keyword in node.keywords if keyword.arg == "dtype"), None)
    if isinstance(dtype_node, ast.Attribute):
        dtype = f"torch.{dtype_node.attr}"
    if not isinstance(shape, list) or not all(isinstance(value, int) and value >= 0 for value in shape):
        return None
    return {"shape": shape, "dtype": dtype}


def reference_input_metadata(entry: dict) -> dict:
    candidates = entry.get("reference_candidates") or []
    if len(candidates) != 1:
        return {"inputs": [], "init_args": []}
    path = REFERENCES / candidates[0]
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        values: dict[str, object] = {}
        functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                try:
                    values[node.targets[0].id] = static_value(node.value, values)
                except (ValueError, TypeError, ZeroDivisionError):
                    pass
        function = functions.get("get_inputs")
        if function is None:
            return {"inputs": [], "init_args": []}
        local = dict(values)
        returned = None
        for statement in function.body:
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
                descriptor = tensor_metadata(statement.value, local)
                if descriptor is not None:
                    local[statement.targets[0].id] = descriptor
            elif isinstance(statement, ast.Return):
                returned = statement.value
        elements = returned.elts if isinstance(returned, (ast.List, ast.Tuple)) else []
        parameters = entry.get("parameters") or []
        inputs = []
        for index, element in enumerate(elements):
            descriptor = tensor_metadata(element, local)
            if descriptor is not None:
                inputs.append({"name": parameters[index] if index < len(parameters) else f"input_{index}", **descriptor})
        init_args = []
        init_function = functions.get("get_init_inputs")
        if init_function:
            init_return = next((statement.value for statement in init_function.body
                                if isinstance(statement, ast.Return)), None)
            if init_return is not None:
                try:
                    value = static_value(init_return, values)
                    init_args = value if isinstance(value, list) else [value]
                except (ValueError, TypeError, ZeroDivisionError):
                    pass
        return {"inputs": inputs, "init_args": init_args}
    except (OSError, SyntaxError):
        return {"inputs": [], "init_args": []}


def markdown_sections(path: Path) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^#{2,3}\s+(?:profile:\s*)?([^#]+?)\s*$", line)
        if match:
            current = match.group(1).strip()
            sections.setdefault(current, [])
        elif current is not None:
            sections[current].append(line)
    return {name: "\n".join(lines).strip() for name, lines in sections.items()}


def planning_contract(project: Path, profile: str) -> dict:
    source = "\n".join(path.read_text(encoding="utf-8", errors="replace")
                       for folder in ("op_host", "op_kernel") for path in source_files(project, folder))
    core_names = ["transfer", "vector", "lifetime", "tiling", "static-work", "pattern-solver", "abi"]
    if (re.search(r"cum(?:sum|prod)|prefix|scan", source, re.I)
            or (re.search(r"carry|running", source, re.I)
                and re.search(r"for\s*\([^)]*(?:\+\+|--)", source))):
        core_names.append("prefix-scan")
    if re.search(r"reduce|argmax|argmin|softmax|norm", source, re.I):
        core_names.extend(["reduction", "residency"])
    if re.search(r"pool|window|reuse|cache", source, re.I):
        core_names.append("residency")
    has_cube = any(token in source for token in ("Matmul<", "Mmad(", "IterateAll("))
    # A Scalar/Vector contraction needs the Cube contract during planning in
    # order to decide whether a first Cube path can be fully closed.  Inject it
    # only for a multiplication-accumulation body with loop structure; ordinary
    # elementwise Mul must not pay this knowledge cost.
    has_scalar_contraction = bool(
        re.search(r"\bfor\s*\(", source)
        and re.search(r"(?:\+=|=\s*[^;]+\+)\s*[^;\n]*\*[^;\n]*;", source)
    )
    if has_cube or has_scalar_contraction:
        core_names.append("cube")
    core = markdown_sections(ROOT / ".codex/kernel-knowledge/implementation-core.md")
    architecture = markdown_sections(ROOT / ".codex/kernel-knowledge/architecture-knowledge.md")
    architecture_names = ["transfer", "vector", "lifetime"]
    return {
        "profile": profile,
        "core": {name: core[name] for name in dict.fromkeys(core_names) if name in core},
        "architecture": {name: architecture[name] for name in architecture_names if name in architecture},
    }


def precision_ok(project: Path) -> bool:
    report = load_json(project / "precision/precision.json") or {}
    current = fingerprint(project)
    workspace = load_json(project / "workspace.json") or {}
    vendor = workspace.get("vendor")
    vendor_env = (VENDORS_ROOT / vendor / "bin/set_env.bash"
                  if isinstance(vendor, str) and vendor else None)
    return bool(current and report.get("status") == "PASS" and report.get("exit_code") == 0
                and report.get("source_fingerprint") == current
                and vendor_env is not None and vendor_env.is_file())


def latency_report(project: Path) -> dict:
    return load_json(project / "performance/latency.json") or {}


def latency(project: Path) -> float | None:
    report = latency_report(project)
    current = fingerprint(project)
    value = report.get("task_duration_us")
    if (current and report.get("source_fingerprint") == current
            and isinstance(value, (int, float)) and math.isfinite(value) and value > 0):
        return float(value)
    return None


def strategy_ok(project: Path) -> bool:
    strategy = load_json(project / "strategy/strategy.json")
    if not strategy or "strategy" not in strategy:
        return False
    commands = [[
        sys.executable, str(ROOT / ".codex/skills/kernel-strategy/scripts/validate_strategy.py"),
        "--strategy", str(project / "strategy/strategy.json"),
        "--project-dir", str(project),
    ]]
    if strategy["strategy"] is not None:
        commands.insert(0, [
            sys.executable, str(ROOT / ".codex/skills/kernel-strategy/scripts/validate_planning.py"),
            "--planning", str(project / "strategy/planning.json"),
            "--project-dir", str(project),
        ])
    return all(subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True).returncode == 0
               for command in commands)


def strategy_value(project: Path) -> dict | None | object:
    report = load_json(project / "strategy/strategy.json")
    if report is None or "strategy" not in report:
        return ...
    return report["strategy"]


def implementation_ok(parent: Path, child: Path) -> bool:
    result = subprocess.run([
        sys.executable,
        str(ROOT / ".codex/skills/kernel-implementation/scripts/validate_implementation.py"),
        "--strategy", str(parent / "strategy/strategy.json"),
        "--implementation", str(child / "strategy/implementation.json"),
        "--parent", str(parent), "--project-dir", str(child), "--require-attempts",
    ], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return result.returncode == 0


def projects(operator: str) -> list[tuple[int, Path]]:
    pattern = re.compile(rf"^{re.escape(operator)}_(\d+)$")
    found = []
    for path in WORKSPACE.glob(f"{operator}_*"):
        match = pattern.fullmatch(path.name)
        if path.is_dir() and match:
            found.append((int(match.group(1)), path))
    return sorted(found)


def best_before(operator: str, version: int) -> tuple[int, float] | None:
    values = [(candidate, latency(path)) for candidate, path in projects(operator)
              if candidate < version]
    valid = [(candidate, value) for candidate, value in values if value is not None]
    return min(valid, key=lambda item: item[1]) if valid else None


def accepted(project: Path) -> bool:
    report = latency_report(project)
    return latency(project) is not None and report.get("accepted") is True


def failure(project: Path) -> dict:
    report = load_json(project / "precision/precision.json") or {}
    result = {key: report.get(key) for key in
              ("failure_stage", "stage", "reason", "message", "exit_code")}
    log = report.get("log_path") or report.get("log")
    if isinstance(log, str):
        path = Path(log) if Path(log).is_absolute() else project / log
        try:
            result["log_tail"] = path.read_text(errors="replace").splitlines()[-60:]
        except OSError:
            pass
    return {key: value for key, value in result.items() if value not in (None, "", [])}


def replan_request(project: Path) -> dict | None:
    value = load_json(project / "strategy/replan.json")
    if not value or set(value) != {"reason", "evidence", "required_design_changes"}:
        return None
    if (not isinstance(value["reason"], str) or not value["reason"].strip()
            or not all(isinstance(value.get(field), list) and value[field]
                       and all(isinstance(item, str) and item.strip() for item in value[field])
                       for field in ("evidence", "required_design_changes"))):
        return None
    return value


def write_failure_replan(project: Path) -> None:
    if replan_request(project) is not None:
        return
    detail = failure(project)
    lines = detail.get("log_tail") if isinstance(detail.get("log_tail"), list) else []
    tokens = ("error", "failed", "invalid", "unalign", "mismatch", "exception", "precision")
    evidence = []
    for line in lines:
        value = str(line).strip()
        if value and any(token in value.lower() for token in tokens) and value not in evidence:
            evidence.append(value[:500])
        if len(evidence) == 8:
            break
    if not evidence:
        evidence = [str(detail.get("reason") or detail.get("message") or "child构建或精度门禁失败")[:500]]
    joined = "\n".join(evidence).lower()
    strategy = load_json(project / "strategy/strategy.json") or {}
    strategy_value = strategy.get("strategy")
    kinds = set(strategy_value.get("kinds", [])) if isinstance(strategy_value, dict) else set()
    required = []
    if any(token in joined for token in ("binarygetfunctionbyentry", "rtsfuncgetbyentry", "funcentry=0")):
        mechanism = "kernel entry/tiling key/code channel/vendor metadata 未生成一致的可部署二进制入口"
    elif "精度不通过" in joined:
        if kinds & {"parallelize", "cube"}:
            mechanism = "核到独立输出tile的所有权、blockIdx解码或A/B/C offset与原数学索引不等价"
            required.append("重新方案必须从shape推导独立输出tile数，逐个闭合blockIdx到输出区间、A/B/C offset和tail；禁止只修改SetDim/SetBlockDim")
        else:
            mechanism = "数学索引、padding、tail、mask 或布局与 reference 语义不等价"
            required.append("重新方案必须指出失败输出对应的索引、padding、tail、mask或布局机制，并用有效范围公式替代该机制")
    elif any(token in joined for token in ("aivec", "mte", "507034", "507035", "507015")):
        mechanism = "GM/UB 地址、容量、对齐、搬运参数或跨流水生命周期非法"
    else:
        mechanism = "直接失败证据指向的已实施源码机制"
    path = project / "strategy/replan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "reason": f"冻结策略实施后的{detail.get('failure_stage') or detail.get('stage') or 'validation'}门禁失败",
        "evidence": evidence,
        "required_design_changes": required + [
            f"已证伪机制：{mechanism}",
            "仅禁止直接证据已否定的机制；其他优化方向仍按父源码全量扫描并闭合任务、容量、地址、dtype、对齐、tail、同步和ABI",
        ],
    }, ensure_ascii=False, indent=2) + "\n")


def reset_rejected_strategy(item: dict) -> bool:
    versions = projects(item["operator"])
    if not versions or versions[-1][0] == 0:
        return False
    version, child = versions[-1]
    request = replan_request(child)
    if request is None:
        return False
    parent = WORKSPACE / f"{item['operator']}_{version - 1}"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = REJECTION_ROOT / stamp / item["operator"]
    destination.mkdir(parents=True, exist_ok=False)
    shutil.move(str(child), str(destination / child.name))
    parent_strategy = parent / "strategy"
    if parent_strategy.exists():
        shutil.move(str(parent_strategy), str(destination / f"{parent.name}_strategy"))
    feedback = RUNS / f"{item['operator']}.replan-feedback.json"
    feedback.write_text(json.dumps({"rejected_version": version, **request}, ensure_ascii=False, indent=2) + "\n")
    item.setdefault("replan_history", []).append({"version": version, "archive": str(destination),
                                                    "feedback": str(feedback), "at": now()})
    item["stage_attempts"] = {}
    item["agent_thread_ids"] = {}
    item.update(state="pending", current_stage=f"v{version - 1}:replan", returncode=0,
                result_reason="rejected frozen strategy returned to planning")
    return True


def environment_failure(project: Path) -> bool:
    report = load_json(project / "precision/precision.json") or {}
    values = [report.get(key) for key in ("reason", "message", "stage", "failure_stage")]
    log_names = {"precision/opp_build.log", "precision/precision.log", "precision/precision_run.log"}
    reported_log = report.get("log")
    if isinstance(reported_log, str) and reported_log:
        log_names.add(reported_log)
    for name in sorted(log_names):
        path = project / name
        if path.is_file():
            values.append(path.read_text(encoding="utf-8", errors="replace"))
    text = "\n".join(str(value).lower() for value in values if value is not None)
    return any(token in text for token in ENVIRONMENT_FAILURE_TOKENS)


def recoverable_performance_failure(project: Path) -> bool:
    """Retry known controller/tooling failures after their implementation is fixed."""
    workspace = load_json(project / "workspace.json") or {}
    reason = str((workspace.get("performance") or {}).get("reason") or "")
    return any(token in reason for token in (
        "未生成 PipeUtilization CSV",
        "too many values to unpack (expected 5)",
    ))


def architecture_profile() -> tuple[str, list[str]]:
    # All suites execute on the same physical 910B device.  Keep one measured
    # hardware identity instead of duplicating (and potentially drifting)
    # architecture facts in each suite workspace.
    device_path = HARDWARE_ROOT / "device_0.json"
    device = load_json(device_path) or {}
    soc = (device.get("identity") or {}).get("soc")
    index = load_json(ROOT / ".codex/kernel-knowledge/architecture-index.json") or {}
    entry = index.get(soc) if isinstance(soc, str) else None
    profile = entry.get("profile") if isinstance(entry, dict) else None
    if not isinstance(profile, str) or not profile:
        raise RuntimeError(f"无法从 {device_path} 的精确 SoC 身份解析架构 profile")
    return profile, [
        str(ROOT / ".codex/kernel-knowledge/implementation-core.md"),
        str(ROOT / ".codex/kernel-knowledge/architecture-knowledge.md"),
    ]


def next_action(operator: str) -> dict:
    versions = projects(operator)
    if not versions:
        return {"kind": "terminal", "state": "prepare_failed", "reason": "missing _0"}
    version, project = versions[-1]
    workspace = load_json(project / "workspace.json") or {}
    durable_best = (workspace.get("status") == "PERFORMANCE_DONE"
                    and precision_ok(project) and latency(project) is not None
                    and accepted(project))
    if version > 0 and not durable_best:
        parent = WORKSPACE / f"{operator}_{version - 1}"
        if not strategy_ok(parent):
            return {"kind": "agent", "stage": "planning", "version": version - 1,
                    "project": parent}
        if not implementation_ok(parent, project):
            return {"kind": "agent", "stage": "implementation_fix", "version": version,
                    "project": project, "parent": parent}
    if not precision_ok(project):
        status = workspace.get("status")
        report = load_json(project / "precision/precision.json") or {}
        if status in {"ADAPTER_MISSING", "BUILD_FAILED", "PRECISION_FAILED", "ENVIRONMENT_FAILED"} and report.get("source_fingerprint") == fingerprint(project):
            if status == "ENVIRONMENT_FAILED":
                return {"kind": "command", "stage": "precision", "version": version,
                        "project": project, "environment_retry": True}
            if environment_failure(project):
                return {"kind": "command", "stage": "precision", "version": version,
                        "project": project, "environment_retry": True}
            if version > 0:
                if status == "BUILD_FAILED":
                    return {"kind": "agent", "stage": "compile_fix", "version": version,
                            "project": project,
                            "parent": WORKSPACE / f"{operator}_{version - 1}"}
                write_failure_replan(project)
                return {"kind": "replan", "stage": "replan", "version": version,
                        "project": project, "parent": WORKSPACE / f"{operator}_{version - 1}"}
            return {"kind": "terminal", "state": status.lower(),
                    "reason": f"_{version} {status}"}
        return {"kind": "command", "stage": "precision", "version": version, "project": project}
    if latency(project) is None:
        if workspace.get("status") == "PERFORMANCE_FAILED":
            if recoverable_performance_failure(project):
                return {"kind": "command", "stage": "performance", "version": version,
                        "project": project, "environment_retry": True}
            return {"kind": "terminal", "state": "performance_failed", "reason": f"_{version} performance failed"}
        return {"kind": "command", "stage": "performance", "version": version, "project": project}
    if version > 0 and not accepted(project):
        report = latency_report(project)
        return {"kind": "terminal", "state": "stopped_no_improvement",
                "reason": f"_{version} improvement={report.get('improvement_percent')}% <= {MIN_IMPROVEMENT}%"}
    if not strategy_ok(project):
        return {"kind": "agent", "stage": "planning", "version": version, "project": project}
    strategy = strategy_value(project)
    if strategy is None:
        return {"kind": "terminal", "state": "stopped_no_strategy",
                "reason": f"_{version}: no currently closed source strategy"}
    if version >= MAX_ROUNDS:
        return {"kind": "terminal", "state": "completed_max_rounds",
                "reason": f"_{version}: four accepted optimization rounds complete"}
    child = WORKSPACE / f"{operator}_{version + 1}"
    return {"kind": "agent", "stage": "implementation", "version": version,
            "project": project, "child": child}


def initial_implementation(strategy: dict) -> dict:
    value = strategy["strategy"]
    files = sorted({target.split("::", 1)[0] for target in value["targets"]})
    return {
        "strategy_kinds": value["kinds"], "summary": ["待实施。"],
        "modified_files": files,
        "attempts": [{"attempt": 1, "kind": "initial", "trigger": None,
                      "summary": "待实施。", "modified_files": files}],
    }


def prepare_child(parent: Path, child: Path, operator: str) -> None:
    if child.exists():
        raise RuntimeError(f"child already exists: {child}")
    strategy = load_json(parent / "strategy/strategy.json")
    if not strategy or strategy.get("strategy") is None:
        raise RuntimeError("parent strategy missing")
    shutil.copytree(parent, child, symlinks=True, ignore=shutil.ignore_patterns(
        "precision", "performance", "bottleneck", "strategy", "workspace.json",
        "build", "build_out", "CMakeFiles", "*.log"))
    (child / "strategy").mkdir()
    shutil.copy2(parent / "strategy/strategy.json", child / "strategy/strategy.json")
    (child / "strategy/implementation.json").write_text(
        json.dumps(initial_implementation(strategy), ensure_ascii=False, indent=2) + "\n")
    parent_ws = load_json(parent / "workspace.json") or {}
    version = int(child.name.rsplit("_", 1)[1])
    workspace = {"operator": operator, "version": version,
                 "source_project": parent_ws.get("source_project"),
                 "parent_version": version - 1, "vendor": parent_ws.get("vendor"),
                 "status": "PREPARED", "source_fingerprint": fingerprint(child),
                 "precision": None, "performance": None,
                 "strategy": {"status": "IMPLEMENTING", "result": "strategy/strategy.json"},
                 "updated_at": now()}
    (child / "workspace.json").write_text(json.dumps(workspace, ensure_ascii=False, indent=2) + "\n")


def implementation_knowledge(strategy: dict, include_diagnosis: bool = False) -> list[str]:
    _, refs = architecture_profile()
    if include_diagnosis:
        refs.append(str(ROOT / ".codex/kernel-knowledge/implementation-diagnosis.md"))
    return refs


def stage_input(operator: str, action: dict, attempt: int) -> Path:
    project = action["project"]
    payload = {"operator": operator, "stage": action["stage"],
               "version": action["version"], "project": str(project)}
    if action["stage"] == "planning":
        manifest = load_suite_manifest(ROOT, SUITE)
        entry = manifest.get(operator) or {}
        precision = load_json(project / "precision/precision.json") or {}
        metadata = {"inputs": precision.get("input_metadata") or [],
                    "init_args": precision.get("init_args") or []}
        if not metadata["inputs"]:
            metadata = reference_input_metadata(entry)
        profile, _ = architecture_profile()
        payload.update({
            "op": {"function": entry.get("function"), "parameters": entry.get("parameters"),
                   "inputs": metadata["inputs"], "init_args": metadata["init_args"],
                   "output_shape": precision.get("output_shape"),
                   "output_dtype": precision.get("output_dtype")},
            "host_files": [str(path) for path in source_files(project, "op_host")],
            "kernel_files": [str(path) for path in source_files(project, "op_kernel")],
            "device": str(HARDWARE_ROOT / "device_0.json"),
            "architecture_profile": profile,
            "source_fingerprint": fingerprint(project),
            "knowledge_contract": planning_contract(project, profile),
            "strategy_output_contract": {
                "planning_fields": ["source_fingerprint", "pattern", "engine",
                    "available_cores", "used_cores", "shape_model", "task_mapping",
                    "parameters", "buffers", "transfers", "compute", "work", "proofs"],
                "patterns": ["elementwise", "prefix_scan", "reduction", "norm_softmax", "window_pooling",
                    "scatter_transposed", "existing_cube", "fused_network"],
                "engines": ["aiv", "aic", "mixed"],
                "text_array_fields": ["shape_model", "parameters", "buffers", "transfers",
                    "compute", "proofs"],
                "work_keys": ["gm_bytes", "dma_bursts", "vector_repeat_blocks", "cube_ops",
                    "scalar_iterations", "task_lifecycles", "state_initializations",
                    "sync_events", "writeback_bursts"],
                "strategy_fields": ["kinds", "evidence", "reasoning", "targets",
                    "changes", "guards"],
                "reasoning_labels": ["任务", "现状", "问题", "策略", "推导", "边界"],
            },
            "validation_commands": [
                f"python {ROOT / '.codex/skills/kernel-strategy/scripts/validate_planning.py'} --planning {project / 'strategy/planning.json'} --project-dir {project}",
                f"python {ROOT / '.codex/skills/kernel-strategy/scripts/validate_strategy.py'} --strategy {project / 'strategy/strategy.json'} --project-dir {project}",
            ],
            "output": str(project / "strategy/strategy.json"),
            "planning_output": str(project / "strategy/planning.json"),
        })
        feedback = RUNS / f"{operator}.replan-feedback.json"
        if feedback.is_file():
            payload["replan_feedback"] = load_json(feedback)
    else:
        parent = project if action["stage"] == "implementation" else action["parent"]
        child = action.get("child", project)
        strategy = load_json(parent / "strategy/strategy.json") or {}
        payload.update({"parent": str(parent), "child": str(child),
                        "strategy": str(parent / "strategy/strategy.json"),
                        "planning": str(parent / "strategy/planning.json"),
                        "implementation": str(child / "strategy/implementation.json"),
                        "sdk_include_roots": [str(path) for path in SDK_INCLUDE_ROOTS
                                              if path.is_dir()],
                        "knowledge": implementation_knowledge(
                            strategy, action["stage"] == "compile_fix")})
        if action["stage"] == "compile_fix":
            payload["failure"] = failure(child)
    directory = RUNS / f"{operator}.stages"
    directory.mkdir(parents=True, exist_ok=True)
    label = action["stage"]
    path = directory / f"v{action['version']}.{label}.attempt{attempt}.input.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return path


def prompt(input_path: Path, action: dict) -> str:
    common = (f"只处理 {input_path}，遵守 {ROOT / 'AGENTS.md'}；使用绝对路径；"
              "不读取其他算子、历史结果或reference，不运行precision、performance或msprof。")
    if action["stage"] == "planning":
        return common + ("完整读取 kernel-strategy SKILL.md、其直接引用的方法、input列出的全部源码和"
                         "knowledge_contract；一次闭合全部确定问题。若有replan_feedback，不得重复已证伪设计。"
                         "按skill生成并校验planning.json与strategy.json；不修改源码。")
    if action["stage"] == "implementation":
        return common + ("完整读取 kernel-implementation SKILL.md，只实施冻结 strategy。child 和记录骨架"
                         "已经创建；读取 input 中 knowledge，完成源码与 implementation.json 并校验。")
    if action["stage"] == "implementation_fix":
        return common + "按 kernel-implementation 修正冻结策略的缺失实施或记录，不增加新优化。"
    if action["stage"] == "compile_fix":
        return common + ("完整读取 kernel-implementation SKILL.md、input中的冻结strategy、planning、"
                         "failure和knowledge。依据编译器直接证据及implementation-diagnosis，只在原action"
                         "范围内修复API签名、符号作用域、字段类型、转换或对象闭环等落地错误；不得新增、"
                         "删除或重选优化方向。更新implementation.json的repair attempt并运行validator，"
                         "不运行构建、precision或performance。")
    return common + "按 kernel-implementation 修正冻结策略的缺失实施或记录，不增加新优化。"


def thread_id(path: Path) -> str | None:
    for line in path.read_text(errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "thread.started" and isinstance(event.get("thread_id"), str):
            return event["thread_id"]
    return None


def run_agent(item: dict, action: dict) -> int:
    key = f"v{action['version']}:{action['stage']}"
    attempts = item.setdefault("stage_attempts", {})
    attempt = attempts.get(key, 0) + 1
    attempts[key] = attempt
    if action["stage"] == "implementation":
        prepare_child(action["project"], action["child"], item["operator"])
    input_path = stage_input(item["operator"], action, attempt)
    stem = f"{item['position']:03d}_{item['operator']}.{key}.attempt{attempt}"
    run_log, final = RUNS / f"{stem}.jsonl", RUNS / f"{stem}.final.txt"
    args = ["codex", "exec", "--json", "--dangerously-bypass-approvals-and-sandbox", "-o", str(final)]
    if action["stage"] == "planning":
        context_key = f"planning:v{action['version']}"
    else:
        context_version = action["version"] + (1 if action["stage"] == "implementation" else 0)
        context_key = f"implementation:v{context_version}"
    contexts = item.setdefault("agent_thread_ids", {})
    if contexts.get(context_key):
        args = ["codex", "exec", "resume", "--json", "--dangerously-bypass-approvals-and-sandbox",
                "-o", str(final), contexts[context_key]]
    else:
        args += ["-C", str(ROOT)]
    args.append(prompt(input_path, action))
    env = os.environ.copy()
    env.update({"ASCENDC_TMPDIR": str(ASCENDC_TMPDIR), "TMPDIR": str(ASCENDC_TMPDIR)})
    with run_log.open("w") as output:
        code = managed_run(args, cwd=ROOT, env=env, stdout=output,
                           stderr=subprocess.STDOUT, text=True)
    if not contexts.get(context_key):
        contexts[context_key] = thread_id(run_log)
    log_text = run_log.read_text(encoding="utf-8", errors="replace").lower()
    if code != 0 and any(token in log_text for token in AGENT_TRANSPORT_TOKENS):
        # Transport failures are not failures of the generated strategy or
        # implementation.  Preserve a separate retry budget and let the next
        # invocation reuse the same content-attempt number with a fresh thread.
        attempts[key] = max(0, attempts.get(key, 1) - 1)
        transport = item.setdefault("agent_transport_attempts", {})
        transport[key] = transport.get(key, 0) + 1
        contexts.pop(context_key, None)
        time.sleep(min(120, 10 * (2 ** min(transport[key] - 1, 4))))
        return AGENT_TRANSPORT_CODE
    if code == 0:
        item.setdefault("agent_transport_attempts", {}).pop(key, None)
    return code


def run_command(item: dict, action: dict) -> int:
    script = ROOT / f".codex/skills/kernel-{action['stage']}/scripts" / (
        "validate_precision.py" if action["stage"] == "precision" else "collect_performance.py")
    command = ("source /data/lu/activate_evokernel.sh && "
               f"python {shlex.quote(str(script))} {shlex.quote(item['operator'])} "
               f"--project-dir {shlex.quote(str(action['project']))} --suite {shlex.quote(SUITE)}")
    if action["stage"] == "performance":
        command += " --device 0"
    key = f"v{action['version']}:{action['stage']}"
    attempt = item.get("stage_attempts", {}).get(key, 1)
    log = RUNS / f"{item['position']:03d}_{item['operator']}.v{action['version']}.{action['stage']}.attempt{attempt}.log"
    DEVICE_VALIDATION_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with DEVICE_VALIDATION_LOCK.open("a+") as device_lock, log.open("w") as output:
        # Planning/implementation remain independently concurrent per suite,
        # while OPP installation, precision execution and profiling share the
        # single physical device and therefore must be globally serialized.
        fcntl.flock(device_lock, fcntl.LOCK_EX)
        try:
            return managed_run(["bash", "-lc", command], cwd=ROOT, stdout=output,
                               stderr=subprocess.STDOUT, text=True)
        finally:
            fcntl.flock(device_lock, fcntl.LOCK_UN)


def cleanup_cache(operator: str) -> None:
    directory = ASCENDC_TMPDIR / "kernel_precision_reference_cache"
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", operator)
    if directory.is_dir():
        for path in directory.glob(f"{safe}.*.pt*"):
            path.unlink(missing_ok=True)


def finish_item(queue: dict, item: dict, state: str, reason: str) -> None:
    item.update(state=state, result_reason=reason, finished_at=now())
    cleanup_cache(item["operator"])
    save(queue)
    # Formal training data is stable only after this operator's optimization
    # chain has reached a terminal state.  The exporter filters out every
    # non-terminal operator recorded in the queue.
    export_policy_data()


def export_policy_data() -> int:
    output = ROOT / "datasets" / ("kernel_policy_data.jsonl" if SUITE == "KernelBench910B"
                                  else f"{SUITE.lower()}_policy_data.jsonl")
    result = subprocess.run([
        sys.executable,
        str(ROOT / ".codex/skills/kernel-policy-data/scripts/export_policy_data.py"),
        "--suite", SUITE,
        "--output", str(output),
    ], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log = RUNS / "policy_data_export.log"
    with log.open("a") as output:
        output.write(f"{now()}\treturncode={result.returncode}\n{result.stdout}\n")
    return result.returncode


def execute_stage(item: dict, action: dict) -> int:
    return run_agent(item, action) if action["kind"] == "agent" else run_command(item, action)


def stage_key(action: dict) -> str:
    suffix = action["stage"]
    if action["kind"] == "command":
        current = fingerprint(action["project"])
        suffix += ":" + (current[:12] if current else "no-source")
    return f"v{action['version']}:{suffix}"


def run_queue(queue: dict) -> None:
    active: dict = {}
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as pool:
        while True:
            scheduler_progress = False
            if STOP_REQUESTED.is_set() and not active:
                save(queue)
                return
            active_items = {id(details[0]) for details in active.values()}
            device_validation_busy = any(
                details[1].get("kind") == "command" for details in active.values()
            )
            active_agent_count = sum(
                details[1].get("kind") == "agent" for details in active.values()
            )
            for item in queue["items"]:
                if STOP_REQUESTED.is_set():
                    break
                if len(active) >= MAX_CONCURRENCY:
                    break
                if id(item) in active_items:
                    continue
                if (item.get("state") not in TERMINAL
                        and len(item.get("replan_history", [])) >= MAX_REPLANS):
                    finish_item(
                        queue, item, "implementation_blocked",
                        f"replan limit {MAX_REPLANS}: repeated child validation failure",
                    )
                    scheduler_progress = True
                    continue
                if reset_rejected_strategy(item):
                    save(queue)
                    scheduler_progress = True
                    continue
                if item.get("state") in TERMINAL:
                    continue
                action = next_action(item["operator"])
                if action["kind"] == "replan":
                    if not reset_rejected_strategy(item):
                        finish_item(queue, item, "agent_failed", "invalid replan handoff")
                    else:
                        save(queue)
                    scheduler_progress = True
                    continue
                if action["kind"] == "terminal":
                    finish_item(queue, item, action["state"], action["reason"])
                    scheduler_progress = True
                    continue
                if action["kind"] == "command" and device_validation_busy:
                    continue
                if action["kind"] == "agent" and active_agent_count >= MAX_AGENT_CONCURRENCY:
                    continue
                key = stage_key(action)
                if action["kind"] == "agent":
                    attempts = item.setdefault("stage_attempts", {})
                    limit = (MAX_IMPLEMENTATION_REPAIRS if action["stage"] == "compile_fix" else
                             MAX_IMPLEMENTATION_FIXES if action["stage"] == "implementation_fix" else
                             MAX_AGENT_ATTEMPTS)
                    if attempts.get(key, 0) >= limit:
                        if action["stage"] == "compile_fix":
                            write_failure_replan(action["project"])
                            if reset_rejected_strategy(item):
                                save(queue)
                                continue
                            finish_item(queue, item, "implementation_blocked",
                                        f"{key} repair limit {limit}")
                        elif action["stage"] == "implementation_fix":
                            child = action["project"]
                            replan = child / "strategy/replan.json"
                            if not replan.exists():
                                replan.write_text(json.dumps({
                                    "reason": "冻结策略与实施源码校验无法在一次受限修复内闭合",
                                    "evidence": [f"{key} validation remained invalid after {limit} implementation fix"],
                                    "required_design_changes": [
                                        "回到父源码重新统一策略kind、planning静态工作与目标源码结构",
                                        "不得重复已被实施校验否定的同一目标结构",
                                    ],
                                }, ensure_ascii=False, indent=2) + "\n")
                            if reset_rejected_strategy(item):
                                save(queue)
                                continue
                            finish_item(queue, item, "implementation_blocked",
                                        f"{key} fix limit {limit}")
                        else:
                            finish_item(queue, item, "agent_failed", f"{key} attempt limit")
                        continue
                if action["kind"] == "command":
                    attempts = item.setdefault("stage_attempts", {})
                    if attempts.get(key, 0) >= MAX_COMMAND_ATTEMPTS:
                        state = ("environment_failed" if action.get("environment_retry") else
                                 ("precision_failed" if action["stage"] == "precision" else "performance_failed"))
                        finish_item(queue, item, state, f"{key} command attempt limit")
                        continue
                    attempts[key] = attempts.get(key, 0) + 1
                item.update(state="running", current_stage=key, started_at=now())
                save(queue)
                future = pool.submit(execute_stage, item, action)
                active[future] = (item, action, key)
                scheduler_progress = True
                active_items.add(id(item))
                if action["kind"] == "command":
                    device_validation_busy = True
                else:
                    active_agent_count += 1
            if not active:
                if all(item.get("state") in TERMINAL for item in queue["items"]):
                    break
                # A replan handoff or terminal transition can deliberately
                # rewrite durable artifacts without submitting a Future.  Run
                # one more scheduler pass so next_action observes that new
                # state instead of treating valid progress as a deadlock.
                if scheduler_progress:
                    continue
                raise RuntimeError("queue has pending items but no schedulable stage")
            completed, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in completed:
                item, action, key = active.pop(future)
                try:
                    code = future.result()
                except Exception as error:
                    code = 1
                    item["controller_error"] = str(error)
                item.update(state="pending", returncode=code, last_stage_finished_at=now())
                if code == AGENT_TRANSPORT_CODE:
                    failures = item.get("agent_transport_attempts", {}).get(key, 0)
                    if failures >= MAX_AGENT_TRANSPORT_RETRIES:
                        finish_item(
                            queue, item, "agent_failed",
                            f"{key} transport retry limit {MAX_AGENT_TRANSPORT_RETRIES}",
                        )
                save(queue)
                with LOG.open("a") as output:
                    output.write(f"{now()}\t{item['position']}\t{item['operator']}\t{key}\t{code}\n")
    if not STOP_REQUESTED.is_set():
        export_policy_data()


def source_bases() -> list[Path]:
    operators = sorted(path.name for path in SOURCE_PROJECTS.iterdir() if path.is_dir())
    bases = [WORKSPACE / f"{operator}_0" for operator in operators]
    missing = [path.name for path in bases if not path.is_dir()]
    if missing:
        raise SystemExit(f"missing {SUITE} _0 projects: " + ", ".join(missing))
    return bases


def queue_items() -> list[dict]:
    bases = source_bases()
    if not bases:
        raise SystemExit(f"no {SUITE} source projects")
    return [{"position": index, "operator": path.name[:-2], "state": "pending",
             "stage_attempts": {}} for index, path in enumerate(bases, 1)]


def initialize_one(operator: str) -> None:
    base = WORKSPACE / f"{operator}_0"
    source = SOURCE_PROJECTS / operator
    if not source.is_dir() or not base.is_dir():
        raise SystemExit(f"operator 或 _0 不存在：{operator}")
    archive = ARCHIVE_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive.mkdir(parents=True, exist_ok=False)
    for version, path in projects(operator):
        if version > 0:
            shutil.move(str(path), str(archive / path.name))
    for folder in ("strategy", "bottleneck"):
        path = base / folder
        if path.exists():
            shutil.move(str(path), str(archive / f"{base.name}_{folder}"))
    item = {"position": 1, "operator": operator, "state": "pending", "stage_attempts": {}}
    save({"generated_at": now(), "mode": "single_operator_validation",
          "archive": str(archive), "max_rounds": MAX_ROUNDS,
          "concurrency": 1, "min_improvement_percent": MIN_IMPROVEMENT,
          "items": [item]})
    LOG.write_text("timestamp\tposition\toperator\tstage\treturncode\n")
    print(f"initialized=1 operator={operator} queue={QUEUE} archive={archive}")


def initialize_selected(raw: str) -> None:
    operators = list(dict.fromkeys(value.strip() for value in raw.split(",") if value.strip()))
    if not operators:
        raise SystemExit("--initialize-ops requires a comma-separated operator list")
    unknown = [operator for operator in operators
               if not (SOURCE_PROJECTS / operator).is_dir() or not (WORKSPACE / f"{operator}_0").is_dir()]
    if unknown:
        raise SystemExit("operator 或 _0 不存在：" + ", ".join(unknown))
    archive = ARCHIVE_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive.mkdir(parents=True, exist_ok=False)
    for operator in operators:
        operator_archive = archive / operator
        for version, path in projects(operator):
            if version > 0:
                operator_archive.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(operator_archive / path.name))
        base = WORKSPACE / f"{operator}_0"
        for folder in ("strategy", "bottleneck"):
            path = base / folder
            if path.exists():
                operator_archive.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(operator_archive / f"{base.name}_{folder}"))
        feedback = RUNS / f"{operator}.replan-feedback.json"
        if feedback.exists():
            operator_archive.mkdir(parents=True, exist_ok=True)
            shutil.move(str(feedback), str(operator_archive / feedback.name))
    items = [{"position": index, "operator": operator, "state": "pending", "stage_attempts": {}}
             for index, operator in enumerate(operators, 1)]
    save({"generated_at": now(), "mode": "selected_source_strategy_four_rounds",
          "archive": str(archive), "old_versions_archived": True, "max_rounds": MAX_ROUNDS,
          "concurrency": MAX_CONCURRENCY, "min_improvement_percent": MIN_IMPROVEMENT,
          "items": items})
    LOG.write_text("timestamp\tposition\toperator\tstage\treturncode\n")
    print(f"initialized={len(items)} queue={QUEUE} archive={archive}")


def reopen_selected(raw: str) -> None:
    """Re-plan selected operators from their latest durable best version."""
    operators = list(dict.fromkeys(value.strip() for value in raw.split(",") if value.strip()))
    if not operators:
        raise SystemExit("--reopen-ops requires a comma-separated operator list")
    unknown = [operator for operator in operators
               if not (SOURCE_PROJECTS / operator).is_dir()
               or not projects(operator)]
    if unknown:
        raise SystemExit("operator 或工作版本不存在：" + ", ".join(unknown))
    archive = ARCHIVE_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive.mkdir(parents=True, exist_ok=False)
    for operator in operators:
        durable = projects(operator)
        completed = [(version, project) for version, project in durable
                     if latency(project) is not None
                     and (version == 0 or accepted(project))]
        if not completed:
            raise SystemExit(f"没有可继续优化的正式最佳版本：{operator}")
        version, latest = min(completed, key=lambda item: latency(item[1]))
        operator_archive = archive / operator
        for later_version, path in durable:
            if later_version <= version:
                continue
            operator_archive.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(operator_archive / path.name))
        for name in ("strategy/planning.json", "strategy/strategy.json",
                     "strategy/terminal_policy.json", "bottleneck"):
            path = latest / name
            if path.exists():
                operator_archive.mkdir(parents=True, exist_ok=True)
                destination = operator_archive / f"{latest.name}_{name.replace('/', '_')}"
                shutil.move(str(path), str(destination))
        feedback = RUNS / f"{operator}.replan-feedback.json"
        if feedback.exists():
            operator_archive.mkdir(parents=True, exist_ok=True)
            shutil.move(str(feedback), str(operator_archive / feedback.name))
    items = [{"position": index, "operator": operator, "state": "pending",
              "stage_attempts": {}} for index, operator in enumerate(operators, 1)]
    save({"generated_at": now(), "mode": "reopen_latest_best_source_strategy",
          "archive": str(archive), "preserve_existing_versions": True,
          "max_rounds": MAX_ROUNDS, "concurrency": MAX_CONCURRENCY,
          "min_improvement_percent": MIN_IMPROVEMENT, "items": items})
    LOG.write_text("timestamp\tposition\toperator\tstage\treturncode\n")
    print(f"reopened={len(items)} queue={QUEUE} archive={archive}")


def initialize() -> None:
    archive = ARCHIVE_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive.mkdir(parents=True, exist_ok=False)
    for base in source_bases():
        operator = base.name[:-2]
        target = archive / operator
        for version, path in projects(operator):
            if version > 0:
                target.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(target / path.name))
        for folder in ("strategy", "bottleneck"):
            path = base / folder
            if path.exists():
                target.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(target / f"{base.name}_{folder}"))
    save({"generated_at": now(), "mode": "source_strategy_serial_four_rounds",
          "archive": str(archive), "max_rounds": MAX_ROUNDS,
          "concurrency": MAX_CONCURRENCY,
          "min_improvement_percent": MIN_IMPROVEMENT, "items": queue_items()})
    LOG.write_text("timestamp\tposition\toperator\tstage\treturncode\n")
    print(f"initialized={len(queue_items())} queue={QUEUE} archive={archive}")


def rebuild() -> None:
    save({"generated_at": now(), "mode": "source_strategy_serial_four_rounds",
          "preserve_existing_versions": True, "max_rounds": MAX_ROUNDS,
          "concurrency": MAX_CONCURRENCY,
          "min_improvement_percent": MIN_IMPROVEMENT, "items": queue_items()})
    LOG.write_text("timestamp\tposition\toperator\tstage\treturncode\n")
    print(f"rebuilt={len(queue_items())} queue={QUEUE}")


def reconcile() -> None:
    queue = load_json(QUEUE)
    if not queue:
        raise SystemExit("queue missing")
    for item in queue["items"]:
        action = next_action(item["operator"])
        if action["kind"] == "terminal":
            item.update(state=action["state"], result_reason=action["reason"])
        elif item.get("state") in TERMINAL or item.get("state") == "running":
            if item.get("state") == "agent_failed":
                item["stage_attempts"] = {}
                item["agent_thread_ids"] = {}
            item["state"] = "pending"
            item["result_reason"] = "reconciled from durable artifacts"
    save(queue)
    print(json.dumps({state: sum(item["state"] == state for item in queue["items"])
                      for state in sorted({item["state"] for item in queue["items"]})}, ensure_ascii=False))


def recover_environment_failures() -> None:
    queue = load_json(QUEUE)
    if not queue:
        raise SystemExit("queue missing")
    recovered = []
    for item in queue["items"]:
        action = next_action(item["operator"])
        if action.get("kind") != "command" or not action.get("environment_retry"):
            continue
        item.setdefault("stage_attempts", {}).pop(stage_key(action), None)
        item.update(state="pending", current_stage=None, returncode=None,
                    result_reason="environment failure recovered for retry")
        recovered.append(item["operator"])
    save(queue)
    print(json.dumps({"recovered": len(recovered), "operators": recovered}, ensure_ascii=False))


def recover_agent_failures() -> None:
    queue = load_json(QUEUE)
    if not queue:
        raise SystemExit("queue missing")
    recovered = []
    for item in queue["items"]:
        if item.get("state") != "agent_failed":
            continue
        item["stage_attempts"] = {}
        item["agent_transport_attempts"] = {}
        item["agent_thread_ids"] = {}
        item.update(state="pending", current_stage=None, returncode=None,
                    result_reason="agent failure recovered for retry")
        recovered.append(item["operator"])
    save(queue)
    print(json.dumps({"recovered": len(recovered), "operators": recovered}, ensure_ascii=False))


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--initialize-op":
        initialize_one(sys.argv[2])
        return
    if len(sys.argv) == 3 and sys.argv[1] == "--initialize-ops":
        initialize_selected(sys.argv[2])
        return
    if len(sys.argv) == 3 and sys.argv[1] == "--reopen-ops":
        reopen_selected(sys.argv[2])
        return
    modes = {"--initialize": initialize, "--rebuild": rebuild, "--reconcile": reconcile,
             "--recover-environment-failures": recover_environment_failures,
             "--recover-agent-failures": recover_agent_failures}
    if len(sys.argv) == 2 and sys.argv[1] in modes:
        modes[sys.argv[1]]()
        return
    if len(sys.argv) == 2 and sys.argv[1] == "--show-stages":
        queue = load_json(QUEUE) or {}
        for item in queue.get("items", []):
            action = next_action(item["operator"])
            print(json.dumps({"operator": item["operator"], "state": item.get("state"),
                              "next": {key: str(value) if isinstance(value, Path) else value
                                       for key, value in action.items()}}, ensure_ascii=False))
        return
    if len(sys.argv) > 1:
        raise SystemExit("usage: run_kernelbench_optimization.py [--initialize|--initialize-op OP|--initialize-ops OP,...|--reopen-ops OP,...|--rebuild|--reconcile|--recover-environment-failures|--recover-agent-failures|--show-stages]")
    STOP_REQUESTED.clear()
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    RUNS.mkdir(parents=True, exist_ok=True)
    ASCENDC_TMPDIR.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"queue already running: {LOCK}")
        queue = load_json(QUEUE)
        if not queue:
            raise SystemExit("queue missing; use --initialize or --rebuild")
        try:
            run_queue(queue)
        finally:
            durable = load_json(QUEUE) or queue
            for item in durable.get("items", []):
                if item.get("state") in TERMINAL:
                    cleanup_cache(item["operator"])


if __name__ == "__main__":
    main()
