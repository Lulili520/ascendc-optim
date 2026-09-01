#!/usr/bin/env python3
"""Warm once, measure one PipeUtilization Task Duration, and gate improvement."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / ".codex"))
from suite_config import SUITES, load_suite_manifest, suite_paths

SUITE = "KernelBench910B"
PROJECT_ROOT, _, REFERENCES = suite_paths(ROOT, SUITE)
ACTIVATE = Path("/data/lu/activate_evokernel.sh")
VENDORS = Path("/usr/local/Ascend/cann-9.0.0/opp/vendors")
MIN_IMPROVEMENT_PERCENT = 5.0


def quote(value: Path | str) -> str:
    return shlex.quote(str(value))


def load_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def source_fingerprint(project: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for folder in ("op_host", "op_kernel") for path in (project / folder).rglob("*") if path.is_file())
    if not files:
        raise RuntimeError("op_host/op_kernel 中没有源码")
    for path in files:
        digest.update(path.relative_to(project).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def require_precision(project: Path) -> str:
    result = load_json(project / "precision/precision.json")
    current = source_fingerprint(project)
    if not result or result.get("status") != "PASS" or result.get("exit_code") != 0:
        raise RuntimeError("当前版本没有有效精度 PASS")
    if result.get("source_fingerprint") != current:
        raise RuntimeError("源码已变化，请重新验证精度")
    return current


def update_workspace(project: Path, status: str, fingerprint: str, performance: dict) -> None:
    path = project / "workspace.json"
    value = load_json(path) or {}
    value.update({
        "status": status,
        "source_fingerprint": fingerprint,
        "performance": performance,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def latency(project: Path) -> float | None:
    report = load_json(project / "performance/latency.json") or {}
    value = report.get("task_duration_us")
    try:
        current = source_fingerprint(project)
    except RuntimeError:
        return None
    return (float(value) if report.get("source_fingerprint") == current
            and isinstance(value, (int, float)) and math.isfinite(value) and value > 0 else None)


def best_prior(project: Path) -> tuple[int, float] | None:
    match = re.fullmatch(r"(.+)_(\d+)", project.name)
    if not match:
        return None
    operator, version = match.group(1), int(match.group(2))
    values = [(index, latency(project.parent / f"{operator}_{index}")) for index in range(version)]
    valid = [(index, value) for index, value in values if value is not None]
    return min(valid, key=lambda item: item[1]) if valid else None


def resolve_operator(name: str, project_dir: str | None) -> tuple[dict, Path]:
    manifest = load_suite_manifest(ROOT, SUITE)
    if name not in manifest:
        raise RuntimeError(f"未知算子：{name}")
    item = manifest[name]
    project = Path(project_dir).resolve() if project_dir else PROJECT_ROOT / item["project"]
    if not project.is_dir():
        raise RuntimeError(f"算子工程不存在：{project}")
    if not list((project / "CppExtension").glob(f'{item["extension_module"]}*.so')):
        raise RuntimeError("缺少可复用 CppExtension .so")
    return item, project


def vendor_env(item: dict) -> Path:
    path = VENDORS / item["vendor"] / "bin/set_env.bash"
    if not path.is_file():
        raise RuntimeError(f"找不到已安装 vendor：{path}")
    return path


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower()).removesuffix("opt")


def load_python(path: Path):
    spec = importlib.util.spec_from_file_location(f"latency_input_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载输入定义：{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def to_npu(value, torch):
    if isinstance(value, torch.Tensor):
        return value.npu().contiguous()
    if isinstance(value, tuple):
        return tuple(to_npu(item, torch) for item in value)
    if isinstance(value, list):
        return [to_npu(item, torch) for item in value]
    return value


def prepare_custom(name: str, item: dict, project: Path, torch):
    precision_scripts = ROOT / ".codex/skills/kernel-precision/scripts"
    sys.path.insert(0, str(precision_scripts))
    import validate_precision
    # The imported precision module initializes with KernelBench defaults.
    # Keep its reference loader on the suite selected by this subprocess.
    validate_precision.SUITE = SUITE
    _, _, validate_precision.REFERENCES = suite_paths(ROOT, SUITE)
    # Performance only owns the callable and its arguments.  Precision may
    # append reference/cache metadata without changing this interface.
    _, custom, arguments, *_ = validate_precision.prepare_call(
        name, item, project, torch, with_reference=False,
    )
    return custom, arguments


def run_one(name: str, device: int, project_dir: str | None) -> None:
    import torch
    import torch_npu  # noqa: F401
    item, project = resolve_operator(name, project_dir)
    torch.npu.set_device(device)
    torch.manual_seed(1234)
    custom, arguments = prepare_custom(name, item, project, torch)
    with torch.inference_mode():
        custom(*arguments)
    torch.npu.synchronize()


def shell_prefix(item: dict, project: Path) -> str:
    return f"""
set -euo pipefail
source {quote(ACTIVATE)}
set +u
source {quote(vendor_env(item))}
set -u
export PYTHONPATH={quote(project / 'CppExtension')}${{PYTHONPATH:+:${{PYTHONPATH}}}}
"""


def run(script: str, log: Path | None = None) -> None:
    result = subprocess.run(["bash", "-lc", script], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if log:
        log.write_text(result.stdout, encoding="utf-8", errors="replace")
    if result.returncode:
        raise RuntimeError(result.stdout[-4000:])


def latest(root: Path, pattern: str) -> Path | None:
    paths = sorted(root.rglob(pattern), key=lambda path: path.stat().st_mtime_ns)
    return paths[-1] if paths else None


def parse_task_duration(path: Path, operator: str) -> float:
    with path.open(encoding="utf-8", errors="replace", newline="") as handle:
        rows = list(csv.DictReader(handle))
    wanted = norm(operator)
    rows = [row for row in rows
            if any(token in row.get("Task Type", "").upper()
                   for token in ("AI_CORE", "AI_VECTOR", "AI_CUBE", "AIV", "AIC", "MIX"))
            and wanted in norm(row.get("Op Name", ""))]
    if not rows:
        raise RuntimeError(f"PipeUtilization 中没有目标算子 {operator} 的 AI Core 任务")
    if len(rows) != 1:
        raise RuntimeError(f"PipeUtilization 中目标算子任务不唯一：{len(rows)}")
    try:
        value = float(rows[-1].get("Task Duration(us)", ""))
    except ValueError as error:
        raise RuntimeError("PipeUtilization 缺少 Task Duration(us)") from error
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError("Task Duration 必须为有限正数")
    return value


def preserve_pipe_utilization_csv(source: Path, report_dir: Path) -> Path:
    target = report_dir / "op_summary_PipeUtilization.csv"
    shutil.copy2(source, target)
    return target


def msprof_command(output: Path, runner: str) -> str:
    """Build the one-shot profiling command, including CSV export."""
    return (
        f"msprof --output={quote(output)} --export=on --ai-core=on "
        f"--aic-metrics=PipeUtilization --task-time=on --ascendcl=on {runner}"
    )


def collect(name: str, device: int, keep_intermediates: bool, project_dir: str | None) -> Path:
    item, project = resolve_operator(name, project_dir)
    fingerprint = require_precision(project)
    report_dir = project / "performance"
    report_dir.mkdir(exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"kernel_latency_{name}_"))
    runner = f"python {quote(Path(__file__).resolve())} --internal-run {quote(name)} --device {device} --project-dir {quote(project)} --suite {quote(SUITE)}"
    prefix = shell_prefix(item, project)
    try:
        print("[1/2] msprof 外预热1次", flush=True)
        run(prefix + runner)
        print("[2/2] 采集1次 PipeUtilization Task Duration", flush=True)
        output = temporary / "PROF"
        output.mkdir()
        run(prefix + msprof_command(output, runner), output / "msprof.log")
        csv_path = latest(output, "op_summary_*.csv")
        if csv_path is None:
            raise RuntimeError("未生成 PipeUtilization CSV，请检查 performance/msprof.log")
        duration = parse_task_duration(csv_path, name)
        evidence_path = preserve_pipe_utilization_csv(csv_path, report_dir)
        prior = best_prior(project)
        improvement = None if prior is None else (prior[1] - duration) / prior[1] * 100.0
        accepted = prior is None or improvement > MIN_IMPROVEMENT_PERCENT
        report = {
            "operator": name,
            "task_duration_us": duration,
            "source_fingerprint": fingerprint,
            "accepted": accepted,
            "best_version": prior[0] if prior else None,
            "best_task_duration_us": prior[1] if prior else None,
            "improvement_percent": improvement,
            "threshold_percent": MIN_IMPROVEMENT_PERCENT,
            "pipe_utilization_csv": "performance/op_summary_PipeUtilization.csv",
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        path = report_dir / "latency.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if accepted:
            update_workspace(project, "PERFORMANCE_DONE", fingerprint, {"status": "DONE", "task_duration_us": duration, "result": "performance/latency.json", "evidence": "performance/op_summary_PipeUtilization.csv"})
        else:
            update_workspace(project, "PERFORMANCE_SCREENED_NO_IMPROVEMENT", fingerprint, {"status": "SCREENED_NO_IMPROVEMENT", "task_duration_us": duration, "best_version": prior[0], "improvement_percent": improvement, "result": "performance/latency.json", "evidence": "performance/op_summary_PipeUtilization.csv"})
        print(f"task_duration_us={duration:.6f} accepted={str(accepted).lower()} improvement_percent={improvement}", flush=True)
        return path
    except Exception as error:
        update_workspace(project, "PERFORMANCE_FAILED", fingerprint, {"status": "FAILED", "reason": str(error)})
        raise
    finally:
        msprof_log = temporary / "PROF/msprof.log"
        if msprof_log.is_file():
            shutil.copy2(msprof_log, report_dir / "msprof.log")
        if keep_intermediates:
            print(f"intermediates={temporary}")
        else:
            shutil.rmtree(temporary, ignore_errors=True)
            for cache in (project / "CppExtension/__pycache__", Path(__file__).resolve().parent / "__pycache__"):
                if cache.is_dir():
                    shutil.rmtree(cache)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operator", nargs="?")
    parser.add_argument("--suite", choices=sorted(SUITES), default="KernelBench910B")
    parser.add_argument("--project-dir")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--keep-intermediates", action="store_true")
    parser.add_argument("--internal-run", metavar="OPERATOR", help=argparse.SUPPRESS)
    args = parser.parse_args()
    global SUITE, PROJECT_ROOT, REFERENCES
    SUITE = args.suite
    PROJECT_ROOT, _, REFERENCES = suite_paths(ROOT, SUITE)
    if args.internal_run:
        run_one(args.internal_run, args.device, args.project_dir)
    elif args.operator:
        collect(args.operator, args.device, args.keep_intermediates, args.project_dir)
    else:
        parser.error("必须提供 OperatorName")


if __name__ == "__main__":
    main()
