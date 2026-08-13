#!/usr/bin/env python3
"""采集单个 AscendC 算子的完整 msprof 指标并生成本地性能报告。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.util
import inspect
import json
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from probe_hardware import DEFAULT_OUTPUT_ROOT, probe


WORKSPACE = Path(__file__).resolve().parents[4]
PROJECT_ROOT = WORKSPACE / "kernel/KernelBench910B"
REFERENCES = WORKSPACE / "kernel/pytorch-references/KernelBench"
MANIFEST = PROJECT_ROOT / "manifest.json"
ACTIVATE = Path("/data/lu/activate_evokernel.sh")
VENDORS = Path("/usr/local/Ascend/cann-9.0.0/opp/vendors")
METRICS = (
    "PipeUtilization",
    "ArithmeticUtilization",
    "Memory",
    "MemoryL0",
    "MemoryUB",
    "L2Cache",
    "ResourceConflictRatio",
)
HARDWARE_FIELDS = (
    "soc", "aic_core_count", "aiv_core_count",
    "ub_bytes_per_core", "l1_bytes_per_core",
    "l0a_bytes_per_core", "l0b_bytes_per_core", "l0c_bytes_per_core",
    "l2_bytes", "gm_peak_bandwidth_gbps_per_core",
)
NUMERIC_HARDWARE_FIELDS = set(HARDWARE_FIELDS) - {"soc"}


def quote(value: Path | str) -> str:
    return shlex.quote(str(value))


def source_fingerprint(project: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        path for folder in ("op_host", "op_kernel")
        for path in (project / folder).rglob("*") if path.is_file()
    )
    if not files:
        raise RuntimeError("op_host/op_kernel 中没有源码文件")
    for path in files:
        digest.update(path.relative_to(project).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_hardware(device: int, hardware_config: str | None) -> dict:
    if hardware_config:
        path = Path(hardware_config).resolve()
        raw = json.loads(path.read_text(encoding="utf-8"))
        identity = raw.get("identity")
        if isinstance(identity, dict) and identity.get("logical_device") not in (None, device):
            raise RuntimeError(
                f"硬件配置属于 logical device {identity.get('logical_device')}，当前请求 device {device}"
            )
        source = path
    else:
        raw = probe(device)
        source = DEFAULT_OUTPUT_ROOT / f"device_{device}.json"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(
            json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if all(key in raw for key in ("identity", "compute", "memory_capacity", "memory_bandwidth")):
        identity = raw["identity"]
        compute = raw["compute"]
        capacity = raw["memory_capacity"]
        bandwidth = raw["memory_bandwidth"]
        flattened = {
            "soc": identity.get("soc") or identity.get("chip_name"),
            "aic_core_count": compute.get("aic_core_count"),
            "aiv_core_count": compute.get("aiv_core_count"),
            **{key: capacity.get(key) for key in (
                "ub_bytes_per_core", "l1_bytes_per_core", "l0a_bytes_per_core",
                "l0b_bytes_per_core", "l0c_bytes_per_core", "l2_bytes",
            )},
            "gm_peak_bandwidth_gbps_per_core": bandwidth.get(
                "gm_peak_bandwidth_gbps_per_core"
            ),
        }
    else:
        flattened = raw
    hardware = {key: flattened.get(key) for key in HARDWARE_FIELDS}
    if not isinstance(hardware["soc"], str) or not hardware["soc"].strip():
        raise RuntimeError("硬件配置缺少 soc")
    for key in NUMERIC_HARDWARE_FIELDS:
        value = hardware[key]
        if value is not None and (not isinstance(value, (int, float)) or value <= 0):
            raise RuntimeError(f"硬件配置字段 {key} 必须为正数或 null")
    print(f"hardware_config={source}", flush=True)
    return hardware


def add_gm_bandwidth_facts(memory: dict, hardware: dict | None) -> dict:
    """Add objective, per-core GM totals and peak utilization when available."""
    def value(name: str) -> float | None:
        raw = memory.get(name)
        return float(raw) if isinstance(raw, (int, float)) else None

    def total(read_key: str, write_key: str) -> float | None:
        read = value(read_key)
        write = value(write_key)
        if read is None or write is None:
            return None
        return read + write

    aic = total("aic_main_mem_read_bw(GB/s)", "aic_main_mem_write_bw(GB/s)")
    aiv = total("aiv_main_mem_read_bw(GB/s)", "aiv_main_mem_write_bw(GB/s)")
    available = [item for item in (aic, aiv) if item is not None]
    maximum = max(available) if available else None
    peak = (hardware or {}).get("gm_peak_bandwidth_gbps_per_core")
    utilization = None
    if maximum is not None and isinstance(peak, (int, float)) and peak > 0:
        utilization = maximum / peak * 100.0
    return {
        **memory,
        "aic_main_mem_total_bw_gbps_per_core": aic,
        "aiv_main_mem_total_bw_gbps_per_core": aiv,
        "gm_max_path_bw_gbps_per_core": maximum,
        "gm_peak_utilization_percent": utilization,
    }


def require_precision(project: Path) -> str:
    result_path = project / "precision/precision.json"
    if not result_path.is_file():
        raise RuntimeError("缺少 precision/precision.json，请先运行精度验证")
    result = json.loads(result_path.read_text())
    current = source_fingerprint(project)
    if result.get("status") != "PASS" or result.get("exit_code") != 0:
        raise RuntimeError("当前版本没有有效的精度 PASS 结果")
    if result.get("source_fingerprint") != current:
        raise RuntimeError("源码指纹已变化，请重新运行精度验证")
    return current


def update_workspace(project: Path, status: str, **fields: object) -> None:
    path = project / "workspace.json"
    data = json.loads(path.read_text()) if path.is_file() else {}
    data.update({"status": status, "updated_at": datetime.now(timezone.utc).isoformat()})
    data.update(fields)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_operator(name: str, project_dir: str | None = None) -> tuple[dict, Path]:
    manifest = json.loads(MANIFEST.read_text())
    if name not in manifest:
        matches = [key for key in manifest if key.lower() == name.lower()]
        hint = f"；是否是 {matches[0]}" if matches else ""
        raise SystemExit(f"未知算子：{name}{hint}")
    item = manifest[name]
    project = Path(project_dir).resolve() if project_dir else PROJECT_ROOT / item["project"]
    if not project.is_dir():
        raise SystemExit(f"算子工程不存在：{project}")
    if len(item["reference_candidates"]) != 1:
        raise SystemExit(f"{name} 没有唯一输入定义")
    extension = project / "CppExtension"
    if not list(extension.glob(f'{item["extension_module"]}*.so')):
        raise SystemExit(f"缺少可复用扩展 .so：{extension}")
    return item, project


def vendor_env(item: dict) -> Path:
    path = VENDORS / item["vendor"] / "bin/set_env.bash"
    if not path.is_file():
        raise SystemExit(f"找不到已安装 OPP 环境：{path}")
    return path


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower()).removesuffix("opt")


def as_number(value: object) -> float | None:
    try:
        text = str(value).strip().rstrip("\t")
        return None if text in ("", "N/A", "NA", "-") else float(text)
    except (TypeError, ValueError):
        return None


def select_numeric(row: dict[str, str], terms: tuple[str, ...]) -> dict[str, float]:
    selected = {}
    for key, value in row.items():
        if any(term in key.lower() for term in terms):
            parsed = as_number(value)
            if parsed is not None:
                selected[key] = parsed
    return selected


def core_summary(rows: list[tuple[int, int]]) -> dict:
    values = [value for _, value in rows]
    maximum, minimum = max(values), min(values)
    midpoint = len(rows) // 2
    first, second = rows[:midpoint], rows[midpoint:]
    mean = lambda group: sum(value for _, value in group) / len(group) if group else None
    first_mean, second_mean = mean(first), mean(second)
    cluster_gap = None
    if first_mean and second_mean:
        cluster_gap = abs(first_mean - second_mean) / max(first_mean, second_mean) * 100
    return {
        "active_cores": len(rows), "min_cycles": minimum,
        "mean_cycles": sum(values) / len(values), "max_cycles": maximum,
        "imbalance_percent": (maximum - minimum) / maximum * 100,
        "slowest_cores": sorted(rows, key=lambda item: item[1], reverse=True)[:3],
        "fastest_cores": sorted(rows, key=lambda item: item[1])[:3],
        "half_cluster_gap_percent": cluster_gap,
    }


def cache_rates(row: dict[str, str]) -> dict[str, float]:
    result = {}
    for prefix in ("aic", "aiv"):
        hit = miss = 0.0
        for key, value in row.items():
            parsed = as_number(value) or 0.0
            lower = key.lower()
            if not lower.startswith(prefix):
                continue
            if "cache_hit" in lower:
                hit += parsed
            elif "cache_miss" in lower:
                miss += parsed
        if hit + miss > 0:
            result[f"{prefix}_derived_hit_rate_percent"] = hit / (hit + miss) * 100
    return result


def load_python_file(path: Path):
    spec = importlib.util.spec_from_file_location(f"perf_input_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载输入定义：{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def to_npu(value, torch):
    if isinstance(value, torch.Tensor):
        return value.npu().contiguous()
    if isinstance(value, tuple):
        return tuple(to_npu(x, torch) for x in value)
    if isinstance(value, list):
        return [to_npu(x, torch) for x in value]
    return value


def prepare_custom(name: str, item: dict, project: Path, torch):
    reference_path = REFERENCES / item["reference_candidates"][0]
    reference = load_python_file(reference_path)
    init_args = reference.get_init_inputs() if hasattr(reference, "get_init_inputs") else []
    model = reference.Model(*init_args).eval().npu()
    inputs = [to_npu(value, torch) for value in reference.get_inputs()]

    forward_names = list(inspect.signature(model.forward).parameters)
    values = {norm(key): value for key, value in zip(forward_names, inputs)}
    values.update({norm(key): value for key, value in model.named_parameters()})
    values.update({norm(key): value for key, value in model.named_buffers()})
    unused = iter(inputs)
    custom_args = []
    for parameter in item["parameters"]:
        key = norm(parameter)
        if key == "self" and inputs:
            custom_args.append(inputs[0])
        elif key in values:
            custom_args.append(values[key])
        else:
            matches = {
                id(value): value
                for candidate, value in values.items()
                if candidate.endswith(key) or key.endswith(candidate)
            }
            if len(matches) == 1:
                custom_args.append(next(iter(matches.values())))
            else:
                try:
                    custom_args.append(next(unused))
                except StopIteration as error:
                    raise RuntimeError(
                        f"{name} 的参数 {parameter!r} 无法唯一映射"
                    ) from error

    sys.path.insert(0, str(project / "CppExtension"))
    extension = importlib.import_module(item["extension_module"])
    function = item["function"].split("(", 1)[0].strip()
    return getattr(extension, function), custom_args, inputs


def run_one(name: str, device: int, project_dir: str | None = None) -> None:
    import torch
    import torch_npu  # noqa: F401

    item, project = resolve_operator(name, project_dir)
    torch.npu.set_device(device)
    torch.manual_seed(1234)
    custom, custom_args, _inputs = prepare_custom(name, item, project, torch)
    with torch.inference_mode():
        output = custom(*custom_args)
    torch.npu.synchronize()
    if isinstance(output, torch.Tensor):
        print(f"operator={name}, shape={tuple(output.shape)}, dtype={output.dtype}")


def shell_prefix(item: dict, project: Path) -> str:
    return f"""
set -euo pipefail
source {quote(ACTIVATE)}
set +u
source {quote(vendor_env(item))}
set -u
export PYTHONPATH={quote(project / 'CppExtension')}${{PYTHONPATH:+:${{PYTHONPATH}}}}
"""


def run_command(script: str, log: Path | None = None) -> None:
    result = subprocess.run(
        ["bash", "-lc", script], text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    if log is not None:
        log.write_text(result.stdout, errors="replace")
    if result.returncode != 0:
        raise RuntimeError(result.stdout[-4000:])


def find_latest(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.rglob(pattern), key=lambda p: p.stat().st_mtime_ns)
    return matches[-1] if matches else None


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", errors="replace", newline="") as handle:
        return list(csv.DictReader(handle))


def compute_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    selected = []
    for row in rows:
        task_type = row.get("Task Type", "").upper()
        if (
            (task_type.startswith("AI_") and "CORE" in task_type)
            or any(kind in task_type for kind in ("AIV", "MIX"))
        ):
            selected.append(row)
    return selected


def as_float(value: object) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def task_metrics(pipe: dict[str, str]) -> dict[str, object]:
    latency = as_float(pipe.get("Task Duration(us)"))
    core_times = {
        key: as_float(pipe[key])
        for key in ("aicore_time(us)", "aiv_time(us)")
        if pipe.get(key) not in (None, "")
    }
    if not core_times:
        raise RuntimeError("PipeUtilization 缺少 aicore_time(us)/aiv_time(us)，无法计算头开销")
    max_core_time = max(core_times.values())
    overhead = max(0.0, latency - max_core_time)
    return {
        **{
            key: pipe.get(key) for key in
            ("Task Type", "Block Num", "Mix Block Num", "Input Shapes",
             "Input Data Types", "Output Shapes", "Output Data Types")
        },
        "aicore_time_us": core_times.get("aicore_time(us)"),
        "aiv_time_us": core_times.get("aiv_time(us)"),
        "max_core_time_us": max_core_time,
        "head_overhead_us": overhead,
        "head_overhead_ratio": overhead / latency * 100.0 if latency > 0 else 0.0,
    }


def load_core_cycles(sample_root: Path) -> list[tuple[int, int]]:
    databases = []
    for name in ("aicore.db", "ai_vector_core.db"):
        db = find_latest(sample_root, name)
        if db is not None:
            databases.append(db)
    if not databases:
        return []
    combined: dict[int, int] = {}
    for db in databases:
        try:
            with sqlite3.connect(db) as conn:
                tables = [
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                ]
                # Both raw and summary tables contain the same task cycles. Use
                # one canonical table per core type to avoid double counting.
                preferred = next(
                    (table for table in ("EventCount", "AICoreOriginalData") if table in tables),
                    None,
                )
                for table in ([preferred] if preferred is not None else tables):
                    columns = {
                        row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')
                    }
                    if not {"coreid", "task_cyc"}.issubset(columns):
                        continue
                    rows = conn.execute(
                        f'SELECT coreid, SUM(task_cyc) FROM "{table}" '
                        "WHERE task_cyc>0 GROUP BY coreid"
                    ).fetchall()
                    for core, cycles in rows:
                        if core is not None:
                            combined[int(core)] = combined.get(int(core), 0) + int(cycles)
        except sqlite3.Error as error:
            raise RuntimeError(f"无法解析逐核数据：{db}: {error}") from error
    return sorted(combined.items())


def write_report(
    name: str, device: int, report_dir: Path, metric_rows: dict[str, dict[str, str]],
    core_cycles: list[tuple[int, int]], hardware: dict | None,
) -> dict:
    pipe = metric_rows.get("PipeUtilization", {})
    latency = as_float(pipe.get("Task Duration(us)"))
    task = task_metrics(pipe)
    per_core = core_summary(core_cycles)
    memory = add_gm_bandwidth_facts(
        select_numeric(
            metric_rows["Memory"],
            ("_bw", "bandwidth", "datas(", "usage_rate", "instructions"),
        ),
        hardware,
    )
    data = {
        "operator": name,
        "kernel_latency_us": latency,
        "task": task,
        "pipeline": select_numeric(pipe, ("_time", "_ratio", "total_cycles", "miss_rate")),
        "arithmetic": select_numeric(
            metric_rows["ArithmeticUtilization"], ("_ratio", "fops", "instr")
        ),
        "memory": memory,
        "memory_l0": select_numeric(metric_rows["MemoryL0"], ("_bw", "bandwidth")),
        "memory_ub": select_numeric(metric_rows["MemoryUB"], ("_bw", "bandwidth")),
        "l2_cache": {
            **select_numeric(
                metric_rows["L2Cache"], ("cache_hit", "cache_miss", "hit_rate")
            ),
            **cache_rates(metric_rows["L2Cache"]),
        },
        "resource_conflict": select_numeric(
            metric_rows["ResourceConflictRatio"], ("_ratio",)
        ),
        "per_core": per_core,
        "hardware": hardware,
    }
    (report_dir / "performance.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    )

    lines = [
        f"# {name} 性能报告", "", "## 采集配置", "",
        f"- NPU device：{device}", "- 预热：整轮采集前1次",
        "- 正式采集：7组 metrics + 1组 sample-based，每组执行1次算子", "",
        "## 核心结果", "", f"- Kernel latency：{latency:.6f} us",
        f"- 活跃核数：{len(core_cycles)}",
    ]
    lines.append(
        f"- 头开销：{task['head_overhead_us']:.6f} us "
        f"({task['head_overhead_ratio']:.2f}%)"
    )
    lines.append(f"- 逐核 cycle 不均衡度：{per_core['imbalance_percent']:.2f}%")
    lines.extend(["", "## 指标文件", ""])
    for metric in METRICS:
        state = "已生成" if metric in metric_rows else "缺失"
        lines.append(f"- `op_summary_{metric}.csv`：{state}")
    lines.extend(["- `per_core_cycles.csv`：" + ("已生成" if core_cycles else "缺失"), ""])
    (report_dir / "perf_report.md").write_text("\n".join(lines), encoding="utf-8")
    (report_dir / "summary.txt").write_text(
        f"operator={name}\nkernel_latency_us={latency:.6f}\n"
        f"active_cores={len(core_cycles)}\n"
        f"head_overhead_us={task['head_overhead_us']:.6f}\n"
        f"head_overhead_ratio={task['head_overhead_ratio']:.2f}\n",
        encoding="utf-8",
    )
    return data


def collect(
    name: str, device: int, keep_intermediates: bool,
    project_dir: str | None = None, hardware_config: str | None = None,
) -> Path:
    item, project = resolve_operator(name, project_dir)
    fingerprint = require_precision(project)
    hardware = load_hardware(device, hardware_config)
    report_dir = project / "performance"
    report_dir.mkdir(exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix=f"kernel_perf_{name}_"))
    runner = (
        f"python {quote(Path(__file__).resolve())} --internal-run {quote(name)} "
        f"--device {device} --project-dir {quote(project)}"
    )
    prefix = shell_prefix(item, project)

    try:
        print("[1/4] 整轮采集前预热1次", flush=True)
        run_command(prefix + runner)

        metric_rows: dict[str, dict[str, str]] = {}
        for index, metric in enumerate(METRICS, 1):
            output = temp_root / f"PROF_{metric}"
            output.mkdir()
            print(f"[2/4] ({index}/7) 采集 {metric}", flush=True)
            command = (
                prefix
                + f"msprof --output={quote(output)} --ai-core=on "
                + f"--aic-metrics={quote(metric)} --task-time=on --ascendcl=on "
                + runner
            )
            run_command(command, output / "msprof.log")
            source = find_latest(output, "op_summary_*.csv")
            if source is None:
                raise RuntimeError(f"{metric} 未生成 op_summary CSV")
            destination = report_dir / f"op_summary_{metric}.csv"
            shutil.copy2(source, destination)
            rows = compute_rows(read_rows(source))
            if not rows:
                raise RuntimeError(f"{metric} 未找到 {name} 的 AI Core 记录")
            metric_rows[metric] = rows[-1]

        print("[3/4] 采集 sample-based 逐核 cycle", flush=True)
        sample = temp_root / "PROF_Sample"
        sample.mkdir()
        command = (
            prefix
            + f"msprof --output={quote(sample)} --ai-core=on "
            + "--aic-metrics=PipeUtilization --aic-mode=sample-based --aic-freq=100 "
            + "--task-time=on --ascendcl=on "
            + runner
        )
        run_command(command, sample / "msprof.log")
        core_cycles = load_core_cycles(sample)
        if not core_cycles:
            raise RuntimeError("sample-based 未生成有效逐核 cycle")
        with (report_dir / "per_core_cycles.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["coreid", "task_cycles"])
            writer.writerows(core_cycles)

        print("[4/4] 生成性能报告", flush=True)
        report = write_report(name, device, report_dir, metric_rows, core_cycles, hardware)
        if report["kernel_latency_us"] <= 0:
            raise RuntimeError("PipeUtilization 中缺少有效 Task Duration(us)")
        update_workspace(
            project, "PERFORMANCE_DONE", source_fingerprint=fingerprint,
            performance={
                "status": "DONE",
                "kernel_latency_us": report["kernel_latency_us"],
                "active_cores": report["per_core"]["active_cores"],
                "core_imbalance_percent": report["per_core"]["imbalance_percent"],
                "result": "performance/performance.json",
            },
        )
        print(f"performance_report={report_dir}", flush=True)
        return report_dir
    except Exception as error:
        update_workspace(
            project, "PERFORMANCE_FAILED",
            performance={"status": "FAILED", "reason": str(error)},
        )
        raise
    finally:
        if keep_intermediates:
            print(f"intermediates={temp_root}", flush=True)
        else:
            shutil.rmtree(temp_root, ignore_errors=True)
            for cache in (
                project / "CppExtension/__pycache__",
                (REFERENCES / item["reference_candidates"][0]).parent / "__pycache__",
                Path(__file__).resolve().parent / "__pycache__",
            ):
                if cache.is_dir():
                    shutil.rmtree(cache)
            print("[清理] 已删除 PROF、临时日志、SQLite 和 Python 缓存", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operator", nargs="?", help="manifest 中的算子工程名")
    parser.add_argument(
        "--project-dir", help="覆盖 manifest 中的工程目录，用于采集工作区版本"
    )
    parser.add_argument("--device", type=int, default=0, help="容器内逻辑 NPU ID，默认0")
    parser.add_argument(
        "--hardware-config",
        help="显式硬件配置 JSON；省略时自动探测当前 device 并保存",
    )
    parser.add_argument(
        "--keep-intermediates", action="store_true", help="保留原始 PROF 目录用于排错"
    )
    parser.add_argument("--internal-run", metavar="OPERATOR", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.internal_run:
        run_one(args.internal_run, args.device, args.project_dir)
        return
    if not args.operator:
        parser.error("必须提供 OperatorName")
    collect(
        args.operator, args.device, args.keep_intermediates,
        args.project_dir, args.hardware_config,
    )


if __name__ == "__main__":
    main()
