#!/usr/bin/env python3
"""Validate the private source-layout proof behind a compact strategy."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

PATTERNS = {"elementwise", "prefix_scan", "reduction", "norm_softmax", "window_pooling", "scatter_transposed", "existing_cube", "fused_network"}
ENGINES = {"aiv", "aic", "mixed"}
WORK_KEYS = {"gm_bytes", "dma_bursts", "vector_repeat_blocks", "cube_ops", "scalar_iterations", "task_lifecycles", "state_initializations", "sync_events", "writeback_bursts"}

def fingerprint(project: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(path for folder in ("op_host", "op_kernel") for path in (project / folder).rglob("*") if path.is_file()):
        digest.update(path.relative_to(project).as_posix().encode()); digest.update(b"\0"); digest.update(path.read_bytes())
    return digest.hexdigest()

def texts(value: object, field: str) -> None:
    if not isinstance(value, list) or not value or any(not isinstance(item, str) or item != item.strip() or not item or "\n" in item for item in value):
        raise RuntimeError(f"{field} 必须由非空、无首尾空白的单行文本组成")

def validate(path: Path, project: Path) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    fields = {"source_fingerprint", "pattern", "engine", "available_cores", "used_cores", "shape_model", "task_mapping", "parameters", "buffers", "transfers", "compute", "work", "proofs"}
    if not isinstance(value, dict) or set(value) != fields: raise RuntimeError("planning 字段不完整或包含额外字段")
    if value["source_fingerprint"] != fingerprint(project): raise RuntimeError("planning 源码指纹与当前 Host/Kernel 不一致")
    if value["pattern"] not in PATTERNS: raise RuntimeError("pattern 非法")
    if value["engine"] not in ENGINES: raise RuntimeError("engine 非法")
    if any(not isinstance(value[field], int) or isinstance(value[field], bool) or value[field] <= 0 for field in ("available_cores", "used_cores")):
        raise RuntimeError("available_cores/used_cores 必须为正整数")
    if value["used_cores"] > value["available_cores"]: raise RuntimeError("used_cores 不得超过当前计算引擎 available_cores")
    if value["engine"] == "aiv" and value["available_cores"] == 24:
        raise RuntimeError("AIV 方案不得把常见 AIC=24 当作 Vector 可用核数，必须读取设备 aiv_core_count")
    for field in ("shape_model", "parameters", "buffers", "transfers", "compute", "proofs"): texts(value[field], field)
    planning_text = "\n".join(
        item for field in ("shape_model", "parameters", "buffers", "transfers", "compute", "proofs")
        for item in value[field]
    )
    if any(token in planning_text for token in ("未知", "未给出", "unknown dimension")):
        raise RuntimeError("planning 不得省略未知 shape 维度")
    if not isinstance(value["task_mapping"], str) or not value["task_mapping"].strip(): raise RuntimeError("task_mapping 必须是非空文本")
    work = value["work"]
    if not isinstance(work, dict) or set(work) != {"current", "target"}: raise RuntimeError("work 必须包含 current/target")
    for side in ("current", "target"):
        metrics = work[side]
        if not isinstance(metrics, dict) or set(metrics) != WORK_KEYS: raise RuntimeError(f"work.{side} 必须覆盖完整静态工作向量")
        if any(not isinstance(number, int) or isinstance(number, bool) or number < 0 for number in metrics.values()): raise RuntimeError(f"work.{side} 只能包含非负整数")
    if work["current"] == work["target"]: raise RuntimeError("非空策略的目标静态工作量不得与当前完全相同")
    if value["pattern"] == "window_pooling":
        current, target = work["current"], work["target"]
        if current["scalar_iterations"] > 0 and (
            target["vector_repeat_blocks"] == 0
            or target["scalar_iterations"] * 10 >= current["scalar_iterations"] * 9
        ):
            raise RuntimeError("Window/Pooling 不得只优化 GM/DMA 而保留同数量级 Scalar 数学主体")
    if value["pattern"] == "prefix_scan":
        current, target = work["current"], work["target"]
        if current["scalar_iterations"] <= 0:
            raise RuntimeError("Prefix/Scan current 必须记录逐元素状态依赖的 Scalar 工作")
        if (target["vector_repeat_blocks"] <= 0
                or target["scalar_iterations"] * 2 >= current["scalar_iterations"]):
            raise RuntimeError("Prefix/Scan 必须把逐元素 Scalar 依赖降为分块 carry 并由 Vector scan 承担局部主体")
    if value["pattern"] == "existing_cube":
        required = ("A有效字节", "B有效字节", "C有效字节", "A重载次数", "B重载次数")
        missing = [term for term in required if term not in planning_text]
        if missing:
            raise RuntimeError(f"已有 Cube planning 缺少完整 A/B/C 复用账：{','.join(missing)}")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--planning", required=True, type=Path); parser.add_argument("--project-dir", required=True, type=Path); args = parser.parse_args()
    try: validate(args.planning.resolve(), args.project_dir.resolve())
    except (OSError, json.JSONDecodeError, RuntimeError) as error: raise SystemExit(f"INVALID_PLANNING: {error}") from error
    print(f"valid_planning={args.planning.resolve()}")

if __name__ == "__main__": main()
