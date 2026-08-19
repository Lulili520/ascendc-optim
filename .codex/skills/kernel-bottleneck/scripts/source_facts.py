#!/usr/bin/env python3
"""Extract ephemeral, source-grounded facts used to prevent missed bottlenecks."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


def _texts(project: Path, folder: str) -> str:
    return "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in sorted((project / folder).rglob("*")) if path.is_file()
    )


def extract(project: Path) -> dict:
    host, kernel = _texts(project, "op_host"), _texts(project, "op_kernel")
    performance = json.loads((project / "performance/performance.json").read_text())
    hardware = performance.get("hardware") or {}
    task = performance.get("task") or {}
    per_core = performance.get("per_core") or {}
    try:
        block_num = int(task.get("Block Num"))
    except (TypeError, ValueError):
        block_num = None
    cores = per_core.get("active_cores") or hardware.get("aiv_core_count") or hardware.get("aic_core_count")
    waves = math.ceil(block_num / cores) if isinstance(block_num, int) and isinstance(cores, int) and cores > 0 else None
    constants = {
        name: int(value)
        for name, value in re.findall(
            r"(?:constexpr|const)\s+(?:u?int(?:32|64)_t|unsigned\s+int)\s+(\w+)\s*=\s*(\d+)",
            host + "\n" + kernel,
        )
    }
    tile_values = [value for name, value in constants.items() if re.search(r"TILE|CHUNK", name, re.I)]
    input_shape = [int(value) for value in re.findall(r"\d+", str(task.get("Input Shapes") or ""))]
    representative_inner = input_shape[-1] if input_shape else None
    small_work_unit = bool(
        tile_values and representative_inner and min(tile_values) < representative_inner
        and isinstance(hardware.get("ub_bytes_per_core"), int)
        and min(tile_values) * 32 < hardware["ub_bytes_per_core"] // 5
        and isinstance(waves, int) and waves >= 4
    )
    queue_slots = [int(value) for value in re.findall(r"InitBuffer\s*\([^,]+,\s*(\d+)\s*,", kernel)]
    gm_pointers = re.findall(
        r"__gm__\s+(const\s+)?[\w:<>]+\s*\*\s*(\w+)", kernel
    )
    gm_inputs = {name for qualifier, name in gm_pointers if qualifier}
    gm_outputs = {name for qualifier, name in gm_pointers if not qualifier}
    # Follow simple local aliases of a known GM input pointer.  Kernel hot
    # loops commonly bind `p = xPtr_ + base + lane` before indexing `p[r*I]`.
    for alias, source in re.findall(
        r"(?:const\s+)?__gm__\s+[\w:<>]+\s*\*\s*(\w+)\s*=\s*(\w+)", kernel
    ):
        if source in gm_inputs:
            gm_inputs.add(alias)
    indexed_accesses = re.findall(r"\b(\w+)\s*\[([^\]\n]+)\]", kernel)
    gm_reads = [(name, index) for name, index in indexed_accesses if name in gm_inputs]
    gm_writes = [
        (name, index) for name, index in indexed_accesses
        if name in gm_outputs and re.search(
            rf"\b{re.escape(name)}\s*\[\s*{re.escape(index)}\s*\]\s*=", kernel
        )
    ]
    # A scalar GM address whose index multiplies a loop induction value by a
    # fixed row/plane stride is mechanically a regular strided transfer.  It
    # remains distinct from task size: changing blockDim cannot coalesce it.
    strided_scalar_reads = [
        {"pointer": name, "index": index.strip()}
        for name, index in gm_reads
        if "*" in index and re.search(r"\b(?:r|row|k|reduce\w*)\b", index, re.I)
    ]
    scalar_gm_writes = [
        {"pointer": name, "index": index.strip()} for name, index in gm_writes
    ]
    loop_count = len(re.findall(r"\bfor\s*\(", kernel))
    scalar_reduction = bool(
        loop_count >= 2
        and re.search(r"\b(?:cur|max|min|sum|acc)\w*\s*=", kernel, re.I)
        and re.search(r"\bif\s*\([^)]*[<>][^)]*\)", kernel)
        and not re.search(r"\b(?:Reduce\w*|WholeReduce\w*|BlockReduce\w*)\s*\(", kernel)
    )
    vector_state_update = bool(
        re.search(r"\bLocalTensor\s*<", kernel)
        and re.search(r"\b(?:Compare|Select|Reduce\w*|WholeReduce\w*|BlockReduce\w*)\s*\(", kernel)
    )
    if vector_state_update:
        scalar_reduction = False
    output_elements = None
    reduction_depth = None
    if constants:
        outer = next((v for k, v in constants.items() if re.search(r"BATCH|OUTER", k, re.I)), None)
        inner = next((v for k, v in constants.items() if re.search(r"INNER", k, re.I)), None)
        reduction_depth = next((v for k, v in constants.items() if re.search(r"REDUCE|REDUCTION", k, re.I)), None)
        if outer and inner:
            output_elements = outer * inner
    facts = {
        "block_num": block_num,
        "active_cores": cores,
        "waves": waves,
        "ub_bytes_per_core": hardware.get("ub_bytes_per_core"),
        "shape": task.get("Input Shapes"),
        "constants": constants,
        "source_counts": {
            "data_copy": len(re.findall(r"\bDataCopy(?:Pad)?\s*\(", kernel)),
            "get_value": kernel.count("GetValue("),
            "set_value": kernel.count("SetValue("),
            "duplicate": len(re.findall(r"\bDuplicate\s*\(", kernel)),
            "compare": len(re.findall(r"\bCompare\s*\(", kernel)),
            "select": len(re.findall(r"\bSelect\s*\(", kernel)),
            "barrier_or_sync": len(re.findall(r"PipeBarrier<|SyncAll\s*\(|WaitFlag<|SetFlag<", kernel)),
        },
        "queue_slots": queue_slots,
        "loop_count": loop_count,
        "gm_scalar_strided_reads": strided_scalar_reads,
        "gm_scalar_writes": scalar_gm_writes,
        "has_scalar_reduction": scalar_reduction,
        "has_vector_state_update": vector_state_update,
        "scalar_reduction_excluded": vector_state_update,
        "hot_path_multipliers": {
            "output_elements": output_elements,
            "reduction_depth": reduction_depth,
            "scalar_reduction_steps": (
                output_elements * reduction_depth
                if scalar_reduction and output_elements and reduction_depth else None
            ),
        },
        "cost_terms": {
            "scalar_reduction_steps": (
                output_elements * reduction_depth
                if scalar_reduction and output_elements and reduction_depth else None
            ),
            "scalar_strided_read_sites": len(strided_scalar_reads),
            "scalar_write_sites": len(scalar_gm_writes),
            "data_copy_sites": len(re.findall(r"\bDataCopy(?:Pad)?\s*\(", kernel)),
            "vector_compute_sites": len(re.findall(r"\b(?:Compare|Select|Add|Mul|Reduce\w*)\s*\(", kernel)),
            "sync_sites": len(re.findall(r"PipeBarrier<|SyncAll\s*\(|WaitFlag<|SetFlag<", kernel)),
            "task_waves": waves,
        },
        "has_block_index_mapping": "GetBlockIdx(" in kernel,
        "has_scalar_lane_loop": bool(re.search(r"for\s*\([^)]*(?:lane|j)\b[^)]*\)[\s\S]{0,500}(?:GetValue|SetValue)\s*\(", kernel)),
        "has_copy_and_vector": bool(re.search(r"\bDataCopy(?:Pad)?\s*\(", kernel) and re.search(r"\b(?:Compare|Select|Add|Mul|Reduce\w*)\s*\(", kernel)),
        "duplicate_in_loop": bool(re.search(r"for\s*\([^)]*\)[\s\S]{0,800}\bDuplicate\s*\(", kernel)),
        "small_work_unit_with_many_waves": small_work_unit,
    }
    return facts


def candidate_causes(facts: dict) -> set[str]:
    """Return heuristic review hints; candidates never force an issue."""
    candidates: set[str] = set()
    waves = facts.get("waves")
    if (isinstance(waves, int) and waves >= 4 and facts.get("has_block_index_mapping")
            and not facts.get("small_work_unit_with_many_waves")):
        candidates.add("overpartitioned_task_mapping")
    if facts.get("small_work_unit_with_many_waves"):
        candidates.add("inefficient_work_unit_size")
    slots = facts.get("queue_slots") or []
    # A single slot is a review hint, not proof that profitable overlap exists.
    if facts.get("has_scalar_lane_loop"):
        candidates.add("scalar_local_lane_compute")
    if facts.get("duplicate_in_loop"):
        candidates.add("repeated_hot_path_overhead")
    if facts.get("gm_scalar_strided_reads"):
        candidates.add("fragmented_regular_strided_transfer")
    if facts.get("has_scalar_reduction"):
        candidates.add("scalar_reduction")
    if facts.get("gm_scalar_writes"):
        candidates.add("fragmented_global_writeback")
    return candidates


_TAXONOMY_PATH = Path(__file__).resolve().parent.parent / "references/cause-taxonomy.json"
REQUIRED_EVIDENCE_KEYS = {
    cause: value[1] for cause, value in json.loads(_TAXONOMY_PATH.read_text()).items()
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", required=True, type=Path)
    args = parser.parse_args()
    facts = extract(args.project_dir.resolve())
    candidates = sorted(candidate_causes(facts))
    facts["candidate_causes"] = candidates
    facts["candidate_evidence_keys"] = {cause: REQUIRED_EVIDENCE_KEYS[cause] for cause in candidates}
    print(json.dumps(facts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
