#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = ROOT / ".codex/skills/kernel-strategy/scripts"
sys.path.insert(0, str(SCRIPTS))


def load_validator():
    path = SCRIPTS / "validate_strategy.py"
    spec = importlib.util.spec_from_file_location("kernel_strategy_validator_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ActionContractTest(unittest.TestCase):
    def test_edit_accepts_source_object_and_transform(self) -> None:
        validator = load_validator()
        validator.validate_edit(
            1,
            "Process 中逐元素 GetValue 累加循环：改为 UB 分块搬入和 Vector Reduce",
        )

    def test_edit_slot_requires_slot_and_source_object(self) -> None:
        validator = load_validator()
        self.assertEqual(
            validator.edit_slot("scalar_loop/Process循环：从GetValue改为Vector Reduce"),
            "scalar_loop",
        )
        with self.assertRaisesRegex(RuntimeError, "operation_slot"):
            validator.edit_slot("Process循环：从GetValue改为Vector Reduce")

    def test_edit_rejects_operation_paraphrase(self) -> None:
        validator = load_validator()
        with self.assertRaisesRegex(RuntimeError, "只重复 operation"):
            validator.validate_edit(1, "提高并行度")

    def test_edit_rejects_missing_transform(self) -> None:
        validator = load_validator()
        with self.assertRaisesRegex(RuntimeError, "缺少明确变换"):
            validator.validate_edit(1, "Process 中的 GetValue 标量归约循环：保持原实现")

    def test_strategy_allows_semantically_merged_slots(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "Op_0"
            for folder in ("bottleneck", "performance", "strategy", "op_host", "op_kernel"):
                (project / folder).mkdir(parents=True, exist_ok=True)
            (project / "op_kernel/x.cpp").write_text("void Process(){ x.GetValue(0); }\n")
            (project / "performance/performance.json").write_text(json.dumps({
                "hardware": {"ub_bytes_per_core": 196352}
            }))
            bottleneck = {
                "reasoning": ["源码事实：Process 使用 GetValue；执行机制：逐lane访问进入Scalar路径；选择依据：生产主循环按 lane 重复。"],
                "issues": [{
                    "evidence": [{
                        "evidence_key": "source.kernel.scalar_local_lane_compute",
                        "source": "op_kernel/x.cpp::Process",
                        "observation": "生产循环调用GetValue",
                    }],
                    "bottleneck": {
                        "bottleneck_key": "compute.scalar_inefficiency",
                        "cause_key": "scalar_local_lane_compute",
                    },
                }],
            }
            strategy = {
                "reasoning": [
                    "因果映射：scalar_local_lane_compute映射到compute.vectorize_scalar_work；"
                    "修复机制：根据 source.kernel.scalar_local_lane_compute@op_kernel/x.cpp::Process，"
                    "GetValue循环改为UB分块Vector路径并处理tail。"
                ],
                "strategies": [{
                    "cause_key": "scalar_local_lane_compute",
                    "strategy_key": "compute.vectorize_scalar_work",
                    "actions": [{
                        "target": "op_kernel/x.cpp::Process",
                        "operation": "vectorize_scalar_work",
                        "edits": [
                            "scalar_loop/GetValue循环：从逐lane读取改为批量处理",
                            "pattern_and_tiling/输入分块：从单元素改为 pattern=ELEMENTWISE_CONTIGUOUS,branch=CHUNKED,fixed_bytes=0,bytes_per_element=4,usable_ub_ratio=0.8,peak_bytes=fixed_bytes+bytes_per_element*chunk,chunk_rule=max_chunk=floor((usable_ub_bytes-fixed_bytes)/bytes_per_element),alignment=32B,tiling_change=false",
                            "vector_sequence/计算序列：从Scalar转换改为Vector转换",
                            "dtype_tail_abi/尾块与输出：从无边界处理改为 valid_len=min(tile,total-start),input_dtype=fp32,compute_dtype=fp32,output_dtype=fp32,DataCopyPad批量写回",
                        ],
                        "constraints": ["只处理有效元素"],
                    }],
                }],
            }
            bp = project / "bottleneck/bottleneck.json"
            sp = project / "strategy/strategy.json"
            bp.write_text(json.dumps(bottleneck, ensure_ascii=False))
            sp.write_text(json.dumps(strategy, ensure_ascii=False))
            validator.validate(bp, sp)
            strategy["strategies"][0]["actions"][0]["edits"].pop(2)
            sp.write_text(json.dumps(strategy, ensure_ascii=False))
            with self.assertRaisesRegex(RuntimeError, "required operation slots"):
                validator.validate(bp, sp)

    def test_derive_covers_all_issues_in_order(self) -> None:
        from derive_strategy import RULES, derive
        report = {"issues": [
            {"bottleneck": {"cause_key": "scalar_reduction"}},
            {"bottleneck": {"cause_key": "serial_pipeline_stages"}},
        ]}
        result = derive(report)
        self.assertEqual(
            [item["cause_key"] for item in result["strategies"]],
            ["scalar_reduction", "serial_pipeline_stages"],
        )
        self.assertEqual(
            set(RULES),
            set(load_validator().CAUSES),
            "每个具体 cause 必须且只能有一个固定 strategy 映射",
        )

    def test_corrected_causes_have_semantic_strategies(self) -> None:
        from derive_strategy import RULES

        self.assertEqual(
            RULES["atomic_write_contention"],
            "memory.privatize_contended_write",
        )
        self.assertEqual(
            RULES["over_synchronization"],
            "pipeline.reduce_sync_scope",
        )
        self.assertEqual(
            RULES["ub_bank_conflict"],
            "memory.relayout_onchip_buffer",
        )
        self.assertEqual(
            RULES["fragmented_regular_strided_transfer"],
            "memory.batch_strided_transfer",
        )

    def test_cross_strategy_conflicting_object_is_rejected(self) -> None:
        validator = load_validator()
        strategies = [
            {"actions": [{"target": "op_kernel/x.cpp::Process", "edits": ["scalar_loop/main：从A改为B"]}]},
            {"actions": [{"target": "op_kernel/x.cpp::Process", "edits": ["scalar_loop/main：从A改为C"]}]},
        ]
        with self.assertRaisesRegex(RuntimeError, "冲突变换"):
            validator._validate_cross_strategy(strategies)

    def test_resize_work_unit_requires_recomputable_ub_formula(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "Op_0"
            (project / "performance").mkdir(parents=True)
            (project / "performance/performance.json").write_text(json.dumps({
                "hardware": {"ub_bytes_per_core": 196352}
            }))
            edits = [
                "active_buffer_budget/峰值UB：从未计算改为 fixed_bytes=0,bytes_per_element=33,usable_ub_ratio=0.8；peak_bytes=fixed_bytes+bytes_per_element*tile",
                "candidate_work_unit/TILE：从固定64改为 candidate_rule=枚举合法对齐候选；max_tile=floor((usable_ub_bytes-fixed_bytes)/bytes_per_element)",
                "task_parallelism/任务数：从只按tile切分改为 total_tasks=batch*ceil(inner/tile)",
                "alignment_and_tail/边界：从固定整块改为 alignment=可靠搬运对齐；valid_len=min(tile,inner-start)",
            ]
            validator._validate_resize_work_unit(project, edits)
            edits[1] = edits[1].replace("candidate_rule=枚举合法对齐候选", "选择 1408")
            with self.assertRaisesRegex(RuntimeError, "candidate_rule"):
                validator._validate_resize_work_unit(project, edits)


if __name__ == "__main__":
    unittest.main()
