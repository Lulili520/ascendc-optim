#!/usr/bin/env python3
"""Regression tests for kernel-policy-data export contracts."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / ".codex/skills/kernel-policy-data/scripts/export_policy_data.py"


def load_exporter():
    spec = importlib.util.spec_from_file_location("kernel_policy_exporter", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PolicyExportTest(unittest.TestCase):
    def test_exporter_does_not_gate_on_child_bottleneck_causes(self) -> None:
        exporter = load_exporter()
        self.assertFalse(hasattr(exporter, "validate_parent_causes_resolved"))

    def test_policy_knowledge_matches_canonical_contracts(self) -> None:
        exporter = load_exporter()
        knowledge = exporter.load_json(
            ROOT / ".codex/skills/kernel-policy-data/references/policy-knowledge.json"
        )
        exporter.validate_policy_knowledge(knowledge)

    def test_compact_performance_excludes_absolute_times(self) -> None:
        exporter = load_exporter()
        compact = exporter.compact_performance({
            "kernel_latency_us": 10.0,
            "task": {
                "aicore_time_us": 8.0,
                "aiv_time_us": 7.0,
                "max_core_time_us": 8.0,
                "head_overhead_ratio": 20.0,
            },
        })
        self.assertNotIn("kernel_latency_us", compact)
        self.assertNotIn("aicore_time_us", compact["task"])
        self.assertNotIn("aiv_time_us", compact["task"])
        self.assertNotIn("max_core_time_us", compact["task"])
        self.assertEqual(compact["task"]["head_overhead_ratio"], 20.0)
        self.assertEqual(set(compact["per_core"]), {"active_cores", "imbalance_percent"})

    def test_project_policy_merges_reports_without_operation_duplication(self) -> None:
        exporter = load_exporter()
        bottleneck = {
            "reasoning": ["源码事实：GetValue逐lane读取；执行机制：Tensor数据进入Scalar路径；选择依据：主循环按 lane 重复。"],
            "issues": [{
                "evidence": [{"source": "op_kernel/x.cpp::Process", "observation": "逐lane调用GetValue"}],
                "bottleneck": {"bottleneck_key": "compute.scalar_inefficiency", "cause_key": "scalar_local_lane_compute"},
            }],
        }
        strategy = {
            "reasoning": ["因果映射：scalar_local_lane_compute映射到vectorize；修复机制：批量处理替换逐lane访问。"],
            "strategies": [{
                "cause_key": "scalar_local_lane_compute",
                "strategy_key": "compute.vectorize_scalar_work",
                "actions": [{
                    "target": "op_kernel/x.cpp::Process", "operation": "vectorize_scalar_work",
                    "edits": ["scalar_loop/GetValue循环：从逐lane读取改为批量Vector处理"],
                    "constraints": ["tail只处理有效元素"],
                }],
            }],
        }
        projected = exporter.project_policy(bottleneck, strategy)
        self.assertEqual(list(projected), ["policy"])
        self.assertNotIn("operation", projected["policy"][0]["changes"][0])

    def test_implementation_link_matches_real_source_diff(self) -> None:
        exporter = load_exporter()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent, child = root / "Op_0", root / "Op_1"
            for project in (parent, child):
                (project / "op_host").mkdir(parents=True)
                (project / "op_kernel").mkdir()
                (project / "strategy").mkdir()
                (project / "bottleneck").mkdir()
                (project / "op_kernel/x.cpp").write_text("void Process() {}\n")
            (child / "op_kernel/x.cpp").write_text("void Process() { run(); }\n")
            strategy = {"strategies": [{"cause_key": "c", "strategy_key": "k", "actions": [{
                "target": "op_kernel/x.cpp::Process", "operation": "op",
                "edits": ["Process：改为调用run"], "constraints": ["ABI不变"],
            }]}]}
            (parent / "strategy/strategy.json").write_text(json.dumps(strategy))
            implementation = {
                "strategy_keys": ["k"],
                "actions": [{"action_index": 1, "implementation_summary": "调用run"}],
                "modified_files": ["op_kernel/x.cpp"],
            }
            (parent / "bottleneck/bottleneck.json").write_text(json.dumps({
                "issues": [{"bottleneck": {"cause_key": "repeated_hot_path_overhead"}}]
            }))
            (child / "strategy/implementation.json").write_text(json.dumps(implementation))
            exporter.validate_implementation_link(parent, strategy, child)
            implementation["modified_files"] = ["op_host/y.cpp"]
            (child / "strategy/implementation.json").write_text(json.dumps(implementation))
            with self.assertRaisesRegex(ValueError, "真实 diff"):
                exporter.validate_implementation_link(parent, strategy, child)

    def test_exported_samples_have_exact_public_fields_and_no_child_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "sample.jsonl"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--output", str(output),
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout)
            samples = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            for sample in samples:
                self.assertEqual(list(sample), ["ops", "input", "output"])
                tags = ["[PERF]", "[HOST]", "[KERNEL]"]
                positions = [sample["input"].index(tag) for tag in tags]
                self.assertEqual(positions, sorted(positions))
                self.assertNotIn("reference_candidates", sample["input"])
                self.assertNotIn("child_latency", sample["input"])
                self.assertNotIn("schema_version", sample["input"])
                self.assertNotIn('\"kernel_latency_us\"', sample["input"])
                self.assertNotIn('\"aicore_time_us\"', sample["input"])
                self.assertNotIn('\"aiv_time_us\"', sample["input"])
                self.assertNotIn('\"max_core_time_us\"', sample["input"])
                self.assertEqual(set(json.loads(sample["output"])), {"policy"})
            audit = json.loads(
                output.with_suffix(".jsonl.audit.json").read_text(encoding="utf-8")
            )
            eligible = [item for item in audit["items"] if item.get("eligible")]
            self.assertEqual(len(eligible), len(samples))
            self.assertTrue(all(item["reduction_percent"] > 1.0 for item in eligible))
            self.assertEqual(audit["exported"], len(samples))


if __name__ == "__main__":
    unittest.main()
