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

    def test_training_knowledge_is_compact_projection(self) -> None:
        exporter = load_exporter()
        knowledge = exporter.load_json(
            ROOT / ".codex/skills/kernel-policy-data/references/policy-knowledge.json"
        )
        projected = exporter.training_policy_knowledge(knowledge)
        self.assertEqual(
            set(projected),
            {
                "decision_order", "thresholds", "metric_evidence_sources",
                "cause_rule_fields", "cause_rules", "operation_slots",
            },
        )
        self.assertIn("metric.core.active_count", projected["metric_evidence_sources"])
        self.assertIn("increase_parallelism", projected["operation_slots"])

    def test_source_symbols_do_not_select_answer(self) -> None:
        exporter = load_exporter()
        project = ROOT / "kernel_workspace/KernelBench910B/level1/ArgmaxOverADimensionCustom_0"
        symbols = exporter.source_symbols(project)
        self.assertIn("op_host/", symbols)
        self.assertIn("op_kernel/", symbols)
        self.assertIn("KernelArgmaxOverADimensionCustom::Process", symbols)
        self.assertNotIn("op_kernel/argmax_over_a_dimension_custom.cpp::Process\n", symbols)
        self.assertNotIn("answer", symbols.lower())

    def test_implementation_link_matches_real_source_diff(self) -> None:
        exporter = load_exporter()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent, child = root / "Op_0", root / "Op_1"
            for project in (parent, child):
                (project / "op_host").mkdir(parents=True)
                (project / "op_kernel").mkdir()
                (project / "strategy").mkdir()
                (project / "op_kernel/x.cpp").write_text("void Process() {}\n")
            (child / "op_kernel/x.cpp").write_text("void Process() { run(); }\n")
            strategy = {"strategy": {"strategy_key": "k", "actions": [{
                "target": "op_kernel/x.cpp::Process", "operation": "op",
                "edits": ["Process：改为调用run"], "constraints": ["ABI不变"],
            }]}}
            implementation = {
                "strategy_key": "k",
                "actions": [{"action_index": 1, "implementation_summary": "调用run"}],
                "modified_files": ["op_kernel/x.cpp"],
            }
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
                tags = [
                    "[TASK_INSTRUCT]", "[POLICY_KNOWLEDGE_JSON]", "[OPERATOR_JSON]",
                    "[PERFORMANCE_JSON]", "[SOURCE_SYMBOLS]", "[OP_HOST_SOURCE]", "[OP_KERNEL_SOURCE]",
                    "[OUTPUT_FORMAT]",
                ]
                positions = [sample["input"].index(tag) for tag in tags]
                self.assertEqual(positions, sorted(positions))
                self.assertNotIn("reference_candidates", sample["input"])
                self.assertNotIn("child_latency", sample["input"])
                self.assertNotIn("schema_version", sample["input"])
                self.assertNotIn('\"kernel_latency_us\"', sample["input"])
                self.assertNotIn('\"aicore_time_us\"', sample["input"])
                self.assertNotIn('\"aiv_time_us\"', sample["input"])
                self.assertNotIn('\"max_core_time_us\"', sample["input"])
                self.assertIn("[BOTTLENECK_JSON]", sample["output"])
                self.assertIn("[STRATEGY_JSON]", sample["output"])
            audit = json.loads(
                output.with_suffix(".jsonl.audit.json").read_text(encoding="utf-8")
            )
            eligible = [item for item in audit["items"] if item.get("eligible")]
            self.assertEqual(len(eligible), len(samples))
            self.assertTrue(all(item["reduction_percent"] > 1.0 for item in eligible))
            self.assertEqual(audit["exported"], len(samples))


if __name__ == "__main__":
    unittest.main()
