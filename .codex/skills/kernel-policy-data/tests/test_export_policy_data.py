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

    def test_system_contract_is_canonical_and_complete(self) -> None:
        exporter = load_exporter()
        knowledge = exporter.load_json(
            ROOT / ".codex/skills/kernel-policy-data/references/policy-knowledge.json"
        )
        projected = exporter.canonical_policy_contract(knowledge)
        self.assertEqual(
            set(projected),
            {
                "decision_order", "metric_evidence_sources", "cause_rule_fields", "cause_rules",
                "operation_slot_fields", "operation_slots", "action_fields",
            },
        )
        self.assertIn("performance.core.active", projected["metric_evidence_sources"])
        slots = {row[0]: row[1:] for row in projected["operation_slots"]}
        self.assertIn("increase_parallelism", slots)
        self.assertEqual(
            slots["increase_parallelism"][0][0],
            "parallel_limit",
        )
        self.assertTrue(all(len(row) == len(projected["cause_rule_fields"]) for row in projected["cause_rules"]))
        self.assertTrue(all(len(row) == len(projected["operation_slot_fields"]) for row in projected["operation_slots"]))

    def test_policy_merges_by_cause(self) -> None:
        exporter = load_exporter()
        bottleneck = {"issues": [{"bottleneck": {
            "bottleneck_key": "b", "cause_key": "c",
        }, "evidence": ["source:x"]}], "reasoning": ["why"]}
        strategy = {"strategies": [{
            "cause_key": "c", "strategy_key": "s", "actions": [{"target": "x"}],
        }], "reasoning": ["how"]}
        self.assertEqual(exporter.merge_policy(bottleneck, strategy), {"strategies": [{
            "bottleneck_key": "b", "cause_key": "c", "evidence": ["source:x"],
            "strategy_key": "s", "reasoning": ["why", "how"],
            "actions": [{"target": "x"}],
        }]})

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
            strategy = {"strategies": [{"strategy_key": "k", "actions": [{
                "target": "op_kernel/x.cpp::Process", "operation": "op",
                "edits": ["Process：改为调用run"], "constraints": ["ABI不变"],
            }]}]}
            implementation = {
                "strategy_keys": ["k"],
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
        exporter = load_exporter()
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
                self.assertEqual(list(sample), ["system_prompt", "input", "output", "ops"])
                self.assertIn("cause_rules", sample["system_prompt"])
                self.assertIn('"strategies"', sample["system_prompt"])
                tags = ["[OPERATOR]", "[PERF]", "[HOST]", "[KERNEL]"]
                positions = [sample["input"].index(tag) for tag in tags]
                self.assertEqual(positions, sorted(positions))
                for removed in ("[TASK_INSTRUCT]", "[POLICY_KNOWLEDGE_JSON]", "[SOURCE_SYMBOLS]", "[OUTPUT_FORMAT]"):
                    self.assertNotIn(removed, sample["input"])
                self.assertNotIn("reference_candidates", sample["input"])
                self.assertNotIn("child_latency", sample["input"])
                self.assertNotIn("schema_version", sample["input"])
                self.assertNotIn('\"kernel_latency_us\"', sample["input"])
                self.assertNotIn('\"aicore_time_us\"', sample["input"])
                self.assertNotIn('\"aiv_time_us\"', sample["input"])
                self.assertNotIn('\"max_core_time_us\"', sample["input"])
                output_value = json.loads(sample["output"])
                self.assertEqual(list(output_value), ["strategies"])
                for item in output_value["strategies"]:
                    self.assertEqual(list(item), [
                        "bottleneck_key", "cause_key", "evidence", "strategy_key",
                        "reasoning", "actions",
                    ])
                    self.assertEqual(len(item["reasoning"]), 2)
            audit = json.loads(
                output.with_suffix(".jsonl.audit.json").read_text(encoding="utf-8")
            )
            system_prompt = Path(audit["system_prompt"])
            self.assertTrue(system_prompt.is_file())
            prompt_text = system_prompt.read_text(encoding="utf-8")
            self.assertIn("cause_rules", prompt_text)
            self.assertIn('"strategies"', prompt_text)
            self.assertEqual(exporter.sha256_file(system_prompt), audit["system_prompt_sha256"])
            self.assertEqual(exporter.sha256_file(output), audit["dataset_sha256"])
            self.assertTrue(all(sample["system_prompt"] == prompt_text for sample in samples))
            eligible = [item for item in audit["items"] if item.get("eligible")]
            self.assertEqual(len(eligible), len(samples))
            self.assertTrue(all(item["reduction_percent"] > 1.0 for item in eligible))
            self.assertEqual(audit["exported"], len(samples))


if __name__ == "__main__":
    unittest.main()
