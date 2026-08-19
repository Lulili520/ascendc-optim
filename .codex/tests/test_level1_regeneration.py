import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "run_level1_regeneration.py"
SPEC = importlib.util.spec_from_file_location("level1_regeneration", SCRIPT)
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)


class DurableStateMachineTest(unittest.TestCase):
    def test_controller_rejects_unknown_arguments_instead_of_running(self):
        self.assertEqual(RUNNER.requested_mode([]), "run")
        self.assertEqual(RUNNER.requested_mode(["--help"]), "help")
        with self.assertRaises(SystemExit):
            RUNNER.requested_mode(["--unknown"])

    def test_taxonomy_strategy_and_operation_slots_are_complete(self):
        root = SCRIPT.parents[1]
        taxonomy = json.loads((
            root / ".codex/skills/kernel-bottleneck/references/cause-taxonomy.json"
        ).read_text())
        derive_path = root / ".codex/skills/kernel-strategy/scripts/derive_strategy.py"
        derive_spec = importlib.util.spec_from_file_location("derive_strategy_contract", derive_path)
        derive = importlib.util.module_from_spec(derive_spec)
        assert derive_spec.loader is not None
        derive_spec.loader.exec_module(derive)
        slots = json.loads((
            root / ".codex/skills/kernel-strategy/references/operation-slots.json"
        ).read_text())
        self.assertEqual(set(taxonomy), set(derive.RULES))
        self.assertEqual(set(derive.OPERATIONS.values()), set(slots))

    def test_controller_prepares_clean_child_and_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent, child = root / "Op_0", root / "Op_1"
            for folder in ("op_host", "op_kernel", "CppExtension", "precision", "performance", "bottleneck", "strategy"):
                (parent / folder).mkdir(parents=True)
            (parent / "op_kernel/x.cpp").write_text("void Process(){}\n")
            (parent / "op_host/x.cpp").write_text("void Tiling(){}\n")
            (parent / "CppExtension/keep.so").write_text("so")
            (parent / "precision/precision.json").write_text("{}")
            (parent / "performance/performance.json").write_text("{}")
            strategy = {"strategies": [{
                "strategy_key": "compute.vectorize_scalar_work",
                "actions": [{"target": "op_kernel/x.cpp::Process"}],
            }]}
            (parent / "strategy/strategy.json").write_text(json.dumps(strategy))
            (parent / "workspace.json").write_text(json.dumps({
                "source_project": "kernel/Op", "vendor": "vendor_op"
            }))
            RUNNER.prepare_child(parent, child, "Op")
            self.assertFalse((child / "precision").exists())
            self.assertFalse((child / "performance").exists())
            self.assertEqual(json.loads((child / "workspace.json").read_text())["status"], "PREPARED")
            attempt = json.loads((child / "strategy/implementation.json").read_text())["attempts"][0]
            self.assertEqual(set(attempt), {
                "attempt", "kind", "trigger", "knowledge_keys", "action_indices",
                "implementation_summary", "modified_files", "validation",
            })

    def test_planning_prompt_forbids_expensive_gates(self):
        action = {"kind": "agent", "stage": "planning", "version": 0}
        prompt = RUNNER.agent_prompt(Path("/tmp/stage.json"), action)
        self.assertIn("precision、performance、msprof 和 reference 均由控制器执行", prompt)
        self.assertIn("源码只读一次", prompt)
        self.assertIn("source_facts", prompt)
        self.assertIn("最多 3 个兼容 issues", prompt)
        self.assertIn("不建立成本账本", prompt)

    def test_four_round_limit_and_bounded_repairs_have_separate_retry_keys(self):
        self.assertEqual(RUNNER.MAX_OPTIMIZATION_ROUNDS, 4)
        self.assertEqual(
            RUNNER.stage_key({"version": 3, "stage": "repair", "repair": 2}),
            "v3:repair2",
        )
        self.assertEqual(
            RUNNER.stage_key({"version": 3, "stage": "planning"}),
            "v3:planning",
        )

    def test_knowledge_is_selected_by_actions_and_exact_device(self):
        strategy = {"strategies": [{"actions": [
            {"operation": "vectorize_scalar_work"},
            {"operation": "coalesce_global_transfer"},
        ]}]}
        selected = RUNNER.implementation_knowledge(strategy)
        self.assertEqual(selected["device_soc"], "Ascend910B2C")
        self.assertEqual(selected["architecture"], "dav-2201")
        references = selected["references"]
        self.assertIn("implementation-core.md", references[0]["path"])
        self.assertIn("transfer", references[0]["sections"])
        self.assertIn("vector", references[0]["sections"])
        self.assertIn("lifetime", references[0]["sections"])
        self.assertIn("architecture-knowledge.md", references[1]["path"])
        self.assertEqual(references[1]["profile"], "dav-2201")
        self.assertTrue(all("cannbot-skills" not in item["path"] for item in references))

    def test_unknown_device_does_not_load_architecture_overlay(self):
        with tempfile.TemporaryDirectory() as directory:
            level = Path(directory) / "KernelBench910B/level1"
            hardware = level.parent / "hardware"
            hardware.mkdir(parents=True)
            (hardware / "device_0.json").write_text(json.dumps({
                "identity": {"soc": "FutureUnknownSoC"}
            }))
            previous = RUNNER.LEVEL
            RUNNER.LEVEL = level
            try:
                selected = RUNNER.implementation_knowledge({"strategies": [{"actions": [
                    {"operation": "coalesce_global_transfer"}
                ]}]})
            finally:
                RUNNER.LEVEL = previous
        self.assertIsNone(selected["architecture"])
        self.assertEqual(len(selected["references"]), 1)
        self.assertIn("implementation-core.md", selected["references"][0]["path"])

    def test_implementation_prompt_allows_one_local_revalidation(self):
        prompt = RUNNER.agent_prompt(
            Path("/tmp/stage.json"), {"kind": "agent", "stage": "implementation", "version": 0}
        )
        self.assertIn("最多两次", prompt)
        self.assertIn("architecture=null", prompt)
        self.assertNotIn("DAV_2201", prompt)

    def test_agent_context_is_reused_only_within_one_operator(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "run.jsonl"
            log.write_text('{"type":"thread.started","thread_id":"thread-op-a"}\n')
            self.assertEqual(RUNNER.thread_id_from_log(log), "thread-op-a")
        first = RUNNER.agent_command(None, Path("/tmp/final"), "plan")
        resumed = RUNNER.agent_command("thread-op-a", Path("/tmp/final"), "implement")
        self.assertNotIn("--ephemeral", first)
        self.assertNotIn("resume", first)
        self.assertIn("resume", resumed)
        self.assertIn("thread-op-a", resumed)

    def test_improvement_uses_formal_latency(self):
        with tempfile.TemporaryDirectory() as directory:
            parent, child = Path(directory) / "Op_0", Path(directory) / "Op_1"
            for project, latency in ((parent, 100.0), (child, 80.0)):
                (project / "performance").mkdir(parents=True)
                (project / "performance/performance.json").write_text(
                    json.dumps({"kernel_latency_us": latency})
                )
            self.assertAlmostEqual(RUNNER.improvement_percent(parent, child), 20.0)


if __name__ == "__main__":
    unittest.main()
