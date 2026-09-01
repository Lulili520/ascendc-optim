import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/export_policy_data.py"
SPEC = importlib.util.spec_from_file_location("exporter", SCRIPT)
EXPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORTER)


class DataContractTest(unittest.TestCase):
    def test_prompt_teaches_fields_without_performance_input(self):
        prompt = EXPORTER.system_prompt()
        self.assertIn("reasoning", prompt)
        self.assertIn("evidence", prompt)
        self.assertIn("targets", prompt)
        self.assertIn("changes", prompt)
        self.assertIn("guards", prompt)
        self.assertIn("kinds", prompt)
        self.assertIn("GM总字节减少=reuse_onchip", prompt)
        self.assertIn("仅GM事务减少=batch_transfer", prompt)
        self.assertIn("可编译、语义闭合且覆盖全部问题", prompt)
        self.assertIn("禁止将Scalar/Vector contraction转换为Cube", prompt)
        self.assertIn("[任务]", prompt)
        self.assertIn("[边界]", prompt)
        self.assertIn("同一kind使用相同术语和句式", prompt)
        self.assertIn("以分号一一对应", prompt)
        self.assertIn("全部有直接证据且可确定修复的问题", prompt)
        self.assertNotIn('{"strategy":null}', prompt)
        self.assertIn("只包含已有验证收益的非空策略", prompt)
        self.assertIn("直达目标结构", prompt)
        self.assertLess(len(prompt), 1050)
        self.assertNotIn("纯冗余→重复GM→碎片搬运→数学主体→已有Cube→work unit→任务所有权→流水", prompt)
        self.assertNotIn("静态执行总次数降序", prompt)
        self.assertNotIn("reuse_onchip→batch_transfer", prompt)

    def test_exported_op_metadata_does_not_use_level(self):
        source = SCRIPT.read_text()
        self.assertNotIn('"level": entry.get("level")', source)

    def test_discovers_only_flat_parent_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("Op_0", "Op_1", "Op_4", "optimization_runs", "level1"):
                (root / name).mkdir()
            (root / "level1/Legacy_0").mkdir()
            found = [path.name for path in EXPORTER.discover_parents(root, set())]
            self.assertEqual(found, ["Op_0", "Op_1", "Op_4"])

    def test_only_terminal_queue_operators_are_export_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = Path(directory) / "queue.json"
            queue.write_text(json.dumps({"items": [
                {"operator": "Done", "state": "stopped_no_strategy"},
                {"operator": "StillRunning", "state": "running"},
                {"operator": "Retry", "state": "pending"},
                {"operator": "Failed", "state": "agent_failed"},
            ]}))
            self.assertEqual(EXPORTER.queue_states(queue), {
                "Done": "stopped_no_strategy", "StillRunning": "running",
                "Retry": "pending", "Failed": "agent_failed",
            })
            self.assertIsNone(EXPORTER.queue_states(queue.with_name("missing.json")))

    def test_missing_terminal_policy_never_mechanically_composes(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            self.assertIsNone(EXPORTER.load_terminal_policy(parent, "Op_2"))


if __name__ == "__main__":
    unittest.main()
