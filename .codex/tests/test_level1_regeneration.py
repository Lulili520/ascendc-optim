import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "run_level1_regeneration.py"
SPEC = importlib.util.spec_from_file_location("level1_regeneration", SCRIPT)
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)


class DurableStateMachineTest(unittest.TestCase):
    def test_completed_operator_is_terminal_without_agent(self):
        action = RUNNER.durable_action("ArgmaxOverADimensionCustom")
        self.assertEqual(action["kind"], "terminal")
        self.assertEqual(action["state"], "completed_two_rounds")

    def test_precision_pass_resumes_at_performance_command(self):
        action = RUNNER.durable_action("AveragePooling2dCustom")
        self.assertEqual((action["kind"], action["stage"], action["version"]),
                         ("command", "performance", 1))

    def test_exhausted_repairs_do_not_open_another_agent(self):
        action = RUNNER.durable_action("AveragePooling1dCustom")
        self.assertEqual(action["kind"], "terminal")
        self.assertEqual(action["state"], "precision_failed")

    def test_planning_prompt_forbids_expensive_gates(self):
        action = RUNNER.durable_action("AveragePooling3dCustom")
        self.assertEqual((action["kind"], action["stage"]), ("agent", "planning"))
        prompt = RUNNER.agent_prompt(Path("/tmp/stage.json"), action)
        self.assertIn("不要运行 precision、performance", prompt)
        self.assertIn("源码只读取一次", prompt)


if __name__ == "__main__":
    unittest.main()
