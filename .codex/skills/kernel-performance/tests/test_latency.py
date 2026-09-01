import csv
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from collect_performance import msprof_command, parse_task_duration, preserve_pipe_utilization_csv


class LatencyTest(unittest.TestCase):
    def test_msprof_command_exports_csv(self):
        command = msprof_command(Path("/tmp/profile"), "python runner.py")
        self.assertIn("--export=on", command)
        self.assertIn("--aic-metrics=PipeUtilization", command)

    def test_selects_exact_operator_instead_of_last_task(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "op_summary.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["Op Name", "Task Type", "Task Duration(us)"])
                writer.writeheader()
                writer.writerow({"Op Name": "WantedCustom", "Task Type": "AI_VECTOR_CORE", "Task Duration(us)": "12.5"})
                writer.writerow({"Op Name": "Unrelated", "Task Type": "AI_VECTOR_CORE", "Task Duration(us)": "1"})
            self.assertEqual(parse_task_duration(path, "WantedCustom"), 12.5)

    def test_preserves_pipe_utilization_csv_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            report = root / "performance"
            report.mkdir()
            source.write_bytes(b"Op Name,Task Duration(us)\nWantedCustom,12.5\n")
            target = preserve_pipe_utilization_csv(source, report)
            self.assertEqual(target.name, "op_summary_PipeUtilization.csv")
            self.assertEqual(target.read_bytes(), source.read_bytes())


if __name__ == "__main__":
    unittest.main()
