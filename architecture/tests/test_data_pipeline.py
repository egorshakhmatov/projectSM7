"""Integration checks for the canonical 15-minute dataset."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "data-project"
sys.path.insert(0, str(DATA))
from prepare_data import load_data, load_split, valid_origins


class DelayDataChecks(unittest.TestCase):
    def test_config_matches_contract(self):
        config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["input"]["window_size"], 8)
        self.assertEqual(config["input"]["feature_names"], ["delay_A", "delay_B", "delay_C"])
        self.assertEqual(config["input"]["step_minutes"], 15)
        self.assertEqual(config["input"]["horizon_minutes"], 60)
        self.assertEqual(config["training"]["loss"], "MSELoss")
        self.assertEqual(config["model"]["output_size"], 3)

    def test_canonical_csv_and_boundaries(self):
        rows, labels = load_data()
        self.assertEqual(len(rows), 70176)
        self.assertEqual(list(rows[0]), ["timestamp", "is_holiday", "is_weekend", "delay_A", "delay_B", "delay_C", "split"])
        self.assertEqual(sum(label == "observed_proxy" for label in labels), 21053)
        self.assertEqual(sum(1 for _ in valid_origins(rows, labels)), 64638)

    def test_all_splits(self):
        for split, count in (("train", 45040), ("validation", 9746), ("test", 9852)):
            X, y, sources = load_split(split)
            self.assertEqual((len(X), len(y), len(sources)), (count, count, count))
            self.assertEqual((len(X[0]), len(X[0][0]), len(y[0])), (8, 3, 3))

    def test_launcher_uses_canonical_csv(self):
        run = lambda *args: subprocess.run([sys.executable, "-B", str(ROOT / "run.py"), *args],
                                           cwd=ROOT.parent, capture_output=True, text=True, encoding="utf-8")
        status = run("status", "--json")
        self.assertEqual(status.returncode, 0, status.stderr)
        data = json.loads(status.stdout)
        self.assertTrue(data["artifacts"]["input_csv"]["exists"])
        self.assertIn("../data-project/", data["artifacts"]["input_csv"]["path"])
        self.assertNotIn("generate", data["implemented_commands"])
        checked = run("validate")
        self.assertEqual(checked.returncode, 0, checked.stderr)
        self.assertEqual(json.loads(checked.stdout)["rows"], 70176)
        prepared = run("prepare")
        self.assertEqual(prepared.returncode, 0, prepared.stderr)
        self.assertEqual(json.loads(prepared.stdout)["train"]["X_shape"], [45040, 8, 3])
        for command in ("train", "evaluate", "predict"):
            self.assertEqual(run(command).returncode, 2)


if __name__ == "__main__":
    unittest.main()
