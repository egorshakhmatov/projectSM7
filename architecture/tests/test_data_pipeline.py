"""Integration checks for the canonical 15-minute dataset."""
import json
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess
import sys
import unittest
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "data-project"
sys.path.insert(0, str(DATA))
from prepare_data import FIELDS, audit_external, csv_digest, load_data, load_split, valid_origins


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


class ExternalAuditChecks(unittest.TestCase):
    def fixture(self):
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        rows = []
        for i in range(100):
            stamp = start + timedelta(minutes=15 * i)
            rows.append(dict(timestamp=stamp.isoformat(), is_holiday='0',
                             is_weekend=str(int(stamp.weekday() >= 5)),
                             delay_A=str(i % 5 / 10), delay_B=str(i % 4 / 10), delay_C=str(i % 3 / 10),
                             split='train' if i < 70 else 'validation' if i < 85 else 'test'))
        return rows

    def audit(self, rows, fields=FIELDS, cutoff=datetime(2025, 1, 1, tzinfo=timezone.utc)):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'external.csv'
            with path.open('w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            before = csv_digest(path)
            report = audit_external(path, cutoff)
            self.assertEqual(csv_digest(path), before)
            return report

    def test_valid_external_requires_manual_review(self):
        report = self.audit(self.fixture())
        self.assertEqual(report['status'], 'manual_review_required')
        self.assertEqual(report['metrics']['train']['candidate_windows_before_provenance'], 59)
        self.assertIn('PROVENANCE_REQUIRED', {i['code'] for i in report['issues']})
        self.assertFalse(any(i['level'] == 'error' for i in report['issues']))

    def test_reject_future_duplicate_and_split_leakage(self):
        rows = self.fixture()
        rows[10]['timestamp'] = rows[9]['timestamp']
        rows[50]['split'] = 'test'
        rows[99]['timestamp'] = '2026-01-01T00:00:00+00:00'
        report = self.audit(rows)
        self.assertEqual(report['status'], 'rejected')
        self.assertTrue({'FUTURE_OBSERVATION', 'DUPLICATE_TIMESTAMP', 'SPLIT_LEAKAGE', 'TIME_STEP'} <= {i['code'] for i in report['issues']})
        self.assertNotIn('candidate_windows_before_provenance', report['metrics']['train'])

    def test_reject_extra_fields_and_invalid_delays(self):
        rows = self.fixture()
        for row in rows:
            row['future_delay'] = '2'
        rows[0]['delay_A'] = 'nan'
        rows[1]['delay_B'] = '-1'
        rows[2]['is_weekend'] = '1'
        report = self.audit(rows, (*FIELDS, 'future_delay'))
        self.assertEqual(report['status'], 'rejected')
        self.assertTrue({'SCHEMA', 'INVALID_DELAY', 'WEEKEND_MISMATCH'} <= {i['code'] for i in report['issues']})

    def test_protect_canonical_from_overwrite(self):
        before = csv_digest(DATA / 'traffic_delays.csv')
        result = subprocess.run([sys.executable, '-B', str(DATA / 'prepare_data.py'), '--source', 'not_a_migration.csv'],
                                capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(result.returncode, 1)
        self.assertIn('защищён от перезаписи', result.stderr)
        self.assertEqual(csv_digest(DATA / 'traffic_delays.csv'), before)

    def test_audit_cli_rejects_invalid_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'invalid.csv'
            path.write_text('timestamp,delay_A\n2024-01-01T00:00:00Z,1\n', encoding='utf-8')
            before = csv_digest(path)
            result = subprocess.run([sys.executable, '-B', str(DATA / 'prepare_data.py'), '--audit', str(path)],
                                    capture_output=True, text=True, encoding='utf-8')
            self.assertEqual(result.returncode, 1)
            report = json.loads(result.stdout)
            self.assertEqual(report['status'], 'rejected')
            self.assertIn('SCHEMA', {i['code'] for i in report['issues']})
            self.assertEqual(csv_digest(path), before)


if __name__ == "__main__":
    unittest.main()
