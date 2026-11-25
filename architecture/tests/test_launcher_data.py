"""Integration checks for the shared run.py entry point."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.sm7.data_adapter import DEFAULT_CSV, ROOT, _settings, DataError


class LauncherAcceptance(unittest.TestCase):
    def invoke(self, *arguments, cwd=None):
        return subprocess.run([sys.executable, '-B', str(ROOT / 'run.py'), *arguments],
                              cwd=cwd, capture_output=True, text=True, encoding='utf-8', check=False)

    def test_actual_architecture_config(self):
        _settings(ROOT / 'config.json')
        config = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
        config['input']['window_size'] = 8
        with self.assertRaises(DataError):
            _settings(config)

    def test_status_and_reserved_commands(self):
        result = self.invoke('status', '--json', cwd=ROOT.parent)
        self.assertEqual(result.returncode, 0, result.stderr)
        status = json.loads(result.stdout)
        self.assertIn('generate', status['implemented_commands'])
        self.assertIn('validate', status['implemented_commands'])
        self.assertIn('prepare', status['implemented_commands'])
        for command in ('train', 'evaluate', 'predict'):
            self.assertIn(command, status['pending_commands'])
            self.assertEqual(self.invoke(command).returncode, 2)

    def test_valid_invalid_history_from_other_working_directory(self):
        valid = self.invoke('validate', '--csv', 'data/control/history.csv', '--time',
                            '2025-09-13T09:20:00', cwd=ROOT.parent)
        self.assertEqual(valid.returncode, 0, valid.stderr)
        self.assertIn('(12, 3)', valid.stdout)
        invalid = self.invoke('validate', '--csv', 'data/control/invalid_gap.csv', '--time',
                              '2025-09-13T09:20:00', cwd=ROOT.parent)
        self.assertEqual(invalid.returncode, 1)
        self.assertIn('timestamp gap', invalid.stderr)

    def test_generate_and_prepare_via_shared_launcher(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            data = folder / 'regenerated.csv'
            report = folder / 'report.json'
            control = folder / 'control'
            generated = self.invoke('generate', '--csv', str(data), cwd=ROOT.parent)
            self.assertEqual(generated.returncode, 0, generated.stderr)
            self.assertEqual(data.read_bytes(), DEFAULT_CSV.read_bytes())
            prepared = self.invoke('prepare', '--csv', str(data), '--report', str(report),
                                   '--control-dir', str(control), cwd=ROOT.parent)
            self.assertEqual(prepared.returncode, 0, prepared.stderr)
            doc = json.loads(report.read_text(encoding='utf-8'))
            self.assertEqual([doc['splits'][part]['examples'] for part in ('train', 'validation', 'test')],
                             [147357, 31567, 31568])
            self.assertTrue((control / 'history.csv').is_file())
            self.assertTrue((control / 'actual.json').is_file())


if __name__ == '__main__':
    unittest.main()
