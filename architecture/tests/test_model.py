"""Technical model checks, independent of the project's data provider."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch
except ModuleNotFoundError:
    torch = None
from environment import check_environment
if torch is not None:
    from model import TrafficGRU, verify_model


class ModelChecks(unittest.TestCase):
    @unittest.skipIf(torch is None, 'PyTorch is not installed')
    def test_single_example_and_incomplete_batch(self):
        model = TrafficGRU().eval()
        with torch.inference_mode():
            for batch in (1, 7):
                predictions = model(torch.zeros(batch, 8, 3))
                self.assertEqual(tuple(predictions.shape), (batch, 3))
                self.assertTrue(torch.isfinite(predictions).all())
                self.assertTrue((predictions >= 0).all())

    @unittest.skipIf(torch is None, 'PyTorch is not installed')
    def test_reject_wrong_window_or_feature_count(self):
        model = TrafficGRU()
        for shape in ((8, 3), (1, 7, 3), (1, 8, 4)):
            with self.assertRaises(ValueError):
                model(torch.zeros(*shape))

    @unittest.skipIf(torch is None, 'PyTorch is not installed')
    def test_training_and_serialization_preserve_rng(self):
        before = torch.get_rng_state().clone()
        result = verify_model()
        self.assertTrue(result['training_step'])
        self.assertTrue(result['weights_roundtrip'])
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_version_mismatch_is_reported(self):
        with patch('environment.metadata.version', return_value='0.0.0'):
            result = check_environment()
        self.assertFalse(result['ready'])
        self.assertEqual(len(result['errors']), 4)


if __name__ == '__main__':
    unittest.main()
