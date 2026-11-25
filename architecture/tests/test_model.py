"""Technical model checks, independent of the project's data provider."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from environment import check_environment
from model import TrafficGRU, verify_model


class ModelChecks(unittest.TestCase):
    def test_single_example_and_incomplete_batch(self):
        model = TrafficGRU().eval()
        with torch.inference_mode():
            for batch in (1, 7):
                logits = model(torch.zeros(batch, 12, 3))
                self.assertEqual(tuple(logits.shape), (batch, 3))
                self.assertTrue(torch.isfinite(logits).all())

    def test_reject_wrong_window_or_feature_count(self):
        model = TrafficGRU()
        for shape in ((12, 3), (1, 11, 3), (1, 12, 4)):
            with self.assertRaises(ValueError):
                model(torch.zeros(*shape))

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
