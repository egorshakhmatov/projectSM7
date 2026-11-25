"""Data acceptance tests from TZ v2. Run from architecture/ with unittest."""
import csv
import hashlib
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from src.sm7.data_adapter import (
    COLUMNS, DEFAULT_CSV, DataError, ROW_COUNTS, _windows,
    generate_csv, load_history, load_test_split, load_training_splits, validate_csv,
)


class DataAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temp.name)
        cls.times, cls.density = validate_csv(DEFAULT_CSV)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def history(self, filename, rows, header=COLUMNS):
        path = self.folder / filename
        with path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
        return path

    def sample(self):
        return [[(datetime(2025, 12, 1, 11, 5) + timedelta(minutes=5*i)).isoformat(), 70, 40, 40]
                for i in range(13)]

    def test_reproducible_generation(self):
        copy = generate_csv(self.folder / 'regenerated.csv')
        self.assertEqual(hashlib.sha256(copy.read_bytes()).hexdigest(),
                         hashlib.sha256(DEFAULT_CSV.read_bytes()).hexdigest())

    def test_exact_period_and_counts(self):
        self.assertEqual(len(self.times), 210528)
        self.assertEqual(self.times[0].isoformat(), '2024-01-01T00:05:00')
        self.assertEqual(self.times[-1].isoformat(), '2026-01-01T00:00:00')
        self.assertTrue(np.isfinite(self.density).all())
        self.assertTrue(((0 <= self.density) & (self.density <= 200)).all())

    def test_split_contract_and_no_leakage(self):
        train, val = load_training_splits(DEFAULT_CSV)
        test = load_test_split(DEFAULT_CSV)
        start = 0
        for part, size in zip((train, val, test), ROW_COUNTS):
            self.assertEqual(part.X.shape, (size - 12, 12, 3))
            self.assertEqual(part.X.dtype, np.float32)
            self.assertEqual(part.y.shape, (size - 12,))
            self.assertEqual(part.y.dtype, np.int64)
            self.assertEqual(part.forecast_origin[0], self.times[start + 11].isoformat())
            self.assertEqual(part.forecast_origin[-1], self.times[start + size - 2].isoformat())
            np.testing.assert_array_equal(part.X[0], self.density[start:start + 12])
            np.testing.assert_array_equal(part.X[-1], self.density[start + size - 13:start + size - 1])
            np.testing.assert_array_equal(part.y, np.argmin(self.density[start + 12:start + size], axis=1))
            start += size
        self.assertFalse(hasattr(train, 'target_density'))
        self.assertFalse(hasattr(val, 'target_density'))
        np.testing.assert_array_equal(test.target_density, self.density[sum(ROW_COUNTS[:2]) + 12:])

    def test_tie_break(self):
        rows = self.sample()
        part = _windows([datetime.fromisoformat(r[0]) for r in rows],
                        np.asarray([r[1:] for r in rows], dtype=np.float32), test=True)
        self.assertEqual(part.y.tolist(), [1])  # class 1 => light number 2

    def test_normal_history_without_future(self):
        rows = self.sample()[:12]
        X, t = load_history(self.history('normal.csv', rows), rows[-1][0])
        self.assertEqual(X.shape, (12, 3))
        self.assertEqual(t, rows[-1][0])
        np.testing.assert_array_equal(X[0], [70, 40, 40])

    def test_future_row_not_in_features(self):
        rows = self.sample()
        rows[-1][1:] = [200, 200, 200]
        X, _ = load_history(self.history('future.csv', rows), rows[-2][0])
        self.assertEqual(X.max(), 70)

    def test_missing_intervals(self):
        rows = self.sample()
        del rows[4]
        with self.assertRaisesRegex(DataError, 'gap'):
            load_history(self.history('gap.csv', rows), rows[-1][0])

    def test_duplicate_time(self):
        rows = self.sample()
        rows[5][0] = rows[4][0]
        with self.assertRaisesRegex(DataError, 'duplicate'):
            load_history(self.history('duplicate.csv', rows), rows[-1][0])

    def test_shuffled_time(self):
        rows = self.sample()
        rows[0], rows[1] = rows[1], rows[0]
        with self.assertRaisesRegex(DataError, 'not increasing'):
            load_history(self.history('shuffled.csv', rows), rows[-1][0])

    def test_out_of_range(self):
        rows = self.sample()
        rows[0][1] = 201
        with self.assertRaisesRegex(DataError, 'outside'):
            load_history(self.history('range.csv', rows), rows[-1][0])

    def test_non_numeric_inf_nan(self):
        for text in ('text', 'inf', 'nan'):
            rows = self.sample()
            rows[0][1] = text
            with self.subTest(value=text), self.assertRaises(DataError):
                load_history(self.history('bad.csv', rows), rows[-1][0])

    def test_missing_value(self):
        rows = self.sample()
        rows[0][1] = ''
        with self.assertRaisesRegex(DataError, 'missing'):
            load_history(self.history('empty.csv', rows), rows[-1][0])

    def test_missing_column(self):
        rows = self.sample()
        with self.assertRaisesRegex(DataError, 'missing required columns'):
            load_history(self.history('column.csv', [r[:3] for r in rows], COLUMNS[:3]), rows[-1][0])

    def test_short_history(self):
        rows = self.sample()[:11]
        with self.assertRaisesRegex(DataError, 'need 12'):
            load_history(self.history('short.csv', rows), rows[-1][0])

    def test_stale_history(self):
        rows = self.sample()[:12]
        with self.assertRaisesRegex(DataError, 'no observation'):
            load_history(self.history('stale.csv', rows), '2025-12-01T12:05:00')

    def test_historical_t_is_valid(self):
        rows = self.sample()
        X, t = load_history(self.history('historical.csv', rows), '2025-12-01T12:00:00')
        self.assertEqual(t, '2025-12-01T12:00:00')
        self.assertEqual(X.shape, (12, 3))


if __name__ == '__main__':
    unittest.main()
