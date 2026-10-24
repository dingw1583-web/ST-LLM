import unittest

from config import DATASET_CONFIGS, get_config
from training_settings import learning_rate_factor, training_settings


class TrainingSettingsTests(unittest.TestCase):
    def test_warmup_and_decay_reach_configured_bounds(self):
        values = [learning_rate_factor(i, 100, 1e-2, 1e-4, 0.1) * 1e-2
                  for i in range(100)]
        self.assertAlmostEqual(values[0], 1e-4)
        self.assertAlmostEqual(values[10], 1e-2)
        self.assertAlmostEqual(values[-1], 1e-4)
        self.assertTrue(all(a <= b for a, b in zip(values[:10], values[1:11])))
        self.assertTrue(all(a >= b for a, b in zip(values[10:-1], values[11:])))
        self.assertTrue(all(1e-4 <= value <= 1e-2 for value in values))

    def test_short_runs_and_no_warmup_are_finite(self):
        for count in (1, 2, 3, 7):
            for warmup in (0.0, 0.1, 0.9):
                for step in range(count + 1):
                    factor = learning_rate_factor(step, count, 0.01, 0.0001, warmup)
                    self.assertGreaterEqual(factor, 0.01)
                    self.assertLessEqual(factor, 1.0)

    def test_invalid_schedule_settings_are_rejected(self):
        for settings in [(0, 0.01, 0.0001, 0.1), (10, 0.0, 0.0001, 0.1),
                         (10, 0.01, 0.1, 0.1), (10, 0.01, 0.0001, 1.0)]:
            with self.assertRaises(ValueError):
                learning_rate_factor(0, *settings)

    def test_run_metadata_resolves_dataset_overrides(self):
        for dataset in DATASET_CONFIGS:
            cfg = get_config(dataset)
            cfg.LR = 0.002
            cfg.BATCH_SIZE = 8
            data = training_settings(cfg)
            self.assertEqual(data['LR'], 0.002)
            self.assertEqual(data['BATCH_SIZE'], 8)
            self.assertEqual(data['NUM_CLASSES'], len(cfg.CLASS_NAMES))
            self.assertEqual(len(data['SEEDS']), 10)
            self.assertEqual(data['optimizer'], 'Adam')


if __name__ == '__main__':
    unittest.main()
