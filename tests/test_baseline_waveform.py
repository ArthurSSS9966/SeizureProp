"""Synthetic waveform contracts; no participant data required."""
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np

from seizureprop.features.baseline_waveform import prepare_baseline, rich_features, contact_features
from seizureprop.models.baseline_probe import BaselineProbe, pair_gm_score


class BaselineWaveformTests(unittest.TestCase):
    def test_units_and_no_future_samples(self) -> None:
        rng = np.random.default_rng(30)
        x = rng.normal(0, 20, (33 * 512, 3))
        first = prepare_baseline(x, 512, 'uV')
        np.testing.assert_allclose(first, prepare_baseline(x / 1e6, 512, 'V'), atol=1e-11)
        x[32 * 512:] = 1e9
        np.testing.assert_array_equal(first, prepare_baseline(x, 512, 'uV'))
        self.assertEqual(first.shape, (7680, 3))
        with self.assertRaises(ValueError):
            prepare_baseline(x[:100], 512, 'uV')

    def test_spectral_features_follow_known_frequency(self) -> None:
        t = np.arange(7680) / 256
        x = np.stack([np.sin(2 * np.pi * f * t) for f in (6, 20, 80)], axis=1)
        features, names = rich_features(x)
        for j, band in enumerate(['4_8', '13_30', '65_90']):
            self.assertGreater(features[j, names.index('welch_relative_' + band)], .95)
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(features.shape[1], len(names))

    def test_contact_features_cancel_shared_reference_and_permute(self) -> None:
        rng = np.random.default_rng(31)
        x = rng.normal(size=(7680, 4))
        neighbors = [[1], [0, 2], [1, 3], [2]]
        basic, rich, names = contact_features(x, neighbors)
        shared = rng.normal(0, 20, (7680, 1))
        other_basic, other_rich, _ = contact_features(x + shared, neighbors)
        np.testing.assert_allclose(basic, other_basic, atol=2e-6, rtol=1e-5)
        np.testing.assert_allclose(rich, other_rich, atol=2e-6, rtol=1e-5)
        order = [3, 1, 0, 2]; inverse = {old: new for new, old in enumerate(order)}
        reordered = [[inverse[i] for i in neighbors[j]] for j in order]
        _, permuted, _ = contact_features(x[:, order], reordered)
        np.testing.assert_allclose(rich[order], permuted, atol=1e-6)
        self.assertEqual(rich.shape[1], len(names))

    def test_missing_contact_score_remains_unknown(self) -> None:
        p = pair_gm_score([.9, np.nan, .2, 1.1], [.8, .8, .9, .5])
        np.testing.assert_allclose(p[[0, 2]], [.8, .2])
        self.assertTrue(np.isnan(p[[1, 3]]).all())

    def test_probe_width_validation_and_serialization(self) -> None:
        rng = np.random.default_rng(32)
        x = rng.normal(size=(60, 113)); y = np.tile([0, 1], 30)
        model = BaselineProbe().fit(x, y, np.repeat(['A', 'B', 'C'], 20))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'probe.joblib'; joblib.dump(model, path)
            np.testing.assert_array_equal(model.predict(x), joblib.load(path).predict(x))
        with self.assertRaises(ValueError):
            model.predict(x[:, :-1])
        y[0] = -1
        with self.assertRaises(ValueError):
            BaselineProbe().fit(x, y, np.repeat(['A', 'B', 'C'], 20))


if __name__ == '__main__':
    unittest.main()
