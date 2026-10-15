"""Scientific invariants for signal-only channel identity and reconstruction."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from seizureprop.evaluation.predict import predict_arrays
from seizureprop.features.representations import centered_features
from seizureprop.models.registry import build_model
from seizureprop.training.masked import masked_objective, pretrain, reconstruction_mask


class ChannelIdentityContracts(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.x = torch.randn(7, 34, 24)
        self.model = build_model('channel_identity', 34, 8).eval()

    def test_permutation_equivariance_and_variable_population(self) -> None:
        order = torch.randperm(7)
        expected = self.model(self.x, 12)['localization'][order]
        actual = self.model(self.x[order], 12)['localization']
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        self.assertEqual(self.model(self.x[:3], 12)['localization'].shape, (3, 24))

    def test_future_seizure_does_not_change_identity_or_earlier_scores(self) -> None:
        changed = self.x.clone()
        changed[:, :, 16:] = torch.randn_like(changed[:, :, 16:]) * 100
        original = self.model(self.x, 12)['localization']
        actual = self.model(changed, 12)['localization']
        torch.testing.assert_close(actual[:, 12:16], original[:, 12:16])
        torch.testing.assert_close(self.model.channel_identity(changed[:, :, :10]),
                                   self.model.channel_identity(self.x[:, :, :10]))

    def test_hidden_targets_cannot_reach_reconstruction(self) -> None:
        calibration, target = self.x[:, :, :10], self.x[:, :, 12:]
        mask = torch.rand(7, 12) < .5
        changed = target.clone().masked_fill(mask[:, None, :], 1000)
        torch.testing.assert_close(self.model.reconstruct(target, calibration, mask),
                                   self.model.reconstruct(changed, calibration, mask))
        loss = masked_objective(self.model, target, calibration, mask)
        loss.backward()
        self.assertGreater(float(self.model.id_frames[0].weight.grad.abs().sum()), 0.)
        self.assertGreater(float(self.model.population_read.in_proj_weight.grad.abs().sum()), 0.)

    def test_identity_ablation_ignores_calibration(self) -> None:
        model = build_model('channel_attention', 34, 8).eval()
        a = model.encode(self.x[:, :, 12:], self.x[:, :, :10])
        b = model.encode(self.x[:, :, 12:], self.x[:, :, :10] * 100)
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_checkpoint_inference_round_trip(self) -> None:
        rng = np.random.default_rng(1)
        a = rng.normal(size=(24, 7, 17)).astype(np.float32)
        rel = rng.normal(size=a.shape).astype(np.float32)
        saved = {'architecture': 'channel_identity', 'hidden': 8,
                 'mean': torch.zeros(34), 'scale': torch.ones(34),
                 'model_state': self.model.state_dict()}
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / 'model.pt'
            torch.save(saved, path)
            actual = predict_arrays(torch.load(path, weights_only=True), a, rel, 12)
        x = torch.tensor(centered_features(a, rel, 12).transpose(1, 2, 0))
        expected = self.model(x, 12)['localization'][:, 12].detach().numpy()
        np.testing.assert_allclose(actual['localization_onset'], expected, atol=1e-6)

    def test_centered_units_invariant(self) -> None:
        rng = np.random.default_rng(4)
        volts = rng.normal(size=(24, 7, 17)).astype(np.float64)
        uv = volts.copy()
        uv[..., :3] += np.log(1e6)
        uv[..., 4:10] += 2 * np.log(1e6)
        rel = volts.copy()
        np.testing.assert_allclose(centered_features(volts, rel, 12),
                                   centered_features(uv, rel, 12), atol=3e-6)

    def test_pretraining_patient_overlap_rejected(self) -> None:
        record = {'patient': 'synthetic', 'x': self.x[:, :, 12:],
                  'calibration': self.x[:, :, :10]}
        with self.assertRaises(ValueError):
            pretrain(self.model, [record], [record], 17, epochs=1)

    def test_small_masks_always_have_both_roles(self) -> None:
        x = torch.zeros(1, 34, 2)
        for seed in range(100):
            a = reconstruction_mask(x, torch.Generator().manual_seed(seed))
            b = reconstruction_mask(x, torch.Generator().manual_seed(seed))
            self.assertTrue(bool(a.any()) and not bool(a.all()))
            torch.testing.assert_close(a, b)
        with self.assertRaisesRegex(ValueError, "at least two"):
            reconstruction_mask(x[:, :, :1], torch.Generator())

    def test_two_token_pretraining_runs_training_and_validation(self) -> None:
        records = [{'patient': p, 'x': self.x[:1, :, :2],
                    'calibration': self.x[:1, :, 3:10]} for p in ('train', 'validation')]
        _, history, epoch = pretrain(self.model, records[:1], records[1:], 17, epochs=2)
        self.assertEqual(len(history), 3)
        self.assertTrue(all(np.isfinite(r['validation_loss']) for r in history))
        self.assertGreaterEqual(epoch, 0)


if __name__ == '__main__':
    unittest.main()
