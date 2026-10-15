"""Joint-objective invariants and equivalence to the reference trainer."""
import copy
import unittest

import numpy as np
import torch

from seizureprop.config import ExperimentConfig
from seizureprop.models.registry import build_model
from seizureprop.training.identity_pretraining import baseline_views
from seizureprop.training.joint_contrastive import hard_wm_ranking, joint_train, tissue_contrast
from seizureprop.training.localization import train


class JointContrastiveContracts(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_tissue_separation_reduces_contrastive_loss(self) -> None:
        labels = torch.tensor([0, 0, 1, 1])
        separated = torch.tensor([[1., 0.], [1., 0.], [-1., 0.], [-1., 0.]])
        mixed = separated[[0, 2, 1, 3]]
        self.assertLess(tissue_contrast(separated, separated, labels), tissue_contrast(mixed, mixed, labels))

    def test_unknown_has_zero_gradient_and_permutation_is_invariant(self) -> None:
        a = torch.randn(5, 8, requires_grad=True)
        b = torch.randn(5, 8, requires_grad=True)
        labels = torch.tensor([0, 1, -1, 0, 1])
        loss = tissue_contrast(a, b, labels)
        order = torch.tensor([3, 2, 4, 0, 1])
        torch.testing.assert_close(loss, tissue_contrast(a[order], b[order], labels[order]))
        loss.backward()
        self.assertEqual(float(a.grad[2].abs().sum() + b.grad[2].abs().sum()), 0.)
        self.assertGreater(float(a.grad[0].abs().sum()), 0.)

    def test_single_tissue_or_unknown_returns_differentiable_zero(self) -> None:
        for labels in (torch.tensor([0, 0]), torch.tensor([-1, -1])):
            x = torch.randn(2, 4, requires_grad=True)
            loss = tissue_contrast(x, x, labels)
            loss.backward()
            self.assertEqual(float(loss), 0.)
            self.assertEqual(float(x.grad.abs().sum()), 0.)

    def test_hard_negatives_protect_positive_wm_and_ignore_unknown(self) -> None:
        scores = torch.zeros(5, requires_grad=True)
        target = torch.tensor([1., 0., 0., 0., 0.])
        tissue = torch.tensor([0, 0, -1, 1, 0])
        loss = hard_wm_ranking(scores, target, tissue, torch.tensor([True, True, True, True, False]))
        loss.backward()
        self.assertLess(float(scores.grad[0]), 0.)
        self.assertGreater(float(scores.grad[1]), 0.)
        self.assertEqual(float(scores.grad[2:].abs().sum()), 0.)

    def test_invalid_masks_and_no_mined_negatives(self) -> None:
        x = torch.randn(3, requires_grad=True)
        labels = torch.tensor([1., 0., 0.])
        with self.assertRaises(ValueError):
            hard_wm_ranking(x, labels, labels, torch.ones(3))
        loss = hard_wm_ranking(x, labels, labels, torch.zeros(3, dtype=torch.bool))
        loss.backward()
        self.assertEqual(float(loss), 0.)

    def test_contrastive_gradient_does_not_read_ictal_signal(self) -> None:
        model = build_model('channel_identity', 34, 8)
        x = torch.randn(4, 34, 40, requires_grad=True)
        a, b = baseline_views(x, 30)
        loss = tissue_contrast(model.channel_identity(a), model.channel_identity(b), torch.tensor([0, 1, 0, 1]))
        loss.backward()
        self.assertEqual(float(x.grad[:, :, 28:].abs().sum()), 0.)
        self.assertGreater(float(model.id_frames[0].weight.grad.abs().sum()), 0.)
        self.assertIsNone(model.localization_head.weight.grad)

    def records(self, patient: str) -> list[dict]:
        return [{'patient': patient, 'recording': 'record', 'x': torch.randn(4, 34, 40),
                 'nbase': 30, 'y': torch.tensor([1., 0., 0., 0.]),
                 'target': np.array([1., 0., 0., 0.]), 'spread': None, 'y_spread': None}]

    def test_disabled_auxiliaries_exactly_reproduce_reference_training(self) -> None:
        training, validation = self.records('train'), self.records('val')
        config = ExperimentConfig('', '', '', 'synthetic', epochs=3, patience=5)
        model = build_model('channel_identity', 34, 8)
        other = copy.deepcopy(model)
        torch.manual_seed(55)
        np.random.seed(55)
        state, history, epoch = train(model, training, validation, config)
        torch.manual_seed(55)
        np.random.seed(55)
        actual, joint_history, joint_epoch = joint_train(other, training, validation, config)
        self.assertEqual(epoch, joint_epoch)
        for key in state:
            torch.testing.assert_close(state[key], actual[key], atol=0, rtol=0)
        self.assertEqual([h['loss'] for h in history], [h['loss'] for h in joint_history])

    def test_patient_overlap_rejected(self) -> None:
        records = self.records('same')
        with self.assertRaises(ValueError):
            joint_train(build_model('channel_identity', 34, 8), records, records,
                        ExperimentConfig('', '', '', 'synthetic'))


if __name__ == '__main__':
    unittest.main()
