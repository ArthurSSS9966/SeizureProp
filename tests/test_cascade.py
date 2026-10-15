"""Scientific invariants for causal gating and independently trained experts."""
import numpy as np
import torch
import unittest

from seizureprop.models.cascade import EventDetector, ActivityInformedOnset
from seizureprop.training.cascade import (detect_episodes, ictal_participation_loss,
                                         onset_discrimination_loss)


def test_event_causality_and_channel_permutation() -> None:
    torch.manual_seed(3)
    model = EventDetector().eval()
    x = torch.randn(7, 40, 17)
    with torch.no_grad():
        y = model(x)
        changed = x.clone(); changed[:, 20:] += 100
        torch.testing.assert_close(y[:20], model(changed)[:20])
        torch.testing.assert_close(y, model(x[[3, 5, 2, 0, 6, 1, 4]]), atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(y[:20], model(x[:, :20]))


def test_finite_detector_receptive_field() -> None:
    model = EventDetector().eval()
    x = torch.randn(3, 50, 17)
    with torch.no_grad():
        y = model(x)
        # No learned elapsed-time-from-file-start shortcut after the warmup.
        torch.testing.assert_close(y[25:], model(x[:, 10:])[15:], atol=1e-6, rtol=1e-5)


def test_episode_buffer_alarm_and_release() -> None:
    p = np.r_[np.zeros(40), np.ones(10), np.zeros(8)]
    e, = detect_episodes(p, .5, persistence=2, release=5, buffer=30)
    assert (e.start, e.estimated_start, e.alarm, e.stop, e.censored) == (10, 40, 41, 55, False)
    assert not detect_episodes(np.zeros(30), .5)
    assert detect_episodes(p[:45], .5)[0].censored
    assert not detect_episodes(np.array([0., 1., 0.]), .5, persistence=2)


def test_unknown_activity_frames_are_not_negatives() -> None:
    z = torch.zeros(3, 20, requires_grad=True)
    loss = ictal_participation_loss(z, torch.tensor([1., 0., 0.]), torch.tensor([1., 1., 0.]))
    loss.backward()
    assert z.grad[2, 9] > 0  # Explicit ten-second negative.
    assert torch.count_nonzero(z.grad[2, :9]) == 0
    assert torch.count_nonzero(z.grad[2, 10:]) == 0


def test_soft_activity_does_not_delete_channels_or_backpropagate() -> None:
    model = ActivityInformedOnset().eval()
    x = torch.randn(5, 20, 17, requires_grad=True)
    activity = torch.zeros(5, 20, requires_grad=True)
    out = model(x, activity)
    assert out.shape == (5,)
    out.sum().backward()
    assert activity.grad is None and x.grad.abs().sum() > 0


def test_later_participants_are_onset_negatives() -> None:
    z = torch.zeros(4, requires_grad=True)
    loss = onset_discrimination_loss(z, torch.tensor([1., 0., 0., -1.]), torch.tensor([1., 1., 0., -1.]))
    loss.backward()
    assert z.grad[0] < 0 and z.grad[1] > 0 and z.grad[2] > 0 and z.grad[3] == 0


def test_onset_channel_equivariance() -> None:
    torch.manual_seed(8)
    model = ActivityInformedOnset().eval()
    x, a = torch.randn(5, 15, 17), torch.rand(5, 15)
    perm = torch.tensor([4, 2, 0, 1, 3])
    with torch.no_grad():
        torch.testing.assert_close(model(x, a)[perm], model(x[perm], a[perm]), atol=1e-6, rtol=1e-5)


def test_missing_onset_labels_have_finite_zero_gradients() -> None:
    z = torch.zeros(5, requires_grad=True)
    y = torch.tensor([1., 0., float('nan'), -1., float('inf')])
    loss = onset_discrimination_loss(z, y)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(z.grad).all()
    torch.testing.assert_close(z.grad[2:], torch.zeros(3))
    assert z.grad[0] < 0 and z.grad[1] > 0
    for target in (torch.tensor([float('nan')]), torch.tensor([.5])):
        try:
            onset_discrimination_loss(torch.zeros(1), target)
        except ValueError:
            pass
        else:
            raise AssertionError('Empty or nonbinary supervision must be rejected')


def load_tests(loader: unittest.TestLoader, tests: unittest.TestSuite,
               pattern: str | None) -> unittest.TestSuite:
    return unittest.TestSuite(unittest.FunctionTestCase(fn) for name, fn in globals().items()
                              if name.startswith('test_') and callable(fn))
