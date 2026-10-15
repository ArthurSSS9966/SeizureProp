"""Patient isolation and temporal contracts for the staged tissue comparison."""
import numpy as np
import pandas as pd
import unittest
from functools import partial
import torch

from seizureprop.training.staged_tissue import (
    StagedTissueScorer, causal_resample, crossfit_tissue, staged_objective,
)


def tissue_rows() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame([dict(patient=p, pair=f'pair{i}', y=i % 2,
                             stimulated=False, x=rng.normal(size=51))
                         for p in ['a', 'b', 'c', 'test'] for i in range(8)])


def test_crossfit_excludes_all_labels_for_held_patient() -> None:
    rows = tissue_rows()
    first, _, lineage = crossfit_tissue(rows, {'a', 'b', 'c'}, [['a'], ['b'], ['c']])
    rows.loc[rows.patient == 'a', 'y'] = 1 - rows.loc[rows.patient == 'a', 'y']
    changed, _, _ = crossfit_tissue(rows, {'a', 'b', 'c'}, [['a'], ['b'], ['c']])
    np.testing.assert_array_equal(first[rows.patient == 'a'], changed[rows.patient == 'a'])
    assert all(not set(item['fit']) & set(item['predict']) for item in lineage)


def test_external_labels_cannot_change_any_tissue_prediction() -> None:
    rows = tissue_rows()
    first, final, _ = crossfit_tissue(rows, {'a', 'b', 'c'}, [['a'], ['b'], ['c']])
    rows.loc[rows.patient == 'test', 'y'] = 1 - rows.loc[rows.patient == 'test', 'y']
    changed, _, _ = crossfit_tissue(rows, {'a', 'b', 'c'}, [['a'], ['b'], ['c']])
    np.testing.assert_array_equal(first, changed)
    assert final.fit_patients == ['a', 'b', 'c']
    with unittest.TestCase().assertRaises(ValueError):
        crossfit_tissue(rows, {'a', 'b', 'c'}, [['a'], ['a', 'b', 'c']])


def test_resampler_never_uses_future_input(fs: int) -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=(fs * 2, 2))
    expected = causal_resample(x, fs)
    x[fs:] += 100
    np.testing.assert_array_equal(expected[:256], causal_resample(x, fs)[:256])


def test_calibration_onset_causality_and_equivariance() -> None:
    torch.manual_seed(4)
    model = StagedTissueScorer(hidden=4).eval()
    x = torch.randn(5, 34, 40)
    scores = torch.rand(5)
    first = model(x, 30, scores)
    permutation = torch.tensor([4, 2, 0, 1, 3])
    shuffled = model(x[permutation], 30, scores[permutation])
    torch.testing.assert_close(first['onset'][permutation], shuffled['onset'])
    x[:, :, 31:] += 100
    future = model(x, 30, scores)
    torch.testing.assert_close(first['onset'], future['onset'], rtol=0, atol=0)
    x[:, :, 12:] -= 200
    torch.testing.assert_close(first['tissue'], model(x, 30)['tissue'], rtol=0, atol=0)


def test_soft_prior_keeps_all_channels_and_does_not_gate_spread() -> None:
    model = StagedTissueScorer(hidden=4).eval()
    x = torch.randn(5, 34, 40)
    base = model(x, 30)
    gated = model(x, 30, torch.zeros(5))
    assert torch.isfinite(gated['onset']).all() and len(gated['onset']) == 5
    torch.testing.assert_close(base['localization'], gated['localization'], rtol=0, atol=0)
    torch.testing.assert_close(gated['onset'], base['onset'] + np.log(.1))


def test_clinical_positive_wm_remains_positive_and_unknown_is_masked() -> None:
    logits = torch.zeros(4, 40, requires_grad=True)
    onset = torch.zeros(4, requires_grad=True)
    tissue = torch.zeros(4, requires_grad=True)
    r = dict(nbase=30, y=torch.tensor([1., 0., 0., 0.]), y_spread=None,
             stimulated=False, tissue_target=torch.tensor([0., 1., -1., -1.]))
    loss = staged_objective(dict(localization=logits, onset=onset, tissue=tissue), r, .1)
    loss.backward()
    assert onset.grad[0] < 0  # Gradient descent increases clinical-positive WM score.
    assert tissue.grad[2:].abs().sum() == 0


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for function in [test_crossfit_excludes_all_labels_for_held_patient,
                     test_external_labels_cannot_change_any_tissue_prediction,
                     test_calibration_onset_causality_and_equivariance,
                     test_soft_prior_keeps_all_channels_and_does_not_gate_spread,
                     test_clinical_positive_wm_remains_positive_and_unknown_is_masked]:
        suite.addTest(unittest.FunctionTestCase(function))
    for fs in (512, 1000, 2048):
        function = partial(test_resampler_never_uses_future_input, fs)
        function.__name__ = f'test_causal_resample_{fs}'
        suite.addTest(unittest.FunctionTestCase(function))
    return suite
