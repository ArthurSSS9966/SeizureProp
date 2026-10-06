"""Scientific and artifact contracts for the installable package."""
from __future__ import annotations
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from seizureprop.artifacts.io import write_json, file_hash, completed_run, mark_complete
from seizureprop.config import ExperimentConfig
from seizureprop.data.manifest import load_records
from seizureprop.evaluation.predict import predict_arrays
from seizureprop.evaluation.ranking import retrieval
from seizureprop.features.representations import FEATURE_NAMES, SCHEMA_ID, centered_features
from seizureprop.models.registry import build_model, initialize
from seizureprop.schemas import FeatureSpec, validate_splits
from seizureprop.training.localization import selection_metric
from seizureprop.workflows.experiment import run, initialization_lineage


def fixture(directory: Path) -> Path:
    rng = np.random.default_rng(91)
    records = []
    for patient in ['A', 'B', 'C']:
        path = directory / f'{patient}.npz'
        absolute = rng.normal(size=(14, 5, 17)).astype(np.float32)
        np.savez_compressed(path, absolute=absolute, relative=absolute.copy(), nbase=4,
                            names=np.array(['A1', 'A2', 'A3', 'A4', 'A5']),
                            y=np.array([1, 0, 0, 0, 0], dtype=np.float32))
        records.append({'patient': patient, 'recording': patient, 'cache': path.name,
                        'sha256': file_hash(path), 'onset_key': 'y', 'spread_key': None})
    write_json(directory / 'manifest.json', {'schema_version': 1,
               'features': {'schema_id': SCHEMA_ID, 'names': FEATURE_NAMES, 'units': 'V', 'window_seconds': 1.},
               'records': records})
    write_json(directory / 'splits.json', {'train': ['A'], 'validation': ['B'], 'test': ['C']})
    path = directory / 'config.json'
    write_json(path, {'manifest': 'manifest.json', 'splits': 'splits.json', 'output': 'run',
                     'protocol': 'synthetic-contract', 'epochs': 1, 'seeds': [17], 'hidden': 4, 'threads': 1})
    return path


class PackageContracts(unittest.TestCase):
    def test_reordered_features_rejected(self) -> None:
        with self.assertRaises(ValueError):
            FeatureSpec(SCHEMA_ID, tuple(reversed(FEATURE_NAMES)), 'V').validate()

    def test_missing_units_rejected(self) -> None:
        with self.assertRaises(ValueError):
            FeatureSpec(SCHEMA_ID, tuple(FEATURE_NAMES), 'unknown').validate()

    def test_patient_overlap_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_splits({'train': ['A'], 'validation': ['A'], 'test': ['C']}, {'A', 'C'})

    def test_unassigned_patient_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_splits({'train': ['A'], 'validation': ['B'], 'test': ['C']}, {'A', 'B', 'C', 'D'})

    def test_context_zero_initialization_preserves_backbone(self) -> None:
        torch.manual_seed(4)
        base = build_model('combined', 34, 4).eval()
        context = build_model('context', 34, 4).eval()
        initialize(context, base.state_dict())
        x = torch.randn(5, 34, 14)
        torch.testing.assert_close(base(x, 4)['localization'], context(x, 4)['localization'], rtol=0, atol=0)

    def test_context_permutation_and_causality(self) -> None:
        model = build_model('context', 34, 4).eval()
        with torch.no_grad():
            model.context_head.weight.fill_(.1)
        x = torch.randn(5, 34, 14)
        permutation = torch.tensor([3, 0, 4, 1, 2])
        original = model(x, 4)['localization']
        torch.testing.assert_close(model(x[permutation], 4)['localization'], original[permutation])
        x[:, :, 6:] += 100
        torch.testing.assert_close(model(x, 4)['localization'][:, :6], original[:, :6])

    def test_centering_does_not_use_future(self) -> None:
        rng = np.random.default_rng(4)
        absolute = rng.normal(size=(14, 5, 17)).astype(np.float32)
        relative = absolute.copy()
        before = centered_features(absolute, relative, 4)
        absolute[8:] += 100
        np.testing.assert_array_equal(before[:8], centered_features(absolute, relative, 4)[:8])

    def test_selection_is_patient_macro(self) -> None:
        rows = [{'patient': 'A', 'onset_AP': 1.}] * 10 + [{'patient': 'B', 'onset_AP': 0.}]
        self.assertEqual(selection_metric(rows, 'onset'), .5)

    def test_ties_do_not_depend_on_channel_order(self) -> None:
        scores = np.zeros(4)
        y = np.array([1, 0, 0, 0])
        self.assertEqual(retrieval(scores, y)['AccAtN'], .25)

    def test_wrong_config_key_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = fixture(Path(tmp))
            value = json.loads(path.read_text()); value['epohs'] = 10
            write_json(path, value)
            with self.assertRaises(ValueError):
                ExperimentConfig.load(path)

    def test_changed_cache_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp); fixture(directory)
            with (directory / 'A.npz').open('ab') as handle:
                handle.write(b'changed')
            with self.assertRaises(ValueError):
                load_records(directory / 'manifest.json')

    def test_short_spread_window_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp); fixture(directory)
            cache = directory / 'A.npz'
            with np.load(cache) as data:
                value = {k: data[k] for k in data.files}
            value['nbase'] = 8
            np.savez_compressed(cache, **value)
            manifest = json.loads((directory / 'manifest.json').read_text())
            manifest['records'][0].update(sha256=file_hash(cache), spread_key='y')
            write_json(directory / 'manifest.json', manifest)
            with self.assertRaises(ValueError):
                load_records(directory / 'manifest.json')

    def test_incompatible_resume_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            write_json(directory / 'manifest.json', {'signature': 'old'})
            mark_complete(directory)
            with self.assertRaises(ValueError):
                completed_run(directory, 'new')

    def test_partial_run_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            write_json(directory / 'manifest.json', {'signature': 'old'})
            with self.assertRaises(ValueError):
                completed_run(directory, 'old')

    def test_pretraining_patient_alias_overlap_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'provenance.json'
            write_json(path, {'fit_patients': ['sub-DEMO_A'], 'selection_patients': [], 'checkpoints': {}})
            config = ExperimentConfig('a', 'b', 'c', 'test', scaler_checkpoint='x', initialization_provenance=str(path))
            with self.assertRaises(ValueError):
                initialization_lineage(config, {'train': ['DEMO_B'], 'validation': ['DEMO_C'], 'test': ['DEMO_A']})

    def test_synthetic_workflow_and_signal_only_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp); path = fixture(directory)
            summary = run(path)
            self.assertEqual(summary['test_patients'], 1)
            self.assertEqual(run(path), summary)
            saved = torch.load(directory / 'run/seed17/best.pt', weights_only=True)
            with np.load(directory / 'C.npz') as data:
                scores = predict_arrays(saved, data['absolute'], data['relative'], int(data['nbase']))
            self.assertEqual(scores['localization_onset'].shape, (5,))
            # Completion hash catches post-hoc edits to a metric file.
            (directory / 'run/metrics.csv').write_text('changed')
            with self.assertRaises(ValueError):
                run(path)


if __name__ == '__main__':
    unittest.main()
