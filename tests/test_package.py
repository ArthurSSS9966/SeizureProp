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


class TissueMappingTests(unittest.TestCase):
    """Synthetic identity and annotation contracts, without clinical data."""

    def test_unknown_roi_does_not_become_grey(self) -> None:
        from seizureprop.data.tissue import map_contacts
        table = map_contacts([{'label': 'A1', 'roi': 'new-unrecognized-region'},
                              {'label': 'A2', 'roi': 'ctx-lh-unknown'}])
        self.assertTrue(all(item.tissue == 'UNKNOWN' for item in table.values()))

    def test_freesurfer_name_number_validation(self) -> None:
        from seizureprop.data.tissue import atlas_tissue
        self.assertEqual(atlas_tissue('ctx-lh-insula', 1035), ('GM', ()))
        self.assertEqual(atlas_tissue('Right-Hippocampus', 53), ('GM', ()))
        self.assertEqual(atlas_tissue('Right-Amygdala', 54), ('GM', ()))
        self.assertEqual(atlas_tissue('Right-Thalamus-Proper', 49), ('GM', ()))
        self.assertEqual(atlas_tissue('Right-Cerebellum-Cortex', 47), ('GM', ()))
        self.assertEqual(atlas_tissue('Left-Cerebral-White-Matter', 2), ('WM', ()))
        for number in [2035, 1035.5, 'bad']:
            self.assertEqual(atlas_tissue('ctx-lh-insula', number)[0], 'UNKNOWN')

    def test_explicit_annotation_precedence_and_conflict(self) -> None:
        from seizureprop.data.tissue import map_contacts
        table = map_contacts([{'label': 'A1', 'matter': 'grey', 'brain_area': 'Left-Cerebral-White-Matter'},
                              {'label': 'A2', 'matter': 'unknown', 'brain_area': 'ctx-lh-insula'},
                              {'label': 'A3', 'matter': 'CSF', 'brain_area': 'ctx-lh-insula'}])
        self.assertEqual(table[('A', 1)].tissue, 'GM')
        self.assertIn('matter_atlas_conflict', table[('A', 1)].flags)
        self.assertEqual(table[('A', 2)].tissue, 'UNKNOWN')
        self.assertEqual(table[('A', 3)].tissue, 'UNKNOWN')

    def test_identity_collisions_and_unparsed_contacts_rejected(self) -> None:
        from seizureprop.data.tissue import contact_key, map_contacts
        self.assertEqual(contact_key(' aB001 '), ('AB', 1))
        self.assertEqual(contact_key('A2B03'), ('A2B', 3))
        with self.assertRaises(ValueError):
            map_contacts([{'label': 'AB01', 'matter': 'grey'}, {'label': 'ab1', 'matter': 'white'}])
        with self.assertRaises(ValueError):
            map_contacts([{'label': 'EEG A01-REF', 'matter': 'grey'}])

    def test_pair_labels_require_two_known_matching_contacts(self) -> None:
        from seizureprop.data.tissue import map_contacts, pair_tissue
        table = map_contacts([{'label': f'A{i}', 'matter': matter}
                              for i, matter in enumerate(['G', 'A', 'white', 'unknown'], 1)])
        self.assertEqual(pair_tissue(table[('A', 1)], table[('A', 2)]), 1)
        self.assertEqual(pair_tissue(table[('A', 3)], table[('A', 3)]), 0)
        self.assertEqual(pair_tissue(table[('A', 1)], table[('A', 3)]), -1)
        self.assertEqual(pair_tissue(table[('A', 1)], None), -1)
        self.assertEqual(pair_tissue(table[('A', 4)], table[('A', 4)]), -1)


class PairCompositionTests(unittest.TestCase):
    def test_mixed_requires_two_known_tissues_without_conflict(self) -> None:
        from seizureprop.data.tissue import pair_composition
        self.assertEqual(pair_composition('GM', 'WM'), 2)
        self.assertEqual(pair_composition('WM', 'GM'), 2)
        self.assertEqual(pair_composition('GM', 'GM'), 1)
        self.assertEqual(pair_composition('WM', 'WM'), 0)
        for left, right in [('UNKNOWN', 'GM'), ('CSF', 'WM'), ('UNKNOWN', 'UNKNOWN')]:
            self.assertEqual(pair_composition(left, right), -1)
        for left, right in [('GM', 'WM'), ('GM', 'GM'), ('WM', 'WM')]:
            self.assertEqual(pair_composition(left, right, conflicted=True), -1)

    def test_three_class_model_rejects_unknown_and_survives_serialization(self) -> None:
        import joblib
        from seizureprop.models.tissue import PairTissueClassifier
        rng = np.random.default_rng(85)
        x = rng.normal(size=(90, 51))
        y = np.tile([0, 1, 2], 30)
        patients = np.repeat(['A', 'B', 'C'], 30)
        x[:, 0] += y * 6
        model = PairTissueClassifier().fit(x, y, patients)
        scores = model.predict_proba(x)
        np.testing.assert_allclose(scores.sum(axis=1), 1.)
        np.testing.assert_allclose(model.predict(x), scores[:, 1])
        self.assertGreater((scores.argmax(axis=1) == y).mean(), .9)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'pair-model.joblib'
            joblib.dump(model, path)
            np.testing.assert_array_equal(scores, joblib.load(path).predict_proba(x))
        y[0] = -1
        with self.assertRaisesRegex(ValueError, 'all tissue classes'):
            PairTissueClassifier().fit(x, y, patients)

    def test_three_class_patient_weighting_handles_absent_class(self) -> None:
        from seizureprop.models.tissue import patient_class_weights
        y = np.array([0, 1, 2, 2, 0, 1])
        patients = np.array(['A'] * 4 + ['B'] * 2)
        weights = patient_class_weights(patients, y)
        self.assertAlmostEqual(weights[:4].sum(), weights[4:].sum())
        for label in (0, 1, 2):
            self.assertAlmostEqual(weights[(patients == 'A') & (y == label)].sum(), 1.)

    def test_three_class_inference_separates_label_from_retention(self) -> None:
        from seizureprop.evaluation.tissue import predict_tissue_arrays
        from seizureprop.features.tissue import TISSUE_SCHEMA
        from seizureprop.models.tissue import PairTissueClassifier
        rng = np.random.default_rng(86)
        model = PairTissueClassifier().fit(rng.normal(size=(90, 51)),
            np.tile([0, 1, 2], 30), np.repeat(['A', 'B', 'C'], 30))
        saved = {'model': model, 'schema': TISSUE_SCHEMA, 'units': 'uV',
                 'state': 'baseline30', 'threshold': 0., 'classes': ('WM', 'GM', 'MIXED')}
        values = rng.normal(size=(36, 8, 17))
        names = np.array([f'A{i}-A{i + 1}' for i in range(1, 9)])
        before = predict_tissue_arrays(saved, values, names, 'uV', 'baseline', 30)
        self.assertTrue(before.retain.all())
        p = before[['wm_score', 'gm_score', 'mixed_score']].to_numpy()
        np.testing.assert_array_equal(before.channel_label, np.array(saved['classes'])[p.argmax(1)])
        np.testing.assert_allclose(p.sum(1), 1.)
        values[30:] = np.nan
        after = predict_tissue_arrays(saved, values, names, 'uV', 'baseline', 30)
        np.testing.assert_array_equal(before.gm_score, after.gm_score)
        saved['classes'] = ('GM', 'WM', 'MIXED')
        with self.assertRaisesRegex(ValueError, 'declare'):
            predict_tissue_arrays(saved, values, names, 'uV', 'baseline', 30)


class TissueStageTests(unittest.TestCase):
    def test_physical_unit_conversion_matches_raw_waveforms(self) -> None:
        from seizureprop.features.representations import extract_features
        from seizureprop.features.tissue import aggregate_tissue, to_microvolts
        rng = np.random.default_rng(23)
        signal_v = rng.normal(0, 25e-6, (6 * 256, 4))
        volts = extract_features(signal_v, 256)
        microvolts = extract_features(signal_v * 1e6, 256)
        converted = to_microvolts(volts, 'V')
        np.testing.assert_allclose(converted, microvolts, atol=3e-6, rtol=1e-5)
        np.testing.assert_allclose(aggregate_tissue(converted), aggregate_tissue(microvolts), atol=4e-6, rtol=1e-5)
        self.assertFalse(np.shares_memory(converted, volts))

    def test_clipped_features_require_raw_reconstruction(self) -> None:
        from seizureprop.features.representations import extract_features
        from seizureprop.features.tissue import to_microvolts
        values = extract_features(np.zeros((512, 2)), 256)
        with self.assertRaisesRegex(ValueError, 'Floor-clipped'):
            to_microvolts(values, 'V')
        with self.assertRaisesRegex(ValueError, 'Explicit'):
            to_microvolts(values, 'unknown')

    def test_patient_class_weighting_limits_channel_count_bias(self) -> None:
        from seizureprop.models.tissue import patient_class_weights
        patients = np.array(['A'] * 4 + ['B'] * 6)
        target = np.array([0, 1, 1, 1, 0, 0, 0, 0, 0, 1])
        weights = patient_class_weights(patients, target)
        for patient in ('A', 'B'):
            for label in (0, 1):
                self.assertAlmostEqual(weights[(patients == patient) & (target == label)].sum(), len(target) / 4)

    def test_cutoff_uses_macro_retention_and_preserves_ties(self) -> None:
        from seizureprop.models.tissue import retention_threshold
        p = np.array([.1] + [.9] * 99 + [.2, .2, .01])
        y = np.array([1] * 102 + [0])
        patients = np.array(['A'] * 100 + ['B'] * 3)
        threshold = retention_threshold(p, y, patients, .95)
        self.assertEqual(threshold, .2)
        macro = np.mean([(p[(patients == patient) & (y == 1)] >= threshold).mean() for patient in ('A', 'B')])
        self.assertGreaterEqual(macro, .95)

    def test_standalone_prediction_survives_serialization(self) -> None:
        import joblib
        from seizureprop.models.tissue import TissueClassifier
        rng = np.random.default_rng(44)
        x = rng.normal(size=(30, 51))
        y = np.tile([0, 1], 15)
        model = TissueClassifier().fit(x, y, np.repeat(['A', 'B', 'C'], 10))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.joblib'
            joblib.dump(model, path)
            np.testing.assert_array_equal(model.predict(x), joblib.load(path).predict(x))

    def test_tissue_inference_ignores_future_ictal_frames(self) -> None:
        from seizureprop.evaluation.tissue import predict_tissue_arrays
        from seizureprop.features.tissue import TISSUE_SCHEMA
        from seizureprop.models.tissue import TissueClassifier
        rng = np.random.default_rng(52)
        model = TissueClassifier().fit(rng.normal(size=(30, 51)), np.tile([0, 1], 15), np.repeat(['A', 'B', 'C'], 10))
        saved = {'model': model, 'schema': TISSUE_SCHEMA, 'units': 'uV', 'state': 'baseline30', 'threshold': .2}
        values = rng.normal(size=(40, 4, 17))
        names = np.array(['A1-A2', 'A2-A3', 'B1-B2', 'B2-B3'])
        before = predict_tissue_arrays(saved, values, names, 'uV', 'baseline', 30)
        values[30:] = np.nan
        after = predict_tissue_arrays(saved, values, names, 'uV', 'baseline', 30)
        np.testing.assert_array_equal(before.gm_score, after.gm_score)
        self.assertEqual(len(after), 4)
        with self.assertRaisesRegex(ValueError, 'Segment kind'):
            predict_tissue_arrays(saved, values, names, 'uV', 'ictal', 30)

    def test_tissue_command_exports_without_clinical_arrays(self) -> None:
        import joblib
        from seizureprop.evaluation.tissue import predict_tissue
        from seizureprop.features.tissue import TISSUE_SCHEMA
        from seizureprop.models.tissue import TissueClassifier
        rng = np.random.default_rng(53)
        model = TissueClassifier().fit(rng.normal(size=(20, 51)), np.tile([0, 1], 10), np.repeat(['A', 'B'], 10))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            joblib.dump({'model': model, 'schema': TISSUE_SCHEMA, 'units': 'uV', 'state': 'baseline6', 'threshold': .2}, path / 'model.joblib')
            np.savez(path / 'signal.npz', absolute=rng.normal(size=(12, 3, 17)), names=['A1-A2', 'A2-A3', 'A3-A4'], units='uV')
            result = predict_tissue(path / 'model.joblib', path / 'signal.npz', path / 'labels.csv', 'uV', 'baseline')
            self.assertEqual(result['channels'], 3)
            self.assertTrue((path / 'labels.csv').exists())
            with self.assertRaisesRegex(ValueError, 'Preserve'):
                predict_tissue(path / 'model.joblib', path / 'signal.npz', path / 'labels.csv', 'uV', 'baseline')


if __name__ == '__main__':
    unittest.main()
