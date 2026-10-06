"""Synthetic checks that publication guards catch data outside result folders."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('publication_guard', ROOT / 'tools/check_publication.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


class PublicationGuardTests(unittest.TestCase):
    def test_identifier_inside_report_template_is_flagged_without_echoing_value(self) -> None:
        identifier = ('CH' + 'OP' + '987').encode()
        result = guard.content_findings('report.py', b'caption = "' + identifier + b'"')
        self.assertEqual(result, [{'path': 'report.py', 'category': 'coded_participant_identifier'}])

    def test_numeric_cohort_list_is_private(self) -> None:
        result = guard.content_findings('settings.json', b'{"patients": [987, 988]}')
        self.assertEqual(result[0]['category'], 'participant_list')

    def test_reusable_settings_and_synthetic_ids_are_allowed(self) -> None:
        self.assertEqual(guard.content_findings('config.json', b'{"hidden": 32, "seeds": [17,29]}'), [])
        self.assertEqual(guard.content_findings('fixture.py', b'patient = "DEMO001"'), [])

    def test_ignore_patterns_cover_new_locations_without_hiding_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            subprocess.run(['git', 'init', '-q', str(path)], check=True)
            (path / '.gitignore').write_bytes((ROOT / '.gitignore').read_bytes())
            private = ['data/raw.edf', 'scratch/raw.nwb', 'scratch/cache.npz',
                       'checkpoints/BestModels/model.zip', 'new/subdir/model.pt',
                       'docs/new_report.md', 'configs/new_cohort.json',
                       'notebooks/demo.ipynb', '.env.production',
                       'legacy/third_party/s4/extensions/kernels/build/file.obj',
                       'scratch/per_patient.json', 'scratch/splits.json']
            public = ['README.md', 'src/seizureprop/evaluation/predict.py',
                      'configs/package_study/context_onset.json', 'tests/test_package.py',
                      'docs/data_publication_policy.md', 'tools/check_publication.py']
            for name in private + public:
                result = subprocess.run(['git', '-C', str(path), 'check-ignore', '--no-index', '-q', name])
                self.assertEqual(result.returncode, 0 if name in private else 1, name)


if __name__ == '__main__':
    unittest.main()
