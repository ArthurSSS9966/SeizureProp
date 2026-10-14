# Seizure Propagation Project

Research software for sEEG channel localization from prepared signal features
at a supplied seizure onset. It is not a validated continuous seizure detector
or a clinical device.

## Install and test

Use Python 3.10+ in a virtual environment or the existing research environment:

```powershell
python -m pip install -e .
python -m seizureprop --help
python -m unittest discover -s tests -p test_package.py -v
```

Core dependencies are in `pyproject.toml`; optional extras are `.[edf]` and
`.[reports]`. Tests use synthetic signals and invented identifiers.

## Public repository layout

```text
src/seizureprop/       # reusable data/features/models/training/inference package
configs/package_study/ # parameter recipes; local data and split files required
tests/                # synthetic package and publication checks
tools/                # source inventory and publication checks
docs/                 # public technical documentation
legacy/third_party/   # archived third-party source; not installed
```

Historical notebooks, cohort-specific scripts, private waveform code, patient
manifests, study reports, clinical files and checkpoints remain in the local
research workspace and are excluded from publication. Coded identifiers and
per-patient results are treated as private. See the
[data publication policy](docs/data_publication_policy.md).

| Task | Guide |
|---|---|
| Call the model from Python or an app | [Package interface](src/seizureprop/README.md) |
| Browse app design and current boundaries | [Documentation index](docs/README.md) |
| Prepare a local run configuration | [Configuration guide](configs/README.md) |
| Check a proposed publication | [Maintenance guide](tools/README.md) |
| Use retained historical work locally | [Notebook guide](notebooks/README.md), [research scripts](scripts/README.md) |

## Supported workflow

`prepare` validates already prepared, hash-pinned feature caches; it does not
convert raw EDF. A local manifest declares units, canonical features,
channel identities, target definitions and patient partitions.

```powershell
python -m seizureprop prepare --kind multiexpert --units V --source result/local_study/prepare.jsonl --root . --output result/local_study/features.json
python -m seizureprop run --config configs/package_study/pretrained_onset.json
python -m seizureprop predict --checkpoint result/local_run/seed17/best.pt --features result/local_input/features.npz --output result/local_prediction/prediction.npz
```

These paths are examples, not bundled data. Configured paths resolve relative
to the config file. Use new output locations; completed runs require matching
provenance and hashes, and incomplete runs are rejected. Execution currently
supports CPU and does not resume mid-epoch.

Inference reads `absolute`, `relative` and `nbase`, not tissue or clinical labels.

For a separate tissue-classification stage, export one reusable label per bipolar
channel from non-seizure signal features:

```powershell
python -m seizureprop tissue-predict --checkpoint result/tissue_model/baseline30.joblib --features result/local_input/baseline.npz --units uV --segment-kind baseline --output result/local_prediction/tissue_labels.csv
```

This input needs `absolute` (time × channels × 17 signal features) and `names`.
For a combined baseline/ictal cache, `nbase` restricts baseline inference to the
pre-onset frames. Units are explicit; the command converts volts to microvolts
when conversion is recoverable from the cached features. Load trusted project
checkpoints only. Binary checkpoints export a GM score, GM/WM/UNCERTAIN label and
retention decision. Experimental three-class checkpoints also export WM and
mixed-pair scores, with the most likely class as the label. Their GM retention
cutoff is a separate decision, so a predicted mixed pair can still be retained.
Unknown anatomy is excluded from training rather than treated as mixed tissue.
Scores are not guaranteed to be calibrated. Match channel identities before reusing
labels for seizure localization. Ictal tissue inference requires an ictal-trained
checkpoint and is retrospective.
For current research inputs, channels are bipolar pairs. Output arrays retain
input order but omit channel identities/timestamps; the caller must carry those
metadata. Sigmoid scores are not calibrated confidence. Pooled involvement is
not the evaluated ten-second spread endpoint. See the package contract.

## Interpretation and app scope

Split by patient before fitting. Keep baseline calibration separate from ictal
observations and preserve units, feature order and montage. Clinical-onset,
confirmed-grey-matter onset and spread retrieval are different endpoints.
Acc@N uses the known number of annotated targets with fractional boundary ties;
it is a retrospective metric, not a live confidence score. Average recordings
and seeds within patients before averaging patients.

The proposed app is an offline review tool on a separate local computer, using
manually transferred EDF recordings. A validated EDF adapter and UI remain to
be built; current inference consumes prepared features. See the
[offline workflow](docs/offline_edf_workflow.md).
Anatomy may display unchanged scores; anatomy-based filtering is a separate
analysis. This repository does not claim validated recruitment times, directed
propagation pathways, calibrated confidence or automatic white-matter rejection.
