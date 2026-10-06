# Scientific package and app integration

Install from the repository root with `python -m pip install -e .` using Python
3.10+. The package exposes `prepare`, `run` and `predict` through
`python -m seizureprop`. See the [root guide](../../README.md) for commands and the
[offline NWB workflow](../../docs/offline_nwb_workflow.md) for planned UI scope.

## Supported interface

```python
from pathlib import Path
from seizureprop.evaluation.predict import predict

result = predict(
    checkpoint=Path("path/to/best.pt"),
    features=Path("path/to/features.npz"),
    output=Path("result/app_trial/prediction.npz"),
)
```

Use a trusted, fixed checkpoint and its matching feature preparation. Inference
runs on CPU, uses the checkpoint scaler, and does not fit on the uploaded seizure.
The current interface consumes prepared features, not raw NWB/EDF or a live stream.
It neither loads imaging nor requires clinical/tissue labels.

## Input contract

| NPZ key | Meaning |
|---|---|
| `absolute` | Finite array `(T, C, 17)` in canonical feature order |
| `relative` | Same shape; changes normalized using reserved baseline calibration |
| `nbase` | Integer count of baseline frames; `0 < nbase < T` |

Frames are non-overlapping one-second feature windows; `C >= 2`. The first ictal
frame is index `nbase`, so onset must already be supplied during preparation.
Canonical names live in [features/representations.py](features/representations.py).
The model concatenates absolute and relative features into 34 values per frame,
centers selected amplitude features using baseline, then applies the stored scaler.
Do not pass 34 features in each input array or normalize them a second time.

The calling app must retain channel/pair IDs, raw sample rate and units, montage,
baseline interval, frame start/end times, onset mark and preprocessing provenance.
The bare prediction command does not validate or export that metadata. Declared
feature schemas are checked in manifests and newer checkpoints; bare NPZ files
cannot prove their units/order. Matching shape alone is insufficient.

Current multi-expert preparation uses adjacent within-shaft bipolar pairs,
256 Hz processed signals and one-second features. Its baseline calibration and
model-context intervals are distinct; see
the private `scripts/seeg_multiexpert_data.py` research adapter. Do not substitute
the historical 512 Hz waveform/notebook pipeline and assume checkpoint compatibility.

## Output contract

| Key | Shape | Exact current calculation |
|---|---|---|
| `localization_onset` | `(C,)` | Localization logit at frame `nbase` |
| `localization_involved` | `(C,)` | Mean of largest `max(1, floor((T-nbase)/5))` ictal logits per channel |
| `localization_probability` | `(C, T)` | Sigmoid of each localization logit |

Higher onset logits give a higher rank for the clinical-onset endpoint. They do not
estimate a contact's exact recruitment time. The sigmoid output is not a
calibrated probability that a channel is the true seizure generator. Output
order matches input order; the output NPZ contains only these three arrays.

Ten-second spread in [training/localization.py](training/localization.py) is a
different endpoint: mean localization **logit** over frames `nbase:nbase+10`.
It requires ten observed ictal frames and a checkpoint whose spread behavior was
evaluated. The generic `localization_involved` output cannot be relabeled as
`spread10`. A future adapter should expose the exact evaluated calculation from
logits; do not average probabilities or silently shorten the interval.

For bipolar channel `A1-A2`, a score belongs to the pair. Do not assign it to A1
or A2 as a separately validated contact prediction. Optional 3D views should
represent the pair explicitly and document any display-only midpoint convention.

## Boundaries

- Import reusable code from `seizureprop`, not notebooks or `scripts.*`.
- Historical waveform models are local-only and excluded from public packages.
- Activity/tissue heads exist in the backbone but are not exported by this
  predictor. Their existence does not establish a validated detector/tissue model.
- No continuous detector, recruitment estimator, directed propagation network,
  imaging registration, GUI or calibrated confidence API is currently provided.
- The intended app starts with NWB files manually transferred from Natus to a
  separate local computer. A validated NWB-to-feature adapter is required; it is
  not implemented by `prepare` or `predict`. See the
  [offline NWB workflow](../../docs/offline_nwb_workflow.md).
- Existing EDF research adapters remain useful for prior datasets. The app's
  input priority is now NWB; it need not convert NWB to EDF to run inference.

Scientific reports with patient-level results remain local under the
[data policy](../../docs/data_publication_policy.md). Do not treat software compatibility tests
as clinical validation or select a deployment model by its highest test score.
