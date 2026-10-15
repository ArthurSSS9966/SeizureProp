# Independent seizure detection, participation and onset models

This experimental protocol extends the independent experts described in
[FULL_SEIZURE.md](FULL_SEIZURE.md). It does not replace existing checkpoints or
claim that cascading improves localization. Each fitted model has one task.

| Stage | Architecture | Supervision | Output |
| --- | --- | --- | --- |
| Event | Learned channel pooling and maximum pooling, then causal residual TCN | Ictal windows positive; available pre/post-event background negative | One event score per second |
| Participation | Spatial attention and causal GRU | Actual channel-time snapshot labels plus positive-only whole-seizure bags | One participation score per channel and second |
| Onset | Separate spatial attention, GRU and temporal attention pooling | Onset versus non-onset channels, emphasizing known later participants | One independent onset probability per channel |

All stages currently consume the same fixed seventeen baseline-relative signal
features. This isolates the cascade question from new representation pretraining;
these are not foundation-model or SPINT reproductions. Model parameters are
independent. The onset control excludes predicted participation; the experimental
onset model appends it as a detached feature. It never deletes a low-scoring
channel or forces its onset probability to zero. Multiple onset channels are
allowed; the readout uses sigmoid, not a channel-wise softmax.

An additional exploratory onset variant appends the frozen event detector's
per-frame probability as another input feature. It helps the onset model identify
background retained in the safety buffer without hard cropping away early seizure
samples. This adds no task or joint loss. Its training probabilities also come
from patient crossfits, and its validation scores decide whether it is retained.
Because this variant was motivated by development-set boundary diagnostics, it
requires new independent confirmation even if its aggregate scores improve.

## Data and phase semantics

Split patients before fitting. Exclude stimulation-induced and uncertain-origin
events at every stage. Use one consistent causal bipolar/filter/resampling path
and a reserved, earlier baseline for per-channel scaling. The event detector is
applied only after this baseline is available. Selecting a suitable calibration
baseline during continuous deployment remains a separate requirement.

Peri-ictal is an interval containing multiple phases, not a mutually exclusive
class. Clinical onset/offset define the positive ictal interval. Available
pre-onset and post-offset windows are negative controls for *ongoing seizure*,
not a supervised seizure-forecasting phenotype. They can contain interictal
discharges, artifacts or uncertain boundary activity. Do not equate these short
peri-event controls with expert-adjudicated all-day background. A one-second
window crossing the annotated offset is unknown for event fitting/evaluation.

The participation model is fitted only on complete ictal inputs. A channel
positive at onset or at the annotated later snapshot must be positive somewhere
in that seizure. A channel negative at the snapshot is not necessarily negative
later. Unknown channel-time labels have no direct negative loss. The final
partial ictal second uses the prior zero-padded seizure-only representation;
automatic streaming windows instead contain whatever samples were actually
available at their endpoints.

## Cascade fitting and inference

1. Fit the event and participation experts independently on training patients.
   Select checkpoints and gate settings on separate validation patients.
2. Cross-fit those upstream experts within the training patients. Every downstream
   training patient's activity scores and trigger estimates come from models
   that excluded that patient. The fixed feature transform has no cohort-fitted
   representation or scaler, avoiding representation leakage across crossfits.
3. Train onset experts on complete ictal sequences and a buffered-trigger context
   view. Both preserve the entire annotated seizure during training. Missing or
   late upstream triggers never remove supervised training examples; record these
   fallbacks. This does not guarantee robustness to truncation during deployment.
4. At inference, event threshold crossings trigger a retained pre-event buffer.
   Consecutive low scores close an episode; open tails are marked censored.
   Feed each automatically extracted episode to the other two experts. Clinical
   onset and offset must not be used to extract inference episodes.

The event detector's finite fifteen-second receptive field and absence of absolute
position prevent learning a fixed onset offset from the beginning of the file.
Maximum pooling preserves evidence from small focal channel subsets. Threshold,
entry persistence, release duration and buffer size are explicit configuration
choices, not biological constants. An alarm time is different from an estimated
start time. Pre-onset alarms are not evidence of seizure prediction.

## Evaluation contracts

- Report event sensitivity, alarm delays, background alarm counts and exposure.
  A false-alarm rate from short event-adjacent recordings is not a validated
  continuous-monitoring rate.
- Evaluate participation only where clinical channel labels exist. Whole-seizure
  participation accuracy is unidentified with sparse labels.
- Compare onset with oracle clinical segments against the automatic cascade.
  Save every emitted episode, including unmatched false alarms. Matching to
  reference events is an evaluation operation only. Distinguish onset ranking
  conditional on successful detection from end-to-end recall.
- If missed events are assigned zero AP in an aggregate, label that aggregate a
  **miss-penalized AP utility**, not ordinary average precision. Save ordinary
  conditional AP separately. All missed events count as zero onset recall.
- Train on complete seizures; measure growing-history performance after fitting.
  A one-second feature frame does not restrict training to the first second.
- Model family, epoch and thresholds are selected without held-out patients.
  Reused internal development folds remain exploratory. Independent external
  confirmation and long non-seizure recordings are still needed.

Reusable code lives in `models/cascade.py` and `training/cascade.py`. Private
cohort adapters, checkpoint files, patient identifiers, crossfit assignments and
per-patient results stay outside the public source scope.
