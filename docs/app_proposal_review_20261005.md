# Review of the undergraduate sEEG app proposal

Reviewed 2026-10-05. Author of proposal: Ashwitha Surabhi. Source:
[Real-Time SEEG Seizure Localization and Propagation Platform](<Real-Time SEEG Seizure Localization and Propagation Platform.pdf>),
both pages. This is a design review against the current repository, not a code
audit of an app: the submitted document contains no implementation or benchmark.
The original PDF is unchanged.

**Scope clarification, 2026-10-05:** the user identifies the initial file type
as NWB from the Natus workflow, manually transferred to a separate local computer.
The app should operate offline. Installation on Natus, live connections and
stream deployment are outside this project's current scope. This clarification
supersedes the proposal's real-time framing and the initial review's suggestion
of retaining live acquisition as a later milestone. See the
[offline NWB workflow](offline_nwb_workflow.md).

## Assessment

The proposal identifies a useful review workflow: synchronized traces and score
views, comparing seizures from the same patient, and optional anatomical context.
Its retrospective-first sequence and expert-annotation comparison are sound
starting points. I would support a narrower first deliverable: **an offline sEEG
seizure-review app importing local NWB recordings and using a frozen model**.
The NWB-to-feature adapter is a required new component, not an existing capability.

The full proposal combines model development, contact localization, imaging
registration, 3D rendering, uncertainty estimation and live acquisition. Those
are separate deliverables with different dependencies. The current package is
ready to support feature-based score inference, but does not yet supply several
of the scientific quantities the proposed interface would display.

## Prioritized findings

### 1. High: resolve channel-pair versus contact identity before drawing the brain

Page 1 promises contact-level scores; page 2, Methodology 2, maps each channel to
one contact. Our external/localization workflows construct adjacent bipolar
pairs, and model outputs are indexed by those pairs. A score for `A1-A2` cannot
simply become an individually validated score for A1, A2, or both.

Keep an explicit pair table containing output index, pair ID, both contact IDs
and montage. In a 3D view, show the pair as a segment or a clearly labeled
display midpoint. Validate joins by ID rather than array position; missing,
duplicate or unmatched IDs must be visible. If individual-contact predictions
are required, define and validate that aggregation separately.

The distinction is also explicit in the official
[BIDS iEEG specification](https://bids-specification.readthedocs.io/en/stable/modality-specific-files/intracranial-electroencephalography.html):
electrodes are physical contacts, while a channel is a recorded time series that
can represent a difference between two electrodes. Retain coordinate system,
units and transform provenance when displaying coordinates.

Evidence: pair preparation (private research source),
[prediction arrays](../src/seizureprop/evaluation/predict.py).

### 2. High: onset ranking, spread involvement and recruitment timing are different outputs

Page 2, Methodology 3, proposes estimating recruitment times and animating
propagation. The current predictor returns onset logits, per-frame sigmoid
scores, and a pooled involvement score. It does not return recruitment times
or directed connections. The model's temporal training targets do not validate
every threshold crossing as physiological recruitment.

There is a concrete integration trap: `localization_involved` averages the
largest 20% of logits over the **whole supplied ictal interval**, using at least
one frame. The evaluated ten-second spread endpoint instead averages logits
over the first ten ictal frames. They cannot share a UI label. A future adapter
should expose the exact ten-second calculation, and mark it unavailable when
fewer than ten seconds are observed.

First show score-versus-time heatmaps and replay of observed activity. Label any
later recruitment estimator as an additional method with validation against
time annotations. Temporal order alone supplies no directed network edges;
avoid arrows suggesting a proven propagation path or surgical target.

Evidence: [predictor](../src/seizureprop/evaluation/predict.py),
[pooling](../src/seizureprop/objectives/primitives.py),
[evaluated spread calculation](../src/seizureprop/training/localization.py).

### 3. High: replace live acquisition with local NWB import and offline replay

The title and page 1 emphasize real-time operation, while page 2 defers it.
The clarified workflow instead uses manually transferred NWB files. Remove the
live stream connector and Natus installation from the deliverables. Interactive
replay means playback of an existing recording, not live monitoring.
The current workflow starts with a supplied onset and baseline. It has no
continuous event detector, acquisition connector, dropped-packet handling or
streaming preprocessing contract.

Report local file-open, preprocessing, inference and display times as software
performance. They are not live detection delays. Preserve one-second feature
timestamps and require ten observed ictal seconds for the spread endpoint even
when the entire recording is already available.

The backbone is causal over feature windows, but this does not certify the raw
pipeline as stream-safe. If live use is ever separately proposed, test that predictions for already
observed frames do not change when future raw samples are appended, and compare
stateful chunked preprocessing with an explicit offline reference. Filter design
matters here; [MNE's filtering guidance](https://mne.tools/stable/auto_tutorials/preprocessing/25_background_filtering.html)
distinguishes causal filters from delay-compensated noncausal filtering. This is
a design consideration, not a claim that this repository uses MNE's default
filter in its current multi-expert preparation.

### 4. High: specify reuse of this model instead of quietly creating a second analysis pipeline

Page 2, Methodology 1, describes calculating power, line length and change points,
but does not name a checkpoint, feature order, montage, baseline scheme or model
API. Those choices determine whether the app reproduces the research model.

Call the installed [package interface](../src/seizureprop/README.md). Build a
validated NWB reader and preprocessing adapter; use curator-prepared feature
bundles as reference fixtures during development. Retain a fixed checkpoint and
its hashes, ordered pair metadata, and reference outputs. Copying functions out of notebooks or using the
older 512 Hz waveform pipeline would not reproduce the current 17-feature,
one-second prepared inputs. Existing EDF/BIDS research adapters do not provide
NWB import; no intermediate EDF conversion is required by the model.

The output NPZ currently omits identities and timestamps. The app adapter must
carry them from validated input metadata. Its acceptance test should compare
all three prediction arrays against direct package output, not just compare a
plot by eye.

### 5. Medium: keep anatomical display separate from signal-only inference

Page 1 and page 2, Methodology 2, propose CT/MRI registration and 3D anatomy.
That makes the whole application use multiple data types, but the localization
model can still be signal-only if imaging only positions its unchanged scores.
Anatomical filtering, reweighting or selecting model input channels by tissue
would make that analysis anatomy-assisted and should be reported separately.

Accept already reviewed coordinates and a brain mesh in a later milestone;
do not make the student implement CT/MRI registration before a usable score
viewer exists. The app must work without imaging. A display-only anatomy toggle
should leave model scores, candidate set and rank order unchanged. Show measured
tissue annotations separately from any experimental predicted tissue score.

### 6. Medium: separate signal quality, model score and calibrated confidence

Page 1 asks for signal quality and confidence. These are different quantities.
The current sigmoid output is not calibrated confidence, and the predictor does
not export a validated tissue classifier. The presence of a tissue head in the
architecture does not establish that this checkpoint trained it meaningfully.

Version 1 can show trace availability, supplied bad-channel flags and explicitly
defined data-quality checks alongside model scores. Calibrated confidence needs
a separate held-out calibration/assessment protocol. Expert rater agreement is
useful target information, but does not automatically calibrate a model's score.
Do not show a percentage labeled 'confidence in the seizure source' merely by
multiplying the sigmoid by 100.

### 7. Medium: make validation match the promised use

Page 2, Methodology 4, lists agreement, recruitment order, false alarms and
latency without defining them. Split validation into software correctness and
scientific performance:

| Question | Acceptance evidence |
|---|---|
| Does the app reproduce inference? | Same prepared inputs/checkpoint produce matching arrays; IDs and timestamps stay aligned after sorting and selection |
| Does it retrieve annotated onset pairs? | Fixed patient-held-out AP and Acc@N, including the number of targets and candidate policy |
| Does it show ten-second spread correctly? | Exact evaluated logit aggregation; short records marked unavailable |
| Does a new recruitment estimator work? | Error in seconds and ordering on recordings with recruitment-time labels; tied/unknown times handled explicitly |
| Does local NWB import work? | Verified series selection, channel mapping, sample times, amplitude conversion and montage; compare selected samples and model-ready features with a reference |
| Does the UI help review? | Prespecified review tasks and measured completion time/mapping errors with users |

Acc@N uses the number of annotated targets and is a retrospective metric, not a
quantity available for a new unlabeled seizure. Do not display it as live model
confidence. Multiple seizures from one patient do not become independent patients;
keep splits patient-wise and reserve test patients from tuning or calibration.

The latest local matched study (excluded under the [data policy](data_publication_policy.md)) reports a numerical
maximum of 52.07% CHOP clinical-onset Acc@N for context/spread, across ten patients,
with no statistically established primary gain. USC transport comprises only
two patients. These results do not demonstrate 90% signal-only localization or
validated white-matter rejection. The older filtering plot uses a different
evaluation setup. Keep each metric attached to its cohort, endpoint and protocol.

## Proposed undergraduate deliverable

Suggested title: **Offline sEEG Seizure Review and Anatomical Visualization App**.

| Milestone | Student-owned output | Evidence required before advancing |
|---|---|---|
| 1. NWB reader and trace viewer | Open a manually transferred NWB locally, inspect recording metadata and traces, select a seizure/onset and baseline | Representative file plus known reference samples; correct series, channel identities, units, montage, timestamps and missing-data behavior |
| 2. Model adapter | Call fixed package inference; retain metadata and export a result table with checkpoint/input hashes | Numerical parity with direct package output; no local fitting or annotation inputs to inference |
| 3. Event comparison | Compare recordings from the same patient and montage | Join by pair identity; show missing pairs and montage changes rather than silently matching rows |
| 4. Optional anatomy | Display reviewed coordinates and mesh; link pair selection to traces | Explicit pair/contact map and coordinate units; scores unchanged by display toggle; usable without coordinates |

Arthur/model owners should supply a representative NWB recording, the checkpoint
selection rule, prepared reference fixture, preprocessing specification, allowed
data and verified pair mapping. The student's bounded responsibility is the NWB
reader, viewer, model adapter and mapping tests, with preprocessing reviewed by
the model owner. Recruitment-method development, calibration and CT/MRI
reconstruction remain separate research tasks. Hospital stream integration is
outside the current scope.

This work can contribute to the paper through reproducible model inspection,
error analysis and review usability. It does not itself resolve low localization
performance or prove the anatomical-loss hypothesis.

## Suggested replacement scope paragraph

We will build an offline sEEG review application on a separate local computer.
Users will manually transfer NWB recordings from the Natus workflow, open them
locally, and select a seizure and baseline interval. A validated import and
preprocessing adapter will feed a frozen localization model. The app will display synchronized traces,
ranked bipolar-channel scores and score time courses, and allow comparison of
seizures from the same patient. Optional reviewed electrode coordinates will
provide anatomical context without changing signal-only predictions. We will
first validate model-output parity, channel identity and timing, then assess
agreement with held-out expert annotations and usability. Recruitment-time
estimation and calibrated confidence require separate validation. The app will
not require installation on Natus or a live acquisition connection.

## Changes made in this review

Updated the root and directory READMEs, added package/API and documentation
indexes, and documented the score semantics above. No model, application,
checkpoint, patient data or PDF was modified. Findings in this document remain
recommendations; they are not reported as implemented app features.

Historical documentation verification before publication exclusions: 68 local links resolved; package CLI, prediction CLI and audit-tool help commands succeed;
`git diff --check` passes. Both PDF pages were read and visually inspected.
Training and numerical compatibility experiments were not rerun for prose edits.
