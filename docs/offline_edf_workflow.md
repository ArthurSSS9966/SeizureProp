# Offline EDF import and seizure review

Scope corrected by the user on 2026-10-05: the raw recordings are EDF files from
the Natus workflow, manually transferred to a local computer. The application
runs on that computer. This describes the intended input and deployment;
the particular export's metadata and montage still need inspection.

## Intended workflow

```text
Natus recording workflow
    -> existing export/transfer procedure
    -> local EDF file
    -> inspect channels, timing, units and annotations
    -> select seizure onset and baseline
    -> validated montage/preprocessing/feature adapter
    -> frozen seizureprop model
    -> traces, ranked bipolar pairs, score time courses and saved report
```

No app component needs installation on Natus or a live link to it. Replay is
interactive playback of recorded data. Optional reviewed electrode coordinates
and anatomy can be loaded separately for display; the core workflow works
without them. Keep source recordings read-only and save analysis separately.

## First implementation milestone: inspect one representative EDF

Before fixing channel rules or building the adapter, confirm:

- Channel labels, order and signal types; identify sEEG channels and retain an
  explicit mapping from original labels to app IDs. Do not silently infer contact
  identity from array position or renamed labels.
- Reference montage and prior processing. Already bipolar recordings must not
  be bipolar-referenced again.
- Actual sampling rates, recording start time, gaps and the time origin used by
  annotations. Do not assume the earlier baseline MAT files' 2000 Hz rate applies
  to these EDF recordings. Reject unsupported discontinuities explicitly.
- Amplitude units and calibration. Compare imported samples and amplitudes with
  a trusted reference before feature extraction; record any unit conversion.
- Available seizure annotations and sufficient baseline. If annotations are
  absent, accept a separate annotation file or let the user mark onset; the
  current model requires supplied onset and is not a continuous detector.
- Availability of a separate reviewed contact/coordinate table for anatomical
  display. Missing anatomy must not prevent signal viewing or inference.

The existing research loaders use MNE's EDF reader. MNE supports EDF/EDF+ and
imports EDF annotation channels when present, but this does not establish that
a particular export contains seizure labels. Its reader can load on demand.
For mixed sampling rates, MNE upsamples requested signals to the highest loaded
rate and recommends preloading to avoid slice-edge artifacts. The app should
inspect channel rates before choosing its loading strategy and record any
resampling. See the
[MNE EDF reader documentation](https://mne.tools/stable/generated/mne.io.read_raw_edf.html).

## What exists and what must be built

| Component | Current status |
|---|---|
| Prepared-feature model inference | Implemented in `seizureprop.evaluation.predict` |
| Dataset-specific EDF preparation | Existing research scripts; starting point for loading and preprocessing |
| General app-facing EDF import adapter | To integrate and validate against a representative export |
| Waveform-to-model feature workflow | Must match the checkpoint's exact preprocessing contract |
| Local trace viewer, event selection and report | Proposed app work |
| Natus plugin or live stream | Outside this scope |

EDF loading can be reused from the research workflow, but a reader alone does
not establish preprocessing compatibility. The current `prepare` command adopts
feature caches, and `predict` reads feature NPZ files. Neither command currently
accepts raw EDF. See the [package contract](../src/seizureprop/README.md).

## Acceptance criteria

First verify imported channel identities, sample times and amplitudes against
known samples. Then compare extracted features and predictions with an approved
reference. Preserve the EDF source hash, selected channels and intervals, original
sample rates and units, montage, preprocessing configuration, ordered pair IDs
and checkpoint hash in each analysis.

For local performance, measure file opening, selected-interval loading,
preprocessing, inference and UI responsiveness. Live detection latency and
false alarms per hour are not acceptance metrics for this onset-supplied review
tool. The [package contract](../src/seizureprop/README.md) defines the current
model outputs and scientific boundaries. Proposals and their reviews remain local.
