# Offline NWB import and seizure review

Scope confirmed by the user on 2026-10-05: the initial data is an NWB file from
the Natus workflow, manually transferred to a local computer. The application
runs on that computer. This is an input/deployment requirement, not a verified
claim about every Natus system's export capabilities or this file's internal schema.

## Intended workflow

```text
Natus recording workflow
    -> existing export/transfer procedure
    -> local NWB file
    -> inspect signals, metadata and annotations
    -> select seizure onset and baseline
    -> validated montage/preprocessing/feature adapter
    -> frozen seizureprop model
    -> traces, ranked bipolar pairs, score time courses and saved report
```

No app component needs installation on Natus or a live link to it. Replay is
interactive playback of recorded data. Optional reviewed electrode coordinates
and anatomy can be loaded separately for display; the core workflow works
without them. Keep source recordings read-only and save analysis separately.

## First implementation milestone: inspect one representative NWB

The exact export structure has not yet been inspected. Before fixing paths or
writing a converter, inventory the actual file and confirm:

- Which series holds the intended sEEG waveform, and whether it is raw or already
  processed. Do not hardcode a series name or assume the first series is correct.
- Waveform dimensions, channel IDs, electrode-table mapping and reference montage.
  Already bipolar recordings must not be bipolar-referenced again.
- Actual sample rate or explicit timestamps, gaps/discontinuities and the time
  origin used by seizure annotations. Do not assume the earlier baseline MAT
  files' 2000 Hz rate applies to NWB exports.
- Amplitude units and stored conversion factors. Compare a few converted samples
  and amplitudes with a trusted reference before feature extraction.
- Available seizure annotations and sufficient baseline. If onset labels are
  absent, let the user mark onset; do not present the current model as a detector.
- Whether electrode coordinates are usable. Missing anatomy must not prevent
  signal viewing or inference.

For standard NWB electrophysiology, PyNWB's `ElectricalSeries` defines time-first
arrays, electrode references, and timestamps or a starting time/rate. The NWB
schema specifies applying global conversion, optional channel conversion, then
offset to recover physical values. These are checks for the reader, not evidence
that the particular export contains all fields in the expected form.
[PyNWB ElectricalSeries](https://pynwb.readthedocs.io/en/stable/pynwb.ecephys.html),
[NWB format specification](https://nwb-schema.readthedocs.io/en/latest/format.html).

Use a read-only, slice-based reader so users can inspect selected intervals
without loading an entire long recording into memory. PyNWB documents this
dataset-access pattern in its
[electrophysiology tutorial](https://pynwb.readthedocs.io/en/stable/tutorials/domain/ecephys.html).

## What exists and what must be built

| Component | Current status |
|---|---|
| Prepared-feature model inference | Implemented in `seizureprop.evaluation.predict` |
| Dataset-specific EDF preparation | Existing research scripts; useful as preprocessing references |
| NWB inventory/import adapter | To build and validate against a representative export |
| General waveform-to-model feature workflow | To integrate with the checkpoint's exact preprocessing contract |
| Local trace viewer, event selection and report | Proposed undergraduate app work |
| Natus plugin or live stream | Outside this scope |

NWB can feed the package's internal representation directly through an adapter;
there is no need to convert to EDF merely to satisfy existing research scripts.
The current `prepare` command adopts feature caches, and `predict` reads feature
NPZ files. Neither command currently accepts NWB. See the
[package contract](../src/seizureprop/README.md).

## Acceptance criteria

First verify imported channel identities, timestamps and amplitudes against
known samples. Then compare extracted features and predictions with an approved
reference. Preserve the NWB source hash, selected series and intervals, montage,
preprocessing configuration, ordered pair IDs and checkpoint hash in each analysis.

For local performance, measure file opening, selected-interval loading,
preprocessing, inference and UI responsiveness. Live detection latency and
false alarms per hour are not acceptance metrics for this onset-supplied review
tool. The [proposal review](app_proposal_review_20261005.md) defines the remaining
scientific and visualization boundaries.
