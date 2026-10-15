# Controlled tissue-stage comparison

`staged_tissue.py` supports a matched signal backbone with three research arms:

- A: signal-only localization.
- B: patient-cross-fitted baseline tissue scores add a bounded log prior to onset scores. The temporal/spread logits are unchanged by this prior, and no channels are removed.
- C: a baseline-only auxiliary tissue head supplies a training objective. Tissue labels never enter inference.

`crossfit_tissue` excludes every record from the held-out patient when fitting both the feature scaler and classifier. It aggregates repeated spontaneous recordings per patient/pair and balances patients and tissue classes. Final validation/test scores use a classifier fitted only on training patients. Scores are not calibrated anatomical certainty. Mixed, unknown and conflicting pairs use target -1 and are excluded from tissue supervision. Clinical-positive white matter remains an onset-positive target.

The common backbone uses the first 12 baseline frames for identity calibration. Negative training readouts start at frame 16 and stop two frames before the onset anchor. This is retrospective localization with provided calibration, not an autonomous event detector. Readouts cannot use future feature frames. The optional causal FIR resampler retains its filter delay; it never compensates delay by reading future samples. A complete application must also validate referencing, units, filtering, source timestamps and observed endpoint coverage.

The trainer uses fixed epoch/update budgets and selects checkpoints only by patient-macro validation onset AP. The supplied evaluation callback must use the same endpoint and candidate policy for all arms. Unsupported spread targets remain missing; never synthesize exact recruitment times from channel-set labels. Caller-owned artifact logging must save patient splits, seeds, code/input hashes, initialization, fitted scalers and stage-one prediction lineage.

These utilities do not establish a foundation model, causal propagation network, or clinically validated seizure detector. The staged score requires an explicit signal-derived tissue prediction; package `predict_arrays` does not automatically load this experimental model or its separate classifier.
