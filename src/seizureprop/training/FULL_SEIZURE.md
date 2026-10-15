# Independent experts trained on full spontaneous seizures

This protocol supersedes the mixed stimulation/spontaneous, early-observation
pilot as the primary research direction. It does not change or erase those results.
Each model has one responsibility. Onset localization predicts event-level onset
contacts. Channel activity predicts time-resolved seizure activity. They are
trained independently, have separate parameters and checkpoint selection, and do
not gate one another. Combining their outputs is an interpretation step, not a
joint optimization target or a validated propagation-path estimator.

## Cohort and signal support

Accept only records explicitly labeled Spontaneous with empty stimulation fields.
Reject stimulation-induced and unknown-origin records before any fitting. Retain
the existing patient-disjoint partitions after exclusion, including the external
center test set; do not reshuffle to improve results. All pretrained weights are
newly initialized and learned exclusively from spontaneous training patients.
No mixed-cohort checkpoint is loaded, including for feature extraction.

Rebuild inputs from EDF through the clinical seizure offset, including the final
partial second. Retain bipolar montage, causal filters and reserved-baseline
scaling. No experimental impulse suppression is used. One-second frames are
representation units, not a restriction to the first seconds of the seizure.
The final partial frame is zero-padded after normalization; this introduces a
documented boundary approximation. Actual duration is retained for reporting.

Fresh masked-spectral pretraining covers every frame of every training event plus
ten pre-onset seconds in each epoch, partitioned into sixteen-second processing
segments. These segments limit computation, not eligible seizure duration.
Gradient contributions are weighted equally by patient and event. Pretraining
has a maximum of fifteen epochs, validation-loss selection and patience four
after at least five epochs. This is a finite prototype budget, not a claim of
optimal foundation-model training.

The frozen waveform encoder provides thirty-two features per channel/second.
Extraction processes sixteen new frames with up to sixteen preceding warmup
frames, using causal attention and no future data. These features can augment
seventeen baseline-relative signal features. The three downstream controls use
signal features with a simple projection/readout, signal features with spatial
attention and temporal GRU, or both learned and signal features with the same
contextual model. Each sees the entire seizure and ten pre-onset seconds during
every supervised update. Independent task models train for up to sixty epochs,
with patience twelve after at least twenty epochs, validation-patient AP selection,
and validation-only classification thresholds. Three seeds are used.

## Labels at their actual resolution

Onset: a static channel target identifies the onset-contact set for the event.
Attention pooling reads the full supplied recording. Balanced binary cross-entropy
optimizes this one classification task. There is no early-decision penalty or
first-N-second training limit.

Activity: the same classifier emits a logit per channel/second. Available evidence
is used in three forms, averaged equally when available:

- Pre-onset baseline frames are negative controls. This is an explicit pre-onset
  control assumption, not a verified all-day negative annotation.
- The ten-second contact consensus supervises only the frame ending at ten seconds.
- A channel positive in either the onset or ten-second reference must participate
  somewhere in the seizure. Positive-only multiple-instance BCE acts on its
  maximum ictal logit across the complete event. This does not label every frame
  positive. All other whole-event channel bags are unknown, not negative.

These terms supervise the same activity classification task at different temporal
resolutions; they do not jointly train onset, anatomy or propagation objectives.
They are weak supervision, not complete per-channel seizure trajectories. A
ten-second-negative channel may participate later. Missing time labels have zero
direct supervised loss. Positive-bag supervision can exploit unrelated peaks;
sparse labels cannot certify performance throughout the seizure.

## Evaluation after training

Primary onset evaluation uses complete held-out seizures. Activity discrimination
can be measured at its available ten-second labels, with pre-onset controls
reported separately. Whole-seizure activity precision/recall is not identifiable
from these annotations. Do not relabel unknown times using the model itself.

After fitting and validation-only representation selection, run the selected
models with eight-, sixteen-, thirty-two- and sixty-four-second rolling windows
and a growing-history control, at a one-second hop. Windows end at the reported
time and cannot read later samples. Recompute learned representations inside each
raw window; slicing embeddings cached from the full recording retains additional
past encoder context and is not a strict raw-window comparison. Reserved-baseline
normalization still uses its earlier calibration segment. Growing history starts
ten seconds before onset; full-event feature extraction additionally has encoder
warmup, so those conditions are explicitly distinct. These are evaluation conditions, not early
localization training constraints. Onset's static event target allows a
retrospective identification curve. Activity accuracy is unavailable between
annotated anchors, although predictions are saved for review. Compare curves
without selecting a favorable test latency or claiming a clinically reliable
delay. Fixed-window inference differs from full-event training and is measured
as such; known global onset/offset remain retrospective alignment assumptions.

Only three spontaneous external test events remain, and only two have eligible
ten-second annotations. Cohort reuse, center/age differences and sparse labels
preclude independent clinical or foundation-model validation. New independent
spontaneous recordings with denser channel annotations are needed for that claim.
