# Baseline identity pretraining

`identity_pretraining.py` trains only the signal-derived channel identity encoder
of `ChannelIdentityScorer`. The localization architecture, temporal backbone,
attention and localization head retain their initial weights until supervised
fine-tuning. A temporary linear G/W head is saved separately for diagnostics and
is not required for localization inference.
The consistency-only arm does not train this head; its head-based tissue AUROC
is an untrained-head diagnostic, not a tissue-classification result. After
localization fine-tuning, a fixed pretrained head can reveal representation
drift, but its loss of performance does not prove that tissue information is
absent from the embedding.

Three ablations isolate the objectives: baseline consistency only, tissue
supervision only, and their combination. The consistency objective uses matched
channels in non-overlapping baseline windows, plus variance and covariance
penalties inspired by [VICReg](https://arxiv.org/abs/2105.04906). It is a compact
adaptation without a projection network. Anti-collapse diagnostics must be
checked; a low alignment loss alone does not demonstrate useful representations.

Inputs to `baseline_views` are channel-by-feature-by-time tensors of one-second
frames. With 30 baseline frames, the views are frames 0–11 and 16–27. Frames 28–29
and all ictal data are excluded. This tests stability across short preictal
segments, not across days or independently recorded interictal sessions. Existing
feature extraction and normalization have their own temporal support and must
be documented separately.

Tissue supervision requires audited pure GM/WM bipolar-pair labels: 1 for GM,
0 for WM and -1 for excluded mixed, unknown or conflicting annotations. Never
infer a pair label from one contact alone. These are training targets, not
inputs. Keep tissue inputs absent at deployment and do not interpret a learned
score as a calibrated anatomical probability.

Split patients before preprocessing/model fitting. Pretraining validation loss
selects its checkpoint; clinical validation onset AP separately selects the
fine-tuned model. Neither uses the held-out institution or local test patients.
Use matched initialization and fresh optimizer/RNG state for fine-tuning, and
report patient-averaged localization alongside white-matter errors. Archived
reconstruction pretraining updates a larger parameter subset, so its comparison
with identity-only pretraining does not isolate the loss function alone.
