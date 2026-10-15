# Joint tissue contrast and hard white-matter negatives

`joint_contrastive.joint_train` adds optional training objectives to the existing
patient-uniform localization trainer. The inference architecture and its inputs
are unchanged. Tissue is privileged external supervision, never an inference
input; it must not be described as training without anatomical labels.

The supervised contrastive term follows the [SupCon principle](https://arxiv.org/abs/2004.11362):
same-tissue embeddings are positives and opposite-tissue embeddings are
negatives. Two disjoint baseline views supply embeddings from the actual
identity encoder. Self-comparisons are excluded and anchors are averaged within
tissue class before averaging classes. Both classes must be present. There is
no projection head; this is a small tissue-specific adaptation, not a reproduction
of the original image experiment. Baseline views share a recording and test
short-term preictal stability, not cross-day identity.

The hard-negative term ranks clinical onset positives above independently mined
white-matter errors with a softplus margin. Eligible negatives must simultaneously
be pure WM, clinically onset-negative, and present in the supplied mining mask.
Clinical-positive WM is protected. Mixed, unknown and conflicting annotations
must be encoded as -1, never treated as WM.

## Caller contract

- Split patients before fitting a scaler, mining, or training.
- Create mining masks from patient-out-of-fold predictions. The mined patient's
  recordings must be excluded from both fitting and checkpoint selection of its
  mining model. Preserve the fold assignments, checkpoints, and prediction hashes.
- Supply aligned `tissue_target` and boolean `mined` tensors on the model's device
  for each training record when the corresponding loss is enabled.
- Prespecify weights, temperature, margin and mining threshold. Preserve the
  zero-auxiliary control, random initialization, sampling, preprocessing and
  validation endpoint. This trainer rejects train/validation patient overlap.
- Keep held-out cohort error analysis descriptive. It cannot feed mining masks,
  model selection or subsequent changes while retaining an untouched-test claim.

For the exploratory fixed comparison, contrast weight is 0.1, temperature 0.1,
hard-negative weight 0.2 and margin 1.0. A three-fold mining stage divides external
training patients into fitting, inner selection and prediction groups. Errors are
defined strictly by selection at clinical N. Mining can return very few examples;
that is a coverage limitation, not justification to relabel other tissue pairs.

Compare localization, false-WM selection and retention of clinical onset channels.
Improved tissue embedding separation alone does not demonstrate improved
localization. Repeated seizures, seeds and bipolar pairs are not independent
patients, and feature-envelope correlation does not establish volume conduction.
