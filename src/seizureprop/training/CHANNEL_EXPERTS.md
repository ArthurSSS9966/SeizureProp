# Separate channel participation and onset experts

The experiment keeps multichannel waveform input and gives each channel an
independent binary logit. Sigmoid permits several positive channels; no
cross-channel softmax or hard channel exclusion is used. The experts start from
the same pretrained encoder but fine-tune separate copies, with separate losses
and epoch selection. This is a small supervised expert comparison, not a MoE
router or a foundation-model validation.

## Supervision and observation time

- **Participation:** the available target is the clinician consensus of channels
  seizing at ten seconds. Input covers `[+2,+10)` seconds relative to consensus
  onset. This is recognition from an eight-second history available at ten
  seconds; it is not prediction of future participation. Unlike the previous
  pilot's `[+3,+11)` input, it does not read beyond the nominal target instant.
- **Onset:** the target is the clinical onset-contact set. Every onset arm reads
  `[-10,+1)` seconds. The one-second post-onset observation is explicit, and
  onset alignment is provided retrospectively. No online onset-time detection
  or exact contact recruitment-time claim follows.
- Bipolar pairs inherit a positive label when either endpoint is in the relevant
  consensus set. Other retained good contacts are consensus-negative reference
  labels, not proof of physiological noninvolvement. Samples without usable
  labels are excluded. `masked_channel_bce` excludes negative/NaN unknown labels
  and balances known positive/negative classes. There is no transfer of the
  ten-second target to every earlier/later window.
- Ten-second consensus combines raters who may have placed onset at slightly
  different times; an instantaneous exact ground truth is therefore approximate.

## Heads and matched controls

`ChannelExpert` uses the final, mean, and first-to-final change of each channel's
temporal embeddings. A shared two-layer MLP produces one logit per channel. A
linear last-embedding participation head supplies a simpler matched-window
control. Encoder attention still uses the full channel set.

Three onset arms have identical encoder/head shapes, initialization, sampling,
signal coverage, and training budget:

1. Waveform-only: four auxiliary columns are zero.
2. Soft participation: columns contain last score, mean score, score change, and
   maximum over four ordered participation predictions.
3. Shuffled participation: the same four columns are permuted across channels
   together, preserving their distribution but breaking channel correspondence.

Auxiliary predictions use eight-second windows ending at -2, -1, 0 and +1
seconds relative to onset. Their union is `[-10,+1)`, matching the raw waveform
support of all onset arms. No ten-second signal or target is passed to onset.
Training a participation head at ten seconds and applying it near onset is an
explicitly unvalidated temporal transfer; its value must be demonstrated by the
matched onset comparison, not assumed from its ten-second accuracy.

## Patient-excluded training

Training patients are partitioned into three folds. For each fold, **both**
masked pretraining and supervised participation fitting exclude every record of
the held-out patients. Those held-out patients receive auxiliary scores from
that fold's expert. The original external validation partition selects epochs;
the excluded training fold's labels never select its auxiliary expert.

The onset expert then fits all training patients using their excluded-patient
auxiliary scores. Validation/test auxiliary scores come from a final participation
expert fitted on all training patients. Validation and test patients do not enter
pretraining or supervised gradient updates. Fold and final experts have different
training-set sizes, a limitation of this cross-fitting design. Scores are frozen
and detached: onset loss never updates the participation expert.

## Evaluation

Three fixed seeds and unchanged patient partitions are used. Validation-patient
macro AP selects each checkpoint, including epoch zero. A threshold grid from
0.05 through 0.95 selects validation-patient macro F1; test labels never tune the
threshold. Report patient-macro AP, AccAtN, F1, precision, recall and Brier score.
Sigmoid scores, especially under balanced BCE, are not calibrated probabilities
by construction. AP/ranking and fixed-threshold channel decisions are distinct.

The primary contrast is soft versus waveform-only onset AP. Shuffling checks
whether correspondence matters, while comparisons of head types test the
participation readout. The old pilot has a different ten-second input and onset
context, so its headline numbers are not a matched control. Reused small cohorts,
repeated validation selection, and sparse spontaneous events limit conclusions.
No continuous per-channel activity labels or complete propagation paths have
been established.

## Exploratory two-anchor extension

The initial ten-second-only participation comparison is retained unchanged. A
separate follow-up uses the existing clinical onset and ten-second sets as
time-specific participation labels. Each participation update samples one of the
two anchors with equal probability, so the number of updates remains unchanged.
The onset anchor reads `[-7,+1)` seconds and uses the clinical onset-contact set;
the ten-second anchor keeps `[+2,+10)` and its own consensus set. These are two
labeled snapshots per eligible event, not twice as many independent seizures.

Participation epoch selection averages validation-patient AP across both anchors.
The primary participation evaluation and its threshold selection remain at ten
seconds. Onset controls and signal coverage remain unchanged. Fold-specific
pretraining checkpoints are reused with their original patient exclusions, while
each supervised expert is refit. The excluded patient's onset labels never train
the expert producing that patient's auxiliary scores. This avoids circular use
of labels when onset annotations supervise both tasks.

This extension follows the initial exploratory results and is not an independent
test. It addresses the training-time mismatch at the onset anchor but does not
provide labels for every intermediate or pre-onset observation.

Private experiment artifacts retain source/cache/code hashes, fold lineage,
auxiliary arrays, selected checkpoints, validation thresholds and test outputs.
The public API is cohort-agnostic; patient identifiers and experiment outputs
remain excluded from publication.
