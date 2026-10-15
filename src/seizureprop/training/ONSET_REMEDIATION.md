# Matched onset remediation experiments

These experiments test signal processing, model choice and supervision separately.
They use the existing patient partitions, three seeds, clinical onset-contact
reference and complete bipolar candidate sets. No anatomy label enters the model,
and no channel is removed. The primary endpoint observes the eleven-second
interval ending one second after the supplied clinical onset. This remains
retrospective localization, not online seizure detection or prediction.

## Input ablation

The four inputs are original processing, robust baseline normalization only,
transient suppression only, and both interventions. Raw EDF is reread because
previous clipping cannot be reversed. Original causal resampling, bandpass and
notch filtering are retained. Calibration uses the same earlier twenty-second
baseline, ending well before the observation interval.

`QualityConfig` specifies every threshold. Robust normalization uses the median
and 1.4826 times the median absolute deviation, with the original one-microvolt
floor and clipping limit. The experimental transient suppressor acts after
causal resampling and before filtering: an extreme derivative accompanied by an
extreme amplitude triggers a three-sample hold of the preceding valid value.
Thresholds are twelve and eight baseline robust standard deviations, respectively.
It uses no stimulus metadata, channel identity, labels or future interpolation.
Its output is causal after calibration. The method can suppress physiological
spikes; masks and affected sample fractions are retained for inspection.
Repeated triggers can extend the hold well beyond three samples. Report the
fraction of reference-positive and reference-negative channels with prolonged
holds; a higher retrieval score alone does not justify adopting this transform.

The same original pretrained encoder initializes each input arm; there is no
new pretraining specific to an input transform. Each arm receives the same
patient/event sampling and fitting budget. The selected input maximizes mean
validation-patient AP over the three seeds, including the unchanged original as
an eligible choice. Test scores are withheld until all selections finish.

## Older and current models

`LegacyResWaveOnset` reproduces the repository's legacy `ResNet` layer names,
residual gating, BatchNorm, dropout and initial valid-convolution truncation.
An invariant test compares its sigmoid score against the original class-one
softmax after copying all parameters. The new API returns the difference of
the two class logits so that BCE is applied correctly.

The legacy architecture is retrained both on raw normalized waveforms and on
the repository's seventeen per-second signal features. The latter also receives
reserved-baseline feature normalization. Both see the same physical eleven
seconds and the same targets. The raw version uses one input feature and kernel
sixteen; the feature version uses seventeen features and kernel eleven. Both
retain thirty-two filters, three blocks and dropout 0.2. These are explicitly
adapted, retrained baselines, not reproductions of the historical USC checkpoints,
feature selection, hyperparameter optimization or reported clinical scores.

Current-model controls include scratch training and a pretrained linear readout,
alongside the pretrained nonlinear readout from the input ablation. Comparisons
use original and validation-selected inputs, deduplicated if they coincide. The
raw comparison separates model choice from engineered feature extraction; the
feature comparison intentionally evaluates an architecture/representation bundle.
Equal update budgets are not equal compute or exhaustive architecture tuning.

## Onset objectives

The baseline objective gives equal total weight to onset positives and onset
negatives. The grouped-negative variant preserves this positive/negative balance
but divides negative weight equally between known ten-second participants and
the other onset negatives. Unknown ten-second annotations are retained as general
onset negatives; they are not assigned a negative ten-second activity label.

An independent uncertainty ablation uses only labels for which all raters agree
with the binary reference. Combining it with grouped negatives is a fourth
objective condition. The evaluation reference and candidate set never change;
supervision coverage is reported explicitly. This is annotation-based grouping,
not adaptive hard-negative mining, a learned propagation ranking or an assumed
channel recruitment time. No jointly optimized participation head is added.

The objective is selected by three-seed validation-patient AP, with the baseline
eligible. Prespecified two- and five-second observation endpoints are then fitted
using the selected objective. Their scores are reported separately, not used to
select a latency from the test results. Epochs and classification thresholds are
chosen on validation patients only. All selected checkpoints, histories, source
hashes, masks, coverage and per-channel outputs stay in private experiment files.

This is an exploratory, repeatedly used external cohort with few spontaneous
test events and different centers in training and testing. Patient-level paired
intervals quantify descriptive uncertainty; they do not replace independent
confirmation. Old high USC involvement retrieval scores are a different endpoint
and cannot serve as a numerical target for this all-channel onset benchmark.
