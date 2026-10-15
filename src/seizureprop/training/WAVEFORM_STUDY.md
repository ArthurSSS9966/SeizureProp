# Shared multichannel waveform pilot

This is a compact signal-only experiment, not an implementation of SPINT or a
validated foundation model. The encoder learns temporal patterns and functional
relationships among channels. It has no anatomical coordinates or channel-ID
embeddings and cannot infer a unique physical electrode position from these inputs.

## Four separate stages

1. **Shared encoder pretraining:** mask channel/time patches and reconstruct their
   log-magnitude spectra. This supplies a label-free learning signal without
   inventing propagation sequences or forcing all seizure morphologies into one
   cluster. Masked values are removed before tokenization.
2. **Global seizure detection:** a separate binary head pools the channel set.
   Recording-level seizure intervals supervise a global prediction; they do not
   make every channel ictal. Positive crops sample the full available seizure,
   with equal training weight for sampled pre-onset controls.
3. **Onset localization:** another copy of the encoder and a per-channel binary
   head use the clinical onset-contact consensus. Equal positive/negative class
   weighting handles sparse annotated contacts without a ranking target.
4. **Ten-second participation:** a third copy uses only the contact consensus at
   ten seconds. This is a separate snapshot endpoint, not onset-union-spread,
   recruitment time, a causal edge, or a full propagation pathway.

Each downstream model is fitted separately; there is no weighted sum of these
three task losses. Pretraining uses training patients only. Validation patients
select reconstruction loss or task-specific average precision, including the
untrained checkpoint. Test labels do not select epochs or hyperparameters.

## Signal flow and pilot choices

`C channels × 8 seconds × 256 samples/second`
→ shared convolutional patch tokenizer
→ `C × 8 × 32` embeddings
→ two alternating causal time-attention / channel-attention blocks
→ final-time embeddings
→ global pooled detection or per-channel localization head.

- One-second patches retain the earlier analysis resolution. Eight-second input
  contexts provide history at modest memory cost; this is a fixed pilot choice,
  not an established optimal duration. Random training endpoints can yield
  overlapping contexts; these remain dependent observations from one patient.
- Channel attention learns signal relationships while accepting variable montage
  sizes. Channel permutation changes the order of channel outputs, not their
  meaning. Attention is not used as a biological connectivity ground truth.
- Causal filtering and temporal attention avoid future-patch leakage. A prediction
  becomes available at the end of its one-second patch; retained filter delays
  and clinical alignment uncertainty remain relevant to onset timing.
- Record-specific scaling uses an earlier reserved 20-second pre-onset baseline,
  with a 1-microvolt standard-deviation floor and clipping at 20 standard
  deviations. This is an explicitly retrospective baseline availability assumption,
  not a learned population transform or an online calibration validation.
- The onset input is `[-7,+1)` seconds relative to consensus onset. The ten-second
  input is `[+3,+11)`, so its last patch covers the annotated time. The one-second
  look beyond the nominal ten-second instant must be retained in interpretation.
- Width 32, four attention heads, two blocks, dropout 0.1, 25% masking, AdamW at
  0.001, and 20 epochs form a fixed feasibility budget. One patient-balanced
  update per training patient per epoch prevents patients with many seizures from
  dominating; it does not amount to exhaustive exposure to the available signal.
- Temporal-only scratch, multichannel scratch, frozen pretrained, and fine-tuned
  pretrained encoders use matched downstream sampling, task budgets, seeds, and
  initial shared parameters where architectures permit. Epoch selection is
  task-specific. Pretrained arms have additional label-free computation.

`EncoderConfig` and `FitConfig` expose architecture and training parameters.
`fit_pretraining` and `fit_task` accept cohort-specific sampling and validation
callbacks. Cohort identifiers, annotations, caches, private drivers, checkpoint
weights, and per-patient predictions remain outside the public package.

## Interpretation and verification

The current pilot uses fixed patient partitions, with an adult development cohort
and pediatric evaluation cohort. Patient-macro metrics first average events
within each patient. Seed variability is not a patient confidence interval.
These data have already supported model development and are not an untouched
confirmatory cohort. Stimulation-induced and spontaneous events must also be
reported separately.

Detection is evaluated on sampled pre-onset and ictal snippets, not continuous
all-day monitoring; average precision depends on that sampling prevalence. It
does not establish event sensitivity, latency, or false alarms per hour.
Localization uses known onset alignment and clinical contact consensus, which
does not establish complete labels for every nonselected channel. Bipolar pairs
are positive if either endpoint belongs to the corresponding consensus set.
This mapping can include adjacent contacts and should not be interpreted as
exact single-contact localization.

Verification covers channel equivariance, variable montage sizes, causal temporal
behavior, masking leakage, spatial gradients, and frozen encoder behavior. The
private experiment stores protocol/source/cache/code hashes, selected epochs,
learning curves, checkpoints and replayable predictions. No G/W filtering, graph
pseudolabels, or fabricated propagation ranking is introduced in this pilot.
