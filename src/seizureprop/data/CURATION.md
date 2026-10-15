# Patient-level data curation

`curation.py` provides reproducible patient-disjoint partitions, metadata coverage
tables, and auditable patient/subtype training mixtures. It does not clean signal
artifacts, infer missing diagnoses, or turn correlated windows into new patients.

## Cohort contract

Create one metadata row per patient before splitting. Keep a separate event table
with recording identity, source hashes, source intervals, acquisition settings,
annotation provenance and eligibility. Reject cross-role patient membership and
audit duplicate signals/source intervals. Exact hashes alone do not detect every
near duplicate. Store individual-level manifests and outputs in ignored folders.

Choose primary strata before fitting. Each stratum needs enough patients for all
requested folds; otherwise report missing coverage or use fewer folds. Secondary
metadata can guide split balance, but must not use model scores or signal features.
Do not pick a favorable seed after seeing model performance. Sparse secondary
levels may remain absent from some folds and must be reported explicitly.

Example with an existing patient metadata table:

```python
from seizureprop.data.curation import stratified_patient_folds, patient_epoch_schedule

# patient_table has one row per patient and explicit subtype labels.
fold_by_patient = stratified_patient_folds(
    patient_table, stratum="subtype", numeric=["age", "channels"], seed=47,
)
# train_patient_groups contains ONLY the current fold's training patients.
schedule = patient_epoch_schedule(
    train_patient_groups, policy="subtype",
    draws=2 * len(train_patient_groups), seed=101,
)
```

The quota budget in this example assumes two subtypes. Create inner validation
with `stratified_validation_patients` before constructing training groups.

For each development fold, keep separate training, inner validation and held-out
internal evaluation patients. Fit every encoder, including self-supervised
pretraining, on that fold's training patients only. Select checkpoints and decision
thresholds on inner validation. A predeclared per-record baseline calibration can
be applied to an unseen record without optimizing model parameters. Do not pool
normalization statistics across held-out patients.

## Training contribution

`patient` gives every patient equal intended weight. `subtype` gives every subtype
equal total weight and divides it equally among patients in that subtype. Process
all complete eligible seizures of a sampled patient and average their losses, so
more seizures, channels or windows do not automatically give a patient more weight.

`patient_epoch_schedule` uses integer quotas and includes every training patient
each epoch when the draw budget permits. Log intended probabilities, realized
draws, event exposures and processed signal duration. Repeated exposure is not new
independent evidence. Keep optimizer-update and epoch budgets matched between
policies; exposure duration and actual compute may still differ.

For a two-subtype cohort, twice the training-patient count per epoch permits both
policies to cover all patients while assigning equal subtype quotas. This is a
controlled design choice, not a universal optimum. Changes to diagnosis groups,
coverage or budget should be versioned and checked again.

## Evaluation contract

Report patient-macro performance, subtype-macro performance, the weaker subtype,
and patient/event counts for every subgroup. Choose checkpoints using a declared
validation metric; do not reselect on internal held-out scores. If held-out folds
are later used to choose the training policy, their results are development model
comparison, not an untouched final test. Small subgroups do not support strong
claims from narrow-looking window-level confidence intervals.

Reserve an independent external cohort for confirmation. A center repeatedly used
for diagnosis is exploratory. Do not make its feature distribution artificially
match training to manufacture a favorable split. New age groups, disease subtypes
or acquisition domains need actual additional patients when support is absent.

Keep complete-seizure onset classification and channel-activity classification as
independent tasks. Sparse channel-time labels remain sparse; whole-seizure input
does not create dense whole-seizure activity ground truth. Rolling latency is a
post-training evaluation, not an implicit early-onset training objective.
