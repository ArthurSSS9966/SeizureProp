# Data publication policy

Coded or de-identified participant identifiers and individual results are private
for this repository. Replacing names with codes does not make a report approved
for publication. This policy also covers public-dataset participant lists used
in local studies, contact mappings, annotations and patient-level provenance.

## Retained locally, excluded from publication

- Raw recordings and imaging, clinical spreadsheets, feature caches and metadata.
- Per-patient/per-recording/contact-level predictions, metrics, labels and ID maps.
- Notebook source and output, including embedded images and clinical text.
- Study-specific JSON configurations, cohort splits, manifests and provenance.
- Historical reports, manuscript drafts and audit inventories that contain IDs.
- Project proposals and proposal reviews, even when they contain no patient data.
- Cohort-specific scripts and private waveform code containing clinical details.
- All checkpoint folders, model weights, archives, logs and tracking databases.
- Secrets, local agent/editor configuration, bytecode and build outputs.

Private files stay on disk. `.gitignore` does not redact content or untrack an
existing file. The publication audit removes ignored paths from the Git index
with `git rm --cached`, without deleting working files. Source-controlled guides,
the reusable package, synthetic tests and reviewed parameter recipes remain.

The broad private defaults for docs, scripts, configs and tools are deliberate:
add exceptions only after reviewing the complete contents. Do not blanket-ignore
all JSON or Markdown: public model settings, schemas and general docs use them.
A report-generating Python string can expose the same data as a saved report.

## Before publishing

1. Review `git diff --cached` and run `python tools/check_publication.py`.
2. Confirm ignored paths are absent from the index and public local links resolve.
3. Review any warnings and all newly admitted documents manually. The scanner
   detects selected patterns, not every possible identifier or clinical fact.
4. Test a clean exported tree; the local research environment can hide missing files.
5. Audit all commits being pushed, not only the latest file tree. Do not push the
   old local experiment branch: its unpublished ancestors include private reports.

Publish a reviewed snapshot based directly on the existing public default branch,
without including the unpublished experiment commits as ancestors. Keep the local
research branch and files intact. No force push or historical rewrite is implied.

## Existing public history

The earlier public repository already contained notebooks and model checkpoints.
Removing them from the new tree does not remove earlier commits, clones or caches.
A separate owner-approved history-removal process would be needed to address that
prior exposure. This audit does not claim those historical copies have disappeared.
