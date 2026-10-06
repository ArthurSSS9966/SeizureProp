# Research scripts

`seeg_external_predict.py` is the public compatibility wrapper around
`seizureprop.evaluation.predict`. Prefer the [package interface](../src/seizureprop/README.md).

The remaining historical scripts stay local. Some embed participant IDs,
contact mappings or clinical findings in report templates, so ignoring only
output CSV/Markdown files would still expose those details. General-purpose
functions should be migrated into the package and reviewed before publication.
These private scripts were not deleted or modified by the publication audit.
