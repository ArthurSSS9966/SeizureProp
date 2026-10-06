# Experiment configuration

The public `package_study/` recipes contain model/training settings and relative
paths to local data. They do not distribute participant lists, manifests,
checkpoint weights or provenance files. Copy a recipe to `configs/local.json`
and adapt local paths when starting new work; that filename is ignored.

Clinical cohort configurations, channel aliases, split lists and patient
mappings remain local. Store new metadata under `private/` or `result/`.
Only publish deliberately reviewed examples with invented identifiers. The
package validates patient separation, but validation does not authorize release
of the split list itself. See [policy](../docs/data_publication_policy.md).
