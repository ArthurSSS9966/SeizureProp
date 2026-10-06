# Maintenance tools

Run from the repository root:

```powershell
python tools/check_publication.py
python tools/audit_repository.py --output result/local_audit
python -m unittest discover -s tests -p test_package.py -v
python -m unittest discover -s tests -p test_publication_guard.py -v
```

The publication check inspects the Git index for ignored files, selected identifier
and credential patterns, and broken links in public guides. It is a guardrail,
not automatic de-identification or permission to publish. It prints findings by
path/category without echoing matched patient data or credentials.

The source inventory may include private paths; keep its outputs under ignored
`result/`. Historical study/compatibility commands remain local because they
reference specific research artifacts. See the [policy](../docs/data_publication_policy.md).
