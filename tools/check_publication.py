"""Check the Git index without printing matched identifiers or secret values."""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {'.py', '.md', '.json', '.toml', '.txt', '.sh', '.yml', '.yaml'}
PATTERNS = {
    'coded_participant_identifier': re.compile(
        rb'(?i)(?<![a-z0-9])(?:P0?\d{2,3}|(?:HUP|CHOP|RID)[-_]?\d{2,}|sub-[a-z]*\d{2,})(?![a-z0-9])'),
    'credential': re.compile(
        rb'gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'),
}


def content_findings(path: str, data: bytes) -> list[dict[str, str]]:
    """Detect selected hazards without treating the result as de-identification."""
    issues = []
    for category, pattern in PATTERNS.items():
        scanned = data
        if category == 'coded_participant_identifier' and path in {
            'tools/check_publication.py',
            'legacy/third_party/s4/models/baselines/vit_all.py',
            'legacy/third_party/s4/src/models/baselines/vit_all.py',
        }:
            # Verified upstream ViT patch-size names, not participant identifiers.
            scanned = scanned.replace(b'vit_small_p16_224', b'vit_small_PATCH_224')
            scanned = scanned.replace(b'jx_vit_base_p16_224', b'jx_vit_base_PATCH_224')
        if pattern.search(scanned) or pattern.search(path.encode()):
            issues.append({'path': path, 'category': category})
    if path.endswith('.json'):
        value = json.loads(data)

        def visit(node: object) -> bool:
            if isinstance(node, dict):
                for key, child in node.items():
                    if key in {'patients', 'train_patients', 'validation_patients', 'test_patients', 'fit_patients', 'selection_patients'} and isinstance(child, list) and child:
                        return True
                    if visit(child):
                        return True
            elif isinstance(node, list):
                return any(visit(child) for child in node)
            return False

        if visit(value):
            issues.append({'path': path, 'category': 'participant_list'})
    return issues


def git(*args: str, data: bytes | None = None) -> bytes:
    """Run read-only Git commands against this checkout."""
    return subprocess.check_output(['git', '-c', f'safe.directory={ROOT.as_posix()}', *args], cwd=ROOT, input=data)


def check() -> dict:
    """Inspect staged content, ignored tracked files and published guide links."""
    names = set(git('ls-files', '-z').decode().strip('\0').split('\0')) - {''}
    ignored = git('ls-files', '-ci', '--exclude-standard', '-z').decode().strip('\0').split('\0')
    findings = [{'path': name, 'category': 'ignored_but_tracked'} for name in ignored if name]
    blobs = {}
    for name in sorted(names):
        if PurePosixPath(name).suffix in TEXT_SUFFIXES:
            data = git('show', ':' + name)
            blobs[name] = data
            findings.extend(content_findings(name, data))
    # Historical third-party documentation may intentionally reference upstream files.
    guides = [name for name in blobs if name.endswith('README.md') and not name.startswith('legacy/third_party/')]
    guides += [name for name in blobs if name.startswith('docs/') and name.endswith('.md')]
    import posixpath
    for name in set(guides):
        for target in re.findall(r'\]\((<[^>]+>|[^)]+)\)', blobs[name].decode('utf-8')):
            target = unquote(target.strip('<>')).split('#')[0]
            if not target or '://' in target:
                continue
            resolved = posixpath.normpath(posixpath.join(posixpath.dirname(name), target))
            if resolved not in names:
                findings.append({'path': name, 'category': 'link_to_unpublished_path', 'target': resolved})
    return {'passed': not findings, 'indexed_files': len(names), 'findings': findings,
            'scope': 'Index only. Manual document review and outgoing-history review are also required.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = check()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
