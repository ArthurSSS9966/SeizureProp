"""Build a static, non-executing inventory of research code and notebooks.

No project module is imported, and no patient data or notebook output is read.
Results describe lexical structure, not measured runtime coverage or correctness.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
CODE_DIRS = ('scripts', 'tests', 'src', 'legacy')


def category(path: Path) -> str:
    """Classify location without claiming ownership from a folder name alone."""
    part = path.relative_to(ROOT).parts
    if len(part) == 1:
        return 'legacy_root'
    if part[:3] == ('src', 'seizureprop', 'legacy'):
        return 'legacy_waveform'
    if part[:2] == ('src', 'seizureprop'):
        return 'active_package'
    return {'scripts': 'research_scripts', 'tests': 'tests',
            'legacy': 'archived_framework'}.get(part[0], 'other')


def module_name(path: Path) -> str:
    parts = list(path.relative_to(ROOT).with_suffix('').parts)
    if parts[-1] == '__init__':
        parts.pop()
    return '.'.join(parts)


def build(output: Path) -> None:
    """Inventory source files, dependencies, exact clones, and notebook code cells."""
    paths = sorted(list(ROOT.glob('*.py')) + [p for d in CODE_DIRS for p in (ROOT / d).rglob('*.py')
                                             if '__pycache__' not in p.parts])
    modules = {module_name(p): p.relative_to(ROOT).as_posix() for p in paths}
    records, edges, definitions = [], [], []
    bodies: dict[str, list[dict]] = defaultdict(list)
    files_by_hash: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        source = path.read_text(encoding='utf-8-sig')
        rel = path.relative_to(ROOT).as_posix()
        files_by_hash[hashlib.sha256(source.encode()).hexdigest()].append(rel)
        item = {'path': rel, 'category': category(path), 'lines': len(source.splitlines()),
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'imports': [],
                'entrypoint': bool(re.search(r'if __name__\s*==', source)),
                'path_mutation': 'sys.path' in source, 'absolute_path_lines': [],
                'artifact_path_lines': [], 'module_constants': [], 'top_level_calls': []}
        for n, line in enumerate(source.splitlines(), 1):
            if re.search(r'(?<![A-Za-z])[A-Za-z]:[/\\](?!/)|/home/|/mnt/|/Users/', line):
                item['absolute_path_lines'].append(n)
            if re.search(r'(result/|checkpoints/|data/|data_test/|/best\.pt|targets\.csv)', line):
                item['artifact_path_lines'].append(n)
        try:
            tree = ast.parse(source, filename=rel)
        except SyntaxError as error:
            item['parse_error'] = f'{error.msg}:{error.lineno}'
            records.append(item)
            continue
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                names = [t.id for t in targets if isinstance(t, ast.Name)]
                item['module_constants'].extend({'name': name, 'line': node.lineno}
                                                for name in names if name.isupper())
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                item['top_level_calls'].append({'line': node.lineno, 'call': ast.unparse(node.value.func)})
        for node in ast.walk(tree):
            imports = []
            if isinstance(node, ast.Import):
                imports = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ''
                if node.level:
                    own = module_name(path).split('.')
                    if path.name != '__init__.py':
                        own.pop()
                    own = own[:len(own) - node.level + 1]
                    base = '.'.join(own + ([base] if base else []))
                imports = [base] + [f'{base}.{a.name}' for a in node.names]
            for name in imports:
                item['imports'].append({'module': name, 'line': node.lineno})
                if name in modules and modules[name] != rel:
                    edges.append({'from': rel, 'to': modules[name], 'line': node.lineno})
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                desc = {'path': rel, 'name': node.name, 'line': node.lineno,
                        'lines': node.end_lineno - node.lineno + 1}
                definitions.append(desc)
                if desc['lines'] >= 8:
                    body = list(node.body)
                    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                        body = body[1:]
                    key = hashlib.sha256(ast.dump(ast.Module(body=body, type_ignores=[])).encode()).hexdigest()
                    bodies[key].append(desc)
        records.append(item)
    unique_edges = {(e['from'], e['to'], e['line']): e for e in edges}
    edges = list(unique_edges.values())
    imports_count = Counter(e['to'] for e in edges)
    notebooks = []
    for path in sorted((ROOT / 'notebooks').rglob('*.ipynb')):
        doc = json.loads(path.read_text(encoding='utf-8-sig'))
        cells = [c for c in doc['cells'] if c['cell_type'] == 'code']
        text = '\n'.join(''.join(c['source']) for c in cells)
        notebooks.append({'path': path.relative_to(ROOT).as_posix(), 'code_cells': len(cells), 'code_lines': len(text.splitlines()),
                          'definition_lines': len(re.findall(r'^\s*(?:class|def)\s+', text, re.M)),
                          'absolute_path_lines': sum(bool(re.search(r'(?<![A-Za-z])[A-Za-z]:[/\\](?!/)|/home/|/mnt/', s)) for s in text.splitlines()),
                          'models_import': bool(re.search(r'(?:from|import) models', text))})
    summary = {'files': len(records), 'lines': sum(r['lines'] for r in records),
               'groups': {g: {'files': len(rs), 'lines': sum(x['lines'] for x in rs)}
                          for g in sorted({r['category'] for r in records})
                          for rs in [[r for r in records if r['category'] == g]]},
               'parse_errors': [r for r in records if 'parse_error' in r],
               'path_mutation_files': sum(r['path_mutation'] for r in records),
               'absolute_path_files': sum(bool(r['absolute_path_lines']) for r in records),
               'artifact_path_files': sum(bool(r['artifact_path_lines']) for r in records),
               'entrypoints': sum(r['entrypoint'] for r in records),
               'notebooks': len(notebooks), 'notebook_code_lines': sum(n['code_lines'] for n in notebooks),
               'import_hubs': imports_count.most_common(15)}
    clones = [v for v in bodies.values() if len({x['path'] for x in v}) > 1]
    same_files = [v for v in files_by_hash.values() if len(v) > 1]
    result = {'scope': 'Root Python plus CODE_DIRS; research notebook code cells only. Excludes data, outputs, tools/auditor itself and archived framework notebooks. Static imports do not prove runtime usage.',
              'summary': summary, 'files': records, 'dependencies': edges,
              'longest_functions': sorted(definitions, key=lambda x: x['lines'], reverse=True)[:35],
              'exact_function_body_clones': clones, 'identical_source_files': same_files,
              'notebooks': notebooks}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'repository_inventory.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    lines = ['# Repository source inventory', '', 'Generated by `tools/audit_repository.py`. Line counts include comments and blank lines. Static findings need contextual review.', '',
             '| File | Group | Lines | Entry point | Path mutation |', '|---|---|---:|---|---|']
    lines += [f"| `{r['path']}` | {r['category']} | {r['lines']} | {r['entrypoint']} | {r['path_mutation']} |" for r in records]
    lines += ['', '## Notebook inventory', '', '| Notebook | Code cells | Code lines | Definition lines |', '|---|---:|---:|---:|']
    lines += [f"| `{n['path']}` | {n['code_cells']} | {n['code_lines']} | {n['definition_lines']} |" for n in notebooks]
    (output / 'repository_inventory.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(f'Exact function-body clone groups: {len(clones)}; identical-file groups: {len(same_files)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    build(parser.parse_args().output)
