"""Assemble the reviewed requirements and their readable hierarchy."""

import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
VISUAL = Path('/Users/nicarq/.codex/visualizations/2026/09/07/01a07d70-7a91-7710-8713-1bb03e65bf65/nightstream-requirements.html')
COMMIT = '4fc02857c3aa4207f3290739cc62d794ba86f5f9'
SHARDS = ['foundations.json', 'folding.json', 'hypernova.json', 'implementation.json']
LABELS = {
    'proved': 'local proof exists', 'partial': 'partial', 'definition': 'defined',
    'assumption': 'explicit assumption', 'not_required': 'not required at this level',
    'not_reviewed': 'not verified by this review', 'connected': 'connected locally',
    'open': 'open', 'tested_scoped': 'scoped tests passed',
    'recorded_only': 'recorded evidence only', 'implemented': 'code exists',
}
ORIGINS = {
    'paper': 'Paper', 'profile': 'Nightstream profile', 'implementation': 'Implementation',
    'assumption': 'External assumption', 'out_of_scope': 'Outside selected Stage 1',
}
nodes = [{'id': 'root', 'parent': None, 'label': 'Nightstream Stage 1', 'kind': 'group',
          'origin': 'implementation', 'requirement': 'Selected SuperNeo v1.1 folding and HyperNova F′ system.'}]
for shard in SHARDS:
    nodes.extend(json.loads((HERE / shard).read_text()))
by_id = {node['id']: node for node in nodes}
assert len(by_id) == len(nodes), 'Duplicate node ID'
children = {}
references = 0
for node in nodes:
    if node['id'] == 'root':
        continue
    assert node['parent'] in by_id, (node['id'], 'missing parent', node['parent'])
    assert by_id[node['parent']]['kind'] == 'group', (node['id'], 'parent is not a group')
    assert node['origin'] in ORIGINS, (node['id'], 'origin', node['origin'])
    children.setdefault(node['parent'], []).append(node)
    if node['kind'] == 'leaf':
        for axis in ['proof', 'connection', 'rust']:
            assert node[axis] in LABELS, (node['id'], axis, node[axis])
        assert node.get('requirement'), (node['id'], 'missing requirement')
    for dependency in node.get('depends_on', []):
        assert dependency in by_id, (node['id'], 'missing dependency', dependency)
    for source in node.get('paper', []) + node.get('code', []):
        path = ROOT / source['path']
        assert path.is_file(), (node['id'], 'missing source', str(path))
        assert 0 < source['line'] <= len(path.read_text().splitlines()), (node['id'], 'bad source line', source)
        references += 1
    ancestors = {node['id']}
    parent = node['parent']
    while parent is not None:
        assert parent not in ancestors, (node['id'], 'parent cycle')
        ancestors.add(parent)
        parent = by_id[parent]['parent']
root_order = ['F', 'T', 'C', 'R', 'D', 'N', 'H', 'L', 'P', 'O']
assert set(root_order) == {node['id'] for node in children['root']}
children['root'].sort(key=lambda node: root_order.index(node['id']))
ordered = []

def walk(node):
    ordered.append(node)
    for child in children.get(node['id'], []):
        walk(child)

walk(by_id['root'])
assert len(ordered) == len(nodes), 'Unreachable node'
data = {'schema': 1, 'commit': COMMIT, 'scope': 'Selected Nightstream Stage 1; local normative paper versions.',
        'nodes': ordered}
(HERE / 'requirements.json').write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')) + '\n')

def source_link(source):
    label = source.get('section') or source.get('symbol') or Path(source['path']).name
    label = str(label).replace('[', '').replace(']', '')
    return f"[{label}]({ROOT / source['path']}:{source['line']})"

header = [
    '# Nightstream Stage 1 — paper requirements and implementation map', '',
    f'Reviewed source: `{COMMIT}`. The map uses the local SuperNeo v1.1 and HyperNova text selected by the owner goal.', '',
    'Each leaf is an individual operation, equation, check, serialization rule, or proof connection. Indexed coefficient/row families share one leaf with their dimensions. Shared primitives are referenced rather than required as separate implementations.', '',
    '**Proof** reports a local Lean result. **Link** reports connection to that item\'s local consuming contract. **Rust** reports execution evidence. A local proof does not imply full-system soundness or approved production conformance. “Not verified by this review” is unknown coverage, not proof that code is absent.', '',
    'Paper requirements, Nightstream profile choices, implementation requirements, external assumptions, and material outside the selected stage are marked separately. This map does not change the existing goal, proof assumptions, or approved conformance criteria.', '',
    'Source status is from declaration/implementation inspection and the earlier same-commit validation. No new cryptographic verification or complete conformance run was performed to make this map.', '',
]
lines = header[:]

def emit(node, depth):
    indent = '    ' * depth
    text = f"{indent}- **{node['id']} — {node['label']}** ({ORIGINS[node['origin']]})"
    if node['kind'] == 'leaf':
        text += f" — Proof: {LABELS[node['proof']]}; Link: {LABELS[node['connection']]}; Rust: {LABELS[node['rust']]}."
    if node.get('requirement'):
        text += ' ' + node['requirement'].replace('\n', ' ')
    if node.get('remaining'):
        text += ' **Remaining:** ' + node['remaining'].replace('\n', ' ')
    sources = node.get('paper', []) + node.get('code', [])
    if sources:
        text += ' Sources: ' + ', '.join(source_link(source) for source in sources) + '.'
    if node.get('depends_on'):
        text += ' Uses: ' + ', '.join('`' + item + '`' for item in node['depends_on']) + '.'
    lines.extend([text, ''])
    for child in children.get(node['id'], []):
        emit(child, depth + 1)

for node in children['root']:
    emit(node, 0)
(HERE / 'MAP.md').write_text('\n'.join(lines) + '\n')

gap_lines = ['# Open and conditional proof connections', '',
             'This is a view of the full map. It does not create new approved work or treat unreviewed coverage as a proved defect. External assumptions are separate from missing deterministic connections.', '']
for parent in children['root']:
    leaves = [node for node in ordered if node['kind'] == 'leaf' and (node['id'].startswith(parent['id'] + '.')) and node['origin'] != 'out_of_scope']
    selected = [node for node in leaves if node['proof'] in ('partial', 'assumption') or node['connection'] in ('partial', 'open')]
    if not selected:
        continue
    gap_lines += ['## ' + parent['label'], '']
    for node in selected:
        refs = node.get('paper', []) + node.get('code', [])
        gap_lines += [f"- **{node['id']} — {node['label']}**. " + (node.get('remaining') or node['requirement']) + ' ' + ', '.join(source_link(ref) for ref in refs), '']
(HERE / 'OPEN_ITEMS.md').write_text('\n'.join(gap_lines) + '\n')

fragment = VISUAL.read_text()
payload = json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c')
pattern = r'(<script type="application/json" id="requirements-data">).*?(</script>)'
fragment, changed = re.subn(pattern, lambda match: match.group(1) + payload + match.group(2), fragment, count=1, flags=re.S)
assert changed == 1, 'Missing visualization data slot'
assert len(fragment.encode()) < 1_000_000, 'Visualization platform size contract'
VISUAL.write_text(fragment)
summary = {'source_references_checked': references, 'nodes': len(nodes),
           'leaves': sum(node['kind'] == 'leaf' for node in nodes),
           'origins': dict(Counter(node['origin'] for node in nodes if node['kind'] == 'leaf')),
           'visual_bytes': len(fragment.encode())}
(HERE / 'validation.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary))
