"""Status and quantitative views of the recorded requirement snapshot."""

from collections import Counter
from fractions import Fraction
import math
import re

AXES = [('proof', 'Proof'), ('connection', 'Link'), ('rust', 'Rust')]
CATEGORIES = {
    'proof': [('proved', 'Proved'), ('assumed', 'Assumed'), ('open', 'Open'), ('not_applicable', 'N/A')],
    'connection': [('connected', 'Connected'), ('open', 'Open'), ('not_applicable', 'N/A')],
    'rust': [('tested', 'Scoped tests'), ('implemented', 'Code only'), ('recorded', 'Recorded'),
             ('open', 'Open'), ('not_applicable', 'N/A')],
}
SCOPES = {'model', 'interactive', 'implementation', 'production', 'outside_stage_1'}
STATUSES = {'proved', 'assumption', 'partial', 'definition', 'not_required', 'not_reviewed',
            'connected', 'open', 'tested_scoped', 'recorded_only', 'implemented'}


def category(axis, status):
    if status == 'not_required':
        return 'not_applicable'
    if axis == 'proof':
        return {'proved': 'proved', 'assumption': 'assumed'}.get(status, 'open')
    if axis == 'connection':
        return 'connected' if status == 'connected' else 'open'
    return {'tested_scoped': 'tested', 'implemented': 'implemented',
            'recorded_only': 'recorded'}.get(status, 'open')


def counts(leaves, axis):
    found = Counter(category(axis, n[axis]) for n in leaves if n['origin'] != 'out_of_scope')
    return {key: found[key] for key, _ in CATEGORIES[axis]}


def count_text(leaves, axis):
    values = counts(leaves, axis)
    return ' · '.join(f'{label} {values[key]}' for key, label in CATEGORIES[axis])


def error_scenario(budget, uses):
    if not re.fullmatch(r'[0-9]+', str(uses)):
        raise ValueError('The number of uses must be a positive integer')
    uses = int(uses)
    if uses < 1:
        raise ValueError('The number of uses must be a positive integer')
    p = budget['parameters']
    joint = p['running_sources'] * p['coefficient_lanes'] * (p['matrices'] + 1)
    joint += 2 * p['fresh_sources'] + p['running_sources']
    numerator = p['rounds'] * p['message_width'] + joint - 1 + p['rounds']
    denominator = int(p['q']) ** 2
    per_test = Fraction(numerator, denominator)
    accumulated = min(Fraction(1), uses * per_test)
    return {'uses': str(uses), 'joint_coefficient_count': joint, 'numerator': str(numerator),
            'denominator': str(denominator), 'per_test': float(per_test),
            'per_test_bits': -math.log2(float(per_test)), 'bound': float(accumulated),
            'bound_bits': -math.log2(float(accumulated))}


def validate_data(data):
    nodes = data['nodes']
    by_id = {n['id']: n for n in nodes}
    if len(by_id) != len(nodes):
        raise ValueError('Duplicate requirement ID')
    for node in nodes:
        if node['parent'] is not None and node['parent'] not in by_id:
            raise ValueError('Unknown parent: ' + node['parent'])
        if node['kind'] == 'leaf':
            for axis, _ in AXES:
                if node.get(axis) not in STATUSES:
                    raise ValueError(f'Unknown {axis} status: {node["id"]}')
        for dependency in node.get('depends_on', []):
            if dependency not in by_id:
                raise ValueError('Unknown dependency: ' + dependency)
        visited, current = set(), node['id']
        while current is not None:
            if current in visited:
                raise ValueError('Cyclic hierarchy: ' + current)
            visited.add(current)
            current = by_id[current]['parent']
        if 'root' not in visited:
            raise ValueError('Requirement outside root: ' + node['id'])
        if data.get('schema', 1) >= 2 and (not node.get('scope') or not set(node['scope']) <= SCOPES):
            raise ValueError('Invalid scope: ' + node['id'])
    assumptions = {a['id']: a for a in data.get('assumptions', [])}
    if len(assumptions) != len(data.get('assumptions', [])):
        raise ValueError('Duplicate assumption ID')
    covered = set()
    for entry in assumptions.values():
        for dependency in entry.get('depends_on', []):
            if dependency not in assumptions:
                raise ValueError('Unknown assumption dependency: ' + dependency)
        for record in entry['records'] + entry.get('inherited_by', []):
            if record not in by_id:
                raise ValueError('Unknown assumption record: ' + record)
        covered.update(entry['records'])
        if not entry.get('approval', {}).get('state'):
            raise ValueError('Missing assumption approval state: ' + entry['id'])
    for node in nodes:
        if data.get('schema', 1) >= 2 and node.get('proof') == 'assumption' and node['id'] not in covered:
            raise ValueError('Assumption missing from ledger: ' + node['id'])
        for premise in node.get('assumption_ids', []):
            if premise not in assumptions:
                raise ValueError('Unknown assumption: ' + premise)
    for panel in data.get('readiness', []):
        for item in panel['items']:
            if item['status'] not in {'passed', 'assumed', 'open'}:
                raise ValueError('Invalid readiness status: ' + item['id'])
            for record in item['records']:
                if record not in by_id:
                    raise ValueError('Unknown readiness record: ' + record)
    evidence_ids = {e['id'] for e in data.get('provenance', {}).get('evidence', [])}
    for item in nodes + [i for panel in data.get('readiness', []) for i in panel['items']]:
        if not set(item.get('evidence_ids', [])) <= evidence_ids:
            raise ValueError('Unknown evidence record: ' + item['id'])
    if data.get('schema', 1) >= 2:
        budget = data['error_budget']
        if budget['code_commit'] != data['provenance']['code_commit']:
            raise ValueError('Error budget belongs to a different code revision')
        error_scenario(budget, budget['example_uses'])
    return by_id
