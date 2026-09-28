"""Bind protocol operations to the existing requirement records without changing status."""
from copy import deepcopy
from site_model import AXES, count_text
from markdown_export import replay_markdown


def resolve_flow(source, data):
    result = deepcopy(source)
    by_id = {n['id']: n for n in data['nodes']}
    owners = {}
    element_ids = set()

    def leaves(key):
        if key not in by_id:
            raise ValueError('Unknown protocol requirement: ' + key)
        if by_id[key]['kind'] == 'leaf':
            return [key]
        return [leaf for n in data['nodes'] if n['parent'] == key for leaf in leaves(n['id'])]

    for item in result['items']:
        if item['id'] in element_ids:
            raise ValueError('Duplicate protocol element: ' + item['id'])
        element_ids.add(item['id'])
        excluded = {leaf for key in item.get('except', []) for leaf in leaves(key)}
        records = [leaf for key in item.pop('records') for leaf in leaves(key) if leaf not in excluded]
        item.pop('except', None)
        item['records'] = list(dict.fromkeys(records))
        for key in item['records']:
            if key in owners:
                raise ValueError('Requirement counted twice in protocol flow: ' + key)
            owners[key] = item['id']
        for key in item.get('uses', []):
            if key not in by_id:
                raise ValueError('Unknown supporting requirement: ' + key)
    required = {n['id'] for n in data['nodes'] if n['kind'] == 'leaf' and n['origin'] != 'out_of_scope'}
    if set(owners) != required:
        raise ValueError('Protocol coverage mismatch: ' + str(sorted(required ^ set(owners))))
    displayed = list(result['support']) + result['circuit']['items']
    for section in result['sections']:
        for key in [key for row in section['rows'] for key in row] + section.get('assurance', []):
            if key not in element_ids:
                raise ValueError('Unknown protocol element in layout: ' + key)
            displayed.append(key)
    if len(displayed) != len(set(displayed)) or set(displayed) != element_ids:
        raise ValueError('Protocol layout must show each counted element once')
    assumptions = {a['id'] for a in data.get('assumptions', [])}
    for condition in result['conditions']:
        if condition['id'] not in assumptions:
            raise ValueError('Unknown protocol assumption: ' + condition['id'])
    for edge in result['edges']:
        if not {edge['from'], edge['to']} <= element_ids:
            raise ValueError('Unknown protocol edge endpoint')
        if edge['kind'] not in {'data', 'circuit', 'check', 'encode', 'base', 'feedback', 'backend'}:
            raise ValueError('Unknown protocol edge kind')
    result['owners'] = owners
    return result


def export_flow(flow, data):
    by_id = {n['id']: n for n in data['nodes']}
    items = {n['id']: n for n in flow['items']}
    lines = ['# Nightstream protocol flow', '', flow['description'], '',
             'Selected Nightstream Goldilocks profile: one fresh claim, 16 carried claims, '
             'b = 2, k_rho = 16, B = 65536.', '',
             'Boxes are operations or claim states. Handoffs connect their exact values. '
             'Phase results cover several operations. These arrows describe execution and value flow, '
             'not the complete Lean theorem dependency graph.', '',
             'Every in-scope requirement has one counted owner. Shared foundations are referenced, '
             'not counted again. Status and evidence retain the scopes in the requirements map.', '',
             '[Open the interactive flow](https://nightstream-requirements.nicarq.chatgpt.site/#protocol-flow)', '',
             flow['feedback_note'], '', flow['circuit']['note'], '', flow['circuit']['base_note'], '',
             '```mermaid', 'flowchart TB']
    for section in flow['sections']:
        lines.append('  subgraph ' + section['id'] + '["' + section['label'] + '"]')
        for row in section['rows']:
            for key in row:
                lines.append('    ' + key.replace('-', '_') + '["' + items[key]['label'] + '"]')
        lines.append('  end')
    lines.append('  subgraph recursive_circuit["HyperNova augmented circuit F′"]')
    for key in flow['circuit']['items']:
        lines.append('    ' + key.replace('-', '_') + '["' + items[key]['label'] + '"]')
    lines.append('  end')
    for edge in flow['edges']:
        arrow = ' -.-> ' if edge['kind'] in {'feedback', 'check', 'base', 'backend'} else ' --> '
        lines.append('  ' + edge['from'].replace('-', '_') + arrow +
                     ('|"' + edge['label'] + '"| ' if edge.get('label') else '') + edge['to'].replace('-', '_'))
    lines += ['```', '', '## Explicit conditions', '']
    for condition in flow['conditions']:
        lines += ['- [' + condition['label'] + '](assumptions.md#assumption-' + condition['id'] +
                  '): ' + condition['detail']]
    lines += ['', 'Folding outputs remain claims. PiDEC checks public digits and recombination; '
              'private digit validity is tested by later folds or the terminal opening checks. '
              'A separate terminal proof backend needs its own approval and validation.', '']
    lines += replay_markdown(data) + ['## Operations, connections and phase results', '']
    for item in flow['items']:
        lines += ['<a id="flow-' + item['id'] + '"></a>', '', '### ' + item['label'], '', item['detail'], '']
        if item.get('formula'):
            lines += ['`' + item['formula'] + '`', '']
        records = [by_id[key] for key in item['records']]
        lines += ['- ' + label + ': ' + count_text(records, axis) for axis, label in AXES]
        lines += ['', '| Requirement | Proof | Link | Rust |', '| --- | --- | --- | --- |']
        for record in records:
            key = record['id']
            lines.append('| [' + record['label'].replace('|', '\\|') + '](markdown/' + key.split('.')[0] +
                         '.md#req-' + key + ') | ' + ' | '.join(record[axis] for axis, _ in AXES) + ' |')
        if item.get('uses'):
            lines += ['', 'Shared support: ' + '; '.join('[' + by_id[key]['label'] + '](markdown/' +
                      key.split('.')[0] + '.md#req-' + key + ')' for key in item['uses']) + '.']
        lines.append('')
    return '\n'.join(lines)
