"""Resolve the curated proof route against the current requirement evidence."""
import copy


def resolve_map(source, data):
    result = copy.deepcopy(source)
    records = {node['id']: node for node in data['nodes']}
    assumptions = {item['id'] for item in data['assumptions']}
    ids = {node['id'] for node in result['nodes']}
    if len(ids) != len(result['nodes']):
        raise ValueError('Duplicate proof-map node')
    placed = [id for layer in result['layers'] for row in layer['rows'] for id in row]
    if len(placed) != len(ids) or set(placed) != ids:
        raise ValueError('Proof-map layers must place every node exactly once')

    def evidence(ref):
        matches = [item for item in records[ref['record']].get('code', [])
                   if item['path'].endswith('/' + ref['file'])
                   and item.get('symbol') == ref['symbol']]
        if len(matches) != 1:
            raise ValueError('Proof-map reference must resolve once: ' + str(ref))
        return matches[0]

    for node in result['nodes']:
        record = records[node['record']]
        if node['kind'] not in ['theorem', 'premise', 'assumption', 'definition', 'open']:
            raise ValueError('Unknown proof-map kind: ' + node['kind'])
        if node.get('assumption') and node['assumption'] not in assumptions:
            raise ValueError('Unknown proof-map assumption: ' + node['assumption'])
        node['code'] = [evidence(ref) for ref in node.get('refs', [])]
        if node['kind'] == 'theorem' and not node['code']:
            raise ValueError('Theorem node needs evidence: ' + node['id'])
        node['statuses'] = {key: record[key] for key in ['proof', 'connection', 'rust']}
    for edge in result['edges']:
        if edge['source'] not in ids or edge['target'] not in ids:
            raise ValueError('Unknown proof-map edge endpoint: ' + str(edge))
        if edge['kind'] not in ['uses', 'supplies', 'requires', 'open']:
            raise ValueError('Unknown proof-map edge kind')
        edge['code'] = [evidence(ref) for ref in edge.get('refs', [])]
        if edge['kind'] in ['uses', 'supplies'] and not edge['code']:
            raise ValueError('Proved connection needs evidence: ' + str(edge))
    return result


def export_map(diagram, data):
    origin = 'https://nightstream-requirements.nicarq.chatgpt.site/'
    code_origin = data['provenance']['repository'] + '/blob/' + data['provenance']['code_commit'] + '/'
    nodes = {node['id']: node for node in diagram['nodes']}
    text = ['# Nightstream proof structure', '',
            '[Interactive map](' + origin + '#proof-map) · [Requirements](requirements.md)', '',
            'A curated route through the checked snapshot. Nodes describe local theorems, statement premises, external assumptions and open obligations. This is not a complete extracted Lean dependency graph and adds no completion credit.', '',
            'Code snapshot: `' + data['provenance']['code_commit'] + '`.', '',
            'Read the main map upward, from verifier acceptance to the security goal. The overview shows the main connections. Select a node for all its direct connections, or turn on All connections. Premises and prover construction have separate columns.', '',
            'Solid arrows use a result under its stated premises. Dashed green arrows supply a premise. Dotted arrows require a premise or assumption. Dashed orange arrows are open connections. Inputs can be joint; one input alone need not imply the result.', '',
            '## Groups', '']
    for layer in diagram['layers']:
        text += ['- **' + layer['title'] + '**: ' + '; '.join(
            ', '.join(' '.join(nodes[id]['label']) for id in row) for row in layer['rows']) + '.']
    text += ['',
            '## Nodes', '']
    for node in diagram['nodes']:
        text += ['### ' + ' '.join(node['label']), '',
                 node['summary'], '',
                 '- Kind: ' + node['kind'] + '.',
                 '- Requirement: [' + node['record'] + '](' + origin + '#req-' + node['record'] + ').',
                 '- Map: [Select node](' + origin + '#proof-map:' + node['id'] + ').',
                 '- Meaning: ' + node['role']]
        if node.get('assumption'):
            text += ['- Assumption: [' + node['assumption'] + '](assumptions.md#assumption-' + node['assumption'] + ').']
        for ref in node['code']:
            text += ['- Lean: [`' + ref['symbol'] + '`](' + code_origin + ref['path'] + '#L' + str(ref['line']) + ').']
        text += ['']
    text += ['## Connections', '', '| From | To | Connection | Evidence and scope |', '| --- | --- | --- | --- |']
    for edge in diagram['edges']:
        refs = ['[`' + r['symbol'] + '`](' + code_origin + r['path'] + '#L' + str(r['line']) + ')' for r in edge['code']]
        note = ('; '.join(refs) + '. ' if refs else '') + edge['note']
        text += ['| ' + ' | '.join([' '.join(nodes[edge['source']]['label']), ' '.join(nodes[edge['target']]['label']), edge['kind'], note.replace('|', '\\|')]) + ' |']
    return '\n'.join(text) + '\n'
