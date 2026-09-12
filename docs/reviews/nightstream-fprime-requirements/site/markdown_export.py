"""Render recorded hierarchy, dependency edges and evidence without changing status."""
import html
import posixpath
import re
from collections import defaultdict
from pathlib import PurePosixPath
from assurance_export import export_assurance
from reference_check import code_commit
from site_model import AXES, count_text, validate_data


SITE_URL = 'https://nightstream-requirements.nicarq.chatgpt.site/'


def prose(value):
    return re.sub(r'([\\`*_\[\]<>|])', r'\\\1', str(value)).replace('\n', ' ')


def export_markdown(data, guide, publication=None, references_checked=None):
    nodes = data['nodes']
    by_id = validate_data(data)
    if len(by_id) != len(nodes):
        raise ValueError('Duplicate requirement IDs')
    children, used_by = defaultdict(list), defaultdict(list)
    for node in nodes:
        if node['parent'] is not None:
            if node['parent'] not in by_id:
                raise ValueError('Unknown parent: ' + node['parent'])
            children[node['parent']].append(node['id'])
        for dependency in node.get('depends_on', []):
            if dependency not in by_id:
                raise ValueError('Unknown dependency: ' + dependency)
            used_by[dependency].append(node['id'])

    paths = {}
    for node in nodes:
        path, current = [], node['id']
        while current is not None:
            if current in path:
                raise ValueError('Cyclic hierarchy: ' + current)
            path.append(current)
            current = by_id[current]['parent']
        paths[node['id']] = list(reversed(path))
        if paths[node['id']][0] != 'root':
            raise ValueError('Requirement outside root: ' + node['id'])

    def filename(node_id):
        path = paths[node_id]
        return 'requirements.md' if len(path) == 1 else 'markdown/' + path[1] + '.md'

    def link(node_id, source):
        target = filename(node_id)
        if target == source:
            target = ''
        else:
            target = posixpath.relpath(target, str(PurePosixPath(source).parent))
        return f'[{node_id} — {prose(by_id[node_id]["label"])}]({target}#req-{node_id})'

    def links(ids, source):
        return '; '.join(link(node_id, source) for node_id in ids) or 'None recorded.'

    def leaves(node_id):
        return [node for node in nodes if node['kind'] == 'leaf' and node_id in paths[node['id']]]

    def counts(node_id):
        return [count_text(leaves(node_id), key) for key, _ in AXES]

    def references(items):
        refs = []
        for item in items:
            location = item['path'] + (':' + str(item['line']) if 'line' in item else '')
            name = prose(item['section']) if item.get('section') else (
                '`' + item['symbol'] + '`' if item.get('symbol') else '')
            refs.append(f'`{location}`' + (f' — {name}' if name else ''))
        return '; '.join(refs) or 'None recorded.'

    def graph_data(group_id):
        members = [node['id'] for node in leaves(group_id)]
        own = set(members)
        edges = [(dependency, node['id']) for node in nodes for dependency in node.get('depends_on', [])
                 if dependency in own or node['id'] in own]
        internal = [(source, target) for source, target in edges if source in own and target in own]
        inputs = {source for source, _ in edges if source not in own}
        outputs = {target for _, target in edges if target not in own}
        ids = list(dict.fromkeys(members + [node_id for edge in edges for node_id in edge]))
        return {'members': members, 'own': own, 'edges': edges, 'internal': internal,
                'inputs': inputs, 'outputs': outputs, 'ids': ids}

    def graph_markdown(group, graph):
        group_id = group['id']
        source = 'markdown/graphs/' + group_id + '.md'
        aliases = {node_id: f'n{index}' for index, node_id in enumerate(graph['ids'])}

        def node_status(node_id):
            node = by_id[node_id]
            return [node[key] for key, _ in AXES] if node['kind'] == 'leaf' else counts(node_id)

        def diagram(ids, edges):
            lines = ['```mermaid', 'flowchart BT']
            for node_id in ids:
                proof, connection, _ = node_status(node_id)
                label = html.escape(by_id[node_id]['label'], quote=True).replace('\n', ' ')
                lines.append(f'  {aliases[node_id]}["{label}<br/>Proof: {proof} · Link: {connection}"]')
            lines.extend(f'  {aliases[dependency]} --> {aliases[consumer]}' for dependency, consumer in edges)
            outside = [aliases[node_id] for node_id in ids if node_id not in graph['own']]
            if outside:
                lines.extend(['  classDef outside stroke-dasharray: 5 5', '  class ' + ','.join(outside) + ' outside'])
            return lines + ['```', '']

        lines = [f'# {prose(group["label"])} — proof connections', '',
                 f'[Interactive proof graph]({SITE_URL}#proof-{group_id}) · '
                 f'[Requirements and evidence](../{group_id}.md#req-{group_id}) · '
                 '[Index and status definitions](../../requirements.md)', '',
                 f'Protocol code commit: `{code_commit(data)}`. Recorded update: {prose(data.get("source_note", "None recorded."))}', '',
                 'The source snapshot is the same as the HTML and requirement records.', '',
                 f'- Group requirements: {len(graph["members"])}.',
                 f'- Internal connections: {len(graph["internal"])}.',
                 f'- Outside inputs: {len(graph["inputs"])} distinct requirements.',
                 f'- Outside consumers: {len(graph["outputs"])} distinct requirements.',
                 f'- With outside requirements: {len(graph["ids"])} boxes and {len(graph["edges"])} connections.', '',
                 'Each box is a requirement. Proof and Link are separate statuses. An arrow points from a prerequisite to a consumer, so consumers appear above their prerequisites. These are recorded uses, not protocol execution order or a certificate of proof closure.', '',
                 'The first diagram matches the default website view. The second includes outside requirements. The tables remain readable when a Markdown viewer does not render Mermaid. Follow the requirement links for exact scopes and theorem evidence.', '',
                 '## Internal proof graph', '']
        lines.extend(diagram(graph['members'], graph['internal']))
        if graph['inputs'] or graph['outputs']:
            lines.extend(['## Graph with outside requirements', '',
                          'This matches “Include outside requirements” on the website. Dashed boxes belong to other groups; each still identifies a specific requirement.', ''])
            lines.extend(diagram(graph['ids'], graph['edges']))
        lines.extend(['## Requirements and status', '',
                      '| Diagram box | Requirement | Proof | Link | Rust | Role in this graph |',
                      '| --- | --- | --- | --- | --- | --- |'])
        for node_id in graph['ids']:
            if node_id in graph['own']:
                role = 'In this group'
            else:
                role = 'Outside ' + ('input and consumer' if node_id in graph['inputs'] & graph['outputs']
                                     else 'input' if node_id in graph['inputs'] else 'consumer')
            statuses = [f'`{status}`' for status in node_status(node_id)]
            lines.append('| ' + ' | '.join([f'`{aliases[node_id]}`', link(node_id, source), *statuses, role]) + ' |')
        lines.extend(['', '## Recorded connections', '',
                      'Each row means **the consumer uses the prerequisite**. Parent/child grouping does not add an edge.', '',
                      '| Prerequisite | Consumer | Connection | Scope note |', '| --- | --- | --- | --- |'])
        for dependency, consumer in graph['edges']:
            role = 'Outside input' if dependency not in graph['own'] else (
                'Outside consumer' if consumer not in graph['own'] else 'Internal')
            note = by_id[consumer].get('dependency_notes', {}).get(dependency, '')
            lines.append('| ' + ' | '.join([link(dependency, source), link(consumer, source), role, prose(note)]) + ' |')
        if not graph['edges']:
            lines.extend(['', 'No dependency edges are recorded for this group. This does not establish independence.'])
        touched = {node_id for edge in graph['edges'] for node_id in edge}
        unconnected = [node_id for node_id in graph['members'] if node_id not in touched]
        lines.extend(['', '## Requirements with no recorded connections', '',
                      'These items stay in the graph; missing records do not prove independence.', '',
                      links(unconnected, source) if unconnected else 'None in this snapshot.', ''])
        return '\n'.join(lines)

    def record(node, source):
        node_id = node['id']
        path = ' → '.join(f'`{item}`' for item in paths[node_id])
        parent = link(node['parent'], source) if node['parent'] else 'None (root).'
        lines = [f'## <a id="req-{node_id}"></a>{node_id} — {prose(node["label"])}', '',
                 f'- Hierarchy: {path}. Parent: {parent}. Kind: `{node["kind"]}`; origin: `{node["origin"]}`.',
                 '- Required result: ' + prose(node['requirement'])]
        if node['kind'] == 'leaf':
            status = '; '.join(f'{label}: `{node[key]}`' for key, label in AXES)
        else:
            status = '; '.join(f'{axis[1]}: {count}' for axis, count in zip(AXES, counts(node_id)))
            status += ' (descendant leaves; no separate group completion claim)'
        lines.append('- Status: ' + status + '. Scope: ' + ', '.join(node.get('scope', [])) + '.')
        if node['kind'] == 'group':
            lines.append('- Contains (direct children): ' + links(children[node_id], source))
        lines.extend([
            '- Depends on (recorded): ' + links(node.get('depends_on', []), source),
            '- Used by (recorded): ' + links(used_by[node_id], source),
            '- Remaining: ' + prose(node.get('remaining') or ('See descendant records.' if node['kind'] == 'group' else 'None recorded.')),
            '- Paper evidence: ' + references(node.get('paper', [])),
            '- Code evidence: ' + references(node.get('code', [])),
            f'- HTML: [Open this requirement]({SITE_URL}#' + ('group-all' if node_id == 'root' else 'req-' + node_id) + ')', '',
        ])
        if node.get('assumption_ids'):
            prefix = '../' if source.startswith('markdown/') else ''
            lines.insert(-1, '- Assumptions: ' + '; '.join(f'[{p}]({prefix}assumptions.md#assumption-{p})' for p in node['assumption_ids']))
        for dependency, note in node.get('dependency_notes', {}).items():
            lines.insert(-1, '- Dependency scope ' + link(dependency, source) + ': ' + prose(note))
        graph_anchor = 'tech-tree' if node_id == 'root' else ('proof-' + node_id if node['kind'] == 'group'
                                                              else 'proof-' + paths[node_id][1] + ':' + node_id)
        lines.insert(-1, f'- Proof graph: [Open the graph' + (' with this requirement selected' if node['kind'] == 'leaf' else '')
                     + f']({SITE_URL}#{graph_anchor})')
        return lines

    groups = [by_id[node_id] for node_id in children['root']]
    graphs = {group['id']: graph_data(group['id']) for group in groups}
    leaf_nodes = [node for node in nodes if node['kind'] == 'leaf']
    excluded = sum(node['origin'] == 'out_of_scope' for node in leaf_nodes)
    index = ['# Nightstream requirements — Markdown index', '',
             'This is the complete, static reading version of the HTML map. JavaScript is not required.', '',
             f'[HTML map]({SITE_URL}) · [Full Stage 1 tree]({SITE_URL}#tech-tree) · [Download all Markdown and source JSON](requirements-markdown.zip) · [Source JSON](requirements.json)', '',
             f'- Scope: {prose(data["scope"])}',
             f'- Protocol code commit: `{code_commit(data)}`.',
             '- Recorded update: ' + prose(data.get('source_note', 'None recorded.')),
             f'- Coverage: {len(nodes)} records; {len(nodes) - len(leaf_nodes)} group records (including root); {len(leaf_nodes)} leaves ({len(leaf_nodes) - excluded} in scope, {excluded} excluded).',
             f'- Recorded dependency edges: {sum(len(node.get("depends_on", [])) for node in nodes)}. Parent/child edges are separate.', '',
             'HTML, Markdown and JSON use the same snapshot. The code commit, map commit and evidence revisions are separate. Read [source and evidence](evidence.md) and [publication metadata](publication.json). This export does not rerun protocol proofs or tests.', '',
             '[Assumption ledger](assumptions.md) · [Readiness](readiness.md) · [Error budget](error-budget.md)', '',
             '## Complete group files', '',
             '| Group | Leaf entries | Proof | Link | Rust | Leaves with no recorded dependencies | Proof graph |',
             '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for group in groups:
        group_leaves = leaves(group['id'])
        missing = sum(not node.get('depends_on') for node in group_leaves)
        index.append('| ' + ' | '.join([link(group['id'], 'requirements.md'), str(len(group_leaves)),
                                       *counts(group['id']), str(missing),
                                       f'[Diagram and connections](markdown/graphs/{group["id"]}.md)']) + ' |')
    index.extend(['', '“No recorded dependencies” is a documentation status. It does not mean that the result needs no other facts.', ''])
    index.extend(re.sub(r'\{\{([\w.]+)\}\}', lambda match: link(match[1], 'requirements.md'), guide).splitlines())
    index.extend(['', 'Complete metadata and retained review qualifications are in the [source JSON](requirements.json).', ''])
    index.extend(record(by_id['root'], 'requirements.md'))
    files = {'requirements.md': '\n'.join(index)}
    for group in groups:
        source = filename(group['id'])
        lines = [f'# {group["id"]} — {prose(group["label"])}', '',
                 '[Index and status definitions](../requirements.md) · [Complete source JSON](../requirements.json) · '
                 f'[Proof diagrams and connections](graphs/{group["id"]}.md)', '',
                 f'Protocol code commit: `{code_commit(data)}`. Recorded update: {prose(data.get("source_note", "None recorded."))}', '',
                 'All subgroups and leaves are expanded below. Dependencies are recorded edges, not a certificate of complete proof closure. Source paths are relative to the repository.', '',
                 f'Proof graph: {len(graphs[group["id"]]["members"])} requirements, '
                 f'{len(graphs[group["id"]]["internal"])} internal connections, '
                 f'{len(graphs[group["id"]]["inputs"])} outside inputs, and '
                 f'{len(graphs[group["id"]]["outputs"])} outside consumers.', '']
        for node in nodes:
            if group['id'] in paths[node['id']]:
                lines.extend(record(node, source))
        files[source] = '\n'.join(lines)
        files['markdown/graphs/' + group['id'] + '.md'] = graph_markdown(group, graphs[group['id']])
    files.update(export_assurance(data, publication or {}, references_checked or {}))
    for name, content in files.items():
        if len(content.splitlines()) > 1500:
            raise ValueError(f'{name} exceeds the repository file size policy')
    return files
