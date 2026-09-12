import copy
import json
import posixpath
import re
import unittest
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from markdown_export import export_markdown
from proof_map import resolve_map, export_map


ROOT = Path(__file__).resolve().parents[1]


class MarkdownExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((ROOT / 'requirements.json').read_text())
        cls.publication = json.loads((ROOT / 'dist/publication.json').read_text())
        cls.references = json.loads((ROOT / 'dist/reference-report.json').read_text())
        cls.files = export_markdown(cls.data, (ROOT / 'reading-guide.md').read_text(), cls.publication, cls.references)
        cls.files['proof-map.md'] = export_map(resolve_map(json.loads((ROOT / 'proof-map.json').read_text()), cls.data), cls.data)
        cls.records = {}
        for content in cls.files.values():
            for node_id, record in re.findall(r'^## <a id="req-([^"]+)"></a>(.*?)(?=^## |\Z)',
                                              content, re.M | re.S):
                if node_id in cls.records:
                    raise AssertionError('Duplicate record: ' + node_id)
                cls.records[node_id] = record

    def test_all_records_and_statuses_are_preserved(self):
        self.assertEqual(set(self.records), {node['id'] for node in self.data['nodes']})
        for node in self.data['nodes']:
            record = self.records[node['id']]
            self.assertIn(f'Kind: `{node["kind"]}`; origin: `{node["origin"]}`', record)
            if node['kind'] == 'leaf':
                self.assertIn(f'Proof: `{node["proof"]}`; Link: `{node["connection"]}`; Rust: `{node["rust"]}`', record)
            for source in node.get('paper', []) + node.get('code', []):
                self.assertIn(source['path'] + ':' + str(source['line']), record)
                if source.get('symbol'):
                    self.assertIn('`' + source['symbol'] + '`', record)
            anchor = 'group-all' if node['id'] == 'root' else 'req-' + node['id']
            self.assertIn('nightstream-requirements.nicarq.chatgpt.site/#' + anchor + ')', record)

    def test_edges_are_complete_in_both_directions(self):
        reverse, children = {}, {}
        for node in self.data['nodes']:
            children.setdefault(node['parent'], []).append(node['id'])
            for dependency in node.get('depends_on', []):
                reverse.setdefault(dependency, []).append(node['id'])
        for node in self.data['nodes']:
            record = self.records[node['id']]
            fields = [('Depends on (recorded)', node.get('depends_on', [])),
                      ('Used by (recorded)', reverse.get(node['id'], []))]
            if node['kind'] == 'group':
                fields.append(('Contains (direct children)', children.get(node['id'], [])))
            for label, expected in fields:
                line = next(line for line in record.splitlines() if line.startswith('- ' + label + ':'))
                actual = re.findall(r'#req-([\w.]+)\)', line)
                self.assertEqual(actual, expected, (node['id'], label))

    def test_all_markdown_links_and_anchors_resolve(self):
        for name, content in self.files.items():
            self.assertLessEqual(len(content.splitlines()), 1500)
            self.assertNotIn('{{', content)
            for href in re.findall(r'\]\(([^)]+)\)', content):
                if href.startswith('https://'):
                    continue
                target, _, anchor = href.partition('#')
                if target:
                    target = str(PurePosixPath(name).parent / target)
                    target = posixpath.normpath(target)
                else:
                    target = name
                if target.endswith(('.json', '.zip')):
                    continue
                self.assertIn(target, self.files, (name, href))
                if anchor:
                    self.assertIn(f'id="{anchor}"', self.files[target], (name, href))

    def test_missing_dependencies_are_not_called_independent(self):
        self.assertIn('- Depends on (recorded): None recorded.', self.records['H.terminal.fresh_opening'])
        self.assertIn('does not mean that the result needs no other facts', self.files['requirements.md'])
        self.assertIn('Each record has a structured `scope`', self.files['requirements.md'])

    def test_records_link_to_their_graph_and_selected_requirement(self):
        for node in self.data['nodes']:
            node_id = node['id']
            if node_id == 'root':
                anchor = 'tech-tree'
            elif node['kind'] == 'group':
                anchor = 'proof-' + node_id
            else:
                owner = node
                by_id = {item['id']: item for item in self.data['nodes']}
                while owner['parent'] != 'root':
                    owner = by_id[owner['parent']]
                anchor = 'proof-' + owner['id'] + ':' + node_id
            self.assertIn('nightstream-requirements.nicarq.chatgpt.site/#' + anchor + ')', self.records[node_id])
        self.assertIn('Proof graph: 39 requirements, 47 internal connections, 19 outside inputs, and 12 outside consumers.', self.files['markdown/C.md'])
        self.assertIn('(graphs/C.md)', self.files['markdown/C.md'])
        self.assertIn('(markdown/graphs/C.md)', self.files['requirements.md'])

    def test_counts_follow_axis_semantics(self):
        sample = {'scope': 'test', 'commit': 'test', 'nodes': [
            {'id': 'root', 'parent': None, 'kind': 'group', 'origin': 'implementation', 'label': 'Root', 'requirement': 'Root'},
            {'id': 'F', 'parent': 'root', 'kind': 'group', 'origin': 'paper', 'label': 'Group', 'requirement': 'Group'},
        ]}
        for node_id, proof, connection, rust, origin in [
            ('F.a', 'proved', 'connected', 'implemented', 'paper'),
            ('F.b', 'definition', 'open', 'recorded_only', 'paper'),
            ('F.c', 'assumption', 'connected', 'not_required', 'assumption'),
            ('F.d', 'not_required', 'not_required', 'tested_scoped', 'paper'),
            ('F.e', 'proved', 'connected', 'implemented', 'out_of_scope'),
        ]:
            sample['nodes'].append({'id': node_id, 'parent': 'F', 'kind': 'leaf', 'origin': origin,
                                    'label': node_id, 'requirement': node_id, 'proof': proof,
                                    'connection': connection, 'rust': rust})
        files = export_markdown(sample, '')
        self.assertIn('Proof: Proved 1 · Assumed 1 · Open 1 · N/A 1', files['markdown/F.md'])
        self.assertIn('Link: Connected 2 · Open 1 · N/A 1', files['markdown/F.md'])
        self.assertIn('Rust: Scoped tests 1 · Code only 1 · Recorded 1 · Open 0 · N/A 1', files['markdown/F.md'])

    def test_invalid_relationships_fail_instead_of_silently_dropping(self):
        data = copy.deepcopy(self.data)
        data['nodes'][1]['depends_on'] = ['missing']
        with self.assertRaisesRegex(ValueError, 'Unknown dependency'):
            export_markdown(data, '')
        data['nodes'][1]['depends_on'] = []
        data['nodes'][1]['parent'] = data['nodes'][1]['id']
        with self.assertRaisesRegex(ValueError, 'Cyclic hierarchy'):
            export_markdown(data, '')

    def test_deployed_files_and_download_share_the_html_snapshot(self):
        page = (ROOT / 'dist/index.html').read_text()
        embedded = re.search(r'<script type="application/json" id="requirements-data">(.*?)</script>', page, re.S)[1]
        self.assertEqual(json.loads(embedded), self.data)
        self.assertEqual(json.loads((ROOT / 'dist/requirements.json').read_text()), self.data)
        with ZipFile(ROOT / 'dist/requirements-markdown.zip') as archive:
            self.assertEqual(set(archive.namelist()), set(self.files) | {'requirements.json', 'publication.json', 'reference-report.json'})
            self.assertEqual(json.loads(archive.read('requirements.json')), self.data)
            for name, expected in [('publication', self.publication), ('reference-report', self.references)]:
                self.assertEqual(json.loads(archive.read(name + '.json')), expected)
                element = 'publication-data' if name == 'publication' else 'reference-data'
                self.assertEqual(json.loads(re.search(r'id="' + element + r'">(.*?)</script>', page, re.S)[1]), expected)
            for name, expected in self.files.items():
                self.assertEqual((ROOT / 'dist' / name).read_text(), expected)
                self.assertEqual(archive.read(name).decode(), expected)
        self.assertIn('href="requirements.md"', page)
        self.assertIn('href="requirements-markdown.zip"', page)
        headers = (ROOT / 'dist/_headers').read_text()
        self.assertIn('/requirements.md\n  Content-Type: text/plain; charset=utf-8', headers)
        self.assertIn('/markdown/*\n  Content-Type: text/plain; charset=utf-8', headers)


if __name__ == '__main__':
    unittest.main()
