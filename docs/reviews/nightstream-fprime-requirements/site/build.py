"""Build the HTML page and static reading files from one requirement snapshot."""
import json
import argparse
import hashlib
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from markdown_export import export_markdown
from reference_check import check_references, code_commit, protocol_repository, publication_record
from site_model import validate_data
from proof_map import resolve_map, export_map

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--paper-root', type=Path, help='Checkout containing the selected local paper Markdown')
parser.add_argument('--publish', action='store_true', help='Require every site input to match HEAD before preparing publication')
args = parser.parse_args()
data = json.loads((root / 'requirements.json').read_text())
validate_data(data)
diagram = resolve_map(json.loads((root / 'proof-map.json').read_text()), data)
publication = publication_record(root, args.publish)
repository = protocol_repository(root, code_commit(data))
reference_data = {**data, 'nodes': data['nodes'] + [
    {'id': 'proof-map-' + str(index), 'code': item['code']}
    for index, item in enumerate(diagram['nodes'] + diagram['edges'])]}
references = check_references(reference_data, repository, args.paper_root or repository)


def payload(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c')


page = (root / 'page.html').read_text()
for marker, content in [
    ('/* SITE_STYLES */', (root / 'styles.css').read_text()),
    ('/* SITE_SCRIPT */', '\n'.join((root / name).read_text() for name in ['proof-graph.js', 'tech-tree.js', 'app.js'])),
    ('/* ASSURANCE_SCRIPT */', (root / 'assurance.js').read_text()),
    ('/* SITE_DATA */', payload(data)),
    ('/* PUBLICATION_DATA */', payload(publication)),
    ('/* REFERENCE_DATA */', payload(references)),
    ('/* PROOF_MAP_DATA */', payload(diagram)),
]:
    assert page.count(marker) == 1, marker
    page = page.replace(marker, content)
for attribute, asset in [('src', 'proof-map.js'), ('href', 'proof-map.css')]:
    version = hashlib.sha256((root / asset).read_bytes()).hexdigest()
    page = page.replace(attribute + '="' + asset + '"', attribute + '="' + asset + '?v=' + version + '"')
assert len(page.splitlines()) <= 1500, 'Repository file size policy'
markdown = export_markdown(data, (root / 'reading-guide.md').read_text(), publication, references)
markdown['proof-map.md'] = export_map(diagram, data)
(root / 'dist').mkdir(exist_ok=True)
(root / 'dist/index.html').write_text(page)
(root / 'index.html').write_text(page)
(root / 'dist/_headers').write_text((root / '_headers').read_text())
for name in ['proof-map.js', 'proof-map.css']:
    (root / 'dist' / name).write_bytes((root / name).read_bytes())
for name, content in markdown.items():
    output = root / 'dist' / name
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content)
snapshot = json.dumps(data, ensure_ascii=False, separators=(',', ':')) + '\n'
json_files = {'requirements.json': snapshot, 'publication.json': json.dumps(publication, indent=2) + '\n',
              'reference-report.json': json.dumps(references, indent=2) + '\n'}
for name, content in json_files.items():
    (root / 'dist' / name).write_text(content)
with ZipFile(root / 'dist/requirements-markdown.zip', 'w', ZIP_DEFLATED) as archive:
    for name, content in markdown.items():
        archive.writestr(name, content)
    for name, content in json_files.items():
        archive.writestr(name, content)
print(f'Built {len(data["nodes"])} requirement nodes: HTML, {len(markdown)} Markdown files, JSON and ZIP. '
      f'Checked {references["references_checked"]} source locations. Source: {publication["source_state"]}.')
