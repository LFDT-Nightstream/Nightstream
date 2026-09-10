"""Build the HTML page and static reading files from one requirement snapshot."""
import json
import argparse
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from markdown_export import export_markdown
from reference_check import check_references, git, publication_record
from site_model import validate_data

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--paper-root', type=Path, help='Checkout containing the selected local paper Markdown')
parser.add_argument('--publish', action='store_true', help='Require every site input to match HEAD before preparing publication')
args = parser.parse_args()
data = json.loads((root / 'requirements.json').read_text())
validate_data(data)
publication = publication_record(root, args.publish)
repository = Path(git(root, 'rev-parse', '--show-toplevel').decode().strip())
references = check_references(data, repository, args.paper_root or repository)


def payload(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c')


page = (root / 'page.html').read_text()
for marker, content in [
    ('/* SITE_STYLES */', (root / 'styles.css').read_text()),
    ('/* SITE_SCRIPT */', (root / 'app.js').read_text()),
    ('/* ASSURANCE_SCRIPT */', (root / 'assurance.js').read_text()),
    ('/* SITE_DATA */', payload(data)),
    ('/* PUBLICATION_DATA */', payload(publication)),
    ('/* REFERENCE_DATA */', payload(references)),
]:
    assert page.count(marker) == 1, marker
    page = page.replace(marker, content)
assert len(page.splitlines()) <= 1500, 'Repository file size policy'
markdown = export_markdown(data, (root / 'reading-guide.md').read_text(), publication, references)
(root / 'dist').mkdir(exist_ok=True)
(root / 'dist/index.html').write_text(page)
(root / 'index.html').write_text(page)
(root / 'dist/_headers').write_text((root / '_headers').read_text())
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
