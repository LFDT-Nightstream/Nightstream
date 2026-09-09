"""Build the HTML page and static reading files from one requirement snapshot."""
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from markdown_export import export_markdown

root = Path(__file__).resolve().parent
data = json.loads((root / 'requirements.json').read_text())
assert len({node['id'] for node in data['nodes']}) == len(data['nodes'])
page = (root / 'page.html').read_text()
for marker, content in [
    ('/* SITE_STYLES */', (root / 'styles.css').read_text()),
    ('/* SITE_SCRIPT */', (root / 'app.js').read_text()),
    ('/* SITE_DATA */', json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c')),
]:
    assert page.count(marker) == 1, marker
    page = page.replace(marker, content)
assert len(page.splitlines()) <= 1500, 'Repository file size policy'
markdown = export_markdown(data, (root / 'reading-guide.md').read_text())
(root / 'dist').mkdir(exist_ok=True)
(root / 'dist/index.html').write_text(page)
(root / 'index.html').write_text(page)
(root / 'dist/_headers').write_text((root / '_headers').read_text())
for name, content in markdown.items():
    output = root / 'dist' / name
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content)
snapshot = json.dumps(data, ensure_ascii=False, separators=(',', ':')) + '\n'
(root / 'dist/requirements.json').write_text(snapshot)
with ZipFile(root / 'dist/requirements-markdown.zip', 'w', ZIP_DEFLATED) as archive:
    for name, content in markdown.items():
        archive.writestr(name, content)
    archive.writestr('requirements.json', snapshot)
print(f'Built {len(data["nodes"])} requirement nodes: HTML, {len(markdown)} Markdown files, JSON and ZIP.')
