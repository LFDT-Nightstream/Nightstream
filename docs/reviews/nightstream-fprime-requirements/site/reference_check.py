"""Check cited locations against the named code commit and local paper corpus."""

from datetime import datetime, timezone
import hashlib
from pathlib import Path, PurePosixPath
import re
import subprocess


def git(root, *args):
    return subprocess.run(['git', '-C', str(root), *args], capture_output=True, check=True).stdout


def code_commit(data):
    return data.get('provenance', {}).get('code_commit', data.get('commit', ''))


def check_references(data, repository, paper_root):
    commit = code_commit(data)
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('A complete code commit is required')
    git(repository, 'cat-file', '-e', commit + '^{commit}')
    cache, locations, issues = {}, {}, []
    checked = 0
    for node in data['nodes']:
        if node.get('origin') == 'out_of_scope':
            continue
        for kind in ('paper', 'code'):
            for ref in node.get(kind, []):
                path = ref['path']
                parsed = PurePosixPath(path)
                if parsed.is_absolute() or '..' in parsed.parts:
                    raise ValueError('Reference leaves the repository: ' + path)
                if path.startswith(('formal/nightstream-lean/', 'formal/deprecated-nightstream-lean/')):
                    raise ValueError('Frozen code cannot supply a current implementation reference: ' + path)
                if path not in cache:
                    try:
                        content = git(repository, 'show', commit + ':' + path)
                        committed = True
                    except subprocess.CalledProcessError:
                        if kind != 'paper' or paper_root is None:
                            raise ValueError('Reference is not in the code commit: ' + path) from None
                        content = (paper_root / path).read_bytes()
                        committed = False
                    cache[path] = content.decode().splitlines()
                    locations[path] = {'sha256': hashlib.sha256(content).hexdigest(),
                                       'repository_committed': committed,
                                       'lines': len(cache[path])}
                line = ref.get('line')
                if not isinstance(line, int) or not 0 < line <= len(cache[path]):
                    issues.append(f'{node["id"]}: {path}:{line} is outside the source')
                checked += 1
    for path, expected in data.get('error_budget', {}).get('source_hashes', {}).items():
        actual = hashlib.sha256(git(repository, 'show', commit + ':' + path)).hexdigest()
        if actual != expected:
            issues.append('Error-budget source changed; review the formula and parameters: ' + path)
    if issues:
        raise ValueError('\n'.join(issues))
    return {'code_commit': commit, 'checked_at': datetime.now(timezone.utc).isoformat(),
            'performed_by': 'Site reference checker', 'scope': 'File and line validity; not proof meaning or protocol conformance',
            'references_checked': checked, 'locations': locations}


def publication_record(site, require_committed=False):
    root = Path(git(site, 'rev-parse', '--show-toplevel').decode().strip())
    relative = site.relative_to(root).as_posix()
    prefix = '' if relative == '.' else relative + '/'
    commit = git(root, 'rev-parse', 'HEAD').decode().strip()
    required = ['requirements.json', 'page.html', 'styles.css', 'app.js', 'assurance.js',
                'build.py', 'markdown_export.py', 'assurance_export.py', 'site_model.py',
                'reference_check.py', 'reading-guide.md', '.openai/hosting.json', '_headers']
    dirty = []
    for name in required:
        try:
            committed = git(root, 'show', commit + ':' + prefix + name)
            if committed != (site / name).read_bytes():
                dirty.append(name)
        except subprocess.CalledProcessError:
            dirty.append(name)
    if require_committed and dirty:
        raise ValueError('Commit the exact site inputs before publication: ' + ', '.join(dirty))
    return {'map_commit': commit if not dirty else None, 'map_path': prefix + 'requirements.json',
            'source_state': 'committed' if not dirty else 'working_tree',
            'source_tree_commit': commit, 'uncommitted_inputs': dirty,
            'map_sha256': hashlib.sha256((site / 'requirements.json').read_bytes()).hexdigest(),
            'artifact_state': 'prepared from committed source' if not dirty else 'local preview',
            'prepared_at': datetime.now(timezone.utc).isoformat(),
            'site_check_authority': 'local build and reference checks; no new independent protocol gate approval'}
