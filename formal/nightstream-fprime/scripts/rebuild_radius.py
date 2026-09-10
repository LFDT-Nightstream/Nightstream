#!/usr/bin/env python3
"""Report source import reachability, not a prediction of Lake rebuilds."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re


def imports(text):
    """Read the import header, ignoring nested Lean comments and strings."""
    clean, index, depth, quoted = [], 0, 0, False
    while index < len(text):
        pair = text[index:index + 2]
        char = text[index]
        if depth:
            if pair == '/-':
                depth += 1
                index += 2
            elif pair == '-/':
                depth -= 1
                index += 2
            else:
                clean.append('\n' if char == '\n' else ' ')
                index += 1
        elif quoted:
            if char == '\\':
                index += 2
            else:
                quoted = char != '"'
                clean.append('\n' if char == '\n' else ' ')
                index += 1
        elif pair == '/-':
            depth = 1
            clean.append(' ')
            index += 2
        elif pair == '--':
            end = text.find('\n', index)
            index = len(text) if end < 0 else end
        elif char == '"':
            quoted = True
            clean.append(' ')
            index += 1
        else:
            clean.append(char)
            index += 1
    names = []
    for line in ''.join(clean).splitlines():
        match = re.fullmatch(r'\s*(?:(?:public|private|meta)\s+)*import\s+(.+?)\s*', line)
        if match:
            names.extend(match.group(1).split())
    return names


def graph(project):
    paths = [project / 'NightstreamFPrime.lean']
    paths += sorted((project / 'NightstreamFPrime').rglob('*.lean'))
    paths += sorted((project / 'tests').rglob('*.lean'))
    source = {'.'.join(path.relative_to(project).with_suffix('').parts): path
              for path in paths if path.is_file()}
    edges, layers = {}, defaultdict(Counter)
    for name, path in source.items():
        text = path.read_text()
        candidates = imports(text)
        internal = {item for item in candidates
                    if item == 'NightstreamFPrime' or item.startswith(('NightstreamFPrime.', 'tests.'))}
        missing = internal - source.keys()
        if missing:
            raise ValueError(f'{name}: missing local imports: {sorted(missing)}')
        edges[name] = internal
        parts = name.split('.')
        layer = parts[1] if len(parts) > 1 and parts[0] == 'NightstreamFPrime' else parts[0]
        layers[layer].update(files=1, lines=len(text.splitlines()))
    return edges, dict(layers)


def dependents(edges, changed):
    if changed not in edges:
        raise ValueError('Unknown module: ' + changed)
    reverse = defaultdict(set)
    for name, imported in edges.items():
        for dependency in imported:
            reverse[dependency].add(name)
    found, pending = set(), list(reverse[changed])
    while pending:
        name = pending.pop()
        if name not in found and name != changed:
            found.add(name)
            pending.extend(reverse[name])
    return sorted(found)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--module', action='append', required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    edges, layers = graph(args.project)
    result = {'scope': 'Local source import reachability; excludes the changed module; not measured rebuilds',
              'modules': len(edges), 'internal_import_edges': sum(map(len, edges.values())),
              'layers': layers,
              'selected': {name: {'dependents': dependents(edges, name)} for name in args.module}}
    for item in result['selected'].values():
        item['count'] = len(item['dependents'])
    output = json.dumps(result, indent=2, sort_keys=True) + '\n'
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end='')


if __name__ == '__main__':
    main()
