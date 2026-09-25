#!/usr/bin/env python3
"""Keep the generic MatrixProgram Export files limited to codecs."""

import argparse
from pathlib import Path
import re

from rebuild_radius import lean_code


NAME = r'[A-Za-z_][A-Za-z0-9_\']*(?:\.[A-Za-z_][A-Za-z0-9_\']*)*'
MODIFIERS = r'(?:(?:private|protected|public|noncomputable|nonrec|partial|unsafe)\s+)*'
ATTRIBUTES = r'(?:@\[[^\]]*\]\s*)*'
DECLARATION = re.compile(ATTRIBUTES + MODIFIERS +
    r'(def|abbrev|structure|inductive|class|theorem|lemma|opaque|instance|axiom)\b')
CODEC = re.compile(ATTRIBUTES + MODIFIERS + r'def\s+' + NAME +
    r'\s*:\s*(?:(?:_root_\.)?NightstreamFPrime\.Export\.Codec\.)?Format\s+(.+)\s+where')
SCAFFOLD = re.compile(
    r'(?:(?:(?:public|private|meta)\s+)*import\s+' + NAME + r'(?:\s+' + NAME + r')*|'
    r'namespace\s+' + NAME + r'|end(?:\s+' + NAME + r')?|'
    r'open\s+(?:scoped\s+)?' + NAME + r'(?:\s+' + NAME + r')*)')


def one_type_argument(text):
    """Accept a named type or one parenthesized type; no outer arrow/product."""
    text = text.strip()
    if re.fullmatch(NAME, text):
        return True
    if not text.startswith('(') or not text.endswith(')'):
        return False
    depth = 0
    for index, char in enumerate(text):
        depth += (char == '(') - (char == ')')
        if depth < 0 or (depth == 0 and index != len(text) - 1):
            return False
    return depth == 0


def check_source(text, path='<source>'):
    """Check this owner's existing codec/where syntax, not arbitrary Lean terms."""
    lines = lean_code(text).splitlines()
    codec_indent = None
    count = index = 0
    while index < len(lines):
        raw = lines[index]
        line = raw.strip()
        number = index + 1
        index += 1
        if not line:
            continue
        indent = len(raw.expandtabs()) - len(raw.expandtabs().lstrip())
        declaration = DECLARATION.match(line)
        if declaration:
            if declaration[1] != 'def':
                raise ValueError(f'{path}:{number}: physical declaration in MatrixProgram codec owner')
            header = line
            while not re.search(r'\bwhere\b|:=', header) and index < len(lines):
                header += ' ' + lines[index].strip()
                index += 1
            codec = CODEC.fullmatch(header)
            if codec is None or not one_type_argument(codec[1]):
                raise ValueError(f'{path}:{number}: expected a Format-typed codec definition')
            codec_indent = indent
            count += 1
        elif SCAFFOLD.fullmatch(line):
            codec_indent = None
        elif codec_indent is None or indent <= codec_indent:
            raise ValueError(f'{path}:{number}: unsupported command in MatrixProgram codec owner')
        # Indented encode/decode/decode_encode fields and their terms/proofs remain opaque.
        # Declaration keywords are checked first, even if a new declaration is indented.
    return count


def check_project(project):
    export = project / 'NightstreamFPrime/Export'
    base = export / 'MatrixProgram.lean'
    files = ([base] if base.is_file() else [])
    files += sorted((export / 'MatrixProgram').rglob('*.lean'))
    count = sum(check_source(path.read_text(), path) for path in files)
    return len(files), count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    try:
        files, definitions = check_project(args.project)
    except (OSError, ValueError) as error:
        parser.exit(1, f'[matrix-codecs] {error}\n')
    print(f'[matrix-codecs] {files} files; {definitions} Format definitions; no physical declarations')


if __name__ == '__main__':
    main()
