#!/usr/bin/env python3
"""Fail when SECURITY_MODEL.md or SPEC.md states an assumption that is not on the allowed list.

The allowed list is the owner's list of 2026-11-13: assumptions on primitives only. A random
oracle on a derived object (the hash, a transcript, a challenge) is not an assumption. It is the
ideal permutation `P2-PERM` plus the named sponge theorem `SPONGE`, with its conditions as Lean
hypotheses, or a Lean theorem.

Rules:
1. One list between the assumption markers. Each entry is a bullet that starts with an allowed id.
2. Each `[ID]` citation names an id of that list.
3. Outside the list, and in SPEC.md, no sentence states an assumption in the forms of `FORBIDDEN`.
4. Each numbered item of `## What you trust` cites an id in its bold title, or the title starts
   with `Gap:` or `Scope:`.
"""
import re
import sys
from pathlib import Path

ALLOWED = {
    'P2-PERM': 'the Poseidon2 permutation is an ideal permutation',
    'SPONGE': 'sponge indifferentiability from an ideal permutation, with its capacity term',
    'KECCAK': 'SHAKE128, SHAKE256 and Keccak-f, for the Ajtai setup and Akita',
    'MSIS': 'Module-SIS under the pinned estimator model',
    'AKITA': 'Akita knowledge soundness with the error epsilon-Akita',
    'HN-A1': 'HyperNova errata Assumption 1',
}
BEGIN = '<!-- assumptions:begin -->'
END = '<!-- assumptions:end -->'
ENTRY = re.compile(r'^- `([^`]+)`')
CITATION = re.compile(r'\[([A-Z][A-Z0-9]+(?:-[A-Z0-9]+)*)\]')
FORBIDDEN = [
    (r'\b(?:as|is|are|be|being)\s+(?:an?\s+|two\s+)?(?:uniform\s+|ideal\s+|independent\s+)?'
     r'random\s+oracles?\b', 'a random oracle on a derived object'),
    (r'\bindependent\s+oracles?\b', 'independent oracles'),
    (r'\brandom-oracle\s+(?:premise|assumption)s?\b', 'a random-oracle premise'),
    (r'\bcollision[\s-]+resist', 'collision resistance of a derived hash'),
    (r'\bhuman-ignorance\b', 'a human-ignorance premise'),
    (r'\bhardness\s+(?:premise|assumption)s?\b', 'a hardness premise outside the list'),
    (r'\b(?:is|are)\s+assumed\b', 'an unnamed assumption'),
    (r'\b(?:we|it|lean|model|result|theorem|premise)\s+assumes?\b', 'an unnamed assumption'),
]
TRUST_ITEM = re.compile(r'^\d+\.\s+\*\*(.+?)\*\*')


def line_of(text, offset):
    return text.count('\n', 0, offset) + 1


def assumption_list(text):
    """The list's span, its ids and its errors (rule 1)."""
    if text.count(BEGIN) != 1 or text.count(END) != 1 or text.index(BEGIN) > text.index(END):
        return None, [], [f'the page needs one list between {BEGIN} and {END}']
    start, end = text.index(BEGIN), text.index(END) + len(END)
    listed, errors = [], []
    for number, line in enumerate(text[start:end].splitlines(), line_of(text, start)):
        if not line.startswith('- '):
            continue
        entry = ENTRY.match(line)
        if not entry or entry.group(1) not in ALLOWED:
            errors.append(f'line {number}: assumption {line[:60]!r} is not on the allowed list '
                          f'({", ".join(sorted(ALLOWED))})')
        elif entry.group(1) in listed:
            errors.append(f'line {number}: assumption `{entry.group(1)}` is listed twice')
        else:
            listed.append(entry.group(1))
    return (start, end), listed, errors


def check_citing(text, listed):
    """Rules 2 and 3 for a page that cites the list. Blank out the list itself first."""
    errors = []
    for match in CITATION.finditer(text):
        if match.group(1) not in listed:
            errors.append(f'line {line_of(text, match.start())}: [{match.group(1)}] '
                          'is not an assumption of the list')
    for pattern, what in FORBIDDEN:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            errors.append(f'line {line_of(text, match.start())}: {match.group(0)!r} states {what}; '
                          'cite an assumption of the list or a Lean theorem')
    return errors


def check(text):
    span, listed, errors = assumption_list(text)
    if span is None:
        return errors
    start, end = span
    outside = text[:start] + ' ' * (end - start) + text[end:]
    errors += check_citing(outside, listed)
    trust = re.search(r'^## What you trust\n(.*?)(?=^## )', outside, re.MULTILINE | re.DOTALL)
    if not trust:
        errors.append('the page needs a `## What you trust` section')
    else:
        offset = trust.start(1)
        for line in trust.group(1).splitlines(keepends=True):
            item = TRUST_ITEM.match(line)
            if item:
                title = item.group(1)
                cited = [c for c in CITATION.findall(title) if c in listed]
                if not cited and not title.startswith(('Gap:', 'Scope:')):
                    errors.append(f'line {line_of(text, offset)}: trust item {title!r} cites no '
                                  'assumption and is not a `Gap:` or a `Scope:`')
            offset += len(line)
    return errors


def main():
    package = Path(__file__).resolve().parents[1]
    model, spec = package / 'SECURITY_MODEL.md', package.parents[1] / 'SPEC.md'
    text = model.read_text()
    errors = [(model, error) for error in check(text)]
    errors += [(spec, error) for error in check_citing(spec.read_text(), assumption_list(text)[1])]
    for path, error in errors:
        print(f'[assumptions] {path.name}: {error}', file=sys.stderr)
    if errors:
        return 1
    print('[assumptions] SECURITY_MODEL.md and SPEC.md state only assumptions of the allowed list')
    return 0


if __name__ == '__main__':
    sys.exit(main())
