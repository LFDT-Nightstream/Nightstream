#!/usr/bin/env python3
"""Compare a fresh canonical Lean binding with the fixture and Rust pins."""

import argparse
import json
from pathlib import Path
import re


PINS = {1: 'POSEIDON2_HASH_CHAIN_V1_STRUCTURAL_IDENTIFIER',
        2: 'POSEIDON2_HASH_CHAIN_V1_PACKAGE_IDENTITY',
        5: 'POSEIDON2_HASH_CHAIN_V1_VERIFICATION_KEY_DIGEST'}


def rust_code(text):
    """Remove nested comments and string literals before reading declarations."""
    output, index, depth = [], 0, 0
    while index < len(text):
        pair = text[index:index + 2]
        if depth:
            if pair == '/*':
                depth += 1
                index += 2
            elif pair == '*/':
                depth -= 1
                index += 2
            else:
                index += 1
        elif pair == '//':
            end = text.find('\n', index)
            index = len(text) if end < 0 else end
        elif pair == '/*':
            depth = 1
            output.append(' ')
            index += 2
        else:
            raw = re.match(r'(?:br|r)(#*)"', text[index:])
            if raw:
                end = text.find('"' + raw[1], index + raw.end())
                if end < 0:
                    raise ValueError('Unterminated Rust raw string')
                index = end + 1 + len(raw[1])
                output.append(' ')
            elif text[index] == '"':
                index += 1
                while index < len(text) and text[index] != '"':
                    index += 2 if text[index] == '\\' else 1
                if index >= len(text):
                    raise ValueError('Unterminated Rust string')
                index += 1
                output.append(' ')
            else:
                output.append(text[index])
                index += 1
    if depth:
        raise ValueError('Unterminated Rust comment')
    return ''.join(output)


def check(actual, expected, rust):
    if not isinstance(actual, list) or len(actual) != 6 or type(actual[0]) is not int or actual[0] != 1:
        raise ValueError('Expected the canonical schema-1 binding value')
    # JSON equality preserves types: a Boolean is not a field element.
    if json.dumps(actual, separators=(',', ':')) != json.dumps(expected, separators=(',', ':')):
        raise ValueError('Canonical binding differs from the committed fixture')
    rust = rust_code(rust)
    for position, name in PINS.items():
        matches = re.findall(r'pub const ' + name + r':\s*\[u64;\s*4\]\s*=\s*\[([\d_,\s]+)\];', rust)
        if len(matches) != 1:
            raise ValueError('Expected one literal Rust pin: ' + name)
        words = [int(word.strip().replace('_', '')) for word in matches[0].split(',') if word.strip()]
        if len(words) != 4 or words != actual[position]:
            raise ValueError('Canonical binding differs from Rust pin: ' + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('emitted', type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parent.parent
    expected = project / 'artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1-binding-v1.json'
    rust = project.parents[1] / 'crates/nightstream-fprime/src/identity.rs'
    check(json.loads(args.emitted.read_text()), json.loads(expected.read_text()), rust.read_text())
    print('[identity] canonical binding, structural identity, package identity and verifier-key pins match')


if __name__ == '__main__':
    main()
