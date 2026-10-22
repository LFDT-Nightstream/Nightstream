"""Check the dependencies of the refinement theorems in Audit.v against audit.expected."""

from pathlib import Path
import re
import subprocess
import sys


NAME = r"[A-Za-z_][A-Za-z_0-9']*(?:\.[A-Za-z_][A-Za-z_0-9']*)*"


def axiom_names(output):
    lines = output.splitlines()
    markers = [i for i, line in enumerate(lines)
               if line in ("Axioms:", "Closed under the global context")]
    if len(markers) != 1:
        raise ValueError("expected exactly one Rocq assumptions report")
    start = markers[0]
    body = [line for line in lines[start + 1:] if line.strip()]
    if lines[start] == "Closed under the global context":
        if body:
            raise ValueError("unexpected output after closed-context report")
        return set()

    names = set()
    has_type = False
    for line in body:
        if line[0].isspace():
            if not names:
                raise ValueError("type continuation without an axiom name")
            has_type = True
            continue
        match = re.fullmatch(rf"({NAME})\s*:\s*(.*)", line)
        if not match or (names and not has_type):
            raise ValueError(f"unrecognized axiom entry: {line}")
        name, type_text = match.groups()
        if name in names:
            raise ValueError(f"duplicate axiom: {name}")
        names.add(name)
        has_type = bool(type_text)
    if not names or not has_type:
        raise ValueError("empty or incomplete axioms report")
    return names


def main():
    scripts = Path(__file__).resolve().parent
    result = subprocess.run(
        ["rocq", "repl", "-batch", "-q", "-Q", ".", "", "-l", "Audit.v"],
        cwd=scripts.parent, capture_output=True, text=True, check=False,
    )
    print(result.stdout, end="", flush=True)
    print(result.stderr, end="", file=sys.stderr, flush=True)
    if result.returncode:
        return 1
    try:
        actual = axiom_names(result.stdout)
        expected_lines = (scripts / "audit.expected").read_text().splitlines()
        if any(not re.fullmatch(NAME, name) for name in expected_lines):
            raise ValueError("audit.expected must contain one axiom name per line")
        expected = set(expected_lines)
        if len(expected) != len(expected_lines):
            raise ValueError("duplicate name in audit.expected")
    except (ValueError, OSError) as error:
        print(f"Audit failed: {error}", file=sys.stderr)
        return 1
    if actual != expected:
        for name in sorted(actual - expected):
            print(f"Added assumption: {name}", file=sys.stderr)
        for name in sorted(expected - actual):
            print(f"Removed assumption: {name}", file=sys.stderr)
        print("Review the changes before editing audit.expected.", file=sys.stderr)
        return 1
    print(f"Audit passed: {len(actual)} expected assumptions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
