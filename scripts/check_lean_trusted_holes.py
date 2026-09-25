#!/usr/bin/env python3
"""Reject trusted-hole tokens in Lean code, but ignore comments and strings."""

from pathlib import Path
import re
import sys


FORBIDDEN = re.compile(r"\b(sorry|axiom|admit|postulate|unsafe)\b")


def code_without_comments_or_strings(source: str) -> str:
    result = list(source)
    index = 0
    block_depth = 0
    in_line_comment = False
    in_string = False
    escaped = False

    while index < len(source):
        current = source[index]
        following = source[index + 1] if index + 1 < len(source) else ""

        if in_line_comment:
            if current == "\n":
                in_line_comment = False
            else:
                result[index] = " "
            index += 1
            continue

        if block_depth:
            if current == "/" and following == "-":
                result[index] = result[index + 1] = " "
                block_depth += 1
                index += 2
            elif current == "-" and following == "/":
                result[index] = result[index + 1] = " "
                block_depth -= 1
                index += 2
            else:
                if current != "\n":
                    result[index] = " "
                index += 1
            continue

        if in_string:
            if current != "\n":
                result[index] = " "
            if escaped:
                escaped = False
            elif current == "\\":
                escaped = True
            elif current == '"':
                in_string = False
            index += 1
            continue

        if current == "-" and following == "-":
            result[index] = result[index + 1] = " "
            in_line_comment = True
            index += 2
        elif current == "/" and following == "-":
            result[index] = result[index + 1] = " "
            block_depth = 1
            index += 2
        elif current == '"':
            result[index] = " "
            in_string = True
            index += 1
        else:
            index += 1

    return "".join(result)


def lean_files(arguments: list[str]):
    for argument in arguments:
        path = Path(argument)
        if path.is_file() and path.suffix == ".lean":
            yield path
        elif path.is_dir():
            for candidate in path.rglob("*.lean"):
                if ".lake" not in candidate.parts:
                    yield candidate


def main() -> int:
    found = False
    for path in sorted(set(lean_files(sys.argv[1:]))):
        source = path.read_text(encoding="utf-8")
        code = code_without_comments_or_strings(source)
        for match in FORBIDDEN.finditer(code):
            line = code.count("\n", 0, match.start()) + 1
            print(f"{path}:{line}:{match.group(1)}")
            found = True
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
