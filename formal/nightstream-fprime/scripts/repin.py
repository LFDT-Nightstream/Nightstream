#!/usr/bin/env python3
"""Write the current fingerprint of each stale pin of SPEC.md (`scripts/validate.sh repin`).

Reads the output of `tests/EndpointCensus.lean` on stdin. Fails when no pin changes.
"""

from pathlib import Path
import re
import sys

SPEC = Path(__file__).resolve().parents[3] / "SPEC.md"
STALE = re.compile(r"`([^`]+)` is pinned at `([^`]*)`, but its statement now has fingerprint `([0-9a-f]{16})`")


def main():
    text = SPEC.read_text(encoding="utf-8")
    new = text
    for name, old, current in STALE.findall(sys.stdin.read()):
        new = new.replace(f"`{name}` (`{old}`)", f"`{name}` (`{current}`)")
        print(f"repin: `{name}` {old} -> {current}")
    if new == text:
        sys.exit("repin: no stale pin to write")
    SPEC.write_text(new, encoding="utf-8")


if __name__ == "__main__":
    main()
