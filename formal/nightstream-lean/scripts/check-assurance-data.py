#!/usr/bin/env python3
"""Parse every committed Nightstream assurance JSON record."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ASSURANCE = ROOT / "assurance"


def main() -> None:
    documents = 0
    records = 0

    for path in sorted(ASSURANCE.glob("*.json")):
        with path.open(encoding="utf-8") as source:
            json.load(source)
        documents += 1

    for path in sorted(ASSURANCE.glob("*.jsonl")):
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
                records += 1

    print(f"[assurance-data] parsed {documents} JSON documents and {records} JSONL records")


if __name__ == "__main__":
    main()
