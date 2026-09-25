#!/usr/bin/env python3
"""Regenerate every derived Nightstream contract view."""

from __future__ import annotations

from contract_model import ModelError, load_model, refresh


def main() -> int:
    try:
        model = load_model(repository_mode=False)
        changed = refresh(model)
    except (ModelError, KeyError, OSError, TypeError, ValueError) as error:
        print(f"contract refresh: FAIL: {error}")
        return 1
    if changed:
        print("contract refresh: PASS")
        for relative in changed:
            print(f"  updated: {relative}")
    else:
        print("contract refresh: PASS (no changes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
