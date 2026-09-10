#!/usr/bin/env python3
"""Build the current map from the canonical site snapshot."""

from pathlib import Path
import subprocess
import sys


if __name__ == '__main__':
    build = Path(__file__).resolve().parent / 'site' / 'build.py'
    raise SystemExit(subprocess.call([sys.executable, '-B', str(build), *sys.argv[1:]]))
