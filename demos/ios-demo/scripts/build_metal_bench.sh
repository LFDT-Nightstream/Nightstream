#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ROOT_DIR="$(cd "$PROJECT_DIR/../.." && pwd)"
PROFILE="release"

case "${1:-}" in
  "") ;;
  --release) PROFILE="release" ;;
  --profiling) PROFILE="profiling" ;;
  -h|--help)
    echo "Usage: $0 [--release|--profiling]"
    exit 0
    ;;
  *)
    echo "Unknown argument: $1" >&2
    exit 2
    ;;
esac

"$ROOT_DIR/scripts/build_metal_bench_xcframework.sh" \
  --profile "$PROFILE" \
  --out "$PROJECT_DIR/Frameworks"
