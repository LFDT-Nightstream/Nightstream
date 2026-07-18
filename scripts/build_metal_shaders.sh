#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Compile Nightstream's shared Metal shaders for one Apple SDK.

Usage:
  ./scripts/build_metal_shaders.sh --sdk macosx|iphoneos|iphonesimulator --out <file.metallib>
EOF
}

SDK=""
OUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sdk)
      SDK="${2:-}"
      shift 2
      ;;
    --out)
      OUT="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${SDK}" in
  macosx|iphoneos|iphonesimulator) ;;
  *)
    echo "Unsupported --sdk '${SDK}'" >&2
    usage >&2
    exit 2
    ;;
esac

if [[ -z "${OUT}" ]]; then
  echo "Missing --out" >&2
  usage >&2
  exit 2
fi

command -v xcrun >/dev/null 2>&1 || {
  echo "Missing xcrun; install Xcode and select it with xcode-select" >&2
  exit 1
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHADER_DIR="${ROOT_DIR}/crates/neo-prover-metal/shaders"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

mkdir -p "$(dirname "${OUT}")"

AIR_FILES=()
for source in field poseidon2; do
  air="${TMP_DIR}/${source}.air"
  xcrun -sdk "${SDK}" metal -I "${SHADER_DIR}" -c "${SHADER_DIR}/${source}.metal" -o "${air}"
  AIR_FILES+=("${air}")
done

xcrun -sdk "${SDK}" metallib "${AIR_FILES[@]}" -o "${OUT}"
echo "Wrote: ${OUT}"
