#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Build the native Metal benchmark XCFramework for iOS.

Usage:
  ./scripts/build_metal_bench_xcframework.sh [--profile release|profiling] [--out <dir>] [--include-x86_64-sim]
EOF
}

PROFILE="release"
OUT_DIR=""
INCLUDE_X86_64_SIM=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILE="${2:-}"
      shift 2
      ;;
    --out)
      OUT_DIR="${2:-}"
      shift 2
      ;;
    --include-x86_64-sim)
      INCLUDE_X86_64_SIM=1
      shift
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

case "$PROFILE" in
  release|profiling) ;;
  *)
    echo "Unsupported profile: $PROFILE" >&2
    exit 2
    ;;
esac

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/dist}"
PACKAGE="neo-prover-metal-bench"
LIB_NAME="libneo_prover_metal_bench.a"
FRAMEWORK_NAME="NeoMetalBench"
HEADER_DIR="$ROOT_DIR/crates/neo-prover-metal-bench/include"

if [[ ! -f "$HEADER_DIR/neo_metal_bench.h" || ! -f "$HEADER_DIR/module.modulemap" ]]; then
  echo "Metal benchmark C headers are incomplete" >&2
  exit 1
fi

if [[ -z "${DEVELOPER_DIR:-}" && -d /Applications/Xcode.app/Contents/Developer/Platforms ]]; then
  export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
fi
command -v xcodebuild >/dev/null 2>&1 || {
  echo "xcodebuild is unavailable; install full Xcode" >&2
  exit 1
}

TARGETS=(aarch64-apple-ios aarch64-apple-ios-sim)
if [[ "$INCLUDE_X86_64_SIM" == 1 ]]; then
  TARGETS+=(x86_64-apple-ios)
fi

for target in "${TARGETS[@]}"; do
  if ! rustup target list --installed | grep -qx "$target"; then
    rustup target add "$target"
  fi
  case "$target" in
    aarch64-apple-ios)
      sdk="iphoneos"
      ;;
    aarch64-apple-ios-sim|x86_64-apple-ios)
      sdk="iphonesimulator"
      ;;
    *)
      echo "No Apple SDK mapping for Rust target: $target" >&2
      exit 2
      ;;
  esac
  sdkroot="$(xcrun --sdk "$sdk" --show-sdk-path)"
  clang="$(xcrun --sdk "$sdk" --find clang)"
  linker_var="CARGO_TARGET_$(printf '%s' "$target" | tr '[:lower:]-' '[:upper:]_')_LINKER"
  env SDKROOT="$sdkroot" "$linker_var=$clang" \
    cargo build -p "$PACKAGE" --lib --profile "$PROFILE" --target "$target"
done

mkdir -p "$OUT_DIR"
rm -rf "$OUT_DIR/$FRAMEWORK_NAME.xcframework"

device_lib="$ROOT_DIR/target/aarch64-apple-ios/$PROFILE/$LIB_NAME"
sim_lib="$ROOT_DIR/target/aarch64-apple-ios-sim/$PROFILE/$LIB_NAME"
if [[ "$INCLUDE_X86_64_SIM" == 1 ]]; then
  universal_dir="$ROOT_DIR/target/ios-universal-sim/$PROFILE"
  mkdir -p "$universal_dir"
  xcrun lipo -create \
    "$sim_lib" \
    "$ROOT_DIR/target/x86_64-apple-ios/$PROFILE/$LIB_NAME" \
    -output "$universal_dir/$LIB_NAME"
  sim_lib="$universal_dir/$LIB_NAME"
fi

xcodebuild -create-xcframework \
  -library "$device_lib" -headers "$HEADER_DIR" \
  -library "$sim_lib" -headers "$HEADER_DIR" \
  -output "$OUT_DIR/$FRAMEWORK_NAME.xcframework"

echo "Wrote: $OUT_DIR/$FRAMEWORK_NAME.xcframework"
