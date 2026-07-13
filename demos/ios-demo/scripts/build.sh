#!/bin/bash

# Build script for TestingWasm Xcode project

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [[ -z "${DEVELOPER_DIR:-}" && -d /Applications/Xcode.app/Contents/Developer/Platforms ]]; then
    export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
fi

# Default values
SCHEME="TestingWasm"
CONFIGURATION="Debug"
DESTINATION=""
BUILD_WASM=0
BUILD_METAL=0
WASM_THREADS=0

pick_default_destination() {
    local id=""

    # Prefer an already-booted iPhone simulator if available.
    id="$(xcrun simctl list devices booted 2>/dev/null | awk -F '[()]' '/iPhone/ {print $2; exit}')"
    if [[ -n "$id" ]]; then
        echo "platform=iOS Simulator,id=$id"
        return
    fi

    # Otherwise pick the first available iPhone simulator.
    id="$(xcrun simctl list devices available 2>/dev/null | awk -F '[()]' '/iPhone/ {print $2; exit}')"
    if [[ -n "$id" ]]; then
        echo "platform=iOS Simulator,id=$id"
        return
    fi

    # Fallback: generic destination (may build both arm64 + x86_64).
    echo "generic/platform=iOS Simulator"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --release)
            CONFIGURATION="Release"
            shift
            ;;
        --wasm)
            BUILD_WASM=1
            shift
            ;;
        --wasm-threads)
            BUILD_WASM=1
            WASM_THREADS=1
            shift
            ;;
        --metal)
            BUILD_METAL=1
            shift
            ;;
        --all)
            BUILD_WASM=1
            BUILD_METAL=1
            shift
            ;;
        --destination)
            DESTINATION="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --release      Build with Release configuration (default: Debug)"
            echo "  --wasm         Rebuild wasm bundle before Xcode build"
            echo "  --wasm-threads Rebuild wasm + wasm-threads bundles (WKWebView) before Xcode build"
            echo "  --metal        Rebuild NeoMetalBench.xcframework before Xcode build"
            echo "  --all          Rebuild wasm + Metal benchmark before Xcode build"
            echo "  --destination  Set build destination (default: first available iPhone simulator)"
            echo "  --help         Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$DESTINATION" ]]; then
    DESTINATION="$(pick_default_destination)"
fi

echo "Building TestingWasm..."
echo "  Configuration: $CONFIGURATION"
echo "  Destination: $DESTINATION"
echo "  Rebuild WASM:  $([[ $BUILD_WASM -eq 1 ]] && echo yes || echo no)"
echo "  WASM Threads:  $([[ $WASM_THREADS -eq 1 ]] && echo yes || echo no)"
echo "  Rebuild Metal: $([[ $BUILD_METAL -eq 1 ]] && echo yes || echo no)"
echo ""

cd "$PROJECT_DIR"

if [[ $BUILD_WASM -eq 1 ]]; then
    if [[ "$CONFIGURATION" == "Release" ]]; then
        if [[ $WASM_THREADS -eq 1 ]]; then
            ./scripts/build_wasm.sh --release --threads
        else
            ./scripts/build_wasm.sh --release
        fi
    else
        if [[ $WASM_THREADS -eq 1 ]]; then
            ./scripts/build_wasm.sh --debug --threads
        else
            ./scripts/build_wasm.sh --debug
        fi
    fi
    echo ""
fi

if [[ $BUILD_METAL -eq 1 || ! -d "$PROJECT_DIR/Frameworks/NeoMetalBench.xcframework" ]]; then
    if [[ "$CONFIGURATION" == "Release" ]]; then
        ./scripts/build_metal_bench.sh --release
    else
        ./scripts/build_metal_bench.sh --profiling
    fi
    echo ""
fi

xcodebuild \
    -project TestingWasm.xcodeproj \
    -scheme "$SCHEME" \
    -configuration "$CONFIGURATION" \
    -destination "$DESTINATION" \
    build

echo ""
echo "Build completed successfully!"
