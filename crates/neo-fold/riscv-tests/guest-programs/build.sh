#!/bin/bash
# Build script for RISC-V guest programs
#
# Prerequisites:
#   - rustup target add riscv32im-unknown-none-elf
#   - cargo +nightly component add rust-src
#
# Usage:
#   ./build.sh fibonacci

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GUEST_NAME="${1:-fibonacci}"
GUEST_DIR="$SCRIPT_DIR/$GUEST_NAME"

if [ ! -d "$GUEST_DIR" ]; then
    echo "Error: Guest directory '$GUEST_DIR' not found"
    exit 1
fi

echo "Building guest program: $GUEST_NAME"
cd "$GUEST_DIR"

# Build the guest program
cargo +nightly build --release -Z build-std=core -Z build-std-features=compiler-builtins-mem

# Copy the binary to a known location
BINARY="target/riscv32im-unknown-none-elf/release/$GUEST_NAME-guest"
if [ -f "$BINARY" ]; then
    cp "$BINARY" "$SCRIPT_DIR/$GUEST_NAME.elf"
    echo "✓ Built: $SCRIPT_DIR/$GUEST_NAME.elf"
    
    # Show binary info
    if command -v riscv64-unknown-elf-objdump &> /dev/null; then
        echo ""
        echo "Disassembly (first 20 lines):"
        riscv64-unknown-elf-objdump -d "$SCRIPT_DIR/$GUEST_NAME.elf" | head -40
    fi
else
    echo "Error: Binary not found at $BINARY"
    exit 1
fi


