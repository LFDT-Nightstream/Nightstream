#!/usr/bin/env bash
set -euo pipefail

# Round-0 end-to-end: dump -> trace -> compare in Sage

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DUMP_PATH="$SCRIPT_DIR/round0_dump.json"
TRACE_PATH="$SCRIPT_DIR/round0_trace.json"

echo "[round0] Repo root : $REPO_ROOT"
echo "[round0] Dump path : $DUMP_PATH"
echo "[round0] Trace path: $TRACE_PATH"

cd "$REPO_ROOT"

LOG_DIR="$SCRIPT_DIR"
DUMP_LOG="$LOG_DIR/round0_dump.rust.log"
TRACE_LOG="$LOG_DIR/round0_trace.rust.log"
SAGE_LOG="$LOG_DIR/round0_compare.sage.log"
COMBINED_LOG="$LOG_DIR/round0_full.log"

echo "\n[round0] Writing dump (Rust) -> $DUMP_LOG"
NEO_ROUND0_DUMP="$DUMP_PATH" \
  cargo test -p neo-fold --test unit --features testing -- \
    write_round0_dump --ignored --nocapture \
    > "$DUMP_LOG" 2>&1

echo "\n[round0] Writing per-X trace (Rust) -> $TRACE_LOG"
NEO_ROUND0_TRACE="$TRACE_PATH" \
  cargo test -p neo-fold --test unit --features testing -- \
    write_round0_trace --ignored --nocapture \
    > "$TRACE_LOG" 2>&1

echo "\n[round0] Comparing in Sage (paper-faithful) -> $SAGE_LOG"
# Defaults: suppress Sage per-X wall (VERBOSE=1) and show side-by-side aggregates
# Override by exporting VERBOSE/COMPARE_VERBOSE/COMPARE_ROWS/COMPARE_MASK before running this script
NEO_NONRES=7 \
  VERBOSE="${VERBOSE:-1}" \
  COMPARE_VERBOSE="${COMPARE_VERBOSE:-1}" \
  COMPARE_ROWS="${COMPARE_ROWS:-16}" \
  COMPARE_MASK="${COMPARE_MASK:-}" \
  sage "$SCRIPT_DIR/q_from_dump_paper_exact.sage" "$DUMP_PATH" "$TRACE_PATH" \
  > "$SAGE_LOG" 2>&1

# Build combined log for easier sharing
cat "$DUMP_LOG" "$TRACE_LOG" "$SAGE_LOG" > "$COMBINED_LOG"

echo "\n[round0] Done. Outputs:"
echo "  dump      : $DUMP_PATH"
echo "  trace     : $TRACE_PATH"
echo "  rust dump : $DUMP_LOG"
echo "  rust trace: $TRACE_LOG"
echo "  sage cmp  : $SAGE_LOG"
echo "  combined  : $COMBINED_LOG"
