#!/usr/bin/env bash
# Validate the matrix reader against the operands from emitWideMatrixCost.
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/lean-wide-matrix-operands.json" >&2
  exit 2
fi
operands=$(realpath "$1")
cd "$(dirname "$0")/../../.."
# The project requires a 300-second cap for each non-Lean test invocation.
timeout --signal=KILL 300 cargo test -p nightstream-fprime --release --lib \
  package::matrix_program::matrix_program_tests::wide_candidate_matrix_operands_decode \
  -- --exact --ignored --nocapture < "$operands"
