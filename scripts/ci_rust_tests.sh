#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# xargs starts the groups in this order, so the four slowest come first. The
# nightstream tests are split by name; the last of them runs the rest.
groups=(
  '-p nightstream lifecycle::'
  '-p nightstream engine::'
  '-p nightstream -- --skip lifecycle:: --skip engine::'
  '-p nightstream-fprime --test component_forms --test package_loader --test per_application_logical_matrix_conformance --test pi_ccs_opening_arithmetic --test pi_ccs_opening_values --test pilot_lean_nonzero_parity --test production_binding --test statement_binding_rows'
  '-p neo-spartan'
  '-p neo-ccs --test packed_witness --test red_team --test poseidon2_round_constants'
  '-p neo-params --test red_team'
  '-p neo-math --test signed_sums'
  '-p neo-ajtai --lib key::tests'
  '-p neo-ajtai --test nightstream_fprime_setup_parity --test nightstream_fprime_signed_commitment --test nightstream_fprime_prefix_commitment --test nightstream_fprime_matrix_commitment --test nightstream_fprime_batch_commitment --test red_team'
  '-p neo-reductions --test matrix_rows --test redteam_*'
  '-p neo-reductions --lib superneo_eval::openings::tests::'
  '-p nightstream-fprime --lib application_records'
  '-p nightstream-fprime --lib package::native_application'
  '-p neo-reductions --test pi_rlc_lean_parity'
  '-p nightstream-fprime --lib package::matrix_program'
  '-p nightstream-fprime --lib package::assignment_transport::quotient_tests'
  '-p nightstream-fprime --lib package::sealed::sealed_tests'
  '-p neo-application'
  '-p neo-wasm'
)

common=(
  --release --locked
  --config 'profile.release.lto="off"'
  --config profile.release.codegen-units=16
)

case "${1:-}" in
  build)
    for group in "${groups[@]}"; do
      read -r -a selected <<< "$group"
      printf 'Building selected tests: %s\n' "$group"
      cargo test "${common[@]}" --no-run "${selected[@]}"
    done
    ;;
  test)
    printf '%s\0' "${groups[@]}" |
      xargs -0 -P 4 -n 1 bash "$0" run
    ;;
  run)
    read -r -a selected <<< "${2:?missing test group}"
    printf 'Running selected tests: %s\n' "$2"
    # One libtest process per binary keeps its shared fixtures alive. The
    # thread count is set by environment so a group can pass its own `--`.
    RUST_TEST_THREADS=1 exec cargo test "${common[@]}" --no-fail-fast "${selected[@]}"
    ;;
  *)
    printf 'Usage: %s {build|test}\n' "$0" >&2
    exit 2
    ;;
esac
