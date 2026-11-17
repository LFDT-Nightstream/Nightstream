#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

groups_slowest_first=(
  '-p nightstream lifecycle::'
  '-p nightstream -- --skip lifecycle::'
  '-p nightstream-fprime'
  '-p neo-ccs --test packed_witness --test red_team --test poseidon2_round_constants --test poseidon2_lean_vectors'
  '-p neo-params --test red_team'
  '-p neo-math --test signed_sums'
  '-p neo-ajtai --lib key::tests'
  '-p neo-ajtai --test nightstream_fprime_setup_parity --test nightstream_fprime_signed_commitment --test nightstream_fprime_prefix_commitment --test nightstream_fprime_matrix_commitment --test nightstream_fprime_batch_commitment --test red_team'
  '-p neo-reductions --test matrix_rows --test redteam_*'
  '-p neo-reductions --lib superneo_eval::openings::tests::'
  '-p neo-reductions --lib factored_oracle_tests::'
  '-p neo-reductions --test pi_rlc_lean_parity'
  '-p ns-compress'
  '-p neo-application'
  '-p neo-wasm'
)

memory_heavy_alone=(
  '-p nightstream --test compression -- --ignored'
  '-p nightstream --test compression_at_cap -- --ignored'
  '-p ns-compress --lib -- --ignored key::tests::the_large_pinned_key_is_the_key_of_the_public_seeds'
)

common=(
  --release --locked
  --config 'profile.release.lto="off"'
  --config profile.release.codegen-units=16
)

shipped=(--release --locked)

case "${1:-}" in
  build)
    for group in "${groups_slowest_first[@]}"; do
      read -r -a selected <<< "$group"
      printf 'Building selected tests: %s\n' "$group"
      cargo test "${common[@]}" --no-run "${selected[@]}"
    done
    for group in "${memory_heavy_alone[@]}"; do
      read -r -a selected <<< "$group"
      printf 'Building selected tests with the shipped profile: %s\n' "$group"
      cargo test "${shipped[@]}" --no-run "${selected[@]}"
    done
    ;;
  test)
    printf '%s\0' "${groups_slowest_first[@]}" |
      xargs -0 -P 4 -n 1 bash "$0" run
    for group in "${memory_heavy_alone[@]}"; do
      bash "$0" run-shipped "$group"
    done
    ;;
  run | run-shipped)
    read -r -a selected <<< "${2:?missing test group}"
    printf 'Running selected tests: %s\n' "$2"
    if [ "$1" = run ]; then flags=("${common[@]}"); else flags=("${shipped[@]}"); fi
    RUST_TEST_THREADS=1 exec cargo test "${flags[@]}" --no-fail-fast "${selected[@]}"
    ;;
  *)
    printf 'Usage: %s {build|test}\n' "$0" >&2
    exit 2
    ;;
esac
