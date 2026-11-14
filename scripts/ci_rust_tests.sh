#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# xargs starts the groups in this order, so the four slowest come first. The
# nightstream tests are split by name; the last of them runs the rest.
groups=(
  '-p nightstream lifecycle::'
  '-p nightstream -- --skip lifecycle::'
  '-p nightstream-fprime'
  '-p neo-ccs --test packed_witness --test red_team --test poseidon2_round_constants'
  '-p neo-params --test red_team'
  '-p neo-math --test signed_sums'
  '-p neo-ajtai --lib key::tests'
  '-p neo-ajtai --test nightstream_fprime_setup_parity --test nightstream_fprime_signed_commitment --test nightstream_fprime_prefix_commitment --test nightstream_fprime_matrix_commitment --test nightstream_fprime_batch_commitment --test red_team'
  '-p neo-reductions --test matrix_rows --test redteam_*'
  '-p neo-reductions --lib superneo_eval::openings::tests::'
  '-p neo-reductions --lib factored_oracle_tests::'
  '-p neo-reductions --test pi_rlc_lean_parity'
  '-p neo-compress'
  '-p neo-application'
  '-p neo-wasm'
)

# Each of these needs up to 30 GB, so each runs alone after the groups.
alone=(
  '-p nightstream --test compression -- --ignored'
  '-p nightstream --test compression_at_cap -- --ignored'
  '-p neo-compress --lib -- --ignored key::tests::the_large_pinned_key_is_the_key_of_the_public_seeds'
)

common=(
  --release --locked
  --config 'profile.release.lto="off"'
  --config profile.release.codegen-units=16
)

# The tests that run alone use the release profile of Cargo.toml, fat LTO,
# which is the build that users run. Under fat LTO, Akita's NTT loops need
# more stack than under `common`.
shipped=(--release --locked)

case "${1:-}" in
  build)
    for group in "${groups[@]}"; do
      read -r -a selected <<< "$group"
      printf 'Building selected tests: %s\n' "$group"
      cargo test "${common[@]}" --no-run "${selected[@]}"
    done
    for group in "${alone[@]}"; do
      read -r -a selected <<< "$group"
      printf 'Building selected tests with the shipped profile: %s\n' "$group"
      cargo test "${shipped[@]}" --no-run "${selected[@]}"
    done
    ;;
  test)
    printf '%s\0' "${groups[@]}" |
      xargs -0 -P 4 -n 1 bash "$0" run
    for group in "${alone[@]}"; do
      bash "$0" run-shipped "$group"
    done
    ;;
  run | run-shipped)
    read -r -a selected <<< "${2:?missing test group}"
    printf 'Running selected tests: %s\n' "$2"
    if [ "$1" = run ]; then flags=("${common[@]}"); else flags=("${shipped[@]}"); fi
    # One libtest process per binary keeps its shared fixtures alive. The
    # thread count is set by environment so a group can pass its own `--`.
    RUST_TEST_THREADS=1 exec cargo test "${flags[@]}" --no-fail-fast "${selected[@]}"
    ;;
  *)
    printf 'Usage: %s {build|test}\n' "$0" >&2
    exit 2
    ;;
esac
