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

# The tests that run alone use the release profile of Cargo.toml, fat LTO,
# which is the build that users run. Under fat LTO, Akita's NTT loops need
# more stack than under `common`. They have their own target folder, because
# Cargo locks a target folder for the whole of a build.
shipped=(--release --locked --target-dir "${CARGO_TARGET_DIR:-target}/shipped")

case "${1:-}" in
  build)
    for group in "${groups_slowest_first[@]}"; do
      read -r -a selected <<< "$group"
      printf 'Building selected tests: %s\n' "$group"
      cargo test "${common[@]}" --no-run "${selected[@]}"
    done
    ;;
  build-shipped)
    for group in "${memory_heavy_alone[@]}"; do
      read -r -a selected <<< "$group"
      printf 'Building selected tests with the shipped profile: %s\n' "$group"
      cargo test "${shipped[@]}" --no-run "${selected[@]}"
    done
    printf 'Shipped tests built at %s\n' "$(date -u +%H:%M:%SZ)"
    ;;
  test)
    shipped_log=$(mktemp)
    bash "$0" build-shipped > "$shipped_log" 2>&1 &
    shipped_build=$!
    trap 'kill "$shipped_build" 2> /dev/null || true' EXIT
    printf '%s\0' "${groups_slowest_first[@]}" |
      xargs -0 -P 4 -n 1 bash "$0" run
    printf 'Groups finished at %s\n' "$(date -u +%H:%M:%SZ)"
    status=0
    wait "$shipped_build" || status=$?
    cat "$shipped_log"
    [ "$status" -eq 0 ]
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
    printf 'Usage: %s {build|build-shipped|test}\n' "$0" >&2
    exit 2
    ;;
esac
