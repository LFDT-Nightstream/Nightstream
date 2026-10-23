#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Keep the branch's package, target, and test-name selections unchanged.
groups=(
  '-p nightstream'
  '-p neo-spartan'
  '-p neo-ccs --test packed_witness'
  '-p neo-math --test signed_sums'
  '-p neo-ajtai --lib key::tests'
  '-p neo-ajtai --test nightstream_fprime_setup_parity --test nightstream_fprime_signed_commitment --test nightstream_fprime_prefix_commitment --test nightstream_fprime_matrix_commitment --test nightstream_fprime_batch_commitment'
  '-p neo-reductions --test matrix_rows'
  '-p neo-reductions --lib superneo_eval::openings::tests::'
  '-p nightstream-fprime --lib application_records'
  '-p nightstream-fprime --lib package::native_application'
  '-p neo-reductions --test pi_rlc_lean_parity'
  '-p nightstream-fprime --lib package::matrix_program'
  '-p nightstream-fprime --lib package::assignment_transport::quotient_tests'
  '-p nightstream-fprime --lib package::sealed::sealed_tests'
)

common=(
  --release --locked
  --config 'profile.release.lto="off"'
  --config profile.release.codegen-units=16
)

cached_group='-p nightstream-fprime --lib package::matrix_program'

nextest_selection() {
  read -r -a selected <<< "${groups[$1]}"
  filter='all()'
  if [[ "${selected[2]:-}" == --lib ]]; then
    # Cargo's positional name filter is a substring match.
    filter="test(~${selected[3]})"
    selected=("${selected[@]:0:3}")
  elif [[ "${selected[1]}" == nightstream ]]; then
    filter='not kind(lib)'
  fi
}

case "${1:-}" in
  build)
    mkdir -p target/ci-nextest
    for index in "${!groups[@]}"; do
      group=${groups[$index]}
      printf 'Building selected tests: %s\n' "$group"
      if [[ "$group" == "$cached_group" ]]; then
        read -r -a selected <<< "$group"
        cargo test "${common[@]}" --no-run "${selected[@]}"
      else
        nextest_selection "$index"
        cargo nextest list "${common[@]}" "${selected[@]}" \
          --list-type binaries-only --message-format json > "target/ci-nextest/$index.json"
      fi
    done
    ;;
  test)
    printf '%s\0' nightstream-lib "${!groups[@]}" |
      xargs -0 -P 4 -n 1 bash "$0" run
    ;;
  run)
    if [[ "${2:?missing test group}" == nightstream-lib ]]; then
      # Keep the compiled circuit, polynomial, and terminal package caches.
      exec cargo test "${common[@]}" --no-fail-fast -p nightstream --lib -- --test-threads=1
    fi
    group=${groups[$2]}
    printf 'Running selected tests: %s\n' "$group"
    if [[ "$group" == "$cached_group" ]]; then
      read -r -a selected <<< "$group"
      exec cargo test "${common[@]}" --no-fail-fast "${selected[@]}" -- --test-threads=1
    fi
    nextest_selection "$2"
    exec cargo nextest run --profile ci --no-fail-fast \
      --binaries-metadata "target/ci-nextest/$2.json" -E "$filter"
    ;;
  docs)
    for package in nightstream neo-spartan; do
      cargo test "${common[@]}" --no-fail-fast -p "$package" --doc -- --test-threads=1
    done
    ;;
  *)
    printf 'Usage: %s {build|test|docs}\n' "$0" >&2
    exit 2
    ;;
esac
