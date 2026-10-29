#!/usr/bin/env bash
# Builds the Lean check executables, then runs the ignored Rust tests that
# compare Rust with Lean, each under its cap. With arguments, runs only those
# checks, in that order. The terminal checks need about 22 GB.
#
# CI runs no Lean, so Rust pins the run digests that lowerCheck and
# terminalCheck print (`digest` on each `match` line). Copy a new digest into
# its `*_digest_is_pinned` test (lowering.rs, terminal_ir.rs) only after the
# check that prints it passes.
set -u
lean_dir=$(cd "$(dirname "$0")/.." && pwd)
repo=$(cd "$lean_dir/../.." && pwd)
checks=(lean_replay lean_setup_root lowering_matches_lean production_layer1_matches_lean terminal_program_matches_lean
  final_program_matches_lean production_runs_rebuild_the_layer1_key production_setup_columns
  production_setup_matches_lean production_runs_match_lean)
[ $# -gt 0 ] && checks=("$@")
failed=0

step() {
  local name=$1
  shift
  if "$@"; then echo "pass $name"; else echo "fail $name"; failed=1; fi
}

cd "$lean_dir" || exit 1
step "lake build" timeout --signal=KILL 1500 lake build
cd "$repo" || exit 1
built=""
for check in "${checks[@]}"; do
  case $check in
    terminal_program_matches_lean | final_program_matches_lean | production_runs_rebuild_the_layer1_key | \
      production_setup_matches_lean | production_runs_match_lean) crate=nightstream ;;
    *) crate=neo-spartan ;;
  esac
  if [[ $built != *" $crate"* ]]; then
    step "build $crate tests" timeout --signal=KILL 300 cargo test --release -p "$crate" --lib --no-run
    built="$built $crate"
  fi
  step "$check" timeout --signal=KILL 300 cargo test --release -p "$crate" --lib "$check" -- --ignored
done
exit $failed
