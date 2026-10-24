#!/usr/bin/env bash
# Builds the Lean check executables, then runs the ignored Rust tests that
# compare Rust with Lean (README.md), one after another, each under its cap.
# With arguments, runs only the named checks, in the order given.
# final_program_matches_lean reads the key file that
# production_layer1_matches_lean writes. production_setup_matches_lean reads
# that key file and the runs file that production_runs_rebuild_the_layer1_key
# writes. The terminal checks need about 22 GB.
set -u
lean_dir=$(cd "$(dirname "$0")/.." && pwd)
repo=$(cd "$lean_dir/../.." && pwd)
checks=(lean_replay lowering_matches_lean production_layer1_matches_lean terminal_program_matches_lean
  final_program_matches_lean production_runs_rebuild_the_layer1_key production_setup_matches_lean)
[ $# -gt 0 ] && checks=("$@")
failed=0

step() {
  local name=$1
  shift
  if "$@"; then echo "pass $name"; else echo "fail $name"; failed=1; fi
}

cd "$lean_dir" || exit 1
step "lake build" timeout --signal=KILL 1500 lake build replayLayer1 lowerCheck terminalCheck setupCheck
cd "$repo" || exit 1
built=""
for check in "${checks[@]}"; do
  case $check in
    terminal_program_matches_lean | final_program_matches_lean | production_runs_rebuild_the_layer1_key | \
      production_setup_matches_lean) crate=nightstream ;;
    *) crate=neo-spartan ;;
  esac
  if [[ $built != *" $crate"* ]]; then
    step "build $crate tests" timeout --signal=KILL 300 cargo test --release -p "$crate" --lib --no-run
    built="$built $crate"
  fi
  step "$check" timeout --signal=KILL 300 cargo test --release -p "$crate" --lib "$check" -- --ignored
done
exit $failed
