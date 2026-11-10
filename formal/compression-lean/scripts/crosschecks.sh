#!/usr/bin/env bash
# Builds the Lean check executables, then runs the ignored Rust tests that
# compare Rust with Lean, each under its cap. With arguments, runs only those
# checks, in that order.
#
# CI runs no Lean, so Rust pins the run digests that lowerCheck and
# boundedDepthCheck print (`digest` on each `match` line). Copy a new digest into
# its `*_digest_is_pinned` test (lowering.rs, bounded_depth_lifecycle/final_ir.rs)
# only after the check that prints it passes. The bounded-depth checks run in
# order: the keys, their JSON and key facts, each key's cover of its step
# package (`coverCheck`, slot 0 light, slot 1 refresh), each relation's runs
# against its step package, the setups, the final program, then its whole-run
# pin.
set -u
lean_dir=$(cd "$(dirname "$0")/.." && pwd)
repo=$(cd "$lean_dir/../.." && pwd)
checks=(lean_replay lean_setup_root lowering_matches_lean bounded_depth_layer1_keys_are_written
  bounded_depth_layer1_matches_lean bounded_depth_covers bounded_depth_runs_rebuild_the_layer1_keys bounded_depth_runs_match_lean bounded_depth_setup_columns
  bounded_depth_setup_matches_lean bounded_depth_final_program_matches_lean bounded_depth_program_digest_is_pinned)
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
  if [[ $check == bounded_depth_covers ]]; then
    # coverCheck on each slot's step package and the layer-1 key JSON of its circuit.
    for slot in 0 1; do
      circuit=light
      [[ $slot == 1 ]] && circuit=refresh
      step "$check $slot" timeout --signal=KILL 300 "$lean_dir/.lake/build/bin/coverCheck" \
        "$repo/formal/nightstream-fprime/artifacts/nightstream-fprime-bounded-depth-$circuit-v1.json" \
        "$repo/target/lean-lowering/bounded_depth_layer1_$slot.json"
    done
    continue
  fi
  case $check in
    bounded_depth_layer1_keys_are_written | bounded_depth_runs_rebuild_the_layer1_keys | bounded_depth_runs_match_lean | \
      bounded_depth_setup_matches_lean | bounded_depth_final_program_matches_lean | bounded_depth_program_digest_is_pinned)
      crate=nightstream ;;
    *) crate=neo-spartan ;;
  esac
  if [[ $built != *" $crate"* ]]; then
    step "build $crate tests" timeout --signal=KILL 300 cargo test --release -p "$crate" --lib --no-run
    built="$built $crate"
  fi
  step "$check" timeout --signal=KILL 300 cargo test --release -p "$crate" --lib "$check" -- --ignored
done
exit $failed
