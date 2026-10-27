#!/usr/bin/env bash
# Builds the Lean check executables, then runs the ignored Rust tests that
# compare Rust with Lean, one after another, each under its cap. With
# arguments, runs only the named checks, in the order given. The tests print
# their sizes; the terminal checks need about 22 GB.
#
# The checks (each test writes its inputs to target/lean-replay or
# target/lean-lowering, then runs the Lean executable on them):
#
# lean_replay (neo-spartan, replayLayer1): proves a toy statement, writes the
#   honest proof and tampered variants to layer1.json, and requires the
#   replay (Ports/Replay/) to draw the same challenges and give the same
#   verdict as Rust. The replay takes the WHIR opened values as `Opens` does,
#   so when Rust rejects inside WHIR, the replay must reject later.
# lean_setup_root (neo-spartan, setupRootCheck): writes the toy key's setup
#   oracle tables and key root to target/lean-replay/setup-root.json, and
#   requires Lean's tableRoot of the tables to be that root: the codeword
#   encoding, leaf order, leaf hash and tree of Bridge/TableRoot.lean and
#   Bridge/Merkle.lean are Rust's on that key.
# lowering_matches_lean (neo-spartan, lowerCheck): records nine programs
#   through record.rs (tests/internal/lowering_stages.rs and the toy layer-1
#   verifier) and writes the statement, the IR and Rust's trace to
#   <program>.txt, with the keys and stage inputs in layer1.json,
#   linear.json, honest.json, late.json and openings.json. lowerCheck lowers
#   the IR and requires the same glue cells, blocks, statement words and rows,
#   term by term. For every program but Edges it also writes Lean's program and
#   requires all of Rust's instructions in order.
# production_layer1_matches_lean (neo-spartan, lowerCheck): the same on the
#   shape run of the layer-1 verifier for the committed production key
#   (crates/nightstream/tests/fixtures/compression/key.bin). It writes
#   production_layer1.txt and production_layer1.json, and requires that the
#   key facts hold and that the key width fits the Poseidon2 hash chain.
# terminal_program_matches_lean (nightstream, terminalCheck): logs
#   FinalProgram::run with a toy layer-1 relation, up to verify_with, to
#   terminal.txt, and requires the builder's instructions to be Rust's: every
#   column source, row and term, and the claim and state handles.
# final_program_matches_lean (nightstream, terminalCheck): the whole final
#   program on the production key and package. It reads the
#   production_layer1.json of production_layer1_matches_lean.
# production_runs_rebuild_the_layer1_key (nightstream): writes the runs that
#   PackageRows::visit_rows pushes for the production circuit to
#   production_runs.bin and requires Key::derive of the file to be the
#   committed key's layer-1 key.
# production_setup_columns (neo-spartan): reads production_runs.bin, requires
#   that its runs derive the committed key, and writes the four run oracles
#   that the key commits after the key rows to production_setup_columns.bin.
# production_setup_matches_lean (nightstream, setupCheck): runs setupCheck on
#   production_runs.bin, production_layer1.json and production_setup_columns.bin
#   and requires `setup facts hold` and `setup columns match`: Lean's setup
#   holds the run oracles of the committed key.
# production_runs_match_lean (nightstream, runsCheck): runs runsCheck on
#   production_runs.bin and requires `runs give the matrices`: no run names a
#   row past F′'s rows, and the runs of each row and matrix give that row of
#   F′'s matrix program of the Poseidon2 hash chain.
#
# Digest pins. CI runs no Lean, so Rust pins the run digest that lowerCheck
# and terminalCheck compute from Lean's programs (`digest` on each `match`
# line). To update a pin, run its check, and only when it passes copy the
# digest into the test: toy_layer1_digest_is_pinned and
# production_layer1_digest_is_pinned (lowering.rs) from lowering_matches_lean
# and production_layer1_matches_lean; terminal_digest_is_pinned and
# final_program_digest_is_pinned (terminal_ir.rs) from
# terminal_program_matches_lean and final_program_matches_lean.
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
