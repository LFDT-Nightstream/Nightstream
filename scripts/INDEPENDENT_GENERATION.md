# Independent Lean generation

This workflow generates PiCCS messages and its verifier-driven transcript from
17 original witness openings, then computes the PiRLC parent and PiDEC child
witnesses, commitments, and separate `Eval_K` and all 14 `Eval_A` families. Rust
intermediates are comparison targets. The selected key, layout and package are
the same ones used by the maintained prover.

**The complete author-run sequence passed:** both independent Lean folds, both
fresh successors and iteration-four terminal acceptance and rejection checks.
The second fold uses the exact first Lean successor. See the
[execution index](../docs/reviews/pirlc-sampler-replacement/INDEPENDENT_EXECUTION.json)
and [implementation report](../docs/reviews/pirlc-sampler-replacement/REPORT.md).
Independent review remains pending.

The selected trace starts with the nonzero iteration-2 bootstrap, generates
2→3, and feeds the exact first Lean result into 3→4. Terminal acceptance and
rejections use iteration 4. The narrower native-proof verification workflow is
[golden conformance](GOLDEN_CONFORMANCE.md).

## Build and inputs

Use the repository instructions and run one Lean or Rust build/test at a time.
Each Lean child runs through `validate.sh` with a 1,500-second cap; native and
Python test children have 300-second caps. Run the coordinator outside the graph
lock. Do not increase the limits to convert a failed check into a pass.

From `formal/nightstream-fprime`, build the retained executables:

```sh
elan run nightstream-lean-4.32.2-3019a32c bash scripts/validate.sh build \
  replayPiCCSFirstRound replayPiCCSPrefix replayPiRLCWitness \
  replayPiDECWitness replayPiDECCommitment replayPiDECEvaluation \
  replayPiDECMatrix mergePiDECMatrix checkPiDECInput \
  emitRecursiveStepFixture replayPhysicalWitness replayFreshAssignment \
  checkFreshRows replayFreshCommitment
```

The requested optimized compiler is `nicarq/lean4-optimized` at
`3019a32cb6f44782ff1e1210676099d683b8d3a8`, based on Lean 4.32.2. Its measured
emitter speedup is not a claim about complete prover runtime. The current replay
also shares proved tensor-weight tables in the Pad generators and uses the
existing native-word arithmetic for original masks and weighted updates. On the same first
fold inputs, the first Pad range's arithmetic fell from 38.78 to 7.17 seconds,
and the first matrix range fell from 191.57 to 137.82 seconds. Every output byte
matched. These timings exclude shared input loading and do not estimate the
whole workflow. The preservation proofs and audits pass on both the requested
compiler and stock Lean 4.32.2.

From the repository root, build the native test executable under the native cap:

```sh
timeout --signal=KILL 300 cargo test -p nightstream --release --lib --no-run
```

Use the executable path printed by Cargo as `BINARY`. A current native run must
supply `native/step-2/` (the envelope, fresh claim/witness, and 16 child witnesses),
`native/fold-2/`, and `native/step-3/`. Generate these with the maintained
`crates/nightstream/tests/run_recursive_phase.py`: base, then the ordered
`sources`, `ccs`, `rlc`, `split`, `openings`, `nifs`, `successor` phases for steps
1 and 2. It enforces the existing time and RSS limits. Never restore an
old-package fixture to provide the bootstrap.

A previously completed native comparison run may be reused only with its actual
receipts and unchanged production source, package, and inputs. Keep that origin
explicit. The independent coordinator rechecks the actual source commitments,
state/public binding and the running child family (`validate_running_children`).
A saved digest or claimed
success receipt does not supply those checks.

## Generate and compare

From the repository root, run one step with a separate output directory:

```sh
elan run nightstream-lean-4.32.2-3019a32c python3.12 -B \
  formal/nightstream-fprime/scripts/generate_independent_folds.py \
  --directory RUN --native RUN/native --binary BINARY --step 2 --phase all
```

The stages are `prepare`, `ccs`, `reductions`, and `successor`; `all` executes them
in that order. Completed checkpoints are reusable only when their exact command,
input files, executable and output bytes still match. Failed attempts must remain
identifiable; do not edit a failure record into a success or use partial output.
Native commands must also show that the exact selected test ran and passed;
a zero-test libtest exit cannot complete or satisfy a checkpoint.
Source snapshots identify source files and the package; they do not prove binary
correctness or protocol semantics.

The command guards count elapsed time during host sleep and reject a successful
exit after the cap. On macOS, prefix a long coordinator command with
`/usr/bin/caffeinate -i` to prevent idle sleep only while that command runs.
This does not change a cap or authorize reuse of an interrupted batch.

The `ccs` producer receives original sources and earlier Lean rounds. Only after
production does it read the native comparison target. Fresh polynomial ranges
count row pairs; carried matrix moments count individual rows. Their separate
complete-range checks prevent an incomplete prefix from being accepted as a
complete round.

For the second comparison chain, use the same native phase runner with `--step 3`
for `sources`, `ccs`, `rlc`, `split`, `openings`, `nifs`, and `successor`. Native
production reads its own `step-3` successor. Then run the independent coordinator
with `--step 3 --phase all`. Its source projection takes the first **Lean** fresh
witness/claim, child claims and complete child-witness ranges. No native computed
intermediate supplies that second Lean source. The first completed handoff must
exist and its exact saved output bytes must still match before projection starts.

The second fold uses the first fold's completed matrix-range timings to group
consecutive ranges. Its soft target is half the existing Lean cap, including
the largest measured batch overhead. This changes only how often the same
producer reloads the parent. All ranges, outputs, ordering and checks remain
required, and every invocation still has the original hard cap. Timing records
guide scheduling; they do not establish the correctness of any generated value.

Each successor compares all caller values, physical witness bytes, logical
assignment coordinates, fresh commitment coefficients and public values. The
scalar canonical-row checker and rejection tests remain separate. Parent and
child witness comparisons include every retained tail coordinate and reject
changed tail values; the complete NIFS check compares proof bytes as well as
all C/R/D values.

Finally run the maintained native phases `terminal`, `mutation`, and `reject`
with `--step 4`, followed by `opening-k-prepare`, `opening-k`,
`opening-a-prepare`, and `opening-a`, also with `--step 4`. Both balanced-opening
checks must pass earlier relation and recomposition checks and then reject at
the production verifier's specified `Eval_K` or `Eval_A` check.

A completed run needs both exact successor handoffs, final iteration-4 scalar
row validation, complete byte/value comparisons, all specified rejections, and
retrievable source-bound output and command evidence. Author-run evidence and
independent reviewer acceptance are separate. Keep the PR in draft while the
required execution or reviews remain incomplete.
