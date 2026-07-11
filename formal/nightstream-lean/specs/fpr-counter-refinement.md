# FPR-COUNTER-REFINE proof contract

property_id: `FPR-COUNTER-REFINE`

claim: For a Construction-2 trace whose transitions satisfy the existing
`FPrime.Envelope.AdvanceCoherent` relation, `chunkCount` equals the number of
F' invocations and `stepCount` equals the sum of the nonempty fresh-batch
cardinalities. The initial state refines the empty schedule. Native Rust rejects
either `u64` counter overflow instead of wrapping.

assumptions:

- Lean `Nat` models the mathematical counter values carried by decoded Rust
  `u64` fields.
- A successful Rust transition uses `checked_add`, so its result agrees with
  natural-number addition; overflow is a rejected transition.
- `AdvanceCoherent` is the active envelope relation. Its content-authority
  non-goals remain unchanged.

non_goals:

- Full Rust AST or binary refinement.
- NIFS correctness, accumulator authority, application semantics, and Nebula
  transition validity.
- Counter encoding inside every surrounding F' R1CS row.
- Availability for traces requiring more than `u64::MAX` chunks or fresh
  instances; Rust rejects before either representation wraps.

paper_sources:

- HyperNova Construction 2, whose paper state uses the single invocation index
  `i`.
- The local direct-F' specialization, where one invocation may install a batch
  of fresh instances and therefore carries both invocation and instance counts.

rust_surfaces:

- `crates/neo-fold-clean/src/paper/construction2/state.rs` (`State` counters).
- `crates/neo-fold-clean/src/paper/construction2/transition.rs`
  (`advance_state`).
- `crates/neo-fold-clean/src/paper/f_prime/native.rs` (prover and verifier
  propagation of transition failures).
- `crates/neo-fold-clean/tests/system/lifecycle_finalization.rs` (public
  lifecycle overflow regressions).

circuit_or_encoding_artifacts:

- The generated no-wrap increment artifact is tracked separately by
  `CIR-U64INC`; this property consumes its no-wrap interpretation but
  does not claim the whole F' circuit correspondence.

failure_class: `code-first`. Release-mode Rust previously wrapped
`u64::MAX + 1` to zero while the R1CS increment relation rejected the final
carry and the Lean envelope used natural-number addition.

counterexample_or_witness:

- Set an honest active proof's `chunk_count` to `u64::MAX`, then extend it by
  one nonempty batch.
- Set an honest active proof's `step_count` to `u64::MAX`, then extend it by a
  one-instance batch.
- Before the repair, both public lifecycle calls succeeded and returned a zero
  counter. After the repair, both return `CounterOverflow`.

lean_theorems:

- `Nightstream.Implementation.FPrime.CounterRefinement.initial_refines`
- `Nightstream.Implementation.FPrime.CounterRefinement.advance_preserves`
- `Nightstream.Implementation.FPrime.CounterRefinement.counter_refinement`

axiom_report: `counter_refinement` depends on `[propext]`, pinned fail-closed by
`tests/Axioms.lean`.

proof_hash: `sha256:437ab004a4ada1e9b4779338cb75c9cf9af6fe1063cd40f19d8bc87a9a10dfe9`

conformance_status: `model-proved`, with public Rust lifecycle regressions for
both overflow branches and executable source anchors for the checked additions.
Full `rust-conformant` status remains under `RUST-REFINE` because no Rust AST or
binary refinement theorem is claimed here.

retest_commands:

```bash
cd formal/nightstream-lean
lake build
lake exe check
rg -n '\b(sorry|admit|axiom|unsafe)\b' Nightstream tests -g '*.lean'

cd ../..
perl -e 'alarm shift; exec @ARGV' 300 cargo test -p neo-fold-clean --release \
  --test system_lifecycle_finalization counter_overflow -- --nocapture
```
