# TRACE-VALID

```text
property_id: TRACE-VALID
claim:
  AcceptedTrace retains the exact input and proof for every Boolean-accepted F'
  invocation. From a true initial state, accepted_trace_sound establishes:
  (1) semantic reachability by exactly schedule.length rich F' edges;
  (2) every schedule entry is nonzero;
  (3) final chunkCount equals the number of invocations;
  (4) final stepCount equals the sum of installed batch cardinalities; and
  (5) every nonempty trace ends in a verifier-pinned authority state.
  accepted_trace_valid_execution connects this exact reachability directly to
  the project's top-level ValidExecution predicate for any checked terminal
  predicate.
assumptions:
  - The initial state satisfies Step.InitialState.
  - Every trace edge carries an invocation for which the closed Step.check
    returns true. A circuit proof must obtain this by combining standalone
    LocalHolds with the next-step or terminal OutgoingLinked rows.
non_goals:
  - Terminal CE/decider validity, Rust trace replay, circuit soundness, encoding
    refinement, bad-event probability, or Fiat-Shamir.
paper_sources:
  - HyperNova Construction 2 compiler and repeated F' execution.
rust_surfaces:
  - crates/neo-fold-clean/src/lifecycle/prove.rs
  - crates/neo-fold-clean/src/lifecycle/verify.rs
  - crates/neo-fold-clean/src/paper/f_prime/native.rs
circuit_or_encoding_artifacts:
  - none; the trace consumes the M3 model relation, not generated rows.
failure_class:
  A digest-only chain omits step evidence, a trace skips/duplicates an edge,
  accepts a zero-sized batch, misstates either counter, or reaches an unpinned
  terminal state.
counterexample_or_witness:
  tests/FPrimeStep.lean constructs a two-invocation schedule [2,1], proves exact
  two-step reachability and counters (2,3), and rejects a forged terminal
  stepCount. All step-level forgery witnesses feed the same trace relation.
lean_theorems:
  - Nightstream.Assurance.FPrimeTrace.accepted_trace_reachable
  - Nightstream.Assurance.FPrimeTrace.accepted_trace_counter_refines
  - Nightstream.Assurance.FPrimeTrace.accepted_trace_final_pinned
  - Nightstream.Assurance.FPrimeTrace.accepted_trace_sound
  - Nightstream.Assurance.FPrimeTrace.accepted_trace_valid_execution
axiom_report:
  accepted_trace_sound and accepted_trace_valid_execution use only [propext],
  guarded fail-closed in tests/Axioms.lean.
proof_hash:
  sha256:bd52d45b433ab4b61fe584a2017dde329b4332af86db6f181ff08b2dc749a30f
conformance_status:
  model-proved; lifecycle Rust refinement and terminal closure remain M5/M6.
  The 2026-07-10 Rust regression run was blocked before tests by pre-existing
  non-exhaustive matches for CcsMatrix::CscWithSeededPhi81 in six unrelated
  sites.
retest_commands:
  - cd formal/nightstream-lean && lake build && lake exe check
  - cd formal/nightstream-lean && lake build tests.FPrimeStep tests.Axioms
  - perl -e 'alarm shift; exec @ARGV' 300 cargo test -p neo-fold-clean --release --test system_lifecycle_f_prime_link
```
