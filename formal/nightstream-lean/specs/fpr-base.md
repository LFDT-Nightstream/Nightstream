# FPR-BASE / FPR-BASE-SPEC

```text
property_ids: FPR-BASE, FPR-BASE-SPEC
claim:
  The NoFold branch is available only from the true verifier-owned initial
  state: pc=1, zero counters, z0=zi=the derived initial boundary, the derived
  public-trace seed, the advertised initial semantic state, the empty running
  digest, the configured initial Nebula lane, and the Initial tag. Stateless
  mode starts from the empty accumulator; stateful mode starts from the
  advertised initial semantic state. A valid standalone step installs a
  nonempty fresh batch, checks semantic/Nebula advance, deterministically
  advances state, and recomputes x_out. The installed batch is linked to that
  output by the next recursive consumer, or by the terminal fold when it is
  the trailing batch. BaseLocalHolds owns the producer obligations;
  BaseHolds additionally owns this delayed outgoing link.

  Rust's RunningInstance::default() is a valid zero-arity product: its claim and
  witness lists are empty, no parent authority is attached, shape is valid, and
  the pointwise relation holds vacuously. This does not claim that an omitted
  entry equals the paper's explicit u_perp pair.
assumptions:
  - Verifier-owned hash and step semantics, including emptyRunning,
    initialNebula, boundary/trace derivation, and fresh-link checker.
  - The paper supplies a default satisfying pair; zero-arity validity itself is
    proved without inspecting or assuming a local accepted-implies-valid field.
non_goals:
  - Generated R1CS soundness, concrete enc_inst refinement, or Rust control flow.
  - NIFS soundness, which is not invoked on the base branch.
paper_sources:
  - HyperNova Construction 2 steps 3.1-3.2 and Definition 12.4 default instance.
rust_surfaces:
  - crates/neo-fold-clean/src/paper/construction2/state.rs (State::base)
  - crates/neo-fold-clean/src/paper/construction2/running.rs
  - crates/neo-fold-clean/src/paper/construction2/transition.rs
    (state_base_case_check, advance_state)
  - crates/neo-fold-clean/src/paper/f_prime/native.rs (Initial/NoFold branch)
circuit_or_encoding_artifacts:
  - CIR-FPR-BASE-PROGRAM covers all 12,498 exact plain/stateless base-step
    rows as a sound and complete checked program.
  - CIR-FPR-BASE-PINS covers verifier-owned full-history base constants.
failure_class:
  A noninitial state takes NoFold, an initial authority coordinate is relabeled,
  an empty/malformed fresh batch is installed, the semantic or Nebula transition
  is skipped, or an empty running product carries parent authority.
counterexample_or_witness:
  tests/FPrimeStep.lean rejects forged z0, public-trace seed, empty accumulator,
  initial Nebula, empty fresh batch, wrong fold variant, semantic digest, x_out,
  and output link; it also rejects parent authority on the empty product.
lean_theorems:
  - Nightstream.Protocol.FPrime.Step.checkLocal_eq_true_iff_localHolds
  - Nightstream.Protocol.FPrime.Step.fPrimeBaseLocal_sound
  - Nightstream.Protocol.FPrime.Step.holds_iff_local_and_outgoing
  - Nightstream.Protocol.FPrime.Step.closeLocal
  - Nightstream.Protocol.FPrime.Step.fPrimeBase_sound
  - Nightstream.HyperNova.Construction2.Default.emptyRunning_realizes_default
  - Nightstream.HyperNova.Construction2.Default.empty_claims_with_parent_rejected
axiom_report:
  fPrimeBase_sound uses [propext]; emptyRunning_realizes_default uses no axioms.
  Both reports are guarded fail-closed in tests/Axioms.lean.
proof_hash:
  Step sha256:4066afda436f1182ed3e6729c3d94c2575afda3d5fd6038040e48312fb61bf2a
  Default sha256:7a6302d666ce6fd6fa6c060a59d4645b14d66f9a6afe120f01a68de5907aa981
conformance_status:
  model-proved with the standalone/local versus delayed-link boundary made
  explicit. M4 has exact base rows and base pins, but must still decode the
  retained checked-program assertions into BaseLocalHolds and derive
  OutgoingLinked from the next consumer or terminal-link rows. Full Rust
  state/branch/error refinement remains M5.
retest_commands:
  - cd formal/nightstream-lean && lake build && lake exe check
  - cd formal/nightstream-lean && lake build tests.FPrimeStep tests.Axioms
  - perl -e 'alarm shift; exec @ARGV' 300 cargo test -p neo-fold-clean --release --test system_lifecycle_f_prime_link
```
