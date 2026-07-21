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

  HyperNova's universal default pair can be replicated exactly as pointwise
  satisfying claim/witness vectors of any arity for every structure. Separately, Rust's
  RunningInstance::default() is a valid zero-arity product: its claim and
  witness lists are empty, no parent authority is attached, shape is valid, and
  the pointwise relation holds vacuously. The empty product does not realize or
  replace the paper's explicit u_perp entry.
assumptions:
  - Verifier-owned hash and step semantics, including emptyRunning,
    initialNebula, boundary/trace derivation, and fresh-link checker.
  - The paper supplies a universally satisfying default pair. Every nonempty
    replicated vector consumes that proof; zero-arity validity is proved
    independently and does not require the pair.
non_goals:
  - Generated R1CS soundness, concrete enc_inst refinement, or Rust control flow.
  - NIFS soundness, which is not invoked on the base branch.
  - Refinement of Rust canonical_zero to the replicated paper default vector,
    or alignment between the native empty sentinel and base-circuit output.
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
  - Nightstream.HyperNova.Construction2.Default.replicatedDefault_allPairs
  - Nightstream.HyperNova.Construction2.Default.emptyRunning_zeroArity
  - Nightstream.HyperNova.Construction2.Default.empty_claims_with_parent_rejected
axiom_report:
  fPrimeBase_sound uses [propext]; replicatedDefault_allPairs and
  emptyRunning_zeroArity use no axioms. All reports are guarded fail-closed
  in tests/Axioms.lean.
proof_hash:
  Step sha256:5f589bcc709a17a4b0a4ae9ac900dd5a87966162023cff3ece95731df2fcc6c3
  Default sha256:7d15f9d77cc714ff25a8f9e0ae945506ad30834f3fa016fe56c2b6009636b912
conformance_status:
  model-proved with the paper replicated vector and Rust zero-arity deviation stated
  separately, and with the standalone/local versus delayed-link boundary made
  explicit. No theorem identifies the empty product with the paper default.
  Rust canonical_zero is the nonempty Construction-2 base accumulator; its
  exact k-child/order/witness/parent refinement to the replicated paper default
  remains open, as does alignment with the native empty pre-chain sentinel. M4
  has exact base rows and base pins, but must still decode the retained checked-
  program assertions into BaseLocalHolds and derive OutgoingLinked from the next
  consumer or terminal-link rows. Full Rust state/branch/error refinement
  remains M5.
retest_commands:
  - cd formal/nightstream-lean && ./scripts/validate.sh build
  - cd formal/nightstream-lean && LEAN_BUILD_TARGET=tests.FPrimeStep ./scripts/validate.sh build
  - cd formal/nightstream-lean && LEAN_BUILD_TARGET=tests.Axioms.Protocol ./scripts/validate.sh build
  - perl -e 'alarm shift; exec @ARGV' 300 cargo test -p neo-fold-clean --release --test system_lifecycle_f_prime_link
```
