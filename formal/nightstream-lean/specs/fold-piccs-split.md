# FOLD-PICCS-SPLIT

```text
property_id: FOLD-PICCS-SPLIT
claim:
  The production-shaped FE/NC split is a registered model-level refinement of
  the paper Pi_CCS relation. Its authoritative input reconstructs the public
  input and source product from canonical openings. Its prover certificate
  contains only FE and NC messages plus downstream payloads; the verifier
  derives both challenge points, materializes y_ring and packed y_zcol, and
  starts NC from the exact FE final transcript state.

  Accepted verification therefore implies the independently stated paper
  relation or an exact FE/NC algebraic event. The block/lane NC residual family
  is equivalent to the paper norm relation, every carrier coordinate has one
  canonical block/lane owner, and the delayed packed projection equals the
  radix recomposition of all authoritative raw running assignments.

  In the ideal-interactive experiment, each FE message is fixed by the prior
  FE challenge prefix. The NC strategy may depend on the completed FE word,
  but each NC message is fixed by the prior NC prefix. Finite root counting,
  successive-coordinate sampling, exact transport of the repository
  FixedPhase.BadChallenge events, and an explicit two-phase union bound prove

      (feRoundCount * Drow) / |C| + (25 * 4) / |C|.

  The existing SumCheckSoundnessContract is not a premise of this production
  theorem. Alpha/gamma mixing events and Fiat-Shamir remain separate.
assumptions:
  - The paper field and ring laws, including no zero divisors, for the
    deterministic norm reduction and finite root-counting theorem.
  - A finite nonempty duplicate-free ideal challenge support whose cardinality
    equals the verifier-owned challengeSetSize.
  - Honest completeness at recursive boundaries uses the registered delayed
    lifecycle premise: the complete pending vector is the radix recomposition
    of the authoritative raw running assignments at the owned old block.
non_goals:
  - A probability bound for FE or NC alpha/gamma mixing-root events.
  - Fiat-Shamir, Poseidon2, random-oracle programming, or transcript-collision
    security.
  - A concrete proof of the Goldilocks-extension no-zero-divisors instance or
    a binding to the production challenge alphabet.
  - Rust, R1CS, generated-artifact, row, column, IR, cost, or minimality
    refinement.
  - Message or transcript identity between production's two SumChecks and the
    paper's displayed one-joint polynomial.
paper_sources:
  - docs/superneo-paper/07-7-neo-s-folding-scheme-for-ccs.md, Section 7.3
    Pi_CCS verifier, output relation, and challenge order.
  - docs/superneo-paper/13-d-deferred-theorems-and-proofs.md, Appendix D.4
    fixed-witness SumCheck and algebraic-loss analysis.
rust_surfaces:
  - crates/neo-fold-clean/src/paper/reductions/pi_ccs_split_nc_circuit/**
  - These paths are mapped for later refinement only; no Rust-conformance claim
    is made by this property.
circuit_or_encoding_artifacts:
  - none. The result is artifact-independent and authorizes no row generation,
    certification, or removal.
failure_class:
  A production certificate supplies an unbound public input, output, restart
  state, or challenge record; omits or duplicates a block/lane coordinate; or
  chooses a round polynomial after seeing its current challenge, allowing an
  invalid paper transition to pass outside the named algebraic events.
counterexample_or_witness:
  The SUM-DEGREE-WIDTH degree-six finite-support countermodel remains the
  necessity witness for exact message width. Here the typed causal strategy
  makes the current/future challenge unavailable, and the exact physical
  certificate-equality and BadChallenge transport theorems prevent replacing
  the submitted FE/NC messages with a proof-only certificate. Transcript
  reset/fork and verifier-output substitution are uninhabited by construction.
lean_theorems:
  - ProductionRefinement.accepted_implies_paper_or_algebraic_failure
  - ProductionRefinement.not_transcriptFailure
  - ProductionRefinement.not_bindingFailure
  - ProductionRefinement.blockLaneCombinedNc_refines_paperNc
  - ProductionRefinement.everyCoordinate_has_exact_owner
  - ProductionRefinement.delayedProjection_refines_rawRecomposition
  - ProductionRefinement.honest_complete_with_output
  - ProductionRefinement.accepted_output_suitable_for_piRlc
  - DelayedPackedYZcol.Lifecycle.Trace.base_owns_no_predecessor
  - DelayedPackedYZcol.Lifecycle.Trace.edge_owns_production_and_consumption
  - DelayedPackedYZcol.Lifecycle.Trace.terminal_owns_discharge
  - DelayedPackedYZcol.Lifecycle.Trace.terminalCount_eq_one
  - DelayedPackedYZcol.Lifecycle.Trace.closedTrace_reduces_to_paper_transitions_or_named_failure
  - CausalFixedPhase.detects_probability_le
  - CausalFixedPhase.badChallenge_implies_detects
  - BlockLane.CausalSoundness.fe_uniformRounds_eq_generated
  - BlockLane.CausalSoundness.fe_roundCollision_implies_detects
  - BlockLane.CausalSoundness.nc_roundCollision_implies_detects
  - BlockLane.CausalSoundness.split_detects_probability_le
  - BlockLaneCombinedNc.CausalSoundness.rawRoundRepresentable
  - BlockLaneCombinedNc.CausalSoundness.splitCollision_implies_detects
  - BlockLaneCombinedNc.CausalSoundness.splitCollision_probability_le
  - ProductionMixingBoundary.goldilocksBaseNoZeroDivisors
  - ProductionMixingBoundary.productionExtensionNoZeroDivisors
  - ProductionMixingBoundary.challengeSetSize_pos_of_aligned
  - ProductionMixingBoundary.zeroChallengeSetSize_has_no_aligned_support
  - ProductionMixingBoundary.replaceCoreChallenges_same_state
  - ProductionMixingBoundary.replaceCoreChallenges_different_challenges
  - ProductionMixingBoundary.derivePreSumcheck_constantCore_shared_gamma
  - ProductionMixingBoundary.feFailure_exact_cases
  - ProductionMixingBoundary.ncFailure_exact_cases
production_mixing_boundary:
  status: kernel-checked carrier obstruction; no production mixing theorem is
    exported from the current interfaces.
  exact_events:
    - FE is exhausted only through ProductionRefinement.FeFailure.sumcheck,
      with the exact nested SumCheck.Fe.BadEvent.mixingRoot and
      SumCheck.Fe.BadEvent.roundCollision constructors.
    - NC is exhausted in the frozen right-nested order laneSelectorRoot,
      blockSelectorRoot, gammaPolynomialRoot, residualWeightRoot,
      roundCollision, without reassociation.
  algebra_boundary:
    - NormRange.GoldilocksModulusEuclid implies the exact production base-field
      no-zero-divisor property.
    - Together with ConcreteCarrier.SevenProjectiveNonresidue it implies the
      exact production extension-field no-zero-divisor property consumed by
      FiniteRootCounting.roots_count_le_degree.
    - The current tree supplies these as explicit arithmetic premises; this
      boundary does not invent closed certificates for them.
  support_obstruction:
    - FiniteUniform.Support carries a duplicate-free, nonempty value list, so
      its cardinality is positive.
    - The production Context carrier stores an unrestricted Nat
      challengeSetSize. Updating only that field to zero preserves the complete
      statement, prior state, transcript schedule, FE coins, and NC coins, but
      makes exact denominator/support alignment impossible.
    - This obstructs derivation from the carrier alone; it does not claim that
      a selected future production constructor must set the denominator to zero.
  schedule_obstruction:
    - The five core values alpha, betaA, betaR, gamma, betaBlock are returned by
      one opaque deriveCore call. The carrier states no internal sampling order,
      finite support, uniform law, or causal visibility boundary between them.
    - Arbitrary replacement, including maximal correlation of every coordinate
      with the shared gamma, preserves the deriveCore state, both explicitly
      delayed squeezes producerBeta then batchWeight, and the state entering FE.
    - betaA and gamma remain single shared FE/NC projections; no independence
      premise or independent FE/NC strategy is introduced.
    - This obstructs derivation from the current Schedule carrier; a refined
      schedule with explicit ordered sampling can discharge it.
  consequence:
    The existing causal splitCollision_probability_le theorem remains intact
    and available once an actual nonempty sampled support aligned with
    challengeSetSize is supplied. It cannot be combined with finite-root bounds
    for the named production mixing events from the present carriers alone.
    Fiat--Shamir remains separate.
axiom_report:
  Every original headline theorem remains guarded fail-closed in
  tests/Axioms/PiCcsSplitNcCausalSoundness.lean. The production mixing boundary
  is separately guarded in
  tests/Axioms/PiCcsSplitNcProductionMixingBoundary.lean. The largest dependency
  set is exactly [propext, Classical.choice, Quot.sound]. There is no sorryAx,
  new axiom, unsafe declaration, or Lean.trustCompiler dependency.
proof_hash:
  Recorded from the final task-owned Lean sources in the evidence ledger.
conformance_status:
  model-proved registered-deviation refinement. This closes deterministic
  Split-NC-to-paper reduction, honest completeness, delayed lifecycle
  ownership, and ideal-interactive FE/NC round-collision probability. It does
  not close mixing-root probability, Fiat-Shamir, concrete field/support, or
  production implementation refinement.
retest_commands:
  - cd formal/nightstream-lean && LEAN_TIMEOUT_SECONDS=900
      LEAN_BUILD_TARGET=tests.PiCcsSplitNcCausalSoundness
      ./scripts/validate.sh build
  - cd formal/nightstream-lean && LEAN_TIMEOUT_SECONDS=900
      LEAN_BUILD_TARGET=tests.Axioms.PiCcsSplitNcCausalSoundness
      ./scripts/validate.sh build
  - cd formal/nightstream-lean && LEAN_TIMEOUT_SECONDS=900
      LEAN_BUILD_TARGET=tests.PiCcsSplitNcProductionMixingBoundary
      ./scripts/validate.sh build
  - cd formal/nightstream-lean && LEAN_TIMEOUT_SECONDS=900
      LEAN_BUILD_TARGET=tests.Axioms.PiCcsSplitNcProductionMixingBoundary
      ./scripts/validate.sh build
  - cd formal/nightstream-lean && LEAN_TIMEOUT_SECONDS=900
      LEAN_BUILD_TARGET=Nightstream.Protocol.FPrime.Frozen
      ./scripts/validate.sh build
  - cd formal/nightstream-lean && ./scripts/validate.sh static
```
