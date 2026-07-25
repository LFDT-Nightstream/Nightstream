# FOLD-NIFS-FS

```text
property_id: FOLD-NIFS-FS
claim:
  The exact typed paper NIFS satisfies both parts of frozen obligation 5
  under the permitted explicit random-oracle contract:

  - deterministic verifier acceptance implies the independent paper
    transition or one of the five named NIFS events;
  - every paper transition has an accepted proof; and
  - on the complete correlated prefix/post-prefix experiment,

      Pr[accepted] - nonInteractiveTotal <= Pr[transition].

  The combined theorem constructs both frozen headline propositions. Neither
  NifsSoundAndCompleteModulo nor NifsNonInteractiveSound is a premise.
transcript_schedule:
  - absorb the complete running/fresh public pair before any PiCCS coin;
  - squeeze alpha in canonical coordinate order, then the shared gamma;
  - absorb each finite polynomial message before its indexed SumCheck coin;
  - absorb the complete PiCCS output before PiRLC;
  - query PiRLC at the literal Fin (K+k) coordinate from the common
    post-output state.
exact_events:
  - accepted PiDEC target-witness extraction failure;
  - PiRLC fork-sampling failure;
  - PiCCS SumCheck collision;
  - PiCCS mixing-root collision;
  - parent-opening binding collision;
  - public-input binding collision;
  - transcript replay collision;
  - transcript final-state collision;
  - output-absorption collision;
  - challenge-sampling failure;
  - multi-fork programming failure.
derived_bounds:
  - challenge-sampling failure has probability zero on the selected support;
  - accepted multi-fork programming failure is bounded by
    (ell + 1) / |C|;
  - the remaining nine exact event bounds compose in the frozen,
    right-nested loss order.
assumptions:
  - paper extraction algebra and strong-sampling-set laws;
  - one exact accepted target-witness extraction-event bound;
  - four exact interactive-event bounds;
  - four exact typed transcript-collision bounds.
non_goals:
  - Poseidon2 input encoding or collision reduction.
  - Construction of a concrete production random-oracle distribution.
  - The bounded production PiRLC rejection sampler, bias, or termination.
  - Concrete Goldilocks/extension arithmetic certificates.
  - Rust, R1CS, IR, encoding, physical rows, columns, or costs.
paper_sources:
  - docs/hypernova-paper/08_3_Multi_folding_schemes.md, Definition 7,
    Definition 10, and Lemma 1.
  - docs/hypernova-paper/26_B_Achieving_non_interactivity_for_multi_folding_schemes.md,
    Construction 3.
  - docs/superneo-paper/06-6-strong-and-weak-interactive-reductions.md.
  - docs/superneo-paper/13-d-deferred-theorems-and-proofs.md,
    Appendices D.3--D.6.
lean_theorems:
  - Frozen.SuperNeo.paperNifsSoundCompleteAndNonInteractive
  - Frozen.SuperNeo.piCcsExecution_coins_eq_replayInput
  - Frozen.SuperNeo.piCcsExecution_outgoingState_eq_postOutput
  - Frozen.SuperNeo.piRlcChallenge_eq_response_after_piCcsOutput
  - PaperNonInteractive.fullOracleAccepted_implies_transition_or_failure
  - PaperNonInteractive.fullOracleMixtureExplicitRandomOracleContract
  - PaperNonInteractive.fullOracleMixtureAccepted_probability_sub_total_le_transition
axiom_report:
  The headline guard permits exactly propext, Classical.choice, and
  Quot.sound. It admits no sorryAx, new axiom, Lean.trustCompiler, or unsafe
  declaration.
conformance_status:
  model-proved. This closes the M2 random-oracle-model theorem. Concrete
  Poseidon2 and production-sampler refinement remain outside M2 and are not
  claimed.
retest_commands:
  - cd formal/nightstream-lean && LEAN_TIMEOUT_SECONDS=900
      LEAN_BUILD_TARGET=tests.NifsPaperExplicitRandomOracleSecurity
      ./scripts/validate.sh build
  - cd formal/nightstream-lean && LEAN_TIMEOUT_SECONDS=900
      LEAN_BUILD_TARGET=tests.Axioms.NifsPaperExplicitRandomOracleSecurity
      ./scripts/validate.sh build
  - cd formal/nightstream-lean && LEAN_TIMEOUT_SECONDS=900
      ./scripts/validate.sh axioms
```
