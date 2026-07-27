# F-prime paper deviations

This register records protocol behavior that is not literally part of the
SuperNeo or HyperNova constructions cited by the frozen F-prime specification.
A deviation is part of the specified protocol only after a kernel-checked
paper-layer theorem reduces its complete verifier transition to the cited
paper relation. Semantic rearrangement, implementation correspondence, or a
matching digest is not sufficient.

## FPR-DEV-PIRLC-AMBIENT-STRICT-HALF

- Paper boundary: SuperNeo Definition 12 and Appendix D.5.
- Literal conflict: Definition 12 uses the strict predicate
  `||z||_infinity < bound`, while Appendix D.5 claims every field element is
  covered at `bound = q / 2`. For odd `q`, the midpoint residues have centered
  magnitude exactly `floor(q / 2)` and are excluded.
- Frozen correction: retain the strict relation and use
  `floor(q / 2) + 1` for the ambient extraction relation.
- Kernel evidence:
  `PiRLC.PaperCorrections.midpointResidue_not_literalAmbientBounded` and
  `PiRLC.PaperCorrections.all_centeredMagnitude_lt_correctedAmbientBound`.
- Classification: paper-level correction. The corrected bound must be used by
  the paper-authoritative Pi_RLC weak reduction and later implementation
  refinement.

## FPR-DEV-PIRLC-COORDINATE-FORK-LOSS

- Paper boundary: SuperNeo Appendix C, Theorem 10, and Appendix D.5.
- Literal conflict: the local Theorem 10 text renders the coordinate-wise
  extraction loss as `ell / |C^ell|`, while Appendix D.5 applies the theorem
  with loss `(ell + 1) / |C|` for `ell = K + k`.
- Kernel obstruction: with `|C| = 3` and `ell = 2`,
  `RenderedBoundObstruction.rendered_denominator_bound_counterexample` proves
  that an adversary can accept exactly `3 / 9` challenges while no accepted
  coordinate fork exists. The rendered loss `2 / 9` would therefore leave a
  strictly positive claimed lower bound, disproving that denominator.
- Selected conservative correction: use Appendix D.5's numerator and
  denominator, `(ell + 1) / |C|`. The cited coordinate-fork lemma actually
  gives the sharper `ell / |C|` loss for the two-special case, so the D.5
  expression is conservative rather than tight. The model-level definition
  `PiRLC.samplingErrorNumerator ell = ell + 1` records only the numerator; it
  is not evidence for the probability bound.
- Kernel closure:
  `PiRLC.PaperWeakFiniteUniform.paperWeak` proves the operational
  finite-uniform weak reduction with the selected `(ell + 1) / |C|` loss,
  derives it from the sharper `ell / |C|` coordinate-fork bound, and proves
  the extractor query bound. Relaxed binding remains the only explicit
  witness-uniqueness security premise.
- Classification: model-proved paper-level correction for the finite-uniform
  game. This does not by itself instantiate Fiat--Shamir or relaxed binding.

## FPR-DEV-PICCS-FIRST-SUCCESS-CONDITIONING

- Paper boundary: SuperNeo Definition 10 and Appendix D.4, extractor Items
  2--8.
- Quantitative conflict: Appendix D.4 rejection-samples the first ambient
  success before drawing a fresh second execution. If ambient success has
  probability `p`, a raw two-run witness-disagreement probability `delta`
  becomes `delta / p` in the first-success-conditioned extractor experiment.
  The qualitative paper argument remains valid because `p` is non-negligible,
  but an exact theorem cannot subtract the unchanged raw `delta`.
- Kernel obstruction:
  `StrongConditioningObstruction.unchanged_raw_uniqueness_budget_counterexample`
  exhibits four uniform executions with ambient success `2/4`, raw
  disagreement `2/16`, and conditioned disagreement `2/8`.
- Finite kernel closure:
  `PiCCS.PaperJoint.StrongExecution.FinitePaperStrong.finitePaperStrong`
  proves the literal finite operational rejection-adjusted strong theorem.
  It fixes the first witness before the fresh alpha/gamma execution and
  charges `rawMismatchBudget / successFloor` exactly once, under separately
  named mixing-root and SumCheck contracts.
  `PaperNonInteractive.CausalFirstSuccessBridge` reindexes that operational
  experiment into the exact causal-prefix × target-extractor seed, proves the
  filtered-first × fresh-second support membership theorem, identifies the
  fixed-first bad event and extracted source with typed NIFS events, and
  transports the full finite extraction inequality without changing its
  premises or budgets.
- Exact finite boundary:
  `FinitePaperStrongBoundary.extractorRuntime_iff_all_finite_cutoffs`
  proves that the current strong game's runtime field is literally the family
  of finite-cutoff bounds; it is not an unbounded expectation.
- Frozen-target obstruction:
  `Frozen.PiCcsAsymptoticObstruction.piCcsStrong_iff_runtime` holds every
  probability, error-budget, and composition coordinate fixed while making
  the exact frozen target equivalent to an arbitrary runtime proposition.
  `frozenTarget_without_samplerLink_countermodel` and
  `not_attemptedBridgeWithoutSamplerLink` prove that no retry-termination
  result follows from an arbitrary game until it carries an operational
  sampler link.
- Unbounded kernel closure:
  `FiniteUniform.FirstSuccessTrace` supplies the specialized countable
  terminating-trace law over the finite one-run alphabet. It proves exact
  finite-prefix mass conservation, residual tail
  `failureProbability^attemptCount`, rational-epsilon tail vanishing,
  normalization, and uniqueness of the event law satisfying the operational
  retry equation. That law is proved equal to the Cartesian product of the
  success-filtered first support and the original unconditioned fresh-second
  support. Its fresh-second marginal is therefore exactly the one-run law.
  The expected retry count is the unique finite solution of
  `E = 1 + failureProbability * E`, namely `1 / successProbability`.
  `FirstSuccessRuntime` owns per-seed run costs, proves every terminating
  trace costs at most its execution count times the one-run bound, and derives
  the exact first-step expectation from the finite one-run mean. Total
  expected work is exactly
  `oneRunExpectedWork * (1 / successProbability + 1)` and is bounded by
  `runCostBound * (1 / successFloor + 1)`. Almost-sure termination and
  polynomiality follow from the positive inverse-polynomial success floor and
  polynomial one-run work.
  `AsymptoticPaperStrong.paperStrong` links that law and runtime family
  definitionally to the PiCCS strong game and retains the intrinsic budget in
  frozen `SumCheck + Schwartz--Zippel` order while charging the raw mismatch
  term as `delta / mu` exactly once.
  `Frozen.PiCcsFirstSuccessBridge.piCcsStrong_of_unboundedFirstSuccess`
  constructs the exact `SuperNeoGames` carrier and proves the frozen
  `PiCcsStrong family.games` target. Almost-sure termination, EPT,
  conditioned-law equality, fresh-second independence, and extraction are
  conclusions rather than premises of that theorem.
- Classification: model-proved paper-level correction and unbounded
  finite-alphabet first-success bridge. The earlier opacity countermodel
  remains the exact boundary for arbitrary unlinked `StrongGame` values. This
  result does not instantiate Fiat--Shamir, Poseidon2, Ajtai, or either named
  fixed-witness algebraic security contract.

## FPR-SPEC-SUPERNEO-COMPOSITION-LINKAGE

- Paper boundary: SuperNeo Theorem 6 and Appendix D.3 require the composed
  execution to run the weak extractor and then the strong extractor over the
  same intermediate execution; Definition 5 then composes that reduction with
  `Pi_DEC`.
- Frozen-spec defect: the original `SuperNeoGames` facade stored the final
  knowledge game as an independent caller field. Component strength,
  weakness, shared-projection equality, and the `Pi_DEC` theorem therefore did
  not constrain the advertised final game.
- Kernel obstruction:
  `Frozen.CompositionLinkageObstruction.unlinked_fields_countermodel`
  instantiates all four component premises while selecting an incomplete final
  game.
- Frozen correction: `SuperNeoGames` now stores the exact strong–weak and
  `Pi_DEC` operational couplings. `strongWeakKnowledgeGame` and
  `superNeoCompositionGame` compute both composed games definitionally, and
  `superNeoPaperObligations_of_components` derives the final theorem from the
  three component reductions plus literal shared-projection equality.
- Quantitative boundary: `InteractiveErrorBudget.total` follows the theorem's
  syntactic composition order and retains `Pi_DEC`'s explicit zero loss.
  `NifsExtractionErrorBudget` separately owns failure to obtain the target
  child witnesses required before Theorem 7 applies; `nifsInteractiveTotal`
  charges that event once before the four nonzero interactive terms.
- Classification: proved specification repair. It changes no paper semantics,
  protocol messages, implementation behavior, or encoding.

## FPR-SPEC-FIAT-SHAMIR-ORACLE-GAP

- Paper boundary: the cited SuperNeo reductions are public-coin protocols;
  the production noninteractive verifier must derive their coins through the
  explicitly assumed random oracle, with the exact transcript input, order,
  labels, and phase handoff.
- Current model boundary: `PaperNonInteractive.Key.absorbPublicInput` receives
  the complete running/fresh public pair before any challenge;
  `PiCCS.PaperJoint.FiatShamir.Oracle` and `ProtocolVerifier.Oracle` then fix
  the typed alpha/gamma/SumCheck schedule and post-output handoff; and the
  literal `Fin (K+k)` index is passed to each `PiRLC` response.  The
  `RandomOracleBoundary` equations kernel-check that order and coordinate
  alignment.  These abstract functions deliberately impose no binding or
  distribution law.
- Kernel obstruction:
  `Frozen.NonInteractiveOracleObstruction.distinct_contexts_same_derived`
  constructs a typed oracle for which two distinct public contexts derive
  identical complete coin views, while
  `distinct_labels_same_squeeze` shows that distinct domain labels can also be
  ignored and `distinct_public_inputs_same_bound_state` shows that the new
  full-public-input absorption call can still discard both public inputs.
- Named boundary: `PublicInputBindingCollision`,
  `TranscriptReplayCollision`, `TranscriptStateCollision`,
  `OutputAbsorptionCollision`, and `PiRlcSamplingSetFailure` remain distinct.
  The ideal paper key proves that the last cannot occur; a bounded production
  sampler must reject or expose its separate shortfall.
- Quantitative target: `FiatShamirContract.ExplicitRandomOracleContract`
  fixes six per-event probability bounds, and
  `anyFailure_probability_le_total` proves their exact nested union is bounded
  by `FiatShamirErrorBudget.total`.  This is a frozen contract, not an
  instantiation or security proof.
- Structural fork bridge:
  `PaperNonInteractive.CoordinateForkBridge` constructs the literal PiRLC
  context and coefficient-complete batch reached from the typed NIFS prefix.
  `nifsEventPredicates` instantiates all six predicates on an aligned fork
  outcome, while `acceptedFork_implies_ambientTargetOpenings` proves that an
  accepted coordinate fork extracts exactly the ambient witness consumed by
  deterministic NIFS soundness.
  `verify_sound_or_residual_or_multiFork` therefore yields the paper
  transition, one of five closed residual interactive events, or the concrete
  programming-failure predicate.  A programmed but rejected coordinate fork
  is owned only by the residual PiRLC fork-sampling event; it is not charged
  again to Fiat--Shamir.  The theorem assumes no probability bound.
- Conditional soundness theorem:
  `PaperNonInteractive.OracleSoundness` gives the PiDEC child-witness failure
  its exact `NifsExtractionErrorBudget` owner, gives the other four residual
  predicates their exact `InteractiveErrorBudget` owners, and specializes the
  six-event oracle contract to `AlignedForkOutcome`.
  `accepted_probability_sub_total_le_transition` kernel-checks the resulting
  subtractive soundness inequality with error exactly
  `nonInteractiveTotal`.  None of its contracts contains acceptance, the
  desired transition, or the final inequality.
- Frozen obligation repair:
  `Frozen.Obligations.NifsSoundAndCompleteModulo` is now explicitly the
  deterministic pointwise core, not the full non-interactive claim.
  `Frozen.Obligations.NifsNonInteractiveSound` separately freezes the exact
  subtractive probability target instantiated by the conditional theorem.
- Continuation-alignment obstruction:
  `AlignedForkOutcome.batchAligned` fixes the PiRLC batch reached after PiCCS,
  but does not bind its adversary oracle to the continuation of the prover
  that produced the NIFS proof.
  `Frozen.NonInteractiveContinuationObstruction.distinct_replacement_oracles_same_nifs_execution`
  kernel-checks this gap by replacing only that oracle with two distinct
  constant response functions while preserving both executable acceptance
  and the independent paper-transition predicate.
- Structural repair:
  `PaperNonInteractive.RewindableContinuation` splits one malicious message
  into a fixed PiCCS prefix and a vector-at-once PiDEC continuation. Its
  projection defines the PiRLC assignment oracle as the straight-line PiDEC
  recomposition of that same continuation; there is no alignment proposition
  to assume. `continuationSuccesses_imply_acceptedFork` proves that successful
  PiDEC continuations at the programmed base and coordinate forks yield the
  literal Appendix-D.5 accepted fork.
  `PaperNonInteractive.RewindableOracleSoundness` then pulls the exact
  eleven-event theorem back to experiments on this owned carrier.
  `rewindable_accepted_probability_sub_total_le_transition` has the frozen
  subtractive target, while
  `piRlcForkSamplingFailure_implies_piDecContinuationFailure` proves that the
  existing fork-sampling event already owns any failed base/fork PiDEC
  continuation; no extra catch-all event is introduced.
- Fixed-key carrier obstruction:
  `Frozen.NonInteractiveFixedKeyObstruction.programmingReceipts_force_same_base`
  proves that two valid programming receipts under one realized key and fixed
  prefix force identical PiRLC base vectors.
  `distinct_bases_force_programming_failure` therefore proves that a
  non-degenerate Appendix-D.5 experiment cannot pretend one fixed
  `Key.piRlcResponse` represents every oracle realization.  The experiment
  must vary a dependent oracle world.
- Conditional Appendix-D.5 experiment:
  `PaperNonInteractive.PostPrefixOracleWorld` reprograms exactly the PiRLC
  vector query after one fixed PiCCS prefix and carries the resulting key and
  continuation together in `RewindablePiRlcWorldOutcome`.
  `PostPrefixForkExperiment` is the deterministic pushforward of the finite
  uniform coordinate-fork seed experiment, preserving seed multiplicity.  Its
  executable acceptance rejects vectors outside the finite alphabet, and its
  accepted programming-failure event is exactly “base accepted and no
  coordinate programming receipt”; rejecting bases are not charged.
  Kernel-checked results give expected query count at most `ell + 1`, the
  sharp programming loss `ell / |C|`, and Appendix D.5's selected
  `(ell + 1) / |C|` loss.
  `PostPrefixWorldSoundness.postPrefixAccepted_probability_sub_total_le_transition`
  composes that actual finite experiment with the eleven exact events.  It
  proves zero ideal strong-set sampling loss and fixes the programming budget
  to `(ell + 1) / |C|`; only one target-witness extraction bound, four
  interactive-reduction bounds, and four exact transcript-collision bounds
  remain premises.
- Global finite experiment:
  `PaperNonInteractive.PiCcsPrefixOracleWorld` owns the initial transcript,
  complete public-input absorption, and preceding PiCCS oracle realization.
  `PiCcsPrefixExperiment` places that world, the malicious prefix and
  continuation, public inputs, and adversary randomness in one dependent
  finite seed carrier.  `FullOracleExperiment.fullOracleForkMixture` then
  composes each realized prefix with the exact post-prefix Appendix-D.5
  experiment without flattening away seed multiplicity.
  `FullOracleSoundness.fullOracleMixtureAccepted_probability_sub_total_le_transition`
  proves the global subtractive theorem, with zero strong-set sampling loss
  and the paper programming loss internal.  Its only premises are the five
  exact interactive-event bounds and four exact transcript-collision bounds.
  This is a theorem for an explicit finite correlated support; it does not
  assert that an arbitrary supplied support realizes an ideal random oracle.
- Interactive-composition alignment:
  `PaperNonInteractive.InteractiveCompositionBridge` constructs the exact
  `StrongExecution.Context`, `PiRlcBatch.CompatibleContext`, and compatible
  `PiDEC` context selected by one NIFS key and public statement.
  `compatibleContext_piRlc` and `compatiblePiDecContext_paper` prove literal
  identity with the NIFS contexts.  Given an explicit
  `CausalPrefixAlignment` receipt,
  `batchOfPrefix_eq_nifsPiRlcBatch` and
  `combinedParent_eq_nifsParent` prove that the causal interactive execution
  reaches exactly the NIFS coefficient-complete batch and parent.  The
  receipt is not an assumption about oracle security: the missing ideal-RO
  coupling must construct it from a fixed prover seed and independent
  verifier coins.
- Explicit causal coupling contract:
  `PaperNonInteractive.CausalPrefixCouplingContract` fixes one public
  statement, stores one sequential interactive
  `Pi_DEC ∘ Pi_RLC ∘ Pi_CCS` adversary, and samples the exact Cartesian
  product of that adversary's prover support with
  `VerifierCoins.support`.  Every realized oracle world must prove that its
  NIFS replay equals the causal interactive prefix and that its continuation
  equals the same adversary's `Pi_DEC` reply.
  `support_eq_product`, `mem_support_iff`, and `support_cardinality` prove the
  seed factorization, while `toPrefixExperiment_prefixAligned` and
  `toPrefixExperiment_batch_eq` show that conversion to the global experiment
  preserves the causal receipt and exact intermediate batch.  This freezes
  the required random-oracle contract shape; it does not construct an ideal
  oracle or prove that production Poseidon2 satisfies the contract.
- Post-prefix D.6 execution bridge:
  `PaperNonInteractive.CausalPostPrefixBridge` pushes one causal product seed
  through the actual post-prefix programming experiment.
  `interactivePiDecExecution_eq_postPrefix` identifies the complete
  interactive `Pi_DEC` execution with the continuation execution in that
  programmed world, while
  `piDecExecutionAt_world_attempt_eq_nifs` identifies its public attempt with
  the attempt checked by the actual NIFS verifier.
  `continuationSuccessAt_world_iff_nifs_target` and
  `interactivePiDecSuccess_iff_postPrefixNifsTarget` therefore prove that the
  Appendix-D.6 success event is exactly NIFS public `Pi_DEC` acceptance
  together with target membership for the same computed children and
  continuation assignments.  This closes the structural execution and event
  alignment; it does not derive target child witnesses from public
  acceptance.
- Finite D.4 causal closure and the remaining D.6 boundary:
  `PaperNonInteractive.CausalFirstSuccessBridge` constructs the exact finite
  carrier required by Appendix D.4 from the explicit causal contract and the
  paper PiRLC target extractor.
  `d4Seed_supportPermutation` exposes the independent causal-prefix × target
  fork support; `mem_d4FirstSuccessFreshSecondSeeds_iff` proves that the first
  coordinate is an actual typed NIFS ambient success and the second is a fresh
  unconditioned run; and
  `fixedFirstBad_iff_nifsD4FixedFirstBad` keeps the first extracted witness
  fixed while checking only the fresh second prefix.
  `d4FixedFirstBad_probability_eq_nifs`,
  `d4SourceExtracted_probability_eq_nifs`, and
  `d4Extraction_after_first_success_nifs` transport the operational D.4
  probabilities and final finite extraction inequality exactly.
  `Frozen.NonInteractiveAdaptiveWitnessObstruction.fixed_witness_bound_does_not_bound_adaptive_existential`
  remains the kernel counterexample explaining why the fixed-first carrier is
  necessary rather than an adaptive existential event.  Separately, Appendix
  D.6 defines success as verifier acceptance together with target child
  witnesses.
  `Frozen.PiDecTargetWitnessObstruction.accepted_without_piDec_target_witness`
  constructs an inhabited typed `Pi_DEC` model whose public equations accept
  but whose computed children have no target witnesses.  The zero-loss
  `Pi_DEC` reduction therefore starts from target-relation success and cannot
  turn bare public verifier acceptance into child assignments.
  `NifsExtractionErrorBudget.piDecTargetWitnessFailure` now owns exactly
  accepted NIFS execution without those target witnesses in every local,
  rewindable, post-prefix, and global soundness total. Rejecting transcripts
  consume no extraction budget. The event is no longer mislabeled as Theorem
  7's zero loss. The exact post-prefix bridge above fixes which target-success
  event must be supplied or bounded; it does not make that stronger event
  follow from acceptance.
- Selected prefix-gate representation with paper-exact degree selection:
  SuperNeo Definition 6, Section 7.3, and Appendix D.4 bound each prover
  message as a polynomial of degree at most the verifier-owned cap; they do
  not require canonical variable-length coefficient trimming.  The paper
  model therefore uses fixed-width coefficient lists and retains high zero
  slots as valid data.  The paper family and NIFS key now select exactly the
  verifier-computed syntax ceiling.  Lean separately proves that ceiling is no
  larger than Appendix D.4's `max(u, 2b+1, 2)` at frozen `b = 2`.
  `SumCheck.Finite.FixedPhase.RawCertificate.{decode_encode,check_encode}`
  proves exact typed/raw round trips without changing message coefficients.
  `ProtocolPolynomial.FixedWidth.accepted_implies_tableTruth_or_badEvent` and
  `StrongReduction.fixedWidthAcceptedProbe_extracts_source_or_badEvent` prove
  the deterministic paper reduction at that width.
  `InteractiveCompositionBridge.acceptedCheck_eq_piCcsCheck` proves literal
  equality of the interactive and NIFS PiCCS gates under replay alignment,
  while `mixingFailure_iff_piCcsMixingRoot` and
  `sumCheckFailure_iff_piCcsSumCheckCollision` identify their two D.4 residual
  events.  `piCcsCheck_extracts_sourceValid_or_badEvent` composes those
  identities with the fixed-width strong reduction, and
  `CausalPrefixCouplingContract.toPrefixExperiment_piCcsCheck_extracts_sourceValid_or_badEvent`
  specializes it pointwise to every realized independent product seed.  The
  separate `CausalFirstSuccessBridge` now performs the exact finite D.4
  first-success conditioning while preserving that fixed-witness order.
  `Frozen.SumCheckEncodingObstruction.fixed_width_acceptance_is_not_canonical_raw_acceptance`
  is retained as the kernel-checked obstruction to the discarded canonical
  convention: a degree-one padded encoding of the zero polynomial is accepted
  by the paper fixed-width gate but rejected by canonical trimming.  It
  explains why coefficient trimming is not the repair.  Independently,
  `PaperJoint.Necessity.SumCheckSoundnessContract.not_sumCheckSoundnessContract_at_paper_budget`
  reaches the exact causal bad-challenge event with probability one for an
  admitted `syntaxDegree <= width` context: a degree-six polynomial vanishes
  on all six sampled values while the Appendix-D.4 ceiling is five.  This
  necessity theorem leaves the frozen loss order unchanged and proves that
  the relaxed context does not satisfy `PaperDegreeWidthExact`.
  `AsymptoticPaperStrong.Point.degreeWidthExact` excludes it from the paper
  strong game, while
  `InteractiveCompositionBridge.strongExecutionContext_paperDegreeWidthExact`
  excludes it from NIFS. It is not a counterexample to SumCheck with its
  degree premise enforced. Exact production serialization, optional
  construction of the permitted SumCheck contract by root counting, and
  transcript-byte refinement remain separate obligations.
- Required closure: construct or characterize an ideal random-oracle support
  for the global prefix experiment and instantiate the explicit causal
  coupling contract for it; establish the D.6 target-success premise or bound
  its failure rather than replacing it with bare acceptance; and prove the
  remaining transition-level parent/child alignment modulo
  specifically named binding events.  Only after those bridges exist can the
  corresponding interactive predicates be bounded by the existing component
  theorems.
  Separately discharge the four exact transcript-collision bounds, bind the
  bounded PiRLC sampler to the actually reached post-PiCCS state, and refine
  exact typed encodings, domain tags, coordinate order, and state transitions
  to production Poseidon2.
  Public-input, replay, state, output-absorption, bounded-sampler shortfall,
  and multi-fork failures remain distinct named events.
- Classification: explicit specification/security-refinement obligation.
  The fixed-prefix D.5 experiment, dependent prefix carrier, and global finite
  composition, deterministic interactive-context alignment, and exact D.6
  execution/target-event alignment are model-proved.  The fixed-width PiCCS
  gate, its D.4 residual-event identities, and the finite causal
  first-success/fresh-second coupling are also model-proved.  These results do
  not establish an ideal random-oracle law, target-witness availability,
  concrete transcript encoding, or the remaining event bounds.

## FPR-SPEC-RUST-DIFFERENTIAL-GAP

- Sequencing boundary: once the canonical executable checker exists, the
  charter requires Rust and Lean to execute shared honest and mutated inputs
  before obligation-set minimality or encoding selection is certified.
- Established control-flow evidence: Rust deterministically regenerates eleven
  native `verify_step` receipts, including honest base/recursive cases and
  mutations, and
  `NativeStep.generated_all_check` proves that the receipt-driven Lean replay
  agrees with every recorded native result while conserving the observed call
  trace. Separately, the one-slot schema executes the frozen canonical step
  and terminal checkers over fourteen proof-free model cases.
- Established shared-input result: the Rust drift gate now generates an exact
  one-slot, one-fresh, stateless linked bit-carrier corpus with the real
  `1 ‖ enc_inst(x_out) ‖ padding` prior public input. It contains honest base
  and recursive steps plus independent initial-state, fold-tag, prior-`pc`,
  prior-link, NIFS-proof, and `x_out` mutations. The generated Lean equality
  quotient preserves every field inspected by the frozen checker, and
  `FPrimeCanonicalSharedInputDifferential.generated_all_agree` kernel-reduces
  `CanonicalVerifier.eval` on all nine cases. Both honest cases accept and all
  seven mutations reject. The Rust regeneration test is fail-closed on JSON or
  Lean drift, and the theorem has a fail-closed axiom audit.
- Established terminal shared-input result: a second drift gate uses the
  production `verify_uncompressed` entry point for the exact one-slot
  `r1cs_ivc_tiny_one_slot_terminal_v1` profile. Its seven cases contain honest
  base and recursive terminals plus independent base-endpoint, recursive
  program-counter, prior-link, running-relation, and fresh-relation mutations.
  The recursive cases separately record the exact Rust link, running-CE, and
  fresh-CCS checks as receipts; the final Rust result is not fed back into
  those receipts. Lean reconstructs the frozen terminal checker from that
  shared payload, and
  `FPrimeCanonicalSharedTerminalDifferential.generated_all_agree`
  kernel-reduces all seven comparisons. Both honest cases accept and all five
  mutations reject. The JSON/Lean regeneration gate and fail-closed axiom
  audit pass.
- Closed sequencing slice: this establishes the charter's shared-input
  differential gate for the bounded production step and terminal profiles. It
  is `rust-conformant` only for those exact acceptance maps and their stated
  carrier preconditions.
- Remaining connection: other production profiles, malformed raw
  `Uncompressed` carriers, proof/serialization parsing, primitive correctness,
  general Rust acceptance equivalence, and the full Rust/R1CS refinement chain
  remain open. The native lifecycle interface also supplies only an arbitrary
  binary fresh-link callback; the paper-required factorization through the
  concrete unary public-input and instance encoders is separately registered
  by `PaperFreshLinkBoundary.currentInterface_admits_nonFactorizingFreshLink`.
  `CanonicalPublicInputLink.equalityFactorization` closes that factorization
  for the typed logical `[1 | enc_inst]` profile.
  `CanonicalPlainCarrierLink.{equalityFactorization,
  check_reduces_to_logicalPaperLink}` closes the plain typed `m_in = 270`
  carrier and proves its thirteen zero-padding coordinates reduce to that
  logical paper relation. `rawCheck_reduces_to_typedCarrier` additionally
  closes the variable-length-list model. `CanonicalPlainCarrierSource`
  fixes the exact affine/body/padding source split and proves pointwise and
  batch reduction. Both native lifecycle acceptance and the paper decider now
  invoke one shared pure Rust predicate over an explicit six-instruction
  program. The Rust exporter emits that exact value; Lean proves it equals the
  canonical typed schedule, has definitional cost 273, and its typed
  interpretation reduces every raw claim to the logical paper link
  (`CanonicalPublicInputLinkProgramRefinement.generated_run_reduces_to_logicalPaperLink`).
  Runtime regressions mutate every body and padding coordinate plus all shape
  fields. Both production callers now pass the verifier-computed typed
  `EncInst` directly into the shared predicate; the helper no longer accepts
  a free 256-bit argument. Production XOut preimage construction now
  interprets a typed schedule shared with its drift exporter. The four
  stateless/stateful × plain/Nebula generated values are kernel-checked equal
  to the independent Lean schedule, begin with the exact state-output domain,
  place the present-only Nebula marker/lane last, and expand to exactly
  `encodeStateXOutPreimage`; their definitional field costs are 23, 28, 27,
  and 32. The frozen `XOut.compute` result is also proved to satisfy the
  generated plain outgoing public-link program at the exact affine-one and
  256 little-endian bit coordinates. For the selected 23-field plain/stateless
  variant, `Poseidon2Sponge.EmissionReceipt` computes the physical core cost
  as `23 + 2 + 600 * 7 = 4225` and requires the actual owner-row slice plus
  the ordered row and fresh-column intervals to equal the reconstructed trace.
  `FPrimeFullHistoryXOutSpongeReceipts` discharges that receipt for the
  captured base output, recursive prior consumer, and recursive output cores,
  and proves their row/column conservation and identical pure schedules.
  This closes only those three nonoptional four-lane sponge cores. The
  totalized wrapper semantics are now frozen more sharply:
  `ProductionHashCallBoundary.paperHash_eq_none_iff` proves rejection is
  exactly absence of the current state or failure of iteration, initial-state,
  running, or program-counter alignment; its coordinate theorems fix the
  all-zero absent and presence-one accepted encodings; and
  `no_nonoptionalCoreRefines` proves an always-present four-lane core cannot
  replace that wrapper. The state/running codecs and physical alignment,
  presence, payload, and typed sponge rows remain open, so neither `hashPrior`
  nor `hashNext` is promoted into the certified `CallRecipe` subset. Current
  whole-program ownership, compiled-Rust
  Poseidon2 parity, and collision resistance also remain open. The logical
  numeric/typed row mismatch is now closed independently:
  `Goldilocks.NumericRowBridge` translates every sparse numeric coefficient,
  source column, and canonical assignment value into the paper Goldilocks
  row semantics, proves satisfaction equivalence for one row and an ordered
  row list, and retains every source occurrence under duplicate-free
  owner-local row identities. The generic translation itself allocates no
  column and emits no call receipt.
  `ProductionPoseidon2PermutationRecipe` now closes the next exact physical
  slice for one width-eight permutation: 600 translated internal SSA rows,
  600 auxiliary temporary coordinates, and eight activation-gated visible
  output-copy rows under one mandatory receipt. Active satisfaction fixes all
  eight visible lanes to the executable production permutation; the explicit
  honest completion writes only those 600 temporaries; inactive completion
  leaves every visible output unconstrained. This does not yet make a sponge
  or either optional hash wrapper into a `CallRecipe`; absorption, padding,
  wrapper alignment/presence, call-site placement, native parity, and
  collision resistance remain explicit obligations.
  `ProductionPoseidon2Sponge23Recipe` now closes the corresponding fused
  typed sponge occurrence for one ordered 23-field input bundle. Its exact
  straight-line core has 4,225 rows and 4,225 auxiliary temporaries; four
  outer activation copies give 4,229 rows total. Active soundness and honest
  active/inactive completion are proved without seven intermediate
  permutation-output gate blocks. `ProductionPoseidon2Sponge23Audit`
  classifies every row dependency, proves allocation ownership, uniqueness,
  and row/column conservation, and proves the fused cost
  `(4229,0,0,4229)` minimum over the exact 128-member class obtained by
  independently retaining or deleting those seven redundant eight-gate
  blocks; the all-retained form costs `(4285,0,0,4229)`. This selection does
  not itself identify a concrete input bundle with XOut serialization.
  `ProductionXOutSponge23InputAlignment` now closes that next boundary for
  the generated plain/stateless source program: explicit four-field lookup
  availability and canonicality yield exactly 23 source fields, exact
  equality with the independent XOut encoder, index-by-index equality with
  normalized sponge inputs `1..23`, and soundness/completeness for the pure
  sponge on that vector. A kernel-checked counterexample shows why lookup
  availability is load-bearing: the empty table passes the older table
  well-formed check but emits only seven fields. This bridge adds no rows.
  It still does not bind optional presence or the totalized hash alignment
  checks, establish a complete hash `CallRecipe`, prove generated
  placement/native parity, or remove the collision event.
  The logical
  selected-context bridge is now universal over typed checked receipts:
  `FixedOneCanonicalAdapter.transition_iff_holds` maps the native
  state/proof/batch representation to the frozen fixed-one Construction-2
  transition in both directions, and
  `nativeAccepted_with_boundaries_and_outgoing_iff_canonicalAccepts` composes
  the producer with the explicitly owned entry, incoming delayed link,
  stateful application, Nebula, and outgoing consumer/terminal obligations.
  `checkedRecorded_with_boundaries_and_outgoing_iff_canonicalAccepts` uses
  exact receipt conservation/replay to substitute the recorded native result.
  This closes the paper-machine context for checked typed receipts; it does
  not make receipt-supplied hash or NIFS results authoritative. Poseidon2
  remains opaque and collision remains named. At the selected physical-recipe
  boundary, `DirectCalls.certifiedSubset` now packages exact recipes for
  `iterationZero`, `stateEqual`, `freshPublic`, `encodeInstance`, and
  `encodedEqual`. Each owns its footprint, rows, temporary completion, and
  active/inactive semantics. The package intentionally excludes `step`,
  `hashPrior`, `hashNext`, `nifsVerify`, `runningCheck`, and `freshCheck`;
  `remainingCalls_exact` keeps those six obligations explicit, so this is
  model-level evidence rather than a complete physical compiler.
  `FixedOneLoweringAdapter.parameters` separately instantiates the typed
  lowering semantics from the same universal native setup and paper machine.
  Its six `CallAlignment` theorems fix the exact application, iteration-ordered
  hash, NIFS argument order, and supplied terminal-check meanings, while
  `stepAccepts_iff_directHolds` and `terminalAccepts_iff_transition` close the
  two intrinsic typed programs back to the frozen relations. Its widths and
  footprints are deliberately untrusted shape inputs; it supplies none of the
  six missing physical recipes, complete codec family, rows, or compiled-Rust
  semantics. `ProductionIterationZeroCallRecipe` separately packages the
  canonical zero test without depending on any unrelated call map: exact
  alignment yields three rows, two one-coordinate auxiliary temporary
  bundles, and a mandatory typed receipt under the permitted field/inversion
  laws. The first concrete codec slice is now separate and explicit:
  `ProductionDigestCodecs.digestCodec_encode_exact` fixes the production
  four-lane Goldilocks order, the digest, optional-digest, and compact adapter
  codecs have exact round trips, and
  `encodeInstance_coordinates_exact` proves five coordinate copies followed by
  verifier-fixed one. `ProductionEncodeInstanceRecipe` realizes that map as
  exactly six caller-owned rows with no temporaries and proves ownership,
  support, active soundness, and active/inactive completeness.
  `ProductionEncodeInstanceCallRecipe` packages that exact implementation as
  a complete typed `CallRecipe` with a nonoptional output/temporary/row
  receipt and a program-derived six-row footprint. It accepts a supplied full
  lowering profile but proves only the two relevant production codecs and the
  `encodeInstance` footprint. `ProductionEncodedEqualCallRecipe` independently
  packages equality over the audited six-coordinate compact codec as exactly
  eighteen rows with auxiliary temporary bundles of widths six, six, and
  five, again with a nonoptional receipt. These slices do not supply
  state/fresh/witness codecs, the nonlinear `freshPublic` implementation,
  complete width agreement, `stateEqual`, the six calls in
  `RemainingRecipes`, or generated-row equality. On the paper's
  singleton fresh input,
  `ProductionFreshPublicSingletonBridge` now proves the compact
  `freshPublic = encodeInstance` equality is exactly the audited
  270-coordinate source predicate, its six-phase 273-obligation program, and
  the logical paper public-input relation.
  `CanonicalPlainCarrierSerialization` proves that flattening the complete
  typed carrier is injective, and
  `FPrimeProductionFreshPublicSingletonRows` composes that fact with the
  selected one-claim terminal-link owner. Its 270 rows are equivalent in both
  directions to the source predicate, source program, and compact equality;
  the isolated current artifact inherits the same result. The exact
  `273 = 270 + 3` boundary keeps expected length, `m_in`, and vector length as
  host/source checks outside the typed row block. The current list/batch
  deviation, those three host checks, producer-row placement, and compiled
  Rust semantics remain explicit.
  `FPrimeFullHistoryProductionDigestCodec.rows_decode_exact_xOut` separately
  closes the exact recursive-output row owner into the selected typed
  four-lane digest codec: the codec coordinates are the physical
  `xOutColumns` in order and decode back to that digest.
  `output_and_terminal_rows_decode_same_digest` composes the exact terminal
  delayed-link rows and proves its `terminalFreshDigest` is the same typed
  value.
  `decodedDigest_eq_logicalLinkDigest` and
  `terminalLogicalPublic_eq_encodePublicInput` prove that this selected digest
  is also exactly the digest whose paper-owned 257-coordinate public input is
  consumed by the captured terminal logical-link rows.
  `FPrimeFullHistoryProductionDigest.fullRows_finalState_latest_digest_and_logical_public`
  lifts both alignments through the exact full-row owner partition and exposes
  the final Construction-2 state's singleton fresh-public payload and
  terminal logical public input as canonical encodings of one typed digest.
  `FPrimeFullHistoryCurrentTerminalLinkCompletion.output_and_snapshot_rows_construct_currentPlainOwner`
  then constructs an explicit assignment for the current isolated 270-row
  owner: it copies the captured affine coordinate and all 256 authoritative
  producer/consumer coordinates, and fixes exactly the thirteen new padding
  coordinates to zero. The full-row and independent `CompilerWitness` lifts
  expose that same digest simultaneously in the current typed claim, the
  final Construction-2 state, and the captured paper-owned public input.
  Separately, the live two-step synthesis now exports only its current
  `terminal.latest_link` range as a bounded generated certificate:
  rows `[9673389, 9673659)`, producer bits `16766..17021`, fresh carrier
  `4090877..4091146`, and exactly 270 rows. Lean proves that all 527 isolated
  columns use the recorded relabel map and that the mapped isolated owner is
  exactly the generated row list. Under the exact recursive-output owner,
  satisfaction of this current range is equivalent to the typed
  zero-completed carrier and frozen logical paper equality.
  `generatedRows_iff_sourceProgram` further proves equivalence with the
  singleton 273-obligation source program, with its three shape obligations
  retained outside the already typed 270-row block, and
  `generatedRows_iff_freshPublic_eq_encodeInstance` proves equivalence with
  the compact fixed-one prior-public equality when the adapter uses that
  source predicate.
  `generatedRows_iff_loweringPriorLinkAccepted` specializes the same result
  to the exact `Terminal.priorLinkAccepted` Boolean in the selected typed
  lowering program, with the terminal proof's fresh value and verifier hash
  aligned explicitly. The program, adapter, and lowering statements use the
  digest reconstructed from the recursive output wires; none accepts a free
  carried digest. The constructive theorem reconstructs the same selected
  digest and codec coordinates.
  `fullRows_and_currentTerminalPlacement_construct_plainOwner` lifts this
  bounded range alongside, but not inside, the captured aggregate.
  This is artifact-checked producer/consumer/paper-public coordinate
  alignment, honest local completion, and one bounded current placement. It
  is not a generated aggregate for every current full-history row, universal
  placement across profiles/batches, a production codec for the complete
  state, physical equality with the selected lowering recipes for
  `freshPublic`, `encodeInstance`, and `encodedEqual`, compiled-Rust
  semantics, or Poseidon2 collision reduction. This prior-link result is not
  the distinct terminal `freshCheck` obligation. At
  the bounded captured-artifact
  boundary, `FPrimeFullHistoryCircuit.exactSteps_of_fullRows_or_bad` exposes
  both exact full-history step witnesses, and
  `FPrimeFullHistoryCanonicalSteps.fullRows_imply_frozenSteps_or_bad` composes
  them with that adapter. Thus satisfaction of all 4,193,134 checked-in rows
  yields both concrete frozen fixed-one checker acceptances or the existing
  `BadEvent`; the reachable branch in this theorem is the named recursive
  PiRLC projection-root event. This is artifact-checked step soundness for the
  stale snapshot, not equality with the selected canonical receipt program,
  current Rust semantics, honest completeness, terminal refinement, or an
  event bound. The isolated terminal-link
  ownership artifact now contains the actual 270 rows—one affine row, 256
  bit links, and thirteen zero pins—and
  `FPrimeTerminalLinkCanonicalRefinement.satisfies_iff_logicalPaperLink`
  reduces those rows exactly to the logical relation under the explicit
  producer-bit alignment proposition.
  `FPrimeEncodingCanonicalBits.publicBit_eq_encodedBit` derives that
  proposition from the exact 532-row output encoder, and
  `satisfies_iff_logicalPaperLink_of_encodingRows` leaves only the concrete
  column-placement map between the two owners. The new bounded current
  placement certificate closes that map for one live two-step synthesis:
  `mapped_rows_eq_generated` proves exact row-list equality,
  `producerBitColumnMap` proves producer alignment, and
  `generatedRows_iff_logicalPaperLink` proves exact current-range semantics.
  The captured full-history snapshot itself still does not close that map for
  current production:
  `FPrimeFullHistoryTerminalLogicalLinkSound.logicalCheck_of_rows` proves only
  that its 532-row producer snapshot plus 257-row terminal-link prefix imply
  the frozen logical equality. Current plain production emits thirteen
  additional verifier-fixed zero-padding rows.
  `FPrimeFullHistoryTerminalLinkDrift.generatedSnapshot_ne_currentPlainOwner`
  and `generatedSnapshot_missingPlainPaddingRows` are the fail-closed
  row-count obstruction (`257 ≠ 270`, deficit `13`). Both current isolated
  Rust drift gates, the bounded live-placement drift gate, and focused Lean
  axiom guards pass; the stale full-history snapshot was not regenerated or
  certified. The constructive completion proves the honest direction by
  copying the 257 captured coordinates and assigning zero to columns
  `514..526` of the isolated owner. The separate generated placement
  certificate records the actual current physical columns without claiming
  that the stale aggregate contains them. The artifact-independent
  `FPrimeTerminalLinkBatch` program lifts the block to every batch size with
  one receipt and one fresh-public column per row, exact `270 * batchSize`
  recurring-row/public-column costs, injective/surjective physical ownership,
  soundness/completeness, and reduction of every claim to the logical paper
  link. Its one-claim row list is kernel-checked equal to the isolated
  artifact. Production interprets one three-instruction affine/body/padding
  schedule per claim in claim-major order. The Rust-exported schedule is
  kernel-checked equal to the selected typed program, costs `270` rows per
  claim, expands to the complete owner order, and yields exact arbitrary-batch
  ownership/cost `270 * batchSize`. Its fail-closed Lean compiler emits rows
  only for the complete selected owner order and returns exactly the
  receipt-owned arbitrary-batch row list. For the generated singleton program,
  acceptance is kernel-checked equivalent to the exact typed Terminal
  `priorLinkAccepted` Boolean under explicit digest, fresh-order, source-link,
  and producer-column alignment. Relabeling that checked singleton compiler
  output is also kernel-checked equal to the generated current full-history
  range; acceptance of the pulled source program is equivalent to both
  satisfaction of those current rows and the output-derived typed
  `priorLinkAccepted` Boolean. This closes generated source-program and
  artifact-checked lowering equality, not compiled-Rust semantics. A separate
  two-claim production
  capture has exactly `540` literal rows and `797` columns; Lean proves the
  generated row list equals `FPrimeTerminalLinkBatch.rows 2`, equals the
  checked compiler output for the Rust-emitted program at batch size two, and
  that every captured row has exactly one typed receipt owner. This is bounded artifact
  evidence, not universal Rust-loop semantics. Formal compiled-Rust
  interpreter semantics, raw production input-to-checked-receipt refinement,
  concrete lowering-codec/recipe instantiation and emitted-row equality (or a
  proved refinement from the distinct production compiler), terminal-checker
  integration, and a whole-current-program or universal full-history physical
  placement refinement remain open.
  NIFS/hash results are still supplied as typed receipts and must not be
  promoted into semantic authority.
- Classification: bounded differential conformance result plus explicit
  remaining implementation-refinement obligations.

## FPR-DEV-BLOCK-LANE-COMBINED-NC

- Added behavior: the production-oriented Pi_CCS model separates FE from a
  block/lane NC representation and later combines NC obligations in its own
  verifier/transcript flow.
- Paper baseline: SuperNeo Section 7.3 uses one joint polynomial `Q` and one
  SumCheck.
- Classification: model-proved registered protocol deviation.
- Typed authority: `ProductionRefinement.AuthoritativeInput` contains only the
  canonical opening-derived carrier and verifier-owned fixed-active context.
  `ProductionRefinement.Certificate` contains only FE/NC messages and later
  PiRLC/PiDEC payloads. The verifier derives FE and NC points, initializes NC
  with FE's exact final state, and materializes `y_ring` and packed `y_zcol`;
  no prover-supplied public input, challenge, transcript state, or output
  sidecar exists.
- Deterministic kernel evidence:
  `accepted_implies_paper_or_algebraic_failure` reduces accepted verification
  to `Semantics.Paper.Holds` or the exact FE/NC algebraic event families.
  `not_transcriptFailure` and `not_bindingFailure` prove that typed
  reset/fork and output-substitution branches are uninhabited.
  `blockLaneCombinedNc_refines_paperNc` proves block/lane residual zero iff
  paper NC truth, and `everyCoordinate_has_exact_owner` proves complete
  carrier coverage. `delayedProjection_refines_rawRecomposition` proves that
  the delayed scalar is the packed radix recomposition of all authoritative
  raw running assignments. `honest_complete_with_output` and
  `accepted_output_suitable_for_piRlc` close honest completeness and the exact
  PiRLC output handoff.
- Residual-axis obstruction and selected refinement:
  `ResidualAlignment.not_literalResidualSlotAlignment` proves that this
  product-domain representation cannot be identified slot-for-slot with the
  paper's row-domain gamma polynomial. At `K = 1`, `k = 1`, and `t = 2`, the
  production carried exponent is `5` while the two paper coefficient slots
  are `4` and `6`; production FE arity is `7` while the paper row arity is
  `1`; and production-relative NC exponent `0` becomes absolute joint-`Q`
  exponent `1`. `carriedCoefficientAxis_is_not_gammaAxis` isolates the reason:
  the Phi81 coefficient coordinate belongs to the lane SumCheck axis, not to
  distinct gamma powers.
- The selected correction retains the production lane axis and relative NC
  indexing. `ncRelativeExponent_eq_paperLocal` and
  `ncJointExponent_eq_paperNormSlot` prove the exact index translation;
  `semanticResidualsZero_iff_paperHolds` replaces false slot identity with
  sound-and-complete semantic refinement; and
  `accepted_implies_paper_or_residual_failure` preserves the exact named
  failure branch. This is evidence for the registered block/lane deviation,
  not a message-, polynomial-, row-, or column-identity claim.
- Challenge ownership: FE and NC share the verifier-derived `betaA` and
  `gamma`; there is no independent-mixing-challenge claim. In the
  ideal-interactive model, FE messages see only the prior FE prefix. The NC
  strategy may depend on the completed FE word, but each NC message sees only
  the prior NC prefix.
- Quantitative kernel evidence:
  `CausalSoundness.rawRoundRepresentable` proves quartic representability for
  the exact ordinary-or-delayed production NC polynomial.
  `splitCollision_implies_detects` transports the exact physical FE and NC
  `FixedPhase.BadChallenge` events, and `splitCollision_probability_le`
  derives the explicit union bound
  `(feRoundCount * Drow)/|C| + (25 * 4)/|C|` from finite root counting and
  successive-coordinate sampling. It assumes neither
  `SumCheckSoundnessContract` nor FE/NC strategy independence.
- Selected mixing-carrier evidence:
  `IdealInteractiveCarrier.support` owns one explicit nonempty,
  duplicate-free product in transcript order: `alpha`, `betaA`, `betaR`, one
  shared `gamma`, `betaBlock`, `producerBeta`, `batchWeight`, FE word, then NC
  word. `input_supportAligned` derives the verifier denominator from that
  support. `IdealInteractiveFeSoundness.mixingRoot_probability_le` and
  `IdealInteractiveNcMixing.ncMixingRoot_probability_le` bound every literal
  FE/NC mixing constructor. `algebraicFailureEvent_eq_namedFailureEvent`
  proves exact two-way transport to the dependent production
  `FeFailure ∨ NcFailure` family, and `namedFailure_probability_le` preserves
  `(feMixingBudget + ncMixingBudget) + splitCollisionBudget`. The earlier
  arbitrary-schedule and zero-denominator countermodels remain valid; the
  positive theorem applies only to this selected constructor.
- Executable boundary: the protocol-owned `piCcsMessageCheck` is exact to the
  public combined-NC message acceptance predicate. Raw-assignment authority is
  supplied by the delayed lifecycle theorem below, never by a child
  `y_zcol` sidecar or digest.
- Required security and implementation closure: Fiat-Shamir/random-oracle
  refinement, closed Goldilocks Euclid and seven-nonresidue certificates,
  binding the selected support to the bounded production sampler/alphabet,
  differential Rust agreement, concrete transcript/dataflow refinement, and
  generated-row realization remain open.
- Excluded claims: this model proof is not Rust/R1CS conformance, generated-row
  authority, a cost result, optimization permission, or message identity with
  the paper's displayed one-joint polynomial.

## FPR-DEV-DELAYED-PACKED-YZCOL

- Added behavior: production carries packed `y_zcol` authority across a
  one-fold delay and closes the predecessor output in a later recursive or
  terminal step.
- Paper baseline: HyperNova Construction 2 performs one selected NIFS fold in
  each recursive `F'_j` invocation and its terminal verifier performs no final
  fold. SuperNeo's CE relation binds its current evaluation claims directly.
- Classification: model-proved registered protocol deviation.
- Executable step evidence: `DelayedPackedYZcol.Checker.check_eq_true_iff_accepted`
  proves exactness of the typed Boolean step checker. The checker replays the
  public combined-NC message, sampler, paper-output, and canonical-parent
  opening obligations; `baseCheck_eq_true_iff_accepted` additionally checks
  that the base carries no pending predecessor.
- Transcript evidence: `ChallengeAuthority.holds` fixes the statement,
  ordered parent/children, pending value, and polynomial bound before sampling;
  gives distinct producer-beta and batch-weight domains; and records their
  statement-to-core-to-producer-to-batch order. It makes no claim about the
  Poseidon2 permutation.
- Lifecycle evidence: `Trace.closedTrace_implies_baseAndAllClosed_or_failure`
  covers the explicit base, every recursive edge, and the terminal raw-child
  opening. Its successful branch proves the base pending value is absent and
  every output has both `PackedYZcolBoundAtBlock` and the independent frozen
  paper-profile transition. Its failure branch contains only located
  `yRingUnbound`, selector/root, SumCheck, Pi_RLC mixing, parent-opening, or
  accumulator-binding events. There is no generic `outputUnbound`,
  `refinementFailure`, source-projection premise, output-column-binding
  premise, or child `CeClaim.y_zcol` authority.
- Explicit ownership evidence:
  `Lifecycle.Trace.base_owns_no_predecessor` fixes the base pending value to
  `none`; `edge_owns_production_and_consumption` identifies each predecessor
  production with its successor carriage and consumption receipt;
  `terminal_owns_discharge` fixes the terminal raw-child discharge;
  `terminalCount_eq_one` proves that every nonempty typed trace has exactly
  one terminal constructor; and
  `closedTrace_reduces_to_paper_transitions_or_named_failure` reduces the
  closed lifecycle to paper transitions or the existing named failures.
- One-fold boundary: a successor's accepted combined-NC proof closes exactly
  its predecessor; the final predecessor is closed by an opening over the
  fourteen ordered raw child assignments. Binding is recomputed from the full
  typed parent/child/pending payload, so a digest is compression rather than
  authority.
- Closed terminal implementation slice: production Π_DEC allocation omits
  child `y_zcol`, and both decider synthesis entrypoints now use the strict
  child CE closure. The pending-family path first projects the same ordered
  raw witness allocations used by terminal Ajtai closure and binds their
  radix recomposition to the pending parent. Focused release tests cover
  ordinary terminal synthesis, the direct projection/Ajtai shared-wire
  invariant, and preservation of the legacy full-claim sidecar rejection.
- Required implementation closure: general differential Rust agreement,
  recursive-circuit parent authority, and the remaining concrete Rust/R1CS
  refinement are open. The terminal slice alone is not end-to-end
  Rust-conformant production authority.
- Excluded claims: no generated-row authority, cost result, or row-removal
  permission follows from this theorem.

## FPR-DEV-PICCS-TARGET-OFFSET

- Paper boundary: SuperNeo Section 7.3 and Appendix D.4, Lemma 7.
- Literal conflict: the definition of `Q` places the carried-evaluation block
  at `gamma^(2K+k) * gamma^I(i,j,l)`, while the displayed claimed sum `T(C)`
  uses `gamma^I(i,j,l)` alone. As printed, the two do not sit at the same
  powers of the mixing indeterminate, so Lemma 7's coefficient-separation step
  cannot match them.
- Reviewed decision (2026-07-24): **`Q` and the Section-7.3 verifier equation
  are canonical.** The displayed `T(C)` is the *local* carried-evaluation
  polynomial; its absolute form is

      T_abs(gamma) = gamma^(2K+k) * T_local(gamma)

  and the printed display omits that offset. Recorded as a paper erratum.
- Why `Q` is the canonical side: the three mixing blocks are exponent-disjoint
  by construction. `F` occupies `0 .. K-1`; `NC` occupies `K .. 2K+k-1`
  because it is scaled by `gamma^K`; `Eval` therefore must begin at `2K+k`.
  The offset is precisely what keeps the `NC` and `Eval` blocks from
  overlapping, which is the property D.4's linear-independence argument
  consumes when it splits Equation (6) into Equations (7), (8), and (9).
  Removing the offset from `Q` would collapse those blocks and would require
  an independent argument that the separation still holds.
- Quantitative consequence: the `(2K+k)` contribution to D.4's
  `epsilon_SZ := (2K + k) * max(log m, ktd) / |K|` is retained. At the
  Appendix B.2 profile (`K = 1`, `k = 14`) that coefficient is `16`.
- Kernel evidence:
  `PaperJoint.TargetPolynomial.evaluateShifted_eq_shift_mul_evaluateLocal`
  proves the local-to-absolute identity;
  `PaperJoint.TargetPolynomial.literalLocal_shifted_support_mismatch_witness`
  exhibits exponent zero as a genuine support-set mismatch under positive
  paper dimensions, so the two displays are provably not interchangeable.
- Open author query: the erratum should still be raised with the SuperNeo
  authors, but per the reviewed decision it does not block formalization.
- Classification: reviewed paper-level erratum decision. It fixes the mixing
  convention only. It is not concrete-field discharge (see
  `ARITH-GOLDILOCKS-FIELD`) and not production residual placement (see
  `FOLD-PICCS-JOINT` and `FOLD-PICCS-SPLIT`).
