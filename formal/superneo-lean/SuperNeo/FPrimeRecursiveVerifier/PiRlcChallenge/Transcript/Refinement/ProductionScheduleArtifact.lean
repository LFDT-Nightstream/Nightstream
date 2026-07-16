import SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.Transcript.Refinement.Generated.ProductionScheduleArtifactData

/-!
Owns: kernel-checked shape, cost-tree, and nonlinear-census theorems for the
generated fixed PiRLC transcript schedule artifact.

Does not own: concrete transcript absorb contents, counter values, Poseidon2
functional equivalence, sampler semantics, a materialized low-norm relation,
or permission to remove rows.

Emits constraints: no.

Authority boundary: these theorems check the generated data exactly. They do
not elevate diagnostic stage names or estimator output into protocol authority.

| Theorem | Rust stage | Guarantee | Evidence tier | Permits row removal? |
|---|---|---|---|---|
| `generated_schedule_order_exact` | `challenge.transcript` | Exactly 15 samples, four rounds each, four lane decompositions each | diagnostic trace | No |
| `generated_event_families_reconcile` | transcript leaves | Per-event sums equal family totals | mixed source/estimate/trace | No |
| `generated_tree_reconciles` | `challenge` | Protocol and phase immediate children sum exactly | trace-reconciled profiler | No |
| `generated_digest_round_cost_formula` | `challenge.transcript.digest_rounds` | Fifteen two-permutation first rounds plus forty-five one-permutation later rounds explain the family total | source plus estimator plus trace | No |
| `generated_nonlinear_census_exact` | Poseidon trace | 78 transcript permutations and 6,708 S-boxes reconcile | diagnostic trace | No |
| `generated_challenge_dimensions_exact` | `challenge` | 127,611 by 121,566 source; 198,567 by 370,383 estimated low norm | source plus estimator | No |
-/

namespace SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.ProductionScheduleArtifact

open ProductionScheduleArtifactData

/-- Generated evidence is tiered and all untraced semantic surfaces stay explicit. -/
theorem generated_evidence_scope_exact :
    schemaVersion = 1 ∧
      sourceCostTier = .materializedSourceR1cs ∧
      encodedCostTier = .traceReconciledEstimate ∧
      nonlinearCensusTier = .diagnosticTrace ∧
      stageOrderTraced = true ∧
      absorbContentsTraced = false ∧
      counterValuesTraced = false ∧
      poseidonFunctionTraced = false := by
  native_decide

/-- Generated paths are the exact stable production ownership names. -/
theorem generated_stage_paths_exact :
    challengePath = "nifs.pi_rlc.challenge" ∧
      transcriptPath = "nifs.pi_rlc.challenge.transcript" ∧
      samplerPath = "nifs.pi_rlc.challenge.sampler" ∧
      bindOutputsDigestPath =
        "nifs.pi_rlc.challenge.transcript.bind_outputs_digest" ∧
      rhoDomainSeparatorPath =
        "nifs.pi_rlc.challenge.transcript.rho_domain_separator" ∧
      samplerInitializePath =
        "nifs.pi_rlc.challenge.sampler.initialize" ∧
      digestRoundsPath =
        "nifs.pi_rlc.challenge.transcript.digest_rounds" ∧
      laneDecompositionPath =
        "nifs.pi_rlc.challenge.transcript.lane_bit_decomposition" := by
  native_decide

/-- The declared Rust constant is visible but is not claimed as traced call content. -/
theorem generated_declared_input_claims_label_exact :
    declaredInputClaimsDigestLabel = "pi_rlc/input_claims_digest" ∧
      absorbContentsTraced = false := by
  native_decide

/-- The generated stage sequence has the exact fixed 15-by-4 geometry. -/
theorem generated_schedule_order_exact : FixedScheduleOrder samples := by
  unfold FixedScheduleOrder
  native_decide

/-- Every per-event cost sums to the corresponding family total. -/
theorem generated_event_families_reconcile :
    EventFamiliesReconcile samples rhoDomainSeparators
      samplerInitializations digestRounds laneDecompositions := by
  unfold EventFamiliesReconcile separatorCosts samplerInitializationCosts
    digestCosts laneCosts sumCosts Cost.add Cost.zero
  native_decide

/-- Protocol -> phase -> family immediate children reconcile componentwise. -/
theorem generated_tree_reconciles :
    TranscriptTreeReconciles bindOutputsDigest rhoDomainSeparators
      digestRounds laneDecompositions transcript sampler challenge := by
  unfold TranscriptTreeReconciles Cost.add
  native_decide

/--
The dominant digest family is exactly fifteen first rounds and forty-five later
rounds. This exposes the multiplication behind every headline count.
-/
theorem generated_digest_round_cost_formula :
    digestRounds.materializedSourceRows = 15 * 1204 + 45 * 604 ∧
      digestRounds.materializedSourceColumns = 15 * 1204 + 45 * 604 ∧
      digestRounds.estimatedLowNormRows = 15 * 3698 + 45 * 1849 ∧
      digestRounds.estimatedLowNormColumns = 15 * 7052 + 45 * 3526 ∧
      digestRounds.tracedPoseidonPermutations = 15 * 2 + 45 * 1 ∧
      digestRounds.tracedSboxes = 86 * (15 * 2 + 45 * 1) ∧
      laneDecompositions.estimatedLowNormColumns = 15 * 4 * 420 := by
  native_decide

/-- Every generated transcript event uses the exact 86-S-box permutation census. -/
theorem generated_nonlinear_census_exact :
    AllNonlinearCensusesConsistent samples bindOutputsDigest
      rhoDomainSeparators digestRounds laneDecompositions transcript := by
  unfold AllNonlinearCensusesConsistent NonlinearCensusConsistent
  native_decide

/-- Exact fixed transcript nonlinear totals, separate from functional equivalence. -/
theorem generated_transcript_nonlinear_totals :
    transcript.tracedPoseidonPermutations = 78 ∧
      transcript.tracedSboxes = 6708 := by
  native_decide

/-- Current source dimensions and low-norm estimate remain separately named. -/
theorem generated_challenge_dimensions_exact :
    challenge.materializedSourceRows = 127611 ∧
      challenge.materializedSourceColumns = 121566 ∧
      challenge.estimatedLowNormRows = 198567 ∧
      challenge.estimatedLowNormColumns = 370383 := by
  native_decide

end SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.ProductionScheduleArtifact
