import SuperNeo.SumCheckPaper

/-!
Contract interface for `SuperNeo.SumCheck`.

Spec: `specs/SumCheck.spec.md`

Paper anchors:
- Definition 6, Section 4, lines 352-355: sum-check protocol.
- Section 7.3 (Π_CCS), lines 440-470: sum-check invocation in CCS folding.
- Section 7.4 (Π_RLC), lines 471-489: sum-check invocation in RLC.
-/

namespace SuperNeo

namespace SumCheckInterface

/-! ## Core Surfaces -/

/-- [Role: Theorem-Target] Curated re-export of `SumCheckInstance`. -/
abbrev SumCheckInstance := SuperNeo.SumCheckInstance

/-- [Role: Theorem-Target] Curated re-export of `SumCheckTranscript`. -/
abbrev SumCheckTranscript := SuperNeo.SumCheckTranscript

/-- [Role: Theorem-Target] Curated re-export of `sumcheckEvalPoly`. -/
abbrev sumcheckEvalPoly := SuperNeo.sumcheckEvalPoly

/-- [Role: Theorem-Target] Curated re-export of `sumcheckRoundConsistent`. -/
abbrev sumcheckRoundConsistent := SuperNeo.sumcheckRoundConsistent

/-- [Role: Theorem-Target] Curated re-export of `sumcheckRoundPolyShape`. -/
abbrev sumcheckRoundPolyShape := SuperNeo.sumcheckRoundPolyShape

/-- [Role: Theorem-Target] Curated re-export of `sumcheckRoundPolyDegreeLe`. -/
abbrev sumcheckRoundPolyDegreeLe := SuperNeo.sumcheckRoundPolyDegreeLe

/-- [Role: Theorem-Target] Curated re-export of `sumcheckRoundDegrees`. -/
abbrev sumcheckRoundDegrees := SuperNeo.sumcheckRoundDegrees

/-- [Role: Theorem-Target] Curated re-export of `sumcheckRoundShapes`. -/
abbrev sumcheckRoundShapes := SuperNeo.sumcheckRoundShapes

/-- [Role: Theorem-Target] Curated re-export of `sumcheckFoldConsistent`. -/
abbrev sumcheckFoldConsistent := SuperNeo.sumcheckFoldConsistent

/-- [Role: Theorem-Target] Curated re-export of `sumcheckInitialRoundConsistent`. -/
abbrev sumcheckInitialRoundConsistent := SuperNeo.sumcheckInitialRoundConsistent

/-- [Role: Theorem-Target] Curated re-export of `sumcheckAcceptedCore`. -/
abbrev sumcheckAcceptedCore := SuperNeo.sumcheckAcceptedCore

/-- [Role: Theorem-Target] Curated re-export of `sumcheckVerifierAccepted`. -/
abbrev sumcheckVerifierAccepted := SuperNeo.sumcheckVerifierAccepted

/-- [Role: Theorem-Target] Curated re-export of `sumcheckAccepted`. -/
abbrev sumcheckAccepted := SuperNeo.sumcheckAccepted

/-- [Role: Theorem-Target] Curated re-export of `SumCheckClaim`. -/
abbrev SumCheckClaim := SuperNeo.SumCheckClaim

/-- [Role: Theorem-Target] Curated re-export of `sumcheckTableSum`. -/
abbrev sumcheckTableSum := SuperNeo.sumcheckTableSum

/-- [Role: Theorem-Target] Curated re-export of `SumCheckStatement`. -/
abbrev SumCheckStatement := SuperNeo.SumCheckStatement

/-- [Role: Theorem-Target] Curated re-export of the Definition-6 theorem witness object. -/
abbrev SumCheckDefinition6Statement := SuperNeo.SumCheckDefinition6Statement

/-- [Role: Theorem-Target] Definition-6 theorem-witness surface for the standalone SumCheck scaffold. -/
abbrev sumcheckPaperClaimTrue := SuperNeo.sumcheckPaperClaimTrue

/-- [Role: Theorem-Target] Curated alias to the Definition-6 theorem-witness surface. -/
abbrev sumcheckClaimTrue := SuperNeo.sumcheckClaimTrue

/-- [Role: Theorem-Target] Curated re-export of `sumcheckLundSoundnessNumerator`. -/
abbrev sumcheckLundSoundnessNumerator := SuperNeo.sumcheckLundSoundnessNumerator

/-- [Role: Theorem-Target] Curated re-export of `sumcheckLundSoundnessDenominator`. -/
abbrev sumcheckLundSoundnessDenominator := SuperNeo.sumcheckLundSoundnessDenominator

/-- [Role: Theorem-Target] Curated re-export of `sumcheckLundSoundnessBound`. -/
abbrev sumcheckLundSoundnessBound := SuperNeo.sumcheckLundSoundnessBound

/-- [Role: Theorem-Target] Curated re-export of `sumcheckFinalOracleConsistent`. -/
abbrev sumcheckFinalOracleConsistent := SuperNeo.sumcheckFinalOracleConsistent

/-- [Role: Theorem-Target] Curated re-export of `sumcheckFinalOracleConsistentWithTable`. -/
abbrev sumcheckFinalOracleConsistentWithTable := SuperNeo.sumcheckFinalOracleConsistentWithTable

/-- [Role: Theorem-Target] Curated re-export of `sumcheckAcceptedForTable`. -/
abbrev sumcheckAcceptedForTable := SuperNeo.sumcheckAcceptedForTable

/-- [Role: Theorem-Target] Curated re-export of `sumcheckStatementTranscriptConsistent`. -/
abbrev sumcheckStatementTranscriptConsistent := SuperNeo.sumcheckStatementTranscriptConsistent

/-- [Role: Theorem-Target] Curated re-export of `sumcheckBaseClaimTrue`. -/
abbrev sumcheckBaseClaimTrue := SuperNeo.sumcheckBaseClaimTrue

/-- [Role: Theorem-Target] Curated re-export of `sumcheckParameterConsistent`. -/
abbrev sumcheckParameterConsistent := SuperNeo.sumcheckParameterConsistent

/-- [Role: Theorem-Target] Curated re-export of `sumcheckDegreeCompatible`. -/
abbrev sumcheckDegreeCompatible := SuperNeo.sumcheckDegreeCompatible

/-! ## Key Theorems -/

/-- [Role: Theorem-Target] Curated theorem surface `sumcheckAccepted_rounds_eq`. -/
abbrev sumcheckAccepted_rounds_eq := @SuperNeo.sumcheckAccepted_rounds_eq

/-- [Role: Theorem-Target] Curated theorem surface `sumcheckAccepted_challenges_eq`. -/
abbrev sumcheckAccepted_challenges_eq := @SuperNeo.sumcheckAccepted_challenges_eq

/-- [Role: Theorem-Target] Curated theorem surface `sumcheckAccepted_fold_step`. -/
abbrev sumcheckAccepted_fold_step := @SuperNeo.sumcheckAccepted_fold_step

/-- [Role: Theorem-Target] Curated theorem surface `sumcheckAccepted_initial_round`. -/
abbrev sumcheckAccepted_initial_round := @SuperNeo.sumcheckAccepted_initial_round

/-- [Role: Theorem-Target] Curated theorem surface `sumcheckAccepted_round_sum_step`. -/
abbrev sumcheckAccepted_round_sum_step := @SuperNeo.sumcheckAccepted_round_sum_step

/-- [Role: Theorem-Target] Curated theorem surface `sumcheckAccepted_not_of_challenge_size_ne`. -/
abbrev sumcheckAccepted_not_of_challenge_size_ne := @SuperNeo.sumcheckAccepted_not_of_challenge_size_ne

/-- [Role: Theorem-Target] Curated theorem surface `sumcheckAccepted_not_of_roundpoly_size_ne`. -/
abbrev sumcheckAccepted_not_of_roundpoly_size_ne := @SuperNeo.sumcheckAccepted_not_of_roundpoly_size_ne

/-- [Role: Theorem-Target] Curated theorem surface `sumcheckAccepted_not_of_bad_round_shape`. -/
abbrev sumcheckAccepted_not_of_bad_round_shape := @SuperNeo.sumcheckAccepted_not_of_bad_round_shape

/-- [Role: Theorem-Target] Curated theorem surface `sumcheckAccepted_not_of_bad_initial_round`. -/
abbrev sumcheckAccepted_not_of_bad_initial_round := @SuperNeo.sumcheckAccepted_not_of_bad_initial_round

/-- [Role: Theorem-Target] Curated theorem surface `sumcheckFinalOracleConsistent_iff_withTable`. -/
abbrev sumcheckFinalOracleConsistent_iff_withTable :=
  @SuperNeo.sumcheckFinalOracleConsistent_iff_withTable

/-- [Role: Theorem-Target] Curated theorem surface `sumcheckVerifierAccepted_of_accepted`. -/
abbrev sumcheckVerifierAccepted_of_accepted :=
  @SuperNeo.sumcheckVerifierAccepted_of_accepted

/-- [Role: Theorem-Target] Curated theorem surface `sumcheckAccepted_of_acceptedForTable`. -/
abbrev sumcheckAccepted_of_acceptedForTable := @SuperNeo.sumcheckAccepted_of_acceptedForTable

/-! ## Constructive Closures -/

/-- [Role: Theorem-Target] Curated theorem surface `sumcheckAccepted_parameter_consistent`. -/
abbrev sumcheckAccepted_parameter_consistent := @SuperNeo.sumcheckAccepted_parameter_consistent

/-- [Role: Theorem-Target] Curated theorem surface `sumcheckAccepted_degree_compatible`. -/
abbrev sumcheckAccepted_degree_compatible := @SuperNeo.sumcheckAccepted_degree_compatible

/-- [Role: Theorem-Target] Standalone structural-closure theorem for SumCheck. -/
abbrev sumcheckSoundness_constructive := SuperNeo.sumcheckSoundness_constructive

/-- [Role: Theorem-Target] Curated theorem surface `sumcheckCompleteness_constructive`. -/
abbrev sumcheckCompleteness_constructive := SuperNeo.sumcheckCompleteness_constructive

/-- [Role: Theorem-Target] Preferred structural soundness alias for the core scaffold. -/
abbrev sumcheckStructuralSoundness_constructive :=
  SuperNeo.sumcheckStructuralSoundness_constructive

/-- [Role: Theorem-Target] Preferred structural completeness alias for the core scaffold. -/
abbrev sumcheckStructuralCompleteness_constructive :=
  SuperNeo.sumcheckStructuralCompleteness_constructive

/-- [Role: Theorem-Target] Curated theorem surface `sumcheckCompleteness_from_baseClaim_constructive`. -/
abbrev sumcheckCompleteness_from_baseClaim_constructive := SuperNeo.sumcheckCompleteness_from_baseClaim_constructive

/-- [Role: Theorem-Target] Curated theorem surface `sumcheckHonestTranscript_accepted_of_baseClaim`. -/
abbrev sumcheckHonestTranscript_accepted_of_baseClaim :=
  SuperNeo.sumcheckHonestTranscript_accepted_of_baseClaim

/-- [Role: Theorem-Target] Curated theorem surface `sumcheckPaperClaimTrue_of_baseClaim_constructive`. -/
abbrev sumcheckPaperClaimTrue_of_baseClaim_constructive := @SuperNeo.sumcheckPaperClaimTrue_of_baseClaim_constructive

/-- [Role: Theorem-Target] Curated theorem surface `sumcheckPaperClaimTrue_of_accepted`. -/
abbrev sumcheckPaperClaimTrue_of_accepted := @SuperNeo.sumcheckPaperClaimTrue_of_accepted

/-- [Role: Theorem-Target] Curated theorem surface `sumcheckPaperClaimTrue_of_statementTranscriptConsistent`. -/
abbrev sumcheckPaperClaimTrue_of_statementTranscriptConsistent :=
  @SuperNeo.sumcheckPaperClaimTrue_of_statementTranscriptConsistent

/-- [Role: Theorem-Target] Curated theorem surface `sumcheckPaperClaimTrue_iff_baseClaim`. -/
abbrev sumcheckPaperClaimTrue_iff_baseClaim := @SuperNeo.sumcheckPaperClaimTrue_iff_baseClaim

/-- [Role: Theorem-Target] Curated theorem surface `sumcheckClaimTrue_iff_baseClaim`. -/
abbrev sumcheckClaimTrue_iff_baseClaim := @SuperNeo.sumcheckClaimTrue_iff_baseClaim

/-- [Role: Theorem-Target] Curated theorem surface `sumcheckFinalOracleConsistent_of_baseClaim_constructive`. -/
abbrev sumcheckFinalOracleConsistent_of_baseClaim_constructive :=
  SuperNeo.sumcheckFinalOracleConsistent_of_baseClaim_constructive

/-- [Role: Theorem-Target] Curated theorem surface `sumcheckStatementTranscriptConsistent_of_baseClaim_constructive`. -/
abbrev sumcheckStatementTranscriptConsistent_of_baseClaim_constructive :=
  SuperNeo.sumcheckStatementTranscriptConsistent_of_baseClaim_constructive

/-- [Role: Theorem-Target] Curated theorem surface `sumcheckStatementTranscriptConsistent_of_accepted_sameTranscript`. -/
abbrev sumcheckStatementTranscriptConsistent_of_accepted_sameTranscript :=
  @SuperNeo.sumcheckStatementTranscriptConsistent_of_accepted_sameTranscript

/-- [Role: Theorem-Target] Curated theorem surface `sumcheckAssumptions_constructive`. -/
abbrev sumcheckAssumptions_constructive := SuperNeo.sumcheckAssumptions_constructive

/-! ## Boundary Surfaces -/

/-- [Role: Boundary] Boundary surface `SumcheckSoundnessAssumption` requiring closure. -/
abbrev SumcheckSoundnessAssumption := SuperNeo.SumcheckSoundnessAssumption

/-- [Role: Boundary] Boundary surface `SumcheckCompletenessAssumption` requiring closure. -/
abbrev SumcheckCompletenessAssumption := SuperNeo.SumcheckCompletenessAssumption

/-- [Role: Boundary] Boundary surface `SumCheckAssumptions` requiring closure. -/
abbrev SumCheckAssumptions := SuperNeo.SumCheckAssumptions

/-- [Role: Boundary] Boundary surface `sumcheckClaimTrue_of_soundness` requiring closure. -/
abbrev sumcheckClaimTrue_of_soundness := @SuperNeo.sumcheckClaimTrue_of_soundness

end SumCheckInterface

end SuperNeo
