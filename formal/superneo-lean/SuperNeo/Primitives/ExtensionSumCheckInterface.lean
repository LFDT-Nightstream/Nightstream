import SuperNeo.Primitives.ExtensionSumCheckPaper

/-!
Contract interface for `SuperNeo.ExtensionSumCheck`.

Spec: `specs/ExtensionSumCheck.spec.md`

Paper anchors:
- Definition 6, Section 4: sum-check protocol.
- Opening convergence Phase 1: the carrier is `SuperNeo.KExt`.
-/

namespace SuperNeo

namespace ExtensionSumCheckInterface

/-- [Role: Theorem-Target] Curated re-export of the extension-field instance object. -/
abbrev ExtensionSumCheckInstance := SuperNeo.ExtensionSumCheckInstance

/-- [Role: Theorem-Target] Curated re-export of the extension-field transcript object. -/
abbrev ExtensionSumCheckTranscript := SuperNeo.ExtensionSumCheckTranscript

/-- [Role: Theorem-Target] Curated re-export of univariate coefficient-array evaluation. -/
abbrev extensionSumcheckEvalPoly := SuperNeo.extensionSumcheckEvalPoly

/-- [Role: Theorem-Target] Curated re-export of transcript shape. -/
abbrev extensionSumcheckRoundConsistent := SuperNeo.extensionSumcheckRoundConsistent

/-- [Role: Theorem-Target] Curated re-export of round polynomial shape. -/
abbrev extensionSumcheckRoundPolyShape := SuperNeo.extensionSumcheckRoundPolyShape

/-- [Role: Theorem-Target] Curated re-export of transcript-wide round polynomial shape. -/
abbrev extensionSumcheckRoundShapes := SuperNeo.extensionSumcheckRoundShapes

/-- [Role: Theorem-Target] Curated re-export of paper-facing degree bound. -/
abbrev extensionSumcheckRoundPolyDegreeLe := SuperNeo.extensionSumcheckRoundPolyDegreeLe

/-- [Role: Theorem-Target] Curated re-export of per-round degree predicates. -/
abbrev extensionSumcheckRoundDegrees := SuperNeo.extensionSumcheckRoundDegrees

/-- [Role: Theorem-Target] Curated re-export of fold-transition consistency. -/
abbrev extensionSumcheckFoldConsistent := SuperNeo.extensionSumcheckFoldConsistent

/-- [Role: Theorem-Target] Curated re-export of initial-round consistency. -/
abbrev extensionSumcheckInitialRoundConsistent := SuperNeo.extensionSumcheckInitialRoundConsistent

/-- [Role: Theorem-Target] Curated re-export of parameter consistency. -/
abbrev extensionSumcheckParameterConsistent := SuperNeo.extensionSumcheckParameterConsistent

/-- [Role: Theorem-Target] Curated re-export of degree compatibility. -/
abbrev extensionSumcheckDegreeCompatible := SuperNeo.extensionSumcheckDegreeCompatible

/-- [Role: Theorem-Target] Curated re-export of core verifier acceptance. -/
abbrev extensionSumcheckAcceptedCore := SuperNeo.extensionSumcheckAcceptedCore

/-- [Role: Theorem-Target] Curated re-export of paper-facing verifier acceptance. -/
abbrev extensionSumcheckVerifierAccepted := SuperNeo.extensionSumcheckVerifierAccepted

/-- [Role: Theorem-Target] Curated re-export of hypercube table sum. -/
abbrev extensionSumcheckTableSum := SuperNeo.extensionSumcheckTableSum

/-- [Role: Theorem-Target] Curated re-export of the paper-facing statement object. -/
abbrev ExtensionSumCheckStatement := SuperNeo.ExtensionSumCheckStatement

/-- [Role: Theorem-Target] Curated re-export of final-oracle consistency. -/
abbrev extensionSumcheckFinalOracleConsistent := SuperNeo.extensionSumcheckFinalOracleConsistent

/-- [Role: Theorem-Target] Curated re-export of table-indexed final-oracle consistency. -/
abbrev extensionSumcheckFinalOracleConsistentWithTable :=
  SuperNeo.extensionSumcheckFinalOracleConsistentWithTable

/-- [Role: Theorem-Target] Curated re-export of verifier acceptance. -/
abbrev extensionSumcheckAccepted := SuperNeo.extensionSumcheckAccepted

/-- [Role: Theorem-Target] Curated re-export of existentially closed verifier acceptance. -/
abbrev extensionSumcheckAcceptedClosed := SuperNeo.extensionSumcheckAcceptedClosed

/-- [Role: Theorem-Target] Curated re-export of fixed-table acceptance. -/
abbrev extensionSumcheckAcceptedForTable := SuperNeo.extensionSumcheckAcceptedForTable

/-- [Role: Theorem-Target] Curated re-export of the packaged claim witness. -/
abbrev ExtensionSumCheckClaim := SuperNeo.ExtensionSumCheckClaim

/-- [Role: Theorem-Target] Curated re-export of the Definition-6 statement object. -/
abbrev ExtensionSumCheckDefinition6Statement := SuperNeo.ExtensionSumCheckDefinition6Statement

/-- [Role: Theorem-Target] Curated re-export of paper-facing claim truth. -/
abbrev extensionSumcheckClaimTrue := SuperNeo.extensionSumcheckClaimTrue

/-- [Role: Theorem-Target] Curated re-export of paper-facing claim truth alias. -/
abbrev extensionSumcheckPaperClaimTrue := SuperNeo.extensionSumcheckPaperClaimTrue

/-- [Role: Theorem-Target] Curated re-export of statement/transcript consistency. -/
abbrev extensionSumcheckStatementTranscriptConsistent :=
  SuperNeo.extensionSumcheckStatementTranscriptConsistent

/-- [Role: Theorem-Target] Curated re-export of soundness assumption surface. -/
abbrev ExtensionSumcheckSoundnessAssumption := SuperNeo.ExtensionSumcheckSoundnessAssumption

/-- [Role: Theorem-Target] Curated re-export of completeness assumption surface. -/
abbrev ExtensionSumcheckCompletenessAssumption := SuperNeo.ExtensionSumcheckCompletenessAssumption

/-- [Role: Theorem-Target] Curated re-export of the constructive assumption bundle. -/
abbrev ExtensionSumCheckAssumptions := SuperNeo.ExtensionSumCheckAssumptions

/-- [Role: Theorem-Target] Curated theorem surface `extensionSumcheckRoundPolyShape_degreeLe`. -/
abbrev extensionSumcheckRoundPolyShape_degreeLe :=
  @SuperNeo.extensionSumcheckRoundPolyShape_degreeLe

/-- [Role: Theorem-Target] Curated theorem surface `extensionSumcheckVerifierAccepted_of_accepted`. -/
abbrev extensionSumcheckVerifierAccepted_of_accepted :=
  @SuperNeo.extensionSumcheckVerifierAccepted_of_accepted

/-- [Role: Theorem-Target] Curated theorem surface `extensionSumcheckFinalOracleConsistent_iff_withTable`. -/
abbrev extensionSumcheckFinalOracleConsistent_iff_withTable :=
  @SuperNeo.extensionSumcheckFinalOracleConsistent_iff_withTable

/-- [Role: Theorem-Target] Curated theorem surface `ExtensionSumCheckClaim.accepted`. -/
abbrev extensionSumcheckClaim_accepted := @SuperNeo.ExtensionSumCheckClaim.accepted

/-- [Role: Theorem-Target] Curated theorem surface `extensionSumcheckAccepted_of_acceptedForTable`. -/
abbrev extensionSumcheckAccepted_of_acceptedForTable :=
  @SuperNeo.extensionSumcheckAccepted_of_acceptedForTable

/-- [Role: Theorem-Target] Curated theorem surface `extensionSumcheckAcceptedClosed_of_acceptedForTable`. -/
abbrev extensionSumcheckAcceptedClosed_of_acceptedForTable :=
  @SuperNeo.extensionSumcheckAcceptedClosed_of_acceptedForTable

/-- [Role: Theorem-Target] Curated theorem surface `extensionSumcheckDefinition6Statement_of_statement`. -/
abbrev extensionSumcheckDefinition6Statement_of_statement :=
  @SuperNeo.extensionSumcheckDefinition6Statement_of_statement

/-- [Role: Theorem-Target] Curated theorem surface `extensionSumcheckPaperClaimTrue_of_baseClaim_constructive`. -/
abbrev extensionSumcheckPaperClaimTrue_of_baseClaim_constructive :=
  @SuperNeo.extensionSumcheckPaperClaimTrue_of_baseClaim_constructive

/-- [Role: Theorem-Target] Curated theorem surface `extensionSumcheckPaperClaimTrue_of_accepted`. -/
abbrev extensionSumcheckPaperClaimTrue_of_accepted :=
  @SuperNeo.extensionSumcheckPaperClaimTrue_of_accepted

/-- [Role: Theorem-Target] Curated theorem surface `extensionSumcheckPaperClaimTrue_of_statementTranscriptConsistent`. -/
abbrev extensionSumcheckPaperClaimTrue_of_statementTranscriptConsistent :=
  @SuperNeo.extensionSumcheckPaperClaimTrue_of_statementTranscriptConsistent

/-- [Role: Theorem-Target] Curated theorem surface `extensionSumcheckCompleteness_constructive`. -/
abbrev extensionSumcheckCompleteness_constructive :=
  @SuperNeo.extensionSumcheckCompleteness_constructive

/-- [Role: Theorem-Target] Curated theorem surface `extensionSumcheckSoundness_constructive`. -/
abbrev extensionSumcheckSoundness_constructive :=
  @SuperNeo.extensionSumcheckSoundness_constructive

/-- [Role: Theorem-Target] Curated theorem surface `extensionSumcheckAssumptions_constructive`. -/
abbrev extensionSumcheckAssumptions_constructive :=
  SuperNeo.extensionSumcheckAssumptions_constructive

end ExtensionSumCheckInterface

end SuperNeo
