import SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra.Refinement.Generated.ProjectionIdentityCertificateData
import SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra.Refinement.ProjectionBatching
import SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra.Semantics.ProjectionBoundary

/-!
Owns: generated shape/schedule/cost checks for the validated production
projection identity and the separate model-level equivalence between exact
53/52-term source evaluations, exact terminal products, and their emitted
carry chains.

Does not own: the concrete row/assignment/decoder refinement that connects the
generated production artifact to the model-level theorem, semantic authority
for diagnostic identity roles, transcript derivation, or the representation
bridge from two base-field limbs to the polynomial ring used by the
exact-or-bad-root reduction.

Emits constraints: no.

Authority boundary: source R1CS rows remain authoritative. Rust replays every
source row, rejects escape and overlap, validates the compact plan, and then
generates the data imported here. The role list labels cost ownership only.

| Predicate/theorem | Rust stage | Guarantee | Assumptions | Permits row removal? |
|---|---|---|---|---|
| source_topology_exact | identities.* | Exact 1916-row/1914-column source layout | Generated trace replay passed | With Rust conformance |
| evaluation_plan_exact | identities.*.evaluations | Sixteen 53-product and one 52-product two-limb schedules | Generated compact plan passed | With Rust conformance |
| retained_bindings_are_diagonal | identities.*.evaluations | Exact 34-by-34 identity retained matrix | Generated compact plan passed | With Rust conformance |
| final_factor_schedule_exact | identities.*.final_limb_checks | Exact ordered W/sign/operand schedule | Generated compact plan passed | With Rust conformance |
| compact_cost_formula_exact | complete identity | 34 retained 41-coordinate fields, 70 synthetic 95-coordinate carries, and 106 product-sum rows produce exactly 5,248 rows by 8,044 columns | Generated plan plus production stage-profile agreement | Accounting only |
| certificateProjectionIdentity_iff_emitted | complete identity model | Abstract source semantics iff abstract emitted carry semantics | Shared authoritative operands; concrete row/decoder refinement open | No for production row removal by itself |
| certificateSource_implies_exact_or_badRoot | projection boundary | Source acceptance reaches exact-or-bad-root | Explicit limb-to-polynomial bridge | No without bridge |
-/

namespace SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra.ProjectionIdentityCertificate

open ProjectionBatchingRefinement
open ProjectionIdentityCertificateData
open ProductSumRefinement

universe u v

def commitmentRoles : List IdentityRole :=
  (List.range 18).map IdentityRole.commitmentLane

def activeXRoles : List IdentityRole :=
  (List.range 5).map IdentityRole.activeXColumn

def yRingRoles : List IdentityRole :=
  (List.range 3).flatMap fun row =>
    (List.range extensionLimbs).map fun limb => IdentityRole.yRingLimb row limb

def yZcolRoles : List IdentityRole :=
  (List.range extensionLimbs).map IdentityRole.yZcolLimb

theorem roles_are_diagnostic : rolesAreDiagnostic = true := by
  decide

theorem roles_exact_partition :
    roles = commitmentRoles ++ activeXRoles ++ yRingRoles ++ yZcolRoles := by
  decide

theorem roles_length_exact : roles.length = identityCount := by
  decide

def strideOffsets (start stride count : Nat) : List Nat :=
  (List.range count).map fun index => start + stride * index

theorem source_topology_exact :
    inputEvaluationRowOffsets = strideOffsets 0 113 pairCount ∧
    inputEvaluationColumnOffsets = strideOffsets 0 113 pairCount ∧
    rhoProductRowOffsets = strideOffsets 108 113 pairCount ∧
    rhoProductColumnOffsets = strideOffsets 108 113 pairCount ∧
    outputEvaluationRowOffset = pairCount * 113 ∧
    quotientEvaluationRowOffset = outputEvaluationRowOffset + 108 ∧
    quotientPhiProductRowOffset = quotientEvaluationRowOffset + 106 ∧
    finalLimbRowOffset = quotientPhiProductRowOffset + 5 ∧
    sourceRowsPerIdentity = finalLimbRowOffset + extensionLimbs ∧
    sourceColumnsPerIdentity = finalLimbRowOffset := by
  decide

theorem evaluation_coefficient_counts_exact :
    evaluationCoefficientCounts =
      List.replicate (pairCount + 1) 54 ++ [53] := by
  decide

def expectedEvaluationKinds : List EvaluationKind :=
  (List.range pairCount).map EvaluationKind.input ++
    [.output, .quotient]

theorem evaluation_plan_exact :
    evaluationPlans.length = pairCount + 2 ∧
    evaluationPlans.map (fun plan => plan.kind) = expectedEvaluationKinds ∧
    evaluationPlans.map (fun plan => plan.coefficientCount) =
      evaluationCoefficientCounts ∧
    evaluationPlans.map (fun plan => plan.sourceRowCount) =
      List.replicate (pairCount + 1) 108 ++ [106] ∧
    evaluationPlans.map (fun plan => plan.coefficientZero) =
      List.replicate (pairCount + 2)
        [.subtractFromResult, .absent] ∧
    evaluationPlans.map (fun plan => plan.productCounts) =
      List.replicate (pairCount + 1) [53, 53] ++ [[52, 52]] ∧
    evaluationPlans.map (fun plan => plan.chunkSizes) =
      List.replicate (pairCount + 1)
        [[18, 18, 17], [18, 18, 17]] ++
        [[[18, 18, 16], [18, 18, 16]]] := by
  decide

theorem evaluation_product_indices_exact
    (plan : EvaluationPlan) :
    plan.productCoefficientIndices =
        (List.range plan.coefficientCount).drop 1 ∧
      plan.powerIndicesByLimb =
        List.replicate extensionLimbs plan.productCoefficientIndices := by
  exact ⟨rfl, rfl⟩

theorem evaluation_retained_boundary_exact :
    (evaluationPlans.flatMap fun plan => plan.retainedOrdinals) =
      List.range compactRetainedFields ∧
    (evaluationPlans.flatMap fun plan => plan.retainedColumnOffsets) =
      retainedColumnOffsets ∧
    retainedColumnOffsets.length = compactRetainedFields := by
  decide

def diagonalRetainedBindings : List RetainedBinding :=
  (List.range compactRetainedFields).map fun index =>
    { identity := index
      retainedOrdinal := index
      coefficient := .one }

theorem retained_bindings_are_diagonal :
    retainedBindings = diagonalRetainedBindings := by
  decide

abbrev RetainedIndex := Fin compactRetainedFields

def certificateRetainedMatrix
    {K : Type u} [CommSemiring K] :
    RetainedIndex → RetainedIndex → K :=
  fun identity retained => if identity = retained then 1 else 0

theorem certificate_retained_matrix_full_column_rank
    {K : Type u} [CommSemiring K] :
    RetainedMatrixFullColumnRank
      (certificateRetainedMatrix (K := K)) := by
  intro left right equalImages
  funext retained
  have pointwise := congrFun equalImages retained
  simpa [retainedIdentityMap, certificateRetainedMatrix] using pointwise

theorem certificate_retained_values_unique
    {K : Type u} [CommSemiring K]
    {left right : RetainedIndex → K}
    (equalImages : ∀ identity,
      retainedIdentityMap (certificateRetainedMatrix (K := K)) left identity =
        retainedIdentityMap (certificateRetainedMatrix (K := K)) right identity) :
    left = right :=
  retainedValues_unique_of_fullColumnRank
    certificate_retained_matrix_full_column_rank equalImages

def expectedFinalLimb0Factors : List FinalFactor :=
  (List.range pairCount).flatMap (fun pair =>
    [ { left := .rhoEvaluation pair 0
        right := .inputEvaluation pair 0
        coefficient := .one }
    , { left := .rhoEvaluation pair 1
        right := .inputEvaluation pair 1
        coefficient := .w }
    ]) ++
  [ { left := .quotientEvaluation 0
      right := .phi 0
      coefficient := .negOne }
  , { left := .quotientEvaluation 1
      right := .phi 1
      coefficient := .negW }
  ]

def expectedFinalLimb1Factors : List FinalFactor :=
  (List.range pairCount).flatMap (fun pair =>
    [ { left := .rhoEvaluation pair 0
        right := .inputEvaluation pair 1
        coefficient := .one }
    , { left := .rhoEvaluation pair 1
        right := .inputEvaluation pair 0
        coefficient := .one }
    ]) ++
  [ { left := .quotientEvaluation 0
      right := .phi 1
      coefficient := .negOne }
  , { left := .quotientEvaluation 1
      right := .phi 0
      coefficient := .negOne }
  ]

theorem final_factor_schedule_exact :
    finalLimbPlans.map (fun plan => plan.factors) =
      [expectedFinalLimb0Factors, expectedFinalLimb1Factors] ∧
    finalLimbPlans.map (fun plan => plan.chunkSizes) =
      [[18, 14], [18, 14]] ∧
    finalLimbPlans.map (fun plan => plan.resultRetainedOrdinal) =
      [30, 31] ∧
    finalLimbPlans.map (fun plan => plan.sourceRowOffset) =
      [1914, 1915] := by
  decide

def operandWithinManifest : FinalOperand → Bool
  | .rhoEvaluation pair limb
  | .inputEvaluation pair limb =>
      decide (pair < pairCount ∧ limb < extensionLimbs)
  | .quotientEvaluation limb
  | .phi limb =>
      decide (limb < extensionLimbs)

theorem final_operands_are_exact_and_bounded :
    ∀ plan ∈ finalLimbPlans,
      ∀ factor ∈ plan.factors,
        operandWithinManifest factor.left = true ∧
          operandWithinManifest factor.right = true := by
  decide

def compactEvaluationFieldsFromPlans : Nat :=
  (evaluationPlans.flatMap fun plan =>
    plan.chunkSizes.map List.length).sum

def compactFinalFieldsFromPlans : Nat :=
  (finalLimbPlans.map fun plan => plan.chunkSizes.length - 1).sum

def compactFinalRowsFromPlans : Nat :=
  (finalLimbPlans.map fun plan => plan.chunkSizes.length).sum

def pairTailRows (coordinates : Nat) : Nat :=
  coordinates / 2 + coordinates % 2

def mixedStageColumns (retainedOrdinary syntheticCanonical : Nat) : Nat :=
  retainedOrdinary * ordinarySlotWidth +
    syntheticCanonical * syntheticCanonicalSlotWidth

def mixedStageRows
    (fieldCounts : Nat × Nat) (productSumRows : Nat) : Nat :=
  pairTailRows (fieldCounts.1 * ordinarySlotWidth) +
    pairTailRows (fieldCounts.2 * syntheticCanonicalSlotWidth) +
    fieldCounts.2 * syntheticCanonicalityPairRows +
    productSumRows

theorem compact_cost_formula_exact :
    compactEvaluationFields = compactEvaluationFieldsFromPlans ∧
    compactFinalFields = compactFinalFieldsFromPlans ∧
    compactRetainedFields = retainedColumnOffsets.length ∧
    compactRetainedFields = compactRetainedFieldsByStage.sum ∧
    compactSyntheticFields = compactSyntheticFieldsByStage.sum ∧
    compactEvaluationFields =
      (compactRetainedFieldsByStage.take 3).sum +
        (compactSyntheticFieldsByStage.take 3).sum ∧
    compactFinalFields =
      (compactRetainedFieldsByStage.drop 3).sum +
        (compactSyntheticFieldsByStage.drop 3).sum ∧
    compactEvaluationProductRows =
      (compactProductSumRowsByStage.take 3).sum ∧
    compactFinalProductRows =
      (compactProductSumRowsByStage.drop 3).sum ∧
    compactFinalProductRows = compactFinalRowsFromPlans ∧
    compactProductSumRows = compactProductSumRowsByStage.sum ∧
    compactOrdinaryCoordinatesByStage =
      compactRetainedFieldsByStage.map
        (fun fields => fields * ordinarySlotWidth) ∧
    compactSyntheticCoordinatesByStage =
      compactSyntheticFieldsByStage.map
        (fun fields => fields * syntheticCanonicalSlotWidth) ∧
    compactOrdinaryCenteredRowsByStage =
      compactOrdinaryCoordinatesByStage.map pairTailRows ∧
    compactSyntheticBooleanRowsByStage =
      compactSyntheticCoordinatesByStage.map pairTailRows ∧
    compactSyntheticCanonicalityRowsByStage =
      compactSyntheticFieldsByStage.map
        (fun fields => fields * syntheticCanonicalityPairRows) ∧
    compactEncodedColumnsByStage =
      List.zipWith mixedStageColumns
        compactRetainedFieldsByStage compactSyntheticFieldsByStage ∧
    compactEncodedRowsByStage =
      List.zipWith mixedStageRows
        (List.zip compactRetainedFieldsByStage compactSyntheticFieldsByStage)
        compactProductSumRowsByStage ∧
    compactEvaluationColumns = (compactEncodedColumnsByStage.take 3).sum ∧
    compactFinalColumns = (compactEncodedColumnsByStage.drop 3).sum ∧
    compactEvaluationRows = (compactEncodedRowsByStage.take 3).sum ∧
    compactFinalRows = (compactEncodedRowsByStage.drop 3).sum ∧
    compactEncodedColumns = compactEncodedColumnsByStage.sum ∧
    compactEncodedRows = compactEncodedRowsByStage.sum ∧
    compactEncodedColumns = 8_044 ∧
    compactEncodedRows = 5_248 ∧
    compactAllIdentityColumns =
      compactEncodedColumns * identityCount ∧
    compactAllIdentityRows =
      compactEncodedRows * identityCount := by
  decide

theorem representative_source_schema_sha256_pinned :
    representativeSourceSchemaSha256 =
      "51a93b1438ba70c6624a1f7d238873dc8b860df49d5e7a0615ad652657faf72f" := by
  decide

theorem compact_plan_sha256_pinned :
    compactPlanSha256 =
      "95f4d55fa0f3596c929b0ecaeeccc133c096fd24369c9faa6713e180be9149d8" := by
  decide

theorem complete_certificate_sha256_pinned :
    completeCertificateSha256 =
      "a7b7970f45f283823baab76fd5aa2d4c93d22e319d0554f85a4956ca2c0cfbd9" := by
  decide

theorem certificate_evaluation53_chunk_lengths
    {K : Type u} [CommRing K]
    (evaluation : Evaluation53 K) :
    threeChunkLengths
        (evaluationProductValues evaluation.coefficients evaluation.powers0) =
      [18, 18, 17] ∧
    threeChunkLengths
        (evaluationProductValues evaluation.coefficients evaluation.powers1) =
      [18, 18, 17] :=
  ⟨evaluation53_chunk_lengths evaluation.coefficients evaluation.powers0,
    evaluation53_chunk_lengths evaluation.coefficients evaluation.powers1⟩

theorem certificate_evaluation52_chunk_lengths
    {K : Type u} [CommRing K]
    (evaluation : Evaluation52 K) :
    threeChunkLengths
        (evaluationProductValues evaluation.coefficients evaluation.powers0) =
      [18, 18, 16] ∧
    threeChunkLengths
        (evaluationProductValues evaluation.coefficients evaluation.powers1) =
      [18, 18, 16] :=
  ⟨evaluation52_chunk_lengths evaluation.coefficients evaluation.powers0,
    evaluation52_chunk_lengths evaluation.coefficients evaluation.powers1⟩

theorem certificate_terminal_chunk_lengths
    {K : Type u} [CommRing K]
    (rhoEvaluations : Fin pairCount → QuadraticValue K)
    (inputEvaluations : Fin pairCount → Evaluation53 K)
    (quotientEvaluation : Evaluation52 K)
    (phi : QuadraticValue K) :
    twoChunkLengths
        (terminalLimb0Terms (karatsubaW : K)
          rhoEvaluations
          (fun input => (inputEvaluations input).output)
          quotientEvaluation.output phi) =
      [18, 14] ∧
    twoChunkLengths
        (terminalLimb1Terms
          rhoEvaluations
          (fun input => (inputEvaluations input).output)
          quotientEvaluation.output phi) =
      [18, 14] :=
  ⟨terminalLimb0_chunk_lengths (karatsubaW : K)
      rhoEvaluations
      (fun input => (inputEvaluations input).output)
      quotientEvaluation.output phi,
    terminalLimb1_chunk_lengths
      rhoEvaluations
      (fun input => (inputEvaluations input).output)
      quotientEvaluation.output phi⟩

theorem certificateProjectionIdentity_iff_emitted
    {K : Type u} [CommRing K]
    (rhoEvaluations : Fin pairCount → QuadraticValue K)
    (inputEvaluations : Fin pairCount → Evaluation53 K)
    (outputEvaluation : Evaluation53 K)
    (quotientEvaluation : Evaluation52 K)
    (phi : QuadraticValue K) :
    SourceExactProjectionIdentity (karatsubaW : K)
        rhoEvaluations inputEvaluations outputEvaluation
        quotientEvaluation phi ↔
      EmittedExactProjectionIdentity (karatsubaW : K)
        rhoEvaluations inputEvaluations outputEvaluation
        quotientEvaluation phi :=
  sourceExactProjectionIdentity_iff_emitted (karatsubaW : K)
    rhoEvaluations inputEvaluations outputEvaluation
    quotientEvaluation phi

theorem certificate_emitted_of_source
    {K : Type u} [CommRing K]
    (rhoEvaluations : Fin pairCount → QuadraticValue K)
    (inputEvaluations : Fin pairCount → Evaluation53 K)
    (outputEvaluation : Evaluation53 K)
    (quotientEvaluation : Evaluation52 K)
    (phi : QuadraticValue K)
    (source :
      SourceExactProjectionIdentity (karatsubaW : K)
        rhoEvaluations inputEvaluations outputEvaluation
        quotientEvaluation phi) :
    EmittedExactProjectionIdentity (karatsubaW : K)
      rhoEvaluations inputEvaluations outputEvaluation
      quotientEvaluation phi :=
  (certificateProjectionIdentity_iff_emitted
    rhoEvaluations inputEvaluations outputEvaluation
    quotientEvaluation phi).mp source

/-- Explicit open boundary between the two-limb R1CS model and the polynomial
evaluation statement used by the exact-or-bad-root theorem. -/
def CompactBoundaryBridge
    {K : Type u} {L : Type v}
    [CommRing K] [CommRing L]
    (rhoEvaluations : Fin pairCount → QuadraticValue K)
    (inputEvaluations : Fin pairCount → Evaluation53 K)
    (outputEvaluation : Evaluation53 K)
    (quotientEvaluation : Evaluation52 K)
    (phi : QuadraticValue K)
    (lhs rhs : Polynomial L) (beta : L) : Prop :=
  EmittedExactProjectionIdentity (karatsubaW : K)
      rhoEvaluations inputEvaluations outputEvaluation
      quotientEvaluation phi →
    ProjectionEvaluationAccepted lhs rhs beta

theorem certificateSource_implies_exact_or_badRoot
    {K : Type u} {L : Type v}
    [CommRing K] [CommRing L]
    (rhoEvaluations : Fin pairCount → QuadraticValue K)
    (inputEvaluations : Fin pairCount → Evaluation53 K)
    (outputEvaluation : Evaluation53 K)
    (quotientEvaluation : Evaluation52 K)
    (phi : QuadraticValue K)
    (lhs rhs : Polynomial L) (beta : L)
    (bridge :
      CompactBoundaryBridge rhoEvaluations inputEvaluations
        outputEvaluation quotientEvaluation phi lhs rhs beta)
    (source :
      SourceExactProjectionIdentity (karatsubaW : K)
        rhoEvaluations inputEvaluations outputEvaluation
        quotientEvaluation phi) :
    lhs = rhs ∨ ProjectionBadRoot lhs rhs beta :=
  projectionEvaluationAccepted_implies_exact_or_badRoot lhs rhs beta
    (bridge (certificate_emitted_of_source
      rhoEvaluations inputEvaluations outputEvaluation
      quotientEvaluation phi source))

end SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra.ProjectionIdentityCertificate
