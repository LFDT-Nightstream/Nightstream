import NightstreamFPrime.Export.Stage1.PiCCSFreshComplete
import NightstreamFPrime.Export.Stage1.PiCCSCarriedComplete
import NightstreamFPrime.Export.Stage1.PiCCSNormComplete

/-! Proof-only composition of the existing complete contribution kernels.
No decoded artifact, expected value, source-validity assumption or runtime
import of this module is needed. The final target is stated separately. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSFirstRoundComposition

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.SumCheck.Finite
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Lifecycle

private abbrev selectedRelation := PerApplicationFixedPoint.relation
  Poseidon2HashChainV1Package.application Poseidon2HashChainV1Package.fits

/-- The existing witness type, with exactly the captured complete assignments. -/
def witness (masks : Array (Array (Nat × Nat))) :
    StrongReduction.OutputWitness productionShape PiCCSSourceImages.shape.carrierWidth :=
  ⟨PiCCSNormSource.assignments masks⟩

noncomputable def sourceData (input : PiCCSPublicReplay.Input)
    (masks : Array (Array (Nat × Nat))) : ProtocolPolynomial.Data K productionShape :=
  ((ProductionKey.key selectedRelation Poseidon2HashChainV1Setup.productionAjtaiKey).statement
    (PiCCSPublicReplay.running input) (PiCCSPublicReplay.fresh input)).sourceProtocolData
      K.embed (witness masks)

/-- Use the actual existing optional fresh/matrix kernels and total Pad/norm
kernels. Failed row loads remain failed computations, not caller premises. -/
def coefficients? (input : PiCCSPublicReplay.Input)
    (masks : Array (Array (Nat × Nat))) : Option (List K) := do
  let coins := PiCCSPublicReplay.pre input
  let verifier := PiCCSPublicReplay.verifierInput input
  let powers := TargetPolynomial.power extensionOps.toOps coins.gamma
  let fresh ← PiCCSFreshComplete.freshRange? input (witness masks)
    (PiCCSNormSource.canonicalLayout ()) powers coins.alpha 0 PiCCSFreshComplete.activePairs
  let matrixLow ← PiCCSCarriedComplete.matrixMoment? input (witness masks) coins.gamma ⟨0, by decide⟩
  let matrixHigh ← PiCCSCarriedComplete.matrixMoment? input (witness masks) coins.gamma ⟨1, by decide⟩
  let padLow := PiCCSCarriedComplete.padMoment input (witness masks) coins.gamma ⟨0, by decide⟩
  let padHigh := PiCCSCarriedComplete.padMoment input (witness masks) coins.gamma ⟨1, by decide⟩
  let innerNorm := PiCCSNormBuckets.finish powers
    (PiCCSNormScan.range
      (NumericBooleanDomain.tensorWeightCoordinates extensionOps coins.alpha.coordinates.tail)
      masks 0 PiCCSSourceImages.blockCount)
  let carried := PiCCSCarriedMoments.carriedPair extensionOps
    (show 2 ≤ verifier.sumcheckDegreeBound from Nat.le_trans (by decide) (Nat.le_max_right _ _))
    (PiCCSCarriedMoments.headSelector extensionOps verifier.priorPoint)
    (powers productionShape.matrixEvaluationOffset) padLow padHigh matrixLow matrixHigh
  let norm := PiCCSNormContribution.normTerm verifier powers
    (PiCCSCarriedMoments.headSelector extensionOps coins.alpha) innerNorm
  return (FixedPolynomial.add extensionOps.toOps carried
    (FixedPolynomial.add extensionOps.toOps fresh norm)).coefficients

private def freshTerm (input : ProtocolPolynomial.VerifierInput K productionShape)
    (data : ProtocolPolynomial.Data K productionShape) (alpha : CubePoint K cubeVariables)
    (gamma : K) (index : Fin (2 ^ 27)) : FixedPolynomial K input.sumcheckDegreeBound :=
  let suffix := NumericBooleanDomain.vertex 27 index
  PiCCSFreshComplete.outerFresh input (TargetPolynomial.power extensionOps.toOps gamma)
    (PiCCSFirstRoundPair.ccsPolynomialWithPowers extensionOps input
      (TargetPolynomial.power extensionOps.toOps gamma)
      (PiCCSFirstRound.equalitySelector extensionOps suffix alpha)
      (ProtocolPolynomial.vertexMessage data
        (PiCCSFirstRound.endpointVertex (arity := cubeVariables) (by decide) false suffix))
      (ProtocolPolynomial.vertexMessage data
        (PiCCSFirstRound.endpointVertex (arity := cubeVariables) (by decide) true suffix)))

private def normTerm (input : ProtocolPolynomial.VerifierInput K productionShape)
    (data : ProtocolPolynomial.Data K productionShape) (alpha : CubePoint K cubeVariables)
    (gamma : K) (index : Fin (2 ^ 27)) : FixedPolynomial K input.sumcheckDegreeBound :=
  let suffix := NumericBooleanDomain.vertex 27 index
  PiCCSNormContribution.normTerm input (TargetPolynomial.power extensionOps.toOps gamma)
    (PiCCSFirstRound.equalitySelector extensionOps suffix alpha)
    (PiCCSFirstRoundPair.normPolynomialWithPowers extensionOps
      (TargetPolynomial.power extensionOps.toOps gamma)
      (ProtocolPolynomial.vertexMessage data
        (PiCCSFirstRound.endpointVertex (arity := cubeVariables) (by decide) false suffix))
      (ProtocolPolynomial.vertexMessage data
        (PiCCSFirstRound.endpointVertex (arity := cubeVariables) (by decide) true suffix)))

private def components (input : ProtocolPolynomial.VerifierInput K productionShape)
    (data : ProtocolPolynomial.Data K productionShape) (alpha : CubePoint K cubeVariables) (gamma : K) :
    FixedPolynomial K input.sumcheckDegreeBound :=
  FixedPolynomial.add extensionOps.toOps
    (FixedPolynomial.widen extensionOps.toOps (show 2 ≤ input.sumcheckDegreeBound from
      Nat.le_trans (by decide) (Nat.le_max_right _ _))
      (FixedPolynomial.mul extensionOps.toOps
        (PiCCSCarriedMoments.headSelector extensionOps input.priorPoint)
        (FixedPolynomial.affine
          (PiCCSCarriedMoments.moment extensionOps data gamma (by decide : productionShape.cubeVariables = 27 + 1) false)
          (extensionOps.sub (PiCCSCarriedMoments.moment extensionOps data gamma (by decide : productionShape.cubeVariables = 27 + 1) true)
            (PiCCSCarriedMoments.moment extensionOps data gamma (by decide : productionShape.cubeVariables = 27 + 1) false)))))
    (FixedPolynomial.add extensionOps.toOps
      (FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices (2 ^ 27)) (freshTerm input data alpha gamma))
      (FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices (2 ^ 27)) (normTerm input data alpha gamma)))

-- Transport the verifier input while it is a symbolic variable. This avoids
-- reducing the selected key or comparing differently indexed polynomials.
private theorem components_coefficients
    (input : ProtocolPolynomial.VerifierInput K productionShape)
    (data : ProtocolPolynomial.Data K productionShape)
    (agreement : input = data.toVerifierInput)
    (alpha : CubePoint K cubeVariables) (gamma : K) :
    (components input data alpha gamma).coefficients =
      (PiCCSFirstRound.firstRound extensionOps data alpha gamma (remaining := 27) (by decide)).coefficients := by
  subst input
  exact congrArg FixedPolynomial.coefficients
    (PiCCSCarriedMoments.full_components_eq_firstRound extensionOps extensionLaws data alpha gamma
      (remaining := 27) (by decide))

private theorem input_agreement (input : PiCCSPublicReplay.Input)
    (masks : Array (Array (Nat × Nat))) :
    PiCCSPublicReplay.verifierInput input = (sourceData input masks).toVerifierInput := by
  exact PiCCSFreshComplete.verifierInput_eq_sourceProtocolData input (witness masks)

private theorem fresh_reference (input : PiCCSPublicReplay.Input)
    (masks : Array (Array (Nat × Nat))) (alpha : CubePoint K cubeVariables) (gamma : K)
    (index : Fin (2 ^ 27)) :
    PiCCSFreshComplete.referencePair input (witness masks)
        (TargetPolynomial.power extensionOps.toOps gamma) alpha index.val =
      freshTerm (PiCCSPublicReplay.verifierInput input) (sourceData input masks) alpha gamma index := by
  have inside : index.val < 2 ^ (cubeVariables - 1) := index.isLt
  simp only [PiCCSFreshComplete.referencePair, dif_pos inside]
  rfl

private theorem fresh_value (input : PiCCSPublicReplay.Input)
    (masks : Array (Array (Nat × Nat))) (alpha : CubePoint K cubeVariables) (gamma : K) :
    PiCCSFreshComplete.freshRange? input (witness masks) (PiCCSNormSource.canonicalLayout ())
        (TargetPolynomial.power extensionOps.toOps gamma) alpha 0 PiCCSFreshComplete.activePairs =
      some (FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices (2 ^ 27))
        (freshTerm (PiCCSPublicReplay.verifierInput input) (sourceData input masks) alpha gamma)) := by
  rw [PiCCSFreshComplete.freshRange_eq_fullPairSum,
    PiCCSPolynomialRange.range_eq_sum extensionOps extensionLaws]
  apply congrArg some
  apply congrArg (FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices (2 ^ 27)))
  funext index
  simpa only [Nat.zero_add] using fresh_reference input masks alpha gamma index

private theorem nonlinear_norm_fields
    (layout : UnifiedSources.ColumnLayout cubeVariables PiCCSSourceImages.shape.carrierWidth)
    (assignments : Fin productionShape.sourceCount → Phi81Relation.Assignment PiCCSSourceImages.shape)
    (vertex : BooleanVertex cubeVariables) (fresh : Vector F ProductionRelation.matrixCount)
    (wanted : ProtocolPolynomial.OutputMessage K productionShape)
    (fields : ∀ source, K.embed (PiCCSSourceImages.assignmentValue layout (assignments source) vertex) =
      wanted.sourceAssignment source) :
    (PiCCSAggregatedImages.nonlinearMessage layout assignments vertex fresh).sourceAssignment =
      wanted.sourceAssignment := by
  funext source
  exact fields source

private theorem norm_fields (input : PiCCSPublicReplay.Input)
    (masks : Array (Array (Nat × Nat))) (vertex : BooleanVertex cubeVariables)
    (fresh : Vector F ProductionRelation.matrixCount) :
    (PiCCSAggregatedImages.nonlinearMessage (PiCCSNormSource.canonicalLayout ())
      (PiCCSNormSource.assignments masks) vertex fresh).sourceAssignment =
      (ProtocolPolynomial.vertexMessage (sourceData input masks) vertex).sourceAssignment := by
  apply nonlinear_norm_fields
  intro source
  exact PiCCSSourceImages.assignment_sourceProtocolData input (witness masks) source vertex

private theorem norm_congr (powers : Nat → K)
    (low high low' high' : ProtocolPolynomial.OutputMessage K productionShape)
    (lowFields : low.sourceAssignment = low'.sourceAssignment)
    (highFields : high.sourceAssignment = high'.sourceAssignment) :
    PiCCSFirstRoundPair.normPolynomialWithPowers extensionOps powers low high =
      PiCCSFirstRoundPair.normPolynomialWithPowers extensionOps powers low' high' := by
  unfold PiCCSFirstRoundPair.normPolynomialWithPowers
  rw [lowFields, highFields]

private theorem norm_reference (input : PiCCSPublicReplay.Input)
    (masks : Array (Array (Nat × Nat))) (alpha : CubePoint K cubeVariables) (gamma : K)
    (index : Fin (2 ^ 27)) :
    PiCCSNormComplete.numericPairNorm (PiCCSPublicReplay.verifierInput input)
        (TargetPolynomial.power extensionOps.toOps gamma) alpha masks
        (fun _ => Vector.replicate ProductionRelation.matrixCount (0 : F))
        (fun _ => Vector.replicate ProductionRelation.matrixCount (0 : F)) index.val =
      normTerm (PiCCSPublicReplay.verifierInput input) (sourceData input masks) alpha gamma index := by
  have inside : index.val < 2 ^ (cubeVariables - 1) := index.isLt
  simp only [PiCCSNormComplete.numericPairNorm, dif_pos inside, normTerm]
  apply congrArg (PiCCSNormContribution.normTerm (PiCCSPublicReplay.verifierInput input)
    (TargetPolynomial.power extensionOps.toOps gamma)
    (PiCCSFirstRound.equalitySelector extensionOps (NumericBooleanDomain.vertex 27 index) alpha))
  exact norm_congr _ _ _ _ _ (norm_fields input masks _ _) (norm_fields input masks _ _)

private theorem headSelector_agreement (alpha : CubePoint K cubeVariables) :
    PiCCSCarriedMoments.headSelector extensionOps alpha = PiCCSNormContribution.headSelector alpha := by
  rcases alpha with ⟨coordinates, length⟩
  cases coordinates <;> rfl

private theorem norm_value (input : PiCCSPublicReplay.Input)
    (masks : Array (Array (Nat × Nat))) (alpha : CubePoint K cubeVariables) (gamma : K) :
    PiCCSNormContribution.normTerm (PiCCSPublicReplay.verifierInput input)
        (TargetPolynomial.power extensionOps.toOps gamma)
        (PiCCSCarriedMoments.headSelector extensionOps alpha)
        (PiCCSNormBuckets.finish (TargetPolynomial.power extensionOps.toOps gamma)
          (PiCCSNormScan.range
            (NumericBooleanDomain.tensorWeightCoordinates extensionOps alpha.coordinates.tail)
            masks 0 PiCCSSourceImages.blockCount)) =
      FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices (2 ^ 27))
        (normTerm (PiCCSPublicReplay.verifierInput input) (sourceData input masks) alpha gamma) := by
  rw [headSelector_agreement]
  rw [PiCCSNormComplete.finished_norm_eq_fullPairSum
    (PiCCSPublicReplay.verifierInput input) gamma alpha masks
    (fun _ => Vector.replicate ProductionRelation.matrixCount (0 : F))
    (fun _ => Vector.replicate ProductionRelation.matrixCount (0 : F)),
    PiCCSPolynomialRange.range_eq_sum extensionOps extensionLaws]
  apply congrArg (FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices (2 ^ 27)))
  funext index
  simpa only [Nat.zero_add] using norm_reference input masks alpha gamma index

/-- Every coefficient of the actual complete constructor is the original
source first-round coefficient, using the Lean-derived public pre-coins.
No expected artifact or contribution/witness-validity hypothesis is accepted. -/
theorem coefficients_eq_firstRound (input : PiCCSPublicReplay.Input)
    (masks : Array (Array (Nat × Nat))) :
    coefficients? input masks =
      some ((PiCCSFirstRound.firstRound extensionOps (sourceData input masks)
        (PiCCSPublicReplay.pre input).alpha (PiCCSPublicReplay.pre input).gamma
        (remaining := 27) (by decide)).coefficients) := by
  let coins := PiCCSPublicReplay.pre input
  let verifier := PiCCSPublicReplay.verifierInput input
  let power := TargetPolynomial.power extensionOps.toOps coins.gamma
  let data := sourceData input masks
  have lowMoment :
      (PiCCSCarriedComplete.matrixMoment? input (witness masks) coins.gamma ⟨0, by decide⟩).map
        (fun matrix => extensionOps.add
          (PiCCSCarriedComplete.padMoment input (witness masks) coins.gamma ⟨0, by decide⟩)
          (extensionOps.mul (power productionShape.matrixEvaluationOffset) matrix)) =
        some (PiCCSCarriedMoments.moment extensionOps data coins.gamma (by decide : productionShape.cubeVariables = 27 + 1) false) := by
    exact PiCCSCarriedComplete.selected_moment input (witness masks) coins.gamma ⟨0, by decide⟩
  have highMoment :
      (PiCCSCarriedComplete.matrixMoment? input (witness masks) coins.gamma ⟨1, by decide⟩).map
        (fun matrix => extensionOps.add
          (PiCCSCarriedComplete.padMoment input (witness masks) coins.gamma ⟨1, by decide⟩)
          (extensionOps.mul (power productionShape.matrixEvaluationOffset) matrix)) =
        some (PiCCSCarriedMoments.moment extensionOps data coins.gamma (by decide : productionShape.cubeVariables = 27 + 1) true) := by
    exact PiCCSCarriedComplete.selected_moment input (witness masks) coins.gamma ⟨1, by decide⟩
  obtain ⟨matrixLow, lowLoaded, lowEqual⟩ := Option.map_eq_some_iff.mp lowMoment
  obtain ⟨matrixHigh, highLoaded, highEqual⟩ := Option.map_eq_some_iff.mp highMoment
  have carried :
      PiCCSCarriedMoments.carriedPair extensionOps
          (show 2 ≤ verifier.sumcheckDegreeBound from Nat.le_trans (by decide) (Nat.le_max_right _ _))
          (PiCCSCarriedMoments.headSelector extensionOps verifier.priorPoint)
          (power productionShape.matrixEvaluationOffset)
          (PiCCSCarriedComplete.padMoment input (witness masks) coins.gamma ⟨0, by decide⟩)
          (PiCCSCarriedComplete.padMoment input (witness masks) coins.gamma ⟨1, by decide⟩)
          matrixLow matrixHigh =
        FixedPolynomial.widen extensionOps.toOps
          (show 2 ≤ verifier.sumcheckDegreeBound from Nat.le_trans (by decide) (Nat.le_max_right _ _))
          (FixedPolynomial.mul extensionOps.toOps
            (PiCCSCarriedMoments.headSelector extensionOps verifier.priorPoint)
            (FixedPolynomial.affine
              (PiCCSCarriedMoments.moment extensionOps data coins.gamma (by decide : productionShape.cubeVariables = 27 + 1) false)
              (extensionOps.sub (PiCCSCarriedMoments.moment extensionOps data coins.gamma (by decide : productionShape.cubeVariables = 27 + 1) true)
                (PiCCSCarriedMoments.moment extensionOps data coins.gamma (by decide : productionShape.cubeVariables = 27 + 1) false)))) := by
    rw [PiCCSCarriedMoments.carriedPair_combined extensionOps extensionLaws, lowEqual, highEqual]
  unfold coefficients?
  dsimp only
  rw [fresh_value input masks coins.alpha coins.gamma, lowLoaded, highLoaded]
  simp only [bind, Option.bind]
  apply congrArg some
  rw [carried]
  calc
    _ = (components verifier data coins.alpha coins.gamma).coefficients := by
      apply congrArg FixedPolynomial.coefficients
      apply congrArg (FixedPolynomial.add extensionOps.toOps _)
      exact congrArg (FixedPolynomial.add extensionOps.toOps _)
        (norm_value input masks coins.alpha coins.gamma)
    _ = _ := components_coefficients verifier data (input_agreement input masks) coins.alpha coins.gamma

end NightstreamFPrime.Export.Stage1.PiCCSFirstRoundComposition
