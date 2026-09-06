import NightstreamFPrime.Layout.ProductionRelation.ZeroRunningOracle
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.NumericCompletionSum
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.ZeroRunningPolynomial

/-!
Owns numeric completion sums of the canonical zero-running PiCCS polynomial.
The executable path folds the plan-derived scalar prefixes and sums only
their stored support. Matrix and norm loops have separate extents. The
Boolean cube and the protocol polynomial occur only on the proof side.
This does not compute the separate full ring output families.
-/

namespace NightstreamFPrime.Layout.ProductionRelation.ZeroRunningRoundSum

open NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open ZeroRunningOracle

private abbrev freshIndex : Fin productionShape.freshCount := ⟨0, by decide⟩
private abbrev freshSource := UnifiedSources.freshSourceIndex freshIndex

private def foldedWidth : Nat → List K → Nat
  | width, [] => width
  | width, _ :: fixed => foldedWidth ((width + 1) / 2) fixed

private theorem folded_size (values : Array K) (fixed : List K) :
    (PrefixFold.foldPrefix extensionOps values fixed).size =
      foldedWidth values.size fixed := by
  induction fixed generalizing values with
  | nil => rfl
  | cons challenge fixed inductionHypothesis =>
      simp only [PrefixFold.foldPrefix, foldedWidth, inductionHypothesis,
        PrefixFold.foldOne_size]

private theorem foldedWidth_mono (fixed : List K) (left right : Nat)
    (ordered : left ≤ right) : foldedWidth left fixed ≤ foldedWidth right fixed := by
  induction fixed generalizing left right with
  | nil => exact ordered
  | cons challenge fixed inductionHypothesis =>
      apply inductionHypothesis
      omega

private theorem foldedWidth_fits (fixed : List K) (width arity : Nat)
    (fits : width ≤ 2 ^ (arity + fixed.length)) :
    foldedWidth width fixed ≤ 2 ^ arity := by
  induction fixed generalizing width with
  | nil => simpa only [foldedWidth, List.length_nil, Nat.add_zero] using fits
  | cons challenge fixed inductionHypothesis =>
      apply inductionHypothesis
      have expanded : width ≤ 2 ^ (arity + fixed.length) * 2 := by
        simpa only [List.length_cons, Nat.add_succ, Nat.pow_succ] using fits
      omega

private theorem matrixPrefix_size_le {logicalWidth : Nat} (plan : Plan logicalWidth)
    (assignment : Fin logicalWidth → F)
    (matrix : Fin Spec.ProductionRelation.matrixCount) :
    (matrixPrefix plan assignment matrix).size ≤ plan.rowCount := by
  cases selected : meaningfulPort? matrix with
  | none => simp [matrixPrefix, selected]
  | some meaningful => simp [matrixPrefix, selected]

private def foldedMatrices {logicalWidth : Nat} (plan : Plan logicalWidth)
    (assignment : Fin logicalWidth → F) (fixed : List K) :
    Vector (Array K) Spec.ProductionRelation.matrixCount :=
  Vector.ofFn fun matrix =>
    PrefixFold.foldPrefix extensionOps (matrixPrefix plan assignment matrix) fixed

private theorem foldedMatrices_get {logicalWidth : Nat} (plan : Plan logicalWidth)
    (assignment : Fin logicalWidth → F) (fixed : List K)
    (matrix : Fin Spec.ProductionRelation.matrixCount) :
    (foldedMatrices plan assignment fixed)[matrix.val]'matrix.isLt =
      PrefixFold.foldPrefix extensionOps (matrixPrefix plan assignment matrix) fixed := by
  simp only [foldedMatrices, Vector.getElem_ofFn]

private def matrixValues
    (matrices : Vector (Array K) Spec.ProductionRelation.matrixCount)
    (index : Nat) : Fin Spec.ProductionRelation.matrixCount → K :=
  fun matrix => if matrix = Spec.ProductionRelation.zeroPort then K.zero
    else (matrices[matrix.val]'matrix.isLt).getD index K.zero

private def polynomial : CCSResidualTable.ConstraintPolynomial K
    Spec.ProductionRelation.matrixCount :=
  ConstraintPolynomialLift.liftConstraintPolynomial K.embed Spec.ProductionRelation.polynomial

private def weight (fixed : List K) (alpha : CubePoint K cubeVariables)
    {arity : Nat} (vertex : BooleanVertex arity) : K :=
  SumCheckTruthPath.pointEqualityCoordinates extensionOps
    (fixed ++ vertex.fieldCoordinates extensionOps) alpha.coordinates

private def matrixTerm (fixed : List K) (arity : Nat)
    (alpha : CubePoint K cubeVariables)
    (matrices : Vector (Array K) Spec.ProductionRelation.matrixCount) (index : Nat) : K :=
  if inside : index < 2 ^ arity then
    extensionOps.mul (weight fixed alpha (NumericBooleanDomain.vertex arity ⟨index, inside⟩))
      (CCSResidualTable.evaluatePolynomial extensionOps polynomial (matrixValues matrices index))
  else K.zero

private def normTerm (fixed : List K) (arity : Nat)
    (alpha : CubePoint K cubeVariables) (values : Array K) (index : Nat) : K :=
  if inside : index < 2 ^ arity then
    extensionOps.mul (weight fixed alpha (NumericBooleanDomain.vertex arity ⟨index, inside⟩))
      (ProtocolPolynomial.strictNormResidual extensionOps (values.getD index K.zero))
  else K.zero

/-- Numeric completion sum from the actual plan and assignment. The finite
matrix vector is cached once. No Boolean-domain index list is allocated. -/
def completionSum {logicalWidth : Nat} (plan : Plan logicalWidth)
    (assignment : Fin logicalWidth → F) (fixed : List K) (arity : Nat)
    (alpha : CubePoint K cubeVariables) (gamma : K) : K :=
  let matrices := foldedMatrices plan assignment fixed
  let values := PrefixFold.foldPrefix extensionOps (freshPrefix assignment) fixed
  SignedJointIdentity.gammaTerm extensionOps gamma 12960
    (extensionOps.add
      (NumericCompletionSum.numericSum extensionOps (foldedWidth plan.rowCount fixed)
        (matrixTerm fixed arity alpha matrices))
      (extensionOps.mul gamma
        (NumericCompletionSum.numericSum extensionOps values.size
          (normTerm fixed arity alpha values))))

private theorem getD_outside (values : Array K) (index : Nat)
    (outside : values.size ≤ index) : values.getD index K.zero = K.zero := by
  simp only [Array.getD_eq_getD_getElem?, Array.getElem?_eq_none outside,
    Option.getD_none]

private theorem embed_literal_zero : K.embed (0 : F) = K.zero :=
  ConcreteCarrier.embed_zero

private theorem polynomial_zero :
    CCSResidualTable.evaluatePolynomial extensionOps polynomial (fun _ => K.zero) = K.zero := by
  have lifted := ConstraintPolynomialLift.Evaluation.evaluatePolynomial_lift
    baseOps extensionOps K.embed constraintEvaluationLaws
    Spec.ProductionRelation.polynomial (fun _ => (0 : F))
  rw [Spec.ProductionRelation.polynomial_zeroImages] at lifted
  simpa only [polynomial, embed_literal_zero] using lifted

private theorem strictNorm_zero :
    ProtocolPolynomial.strictNormResidual extensionOps K.zero = K.zero := by
  change extensionOps.mul
    (extensionOps.mul (extensionOps.add extensionOps.zero extensionOps.one) extensionOps.zero)
    (extensionOps.sub extensionOps.zero extensionOps.one) = extensionOps.zero
  rw [extensionLaws.mul_zero, extensionLaws.mul_comm, extensionLaws.mul_zero]

private theorem matrixTerm_outside {logicalWidth : Nat} (plan : Plan logicalWidth)
    (assignment : Fin logicalWidth → F) (fixed : List K) (arity : Nat)
    (alpha : CubePoint K cubeVariables) (index : Nat)
    (outside : foldedWidth plan.rowCount fixed ≤ index)
    (inside : index < 2 ^ arity) :
    matrixTerm fixed arity alpha (foldedMatrices plan assignment fixed) index = K.zero := by
  have zeroImages : matrixValues (foldedMatrices plan assignment fixed) index =
      fun _ => K.zero := by
    funext matrix
    unfold matrixValues
    by_cases zeroPort : matrix = Spec.ProductionRelation.zeroPort
    · rw [if_pos zeroPort]
    · rw [if_neg zeroPort, foldedMatrices_get]
      apply getD_outside
      rw [folded_size]
      exact Nat.le_trans
        (foldedWidth_mono fixed _ _ (matrixPrefix_size_le plan assignment matrix)) outside
  rw [matrixTerm, dif_pos inside, zeroImages, polynomial_zero]
  exact extensionLaws.mul_zero _

private theorem normTerm_outside (fixed : List K) (arity : Nat)
    (alpha : CubePoint K cubeVariables) (values : Array K) (index : Nat)
    (outside : values.size ≤ index) (inside : index < 2 ^ arity) :
    normTerm fixed arity alpha values index = K.zero := by
  rw [normTerm, dif_pos inside, getD_outside values index outside, strictNorm_zero]
  exact extensionLaws.mul_zero _

private def completionPoint (fixed : List K) {arity : Nat}
    (dimension : arity + fixed.length = cubeVariables)
    (vertex : BooleanVertex arity) : CubePoint K cubeVariables :=
  ⟨fixed ++ vertex.fieldCoordinates extensionOps, by
    rw [List.length_append, BooleanVertex.fieldCoordinates_length]
    omega⟩

private theorem folded_getD_eq_evaluate (values : Array K) (fixed : List K)
    {arity total : Nat} (vertex : BooleanVertex arity)
    (dimension : arity + fixed.length = total)
    (fits : values.size ≤ 2 ^ total) :
    (PrefixFold.foldPrefix extensionOps values fixed).getD
        (NumericBooleanDomain.index vertex) K.zero =
      (PrefixFold.zeroExtend extensionOps total values).evaluate extensionOps
        ⟨fixed ++ vertex.fieldCoordinates extensionOps, by
          rw [List.length_append, BooleanVertex.fieldCoordinates_length]
          omega⟩ := by
  subst total
  have result := PrefixFold.foldPrefix_evaluate extensionOps extensionLaws values fixed
    (vertex.toCubePoint extensionOps) fits
  rw [SumCheckTruthPath.evaluate_toCubePoint_eq_valueAt extensionOps extensionLaws] at result
  simpa only [PrefixFold.zeroExtend, BooleanTable.valueAt_tabulate,
    BooleanVertex.toCubePoint_coordinates] using result

section

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

private theorem freshRead_eq (plan : Plan logicalWidth)
    (cubeFits : Phi81CarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (assignment : Fin logicalWidth → F) (fixed : List K) {arity : Nat}
    (dimension : arity + fixed.length = cubeVariables) (vertex : BooleanVertex arity) :
    (PrefixFold.foldPrefix extensionOps (freshPrefix assignment) fixed).getD
        (NumericBooleanDomain.index vertex) K.zero =
      ((protocol plan cubeFits ajtai fresh assignment).sourceAssignments freshSource).evaluate
        extensionOps (completionPoint fixed dimension vertex) := by
  have fits : (freshPrefix assignment).size ≤ 2 ^ cubeVariables := by
    rw [freshPrefix_size]
    exact cubeFits
  have result := folded_getD_eq_evaluate (freshPrefix assignment) fixed vertex dimension fits
  simpa only [freshPrefix_table plan cubeFits ajtai fresh assignment,
    completionPoint] using result

private theorem matrixRead_eq (plan : Plan logicalWidth)
    (cubeFits : Phi81CarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (assignment : Fin logicalWidth → F) (fixed : List K) {arity : Nat}
    (dimension : arity + fixed.length = cubeVariables) (vertex : BooleanVertex arity)
    (matrix : Fin Spec.ProductionRelation.matrixCount) :
    ((foldedMatrices plan assignment fixed)[matrix.val]'matrix.isLt).getD
        (NumericBooleanDomain.index vertex) K.zero =
      ((protocol plan cubeFits ajtai fresh assignment).freshMatrixImages freshIndex matrix).evaluate
        extensionOps (completionPoint fixed dimension vertex) := by
  rw [foldedMatrices_get]
  have fits : (matrixPrefix plan assignment matrix).size ≤ 2 ^ cubeVariables := by
    exact matrixPrefix_fits plan assignment matrix
  have result := folded_getD_eq_evaluate (matrixPrefix plan assignment matrix)
    fixed vertex dimension fits
  simpa only [matrixPrefix_table plan cubeFits ajtai fresh assignment matrix,
    completionPoint] using result

private theorem weighted_add (gamma value ccs norm : K) :
    extensionOps.add (extensionOps.mul value ccs)
        (extensionOps.mul gamma (extensionOps.mul value norm)) =
      extensionOps.mul value (extensionOps.add ccs (extensionOps.mul gamma norm)) := by
  rw [extensionLaws.left_distrib]
  congr 1
  calc
    extensionOps.mul gamma (extensionOps.mul value norm) =
        extensionOps.mul (extensionOps.mul gamma value) norm :=
      (extensionLaws.mul_assoc _ _ _).symm
    _ = extensionOps.mul (extensionOps.mul value gamma) norm := by
      rw [extensionLaws.mul_comm gamma value]
    _ = extensionOps.mul value (extensionOps.mul gamma norm) := extensionLaws.mul_assoc _ _ _

private theorem term_eq_polynomial (plan : Plan logicalWidth)
    (cubeFits : Phi81CarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (assignment : Fin logicalWidth → F) (fixed : List K) {arity : Nat}
    (dimension : arity + fixed.length = cubeVariables)
    (alpha : CubePoint K cubeVariables) (gamma : K) (vertex : BooleanVertex arity) :
    SignedJointIdentity.gammaTerm extensionOps gamma 12960
        (extensionOps.add
          (matrixTerm fixed arity alpha (foldedMatrices plan assignment fixed)
            (NumericBooleanDomain.index vertex))
          (extensionOps.mul gamma
            (normTerm fixed arity alpha
              (PrefixFold.foldPrefix extensionOps (freshPrefix assignment) fixed)
              (NumericBooleanDomain.index vertex)))) =
      ProtocolPolynomial.polynomial extensionOps (protocol plan cubeFits ajtai fresh assignment)
        alpha gamma (fixed ++ vertex.fieldCoordinates extensionOps) := by
  let point := completionPoint fixed dimension vertex
  have reduction := PiCCS.v1_1.ZeroRunningPolynomial.qAtPoint_eq_fresh
    (source plan cubeFits ajtai fresh assignment)
    (source_running_zero plan cubeFits ajtai fresh assignment)
    (source_matrix13_zero plan cubeFits ajtai fresh assignment) alpha point gamma
  have protocolEq : ProtocolDataRefinement.toProtocolData baseOps K.embed
      ((source plan cubeFits ajtai fresh assignment).toUnifiedInputs baseOps) =
      protocol plan cubeFits ajtai fresh assignment := rfl
  dsimp only at reduction
  rw [protocolEq] at reduction
  dsimp only [ProtocolPolynomial.messageAt] at reduction
  change ProtocolPolynomial.qAtPoint extensionOps
      (protocol plan cubeFits ajtai fresh assignment) alpha gamma point =
    SignedJointIdentity.gammaTerm extensionOps gamma 12960
      (extensionOps.mul (SumCheckTruthPath.pointEquality extensionOps point alpha)
        (extensionOps.add
          (CCSResidualTable.evaluatePolynomial extensionOps
            (protocol plan cubeFits ajtai fresh assignment).constraintPolynomial
            (fun matrix => if matrix = Spec.ProductionRelation.zeroPort then K.zero
              else ((protocol plan cubeFits ajtai fresh assignment).freshMatrixImages
                freshIndex matrix).evaluate extensionOps point))
          (extensionOps.mul gamma
            (ProtocolPolynomial.strictNormResidual extensionOps
              (((protocol plan cubeFits ajtai fresh assignment).sourceAssignments
                freshSource).evaluate extensionOps point))))) at reduction
  have images : matrixValues (foldedMatrices plan assignment fixed)
      (NumericBooleanDomain.index vertex) =
      (fun matrix => if matrix = Spec.ProductionRelation.zeroPort then K.zero
        else ((protocol plan cubeFits ajtai fresh assignment).freshMatrixImages freshIndex matrix).evaluate
          extensionOps point) := by
    funext matrix
    unfold matrixValues
    by_cases zeroPort : matrix = Spec.ProductionRelation.zeroPort
    · rw [if_pos zeroPort, if_pos zeroPort]
    · rw [if_neg zeroPort, if_neg zeroPort]
      exact matrixRead_eq plan cubeFits ajtai fresh assignment fixed dimension vertex matrix
  have freshRead := freshRead_eq plan cubeFits ajtai fresh assignment fixed dimension vertex
  have constraint : (protocol plan cubeFits ajtai fresh assignment).constraintPolynomial = polynomial := rfl
  rw [constraint] at reduction
  have inside := NumericBooleanDomain.index_lt_twoPow vertex
  have pointDimension : (fixed ++ vertex.fieldCoordinates extensionOps).length =
      productionShape.cubeVariables := by
    rw [List.length_append, BooleanVertex.fieldCoordinates_length]
    change fixed.length + arity = cubeVariables
    omega
  rw [ProtocolPolynomial.polynomial, dif_pos pointDimension]
  simp only [matrixTerm, normTerm, dif_pos inside, NumericBooleanDomain.vertex_index]
  rw [weighted_add, images, freshRead]
  exact reduction.symm

/-- The executable numeric prefixes sum the exact production polynomial over
all Boolean completions. The omitted matrix and assignment suffixes are
proved zero from their constructors; no table or relation premise is supplied. -/
theorem completionSum_eq_sumCompletions (plan : Plan logicalWidth)
    (cubeFits : Phi81CarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (assignment : Fin logicalWidth → F) (fixed : List K) (arity : Nat)
    (dimension : arity + fixed.length = cubeVariables)
    (alpha : CubePoint K cubeVariables) (gamma : K) :
    completionSum plan assignment fixed arity alpha gamma =
      SumCheck.Finite.HypercubeTruth.sumCompletions extensionOps.toOps
        (ProtocolPolynomial.polynomial extensionOps (protocol plan cubeFits ajtai fresh assignment)
          alpha gamma) fixed arity := by
  let matrices := foldedMatrices plan assignment fixed
  let values := PrefixFold.foldPrefix extensionOps (freshPrefix assignment) fixed
  have rowsFit : foldedWidth plan.rowCount fixed ≤ 2 ^ arity := by
    apply foldedWidth_fits
    rw [dimension]
    exact plan.rowCount_le
  have valuesFit : values.size ≤ 2 ^ arity := by
    apply PrefixFold.foldPrefix_fits
    rw [freshPrefix_size, dimension]
    exact cubeFits
  have matrixSum := NumericCompletionSum.numericSum_prefix_eq_vertexSum
    extensionOps extensionLaws arity (foldedWidth plan.rowCount fixed)
    (matrixTerm fixed arity alpha matrices) rowsFit
    (matrixTerm_outside plan assignment fixed arity alpha)
  have normSum := NumericCompletionSum.numericSum_prefix_eq_vertexSum
    extensionOps extensionLaws arity values.size
    (normTerm fixed arity alpha values) valuesFit (normTerm_outside fixed arity alpha values)
  change SignedJointIdentity.gammaTerm extensionOps gamma 12960
    (extensionOps.add
      (NumericCompletionSum.numericSum extensionOps (foldedWidth plan.rowCount fixed)
        (matrixTerm fixed arity alpha matrices))
      (extensionOps.mul gamma (NumericCompletionSum.numericSum extensionOps values.size
        (normTerm fixed arity alpha values)))) = _
  rw [matrixSum, normSum,
    SumCheckTruthPath.sumCompletions_eq_vertexSum extensionOps extensionLaws]
  rw [← FiniteSumAlgebra.sumMap_mul_left extensionOps extensionLaws gamma,
    ← FiniteSumAlgebra.sumMap_add extensionOps extensionLaws]
  unfold SignedJointIdentity.gammaTerm
  rw [← FiniteSumAlgebra.sumMap_mul_left extensionOps extensionLaws]
  apply FiniteSumAlgebra.sumMap_congr
  intro vertex _
  exact term_eq_polynomial plan cubeFits ajtai fresh assignment fixed dimension alpha gamma vertex

/-- One round evaluates after the prior challenges and the trial value. -/
def roundSum (plan : Plan logicalWidth) (assignment : Fin logicalWidth → F)
    (challenges : List K) (trial : K) (alpha : CubePoint K cubeVariables) (gamma : K) : K :=
  completionSum plan assignment (challenges ++ [trial])
    (cubeVariables - challenges.length - 1) alpha gamma

/-- Exact numeric round-oracle contract at every available round and every
trial value. This is the canonical nonlinear protocol polynomial. -/
theorem roundSum_eq_sumCompletions (plan : Plan logicalWidth)
    (cubeFits : Phi81CarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (assignment : Fin logicalWidth → F) (challenges : List K)
    (available : challenges.length < cubeVariables)
    (trial : K) (alpha : CubePoint K cubeVariables) (gamma : K) :
    roundSum plan assignment challenges trial alpha gamma =
      SumCheck.Finite.HypercubeTruth.sumCompletions extensionOps.toOps
        (ProtocolPolynomial.polynomial extensionOps (protocol plan cubeFits ajtai fresh assignment)
          alpha gamma) (challenges ++ [trial])
        (cubeVariables - challenges.length - 1) := by
  apply completionSum_eq_sumCompletions
  simp only [List.length_append, List.length_singleton]
  omega

end

end NightstreamFPrime.Layout.ProductionRelation.ZeroRunningRoundSum
