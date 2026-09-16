import NightstreamFPrime.Export.Stage1.PiCCSPolynomialRange
import NightstreamFPrime.Export.Stage1.PiCCSCachedSelector

/-! Mathematical factoring of the complete first-round carried contribution
into two weighted endpoint moments. No source reader or runtime scan is added. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSCarriedMoments

open NightstreamFPrime.Spec.SumCheck.Finite
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open ProtocolPolynomialDegree.Support (polynomialLaws)
open PiCCSPolynomialRange (coefficients_ext add_coefficients scale_coefficients)
open FiniteSumAlgebra (sumMap)

universe uField
variable {Field : Type uField} {Index : Type}

private def affine (ops : InterpolationOps Field) (low high : Field) : FixedPolynomial Field 1 :=
  FixedPolynomial.affine low (ops.sub high low)

private theorem add_shuffle (ops : InterpolationOps Field) (laws : InterpolationEvaluationLaws ops)
    (a b c d : Field) :
    ops.add (ops.add a b) (ops.add c d) = ops.add (ops.add a c) (ops.add b d) := by
  simp only [laws.add_assoc]
  rw [← laws.add_assoc b c d, laws.add_comm b c, laws.add_assoc]

private theorem mul_coefficients (ops : InterpolationOps Field) (a b c d : Field) :
    (FixedPolynomial.mul ops.toOps (FixedPolynomial.affine a b)
      (FixedPolynomial.affine c d)).coefficients =
      [ops.add (ops.mul a c) ops.zero,
       ops.add (ops.mul a d) (ops.add (ops.mul b c) ops.zero), ops.mul b d] := rfl

private theorem mul_add (ops : InterpolationOps Field) (laws : InterpolationEvaluationLaws ops)
    (head left right : FixedPolynomial Field 1) :
    FixedPolynomial.mul ops.toOps head (FixedPolynomial.add ops.toOps left right) =
      FixedPolynomial.add ops.toOps (FixedPolynomial.mul ops.toOps head left)
        (FixedPolynomial.mul ops.toOps head right) := by
  rcases head with ⟨head, headLength⟩
  obtain ⟨a, b, rfl⟩ := List.length_eq_two.mp headLength
  rcases left with ⟨left, leftLength⟩
  obtain ⟨c, d, rfl⟩ := List.length_eq_two.mp leftLength
  rcases right with ⟨right, rightLength⟩
  obtain ⟨e, f, rfl⟩ := List.length_eq_two.mp rightLength
  apply coefficients_ext
  rw [add_coefficients]
  change (FixedPolynomial.mul ops.toOps (FixedPolynomial.affine a b)
      (FixedPolynomial.affine (ops.add c e) (ops.add d f))).coefficients =
    List.zipWith ops.add
      (FixedPolynomial.mul ops.toOps (FixedPolynomial.affine a b) (FixedPolynomial.affine c d)).coefficients
      (FixedPolynomial.mul ops.toOps (FixedPolynomial.affine a b) (FixedPolynomial.affine e f)).coefficients
  rw [mul_coefficients, mul_coefficients, mul_coefficients]
  simp only [List.zipWith_cons_cons, List.zipWith_nil_left, laws.left_distrib, laws.add_zero]
  rw [add_shuffle ops laws (ops.mul a d) (ops.mul a f) (ops.mul b c) (ops.mul b e)]

private theorem mul_scale_right (ops : InterpolationOps Field) (laws : InterpolationEvaluationLaws ops)
    (scalar : Field) (head value : FixedPolynomial Field 1) :
    FixedPolynomial.mul ops.toOps head (FixedPolynomial.scale ops.toOps scalar value) =
      FixedPolynomial.scale ops.toOps scalar (FixedPolynomial.mul ops.toOps head value) := by
  rcases head with ⟨head, headLength⟩
  obtain ⟨a, b, rfl⟩ := List.length_eq_two.mp headLength
  rcases value with ⟨value, valueLength⟩
  obtain ⟨c, d, rfl⟩ := List.length_eq_two.mp valueLength
  apply coefficients_ext
  rw [scale_coefficients]
  change (FixedPolynomial.mul ops.toOps (FixedPolynomial.affine a b)
      (FixedPolynomial.affine (ops.mul scalar c) (ops.mul scalar d))).coefficients =
    (FixedPolynomial.mul ops.toOps (FixedPolynomial.affine a b) (FixedPolynomial.affine c d)).coefficients.map
      (ops.mul scalar)
  rw [mul_coefficients, mul_coefficients]
  have commute (x y : Field) : ops.mul x (ops.mul scalar y) = ops.mul scalar (ops.mul x y) := by
    rw [← laws.mul_assoc, laws.mul_comm x scalar, laws.mul_assoc]
  simp only [List.map_cons, List.map_nil, laws.add_zero, laws.left_distrib, commute]

private theorem mul_scale_left (ops : InterpolationOps Field) (laws : InterpolationEvaluationLaws ops)
    (scalar : Field) (head value : FixedPolynomial Field 1) :
    FixedPolynomial.mul ops.toOps (FixedPolynomial.scale ops.toOps scalar head) value =
      FixedPolynomial.scale ops.toOps scalar (FixedPolynomial.mul ops.toOps head value) := by
  rcases head with ⟨head, headLength⟩
  obtain ⟨a, b, rfl⟩ := List.length_eq_two.mp headLength
  rcases value with ⟨value, valueLength⟩
  obtain ⟨c, d, rfl⟩ := List.length_eq_two.mp valueLength
  apply coefficients_ext
  rw [scale_coefficients]
  change (FixedPolynomial.mul ops.toOps
      (FixedPolynomial.affine (ops.mul scalar a) (ops.mul scalar b)) (FixedPolynomial.affine c d)).coefficients =
    (FixedPolynomial.mul ops.toOps (FixedPolynomial.affine a b) (FixedPolynomial.affine c d)).coefficients.map
      (ops.mul scalar)
  rw [mul_coefficients, mul_coefficients]
  simp only [List.map_cons, List.map_nil, laws.add_zero, laws.left_distrib, laws.mul_assoc]

private theorem widen_zero (ops : InterpolationOps Field) {degree target : Nat} (bound : degree ≤ target) :
    FixedPolynomial.widen ops.toOps bound (FixedPolynomial.zero ops.toOps degree) =
      FixedPolynomial.zero ops.toOps target := by
  apply coefficients_ext
  change List.replicate (degree + 1) ops.zero ++ List.replicate (target - degree) ops.zero =
    List.replicate (target + 1) ops.zero
  rw [← List.replicate_add]
  congr 1
  omega

private theorem widen_add (ops : InterpolationOps Field) (laws : InterpolationEvaluationLaws ops)
    {degree target : Nat} (bound : degree ≤ target) (left right : FixedPolynomial Field degree) :
    FixedPolynomial.widen ops.toOps bound (FixedPolynomial.add ops.toOps left right) =
      FixedPolynomial.add ops.toOps (FixedPolynomial.widen ops.toOps bound left)
        (FixedPolynomial.widen ops.toOps bound right) := by
  apply coefficients_ext
  rw [add_coefficients]
  change (FixedPolynomial.add ops.toOps left right).coefficients ++ List.replicate (target - degree) ops.zero =
    List.zipWith ops.add (left.coefficients ++ List.replicate (target - degree) ops.zero)
      (right.coefficients ++ List.replicate (target - degree) ops.zero)
  rw [add_coefficients, List.zipWith_append (by rw [left.coefficients_length, right.coefficients_length])]
  simp only [List.zipWith_replicate', laws.add_zero]

private theorem widen_scale (ops : InterpolationOps Field) (laws : InterpolationEvaluationLaws ops)
    {degree target : Nat} (bound : degree ≤ target) (scalar : Field) (value : FixedPolynomial Field degree) :
    FixedPolynomial.widen ops.toOps bound (FixedPolynomial.scale ops.toOps scalar value) =
      FixedPolynomial.scale ops.toOps scalar (FixedPolynomial.widen ops.toOps bound value) := by
  apply coefficients_ext
  rw [scale_coefficients]
  change (FixedPolynomial.scale ops.toOps scalar value).coefficients ++ List.replicate (target - degree) ops.zero =
    (value.coefficients ++ List.replicate (target - degree) ops.zero).map (ops.mul scalar)
  rw [scale_coefficients, List.map_append, List.map_replicate, laws.mul_zero]

private def applyHead (ops : InterpolationOps Field) {degree : Nat} (bound : 2 ≤ degree)
    (head value : FixedPolynomial Field 1) : FixedPolynomial Field degree :=
  FixedPolynomial.widen ops.toOps bound (FixedPolynomial.mul ops.toOps head value)

private theorem applyHead_add (ops : InterpolationOps Field) (laws : InterpolationEvaluationLaws ops)
    {degree : Nat} (bound : 2 ≤ degree) (head left right : FixedPolynomial Field 1) :
    applyHead ops bound head (FixedPolynomial.add ops.toOps left right) =
      FixedPolynomial.add ops.toOps (applyHead ops bound head left) (applyHead ops bound head right) := by
  unfold applyHead
  rw [mul_add ops laws, widen_add ops laws]

private theorem applyHead_scale (ops : InterpolationOps Field) (laws : InterpolationEvaluationLaws ops)
    {degree : Nat} (bound : 2 ≤ degree) (head value : FixedPolynomial Field 1) (scalar : Field) :
    applyHead ops bound head (FixedPolynomial.scale ops.toOps scalar value) =
      FixedPolynomial.scale ops.toOps scalar (applyHead ops bound head value) := by
  unfold applyHead
  rw [mul_scale_right ops laws, widen_scale ops laws]

private theorem applyHead_scaled_head (ops : InterpolationOps Field) (laws : InterpolationEvaluationLaws ops)
    {degree : Nat} (bound : 2 ≤ degree) (head value : FixedPolynomial Field 1) (scalar : Field) :
    applyHead ops bound (FixedPolynomial.scale ops.toOps scalar head) value =
      applyHead ops bound head (FixedPolynomial.scale ops.toOps scalar value) := by
  unfold applyHead
  rw [mul_scale_left ops laws, mul_scale_right ops laws]

private theorem applyHead_sum (ops : InterpolationOps Field) (laws : InterpolationEvaluationLaws ops)
    {degree : Nat} (bound : 2 ≤ degree) (head : FixedPolynomial Field 1)
    (indices : List Index) (values : Index → FixedPolynomial Field 1) :
    applyHead ops bound head (FixedPolynomial.sum ops.toOps indices values) =
      FixedPolynomial.sum ops.toOps indices (fun index => applyHead ops bound head (values index)) := by
  induction indices with
  | nil =>
      unfold applyHead
      rw [FixedPolynomial.sum, FixedPolynomial.mul_zero_right ops.toOps (polynomialLaws laws), widen_zero]
      rfl
  | cons index rest ih => rw [FixedPolynomial.sum, FixedPolynomial.sum, applyHead_add ops laws, ih]

private theorem affine_combine (ops : InterpolationOps Field) (laws : InterpolationEvaluationLaws ops)
    (scalar low high otherLow otherHigh : Field) :
    FixedPolynomial.add ops.toOps (affine ops low high)
        (FixedPolynomial.scale ops.toOps scalar (affine ops otherLow otherHigh)) =
      affine ops (ops.add low (ops.mul scalar otherLow)) (ops.add high (ops.mul scalar otherHigh)) := by
  apply coefficients_ext
  change [ops.add low (ops.mul scalar otherLow),
      ops.add (ops.sub high low) (ops.mul scalar (ops.sub otherHigh otherLow))] =
    [ops.add low (ops.mul scalar otherLow),
      ops.sub (ops.add high (ops.mul scalar otherHigh)) (ops.add low (ops.mul scalar otherLow))]
  rw [FiniteSumAlgebra.mul_sub ops laws]
  simp only [InterpolationOps.sub, laws.neg_add]
  rw [add_shuffle ops laws]

private theorem sum_affine (ops : InterpolationOps Field) (laws : InterpolationEvaluationLaws ops)
    (indices : List Index) (weight low high : Index → Field) :
    FixedPolynomial.sum ops.toOps indices (fun index =>
      FixedPolynomial.scale ops.toOps (weight index) (affine ops (low index) (high index))) =
      affine ops (sumMap ops indices (fun index => ops.mul (weight index) (low index)))
        (sumMap ops indices (fun index => ops.mul (weight index) (high index))) := by
  have coefficients : FixedPolynomial.sum ops.toOps indices (fun index =>
      FixedPolynomial.scale ops.toOps (weight index) (affine ops (low index) (high index))) =
      FixedPolynomial.affine (sumMap ops indices (fun index => ops.mul (weight index) (low index)))
        (sumMap ops indices (fun index => ops.mul (weight index) (ops.sub (high index) (low index)))) := by
    induction indices with
    | nil => rfl
    | cons index rest ih => rw [FixedPolynomial.sum, ih]; rfl
  rw [coefficients]
  unfold affine
  apply congrArg (FixedPolynomial.affine (sumMap ops indices (fun index => ops.mul (weight index) (low index))))
  calc
    _ = sumMap ops indices (fun index => ops.sub (ops.mul (weight index) (high index))
          (ops.mul (weight index) (low index))) :=
      FiniteSumAlgebra.sumMap_congr ops _ _ _ (fun index _ => FiniteSumAlgebra.mul_sub ops laws _ _ _)
    _ = _ := FiniteSumAlgebra.sumMap_sub ops laws indices _ _

/-- Exactly the first two addends of pairPolynomialWithNorm. The matrix
scalar is the global matrixEvaluationOffset shift, applied once. -/
def carriedPair (ops : InterpolationOps Field) {degree : Nat} (bound : 2 ≤ degree)
    (selector : FixedPolynomial Field 1) (matrixShift padLow padHigh matrixLow matrixHigh : Field) :
    FixedPolynomial Field degree :=
  FixedPolynomial.add ops.toOps (applyHead ops bound selector (affine ops padLow padHigh))
    (FixedPolynomial.scale ops.toOps matrixShift (applyHead ops bound selector (affine ops matrixLow matrixHigh)))

private theorem carriedPair_factor (ops : InterpolationOps Field) (laws : InterpolationEvaluationLaws ops)
    {degree : Nat} (bound : 2 ≤ degree) (head : FixedPolynomial Field 1)
    (weight matrixShift padLow padHigh matrixLow matrixHigh : Field) :
    carriedPair ops bound (FixedPolynomial.scale ops.toOps weight head)
        matrixShift padLow padHigh matrixLow matrixHigh =
      applyHead ops bound head (FixedPolynomial.scale ops.toOps weight
        (affine ops (ops.add padLow (ops.mul matrixShift matrixLow))
          (ops.add padHigh (ops.mul matrixShift matrixHigh)))) := by
  unfold carriedPair
  rw [← applyHead_scale ops laws, ← applyHead_add ops laws, affine_combine ops laws,
    applyHead_scaled_head ops laws]

/-- General finite carried sum. Repeated indices and empty lists keep their
original meaning. No endpoint-validity or coverage premise is required. -/
theorem sum_carriedPair (ops : InterpolationOps Field) (laws : InterpolationEvaluationLaws ops)
    {degree : Nat} (bound : 2 ≤ degree) (head : FixedPolynomial Field 1)
    (indices : List Index) (weight : Index → Field) (matrixShift : Field)
    (padLow padHigh matrixLow matrixHigh : Index → Field) :
    FixedPolynomial.sum ops.toOps indices (fun index =>
      carriedPair ops bound (FixedPolynomial.scale ops.toOps (weight index) head) matrixShift
        (padLow index) (padHigh index) (matrixLow index) (matrixHigh index)) =
      FixedPolynomial.widen ops.toOps bound (FixedPolynomial.mul ops.toOps head
        (affine ops
          (sumMap ops indices (fun index => ops.mul (weight index)
            (ops.add (padLow index) (ops.mul matrixShift (matrixLow index)))))
          (sumMap ops indices (fun index => ops.mul (weight index)
            (ops.add (padHigh index) (ops.mul matrixShift (matrixHigh index))))))) := by
  simp only [carriedPair_factor ops laws]
  rw [← applyHead_sum ops laws, sum_affine ops laws]
  rfl

/-- The exact head factor of the original equality selector. -/
def headSelector (ops : InterpolationOps Field) {arity : Nat}
    (prior : CubePoint Field arity) : FixedPolynomial Field 1 :=
  match prior.coordinates with
  | [] => FixedPolynomial.zero ops.toOps 1
  | head :: _ => affine ops (ops.sub ops.one head) head

private theorem equalitySelector_factor (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) {arity remaining : Nat}
    (dimension : arity = remaining + 1) (suffix : BooleanVertex remaining)
    (prior : CubePoint Field arity) :
    PiCCSFirstRound.equalitySelector ops suffix prior =
      FixedPolynomial.scale ops.toOps
        (NumericBooleanDomain.tensorWeightCoordinates ops prior.coordinates.tail
          (NumericBooleanDomain.index suffix)) (headSelector ops prior) := by
  rw [← PiCCSCachedSelector.equalitySelector_prepare ops laws dimension suffix prior]
  rcases prior with ⟨coordinates, length⟩
  cases coordinates with
  | nil => simp only [List.length_nil] at length; omega
  | cons first rest =>
      change FixedPolynomial.scale ops.toOps
        (PiCCSTensorWeights.lookup ops rest (PiCCSTensorWeights.prepare ops rest)
          (NumericBooleanDomain.index suffix))
        (affine ops (ops.sub ops.one first) first) = _
      rw [PiCCSTensorWeights.lookup_prepare ops
        (NumericBooleanDomain.WeightProductLaws.ofInterpolationEvaluationLaws laws)]
      rfl

private def padValue (ops : InterpolationOps Field) {shape : Shape} (gamma : Field)
    (message : ProtocolPolynomial.OutputMessage Field shape) : Field :=
  sumMap ops (canonicalPadCoordinates shape) (fun coordinate =>
    ops.mul (TargetPolynomial.power ops.toOps gamma coordinate.localGammaExponent) (message.padImage coordinate))

private def matrixValue (ops : InterpolationOps Field) {shape : Shape} (gamma : Field)
    (message : ProtocolPolynomial.OutputMessage Field shape) : Field :=
  sumMap ops (canonicalMatrixCoordinates shape) (fun coordinate =>
    ops.mul (TargetPolynomial.power ops.toOps gamma coordinate.localGammaExponent) (message.matrixImage coordinate))

/-- Mathematical endpoint moment on the full numeric suffix domain.
The original canonical coordinate families include every carried source. -/
def moment (ops : InterpolationOps Field) {shape : Shape}
    (data : ProtocolPolynomial.Data Field shape) (gamma : Field) {remaining : Nat}
    (dimension : shape.cubeVariables = remaining + 1) (bit : Bool) : Field :=
  sumMap ops (canonicalFinIndices (2 ^ remaining)) fun index =>
    let suffix := NumericBooleanDomain.vertex remaining index
    let message := ProtocolPolynomial.vertexMessage data (PiCCSFirstRound.endpointVertex dimension bit suffix)
    ops.mul (NumericBooleanDomain.tensorWeightCoordinates ops data.priorPoint.coordinates.tail index.val)
      (ops.add (padValue ops gamma message)
        (ops.mul (TargetPolynomial.power ops.toOps gamma shape.matrixEvaluationOffset)
          (matrixValue ops gamma message)))

/-- Full mathematical carried contribution. Instantiating data with the
selected statement's sourceProtocolData uses the original assignments; no
matrix-image equality is an extra premise. The two endpoint moments replace
the full carried pair sum, preserving every fixed-width coefficient. -/
theorem complete_carried_sum (ops : InterpolationOps Field) (laws : InterpolationEvaluationLaws ops)
    {shape : Shape} (data : ProtocolPolynomial.Data Field shape) (gamma : Field)
    {remaining : Nat} (dimension : shape.cubeVariables = remaining + 1) :
    FixedPolynomial.sum ops.toOps (canonicalFinIndices (2 ^ remaining)) (fun index =>
      let suffix := NumericBooleanDomain.vertex remaining index
      let low := ProtocolPolynomial.vertexMessage data (PiCCSFirstRound.endpointVertex dimension false suffix)
      let high := ProtocolPolynomial.vertexMessage data (PiCCSFirstRound.endpointVertex dimension true suffix)
      carriedPair ops (show 2 ≤ data.toVerifierInput.sumcheckDegreeBound from
        Nat.le_trans (by decide) (Nat.le_max_right _ _))
        (PiCCSFirstRound.equalitySelector ops suffix data.priorPoint)
        (TargetPolynomial.power ops.toOps gamma shape.matrixEvaluationOffset)
        (padValue ops gamma low) (padValue ops gamma high)
        (matrixValue ops gamma low) (matrixValue ops gamma high)) =
      FixedPolynomial.widen ops.toOps (show 2 ≤ data.toVerifierInput.sumcheckDegreeBound from
        Nat.le_trans (by decide) (Nat.le_max_right _ _))
        (FixedPolynomial.mul ops.toOps (headSelector ops data.priorPoint)
          (FixedPolynomial.affine (moment ops data gamma dimension false)
            (ops.sub (moment ops data gamma dimension true) (moment ops data gamma dimension false)))) := by
  simp only [equalitySelector_factor ops laws dimension, NumericBooleanDomain.index_vertex]
  rw [sum_carriedPair ops laws]
  rfl

/-- The carriedPair above is the exact carried branch of the existing pair
constructor. The nonlinear branch and its constraint shift are unchanged. -/
theorem pairPolynomialWithNorm_split (ops : InterpolationOps Field) (laws : InterpolationEvaluationLaws ops)
    {shape : Shape} (input : ProtocolPolynomial.VerifierInput Field shape) (powers : Nat → Field)
    (alphaSelector priorSelector : FixedPolynomial Field 1)
    (low high : ProtocolPolynomial.OutputMessage Field shape)
    (padLow padHigh matrixLow matrixHigh : Field) (sourceNorm : FixedPolynomial Field 3) :
    PiCCSFirstRoundPair.pairPolynomialWithNorm ops input powers alphaSelector priorSelector low high
        padLow padHigh matrixLow matrixHigh sourceNorm =
      FixedPolynomial.add ops.toOps
        (carriedPair ops (show 2 ≤ input.sumcheckDegreeBound from
          Nat.le_trans (by decide) (Nat.le_max_right _ _)) priorSelector
          (powers shape.matrixEvaluationOffset) padLow padHigh matrixLow matrixHigh)
        (FixedPolynomial.scale ops.toOps (powers shape.constraintOffset)
          (FixedPolynomial.add ops.toOps
            (FixedPolynomial.widen ops.toOps (Nat.le_max_left _ _)
              (PiCCSFirstRoundPair.ccsPolynomialWithPowers ops input powers alphaSelector low high))
            (FixedPolynomial.scale ops.toOps (powers shape.freshCount)
              (FixedPolynomial.widen ops.toOps (Nat.le_max_right _ _)
                (FixedPolynomial.mul ops.toOps alphaSelector sourceNorm))))) := by
  unfold PiCCSFirstRoundPair.pairPolynomialWithNorm carriedPair applyHead
  exact (PiCCSPolynomialRange.add_assoc ops laws _ _ _).symm

/-- The separate Pad/matrix affine polynomials use the same combined moments
as complete_carried_sum. This is coefficient equality, for arbitrary inputs. -/
theorem carriedPair_combined (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) {degree : Nat} (bound : 2 ≤ degree)
    (head : FixedPolynomial Field 1) (matrixShift padLow padHigh matrixLow matrixHigh : Field) :
    carriedPair ops bound head matrixShift padLow padHigh matrixLow matrixHigh =
      FixedPolynomial.widen ops.toOps bound (FixedPolynomial.mul ops.toOps head
        (FixedPolynomial.affine (ops.add padLow (ops.mul matrixShift matrixLow))
          (ops.sub (ops.add padHigh (ops.mul matrixShift matrixHigh))
            (ops.add padLow (ops.mul matrixShift matrixLow))))) := by
  unfold carriedPair
  rw [← applyHead_scale ops laws, ← applyHead_add ops laws, affine_combine ops laws]
  rfl

/-- The one generic formula join needed by the selected composer: complete
carried moments plus the fresh and norm pair sums give the original complete
first-round polynomial. Every source field is read from the same data. -/
theorem full_components_eq_firstRound (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) {shape : Shape}
    (data : ProtocolPolynomial.Data Field shape)
    (alpha : CubePoint Field shape.cubeVariables) (gamma : Field)
    {remaining : Nat} (dimension : shape.cubeVariables = remaining + 1) :
    let input := data.toVerifierInput
    let powers := TargetPolynomial.power ops.toOps gamma
    let fresh := fun index : Fin (2 ^ remaining) =>
      let suffix := NumericBooleanDomain.vertex remaining index
      let low := ProtocolPolynomial.vertexMessage data (PiCCSFirstRound.endpointVertex dimension false suffix)
      let high := ProtocolPolynomial.vertexMessage data (PiCCSFirstRound.endpointVertex dimension true suffix)
      FixedPolynomial.scale ops.toOps (powers shape.constraintOffset)
        (FixedPolynomial.widen ops.toOps (Nat.le_max_left _ _)
          (PiCCSFirstRoundPair.ccsPolynomialWithPowers ops input powers
            (PiCCSFirstRound.equalitySelector ops suffix alpha) low high))
    let norm := fun index : Fin (2 ^ remaining) =>
      let suffix := NumericBooleanDomain.vertex remaining index
      let low := ProtocolPolynomial.vertexMessage data (PiCCSFirstRound.endpointVertex dimension false suffix)
      let high := ProtocolPolynomial.vertexMessage data (PiCCSFirstRound.endpointVertex dimension true suffix)
      FixedPolynomial.scale ops.toOps (powers shape.constraintOffset)
        (FixedPolynomial.scale ops.toOps (powers shape.freshCount)
          (FixedPolynomial.widen ops.toOps (Nat.le_max_right _ _)
            (FixedPolynomial.mul ops.toOps (PiCCSFirstRound.equalitySelector ops suffix alpha)
              (PiCCSFirstRoundPair.normPolynomialWithPowers ops powers low high))))
    FixedPolynomial.add ops.toOps
      (FixedPolynomial.widen ops.toOps (show 2 ≤ input.sumcheckDegreeBound from
        Nat.le_trans (by decide) (Nat.le_max_right _ _))
        (FixedPolynomial.mul ops.toOps (headSelector ops data.priorPoint)
          (FixedPolynomial.affine (moment ops data gamma dimension false)
            (ops.sub (moment ops data gamma dimension true) (moment ops data gamma dimension false)))))
      (FixedPolynomial.add ops.toOps
        (FixedPolynomial.sum ops.toOps (canonicalFinIndices (2 ^ remaining)) fresh)
        (FixedPolynomial.sum ops.toOps (canonicalFinIndices (2 ^ remaining)) norm)) =
      PiCCSFirstRound.firstRound ops data alpha gamma dimension := by
  dsimp only
  rw [← complete_carried_sum ops laws data gamma dimension,
    ← PiCCSPolynomialRange.range_eq_firstRound ops data alpha gamma dimension,
    PiCCSPolynomialRange.range_eq_sum ops laws]
  apply PiCCSPolynomialRange.coefficient_ext
  intro coefficient
  simp only [PiCCSPolynomialRange.coefficient_add,
    PiCCSPolynomialRange.coefficient_sum]
  rw [← FiniteSumAlgebra.sumMap_add ops laws,
    ← FiniteSumAlgebra.sumMap_add ops laws]
  apply FiniteSumAlgebra.sumMap_congr
  intro index _
  simp only [Nat.zero_add, PiCCSFirstRound.numericPair, dif_pos index.isLt,
    PiCCSFirstRound.vertexPolynomial, PiCCSFirstRoundPair.pairPolynomial]
  have scale_add {degree : Nat} (scalar : Field)
      (left right : FixedPolynomial Field degree) :
      FixedPolynomial.scale ops.toOps scalar (FixedPolynomial.add ops.toOps left right) =
        FixedPolynomial.add ops.toOps (FixedPolynomial.scale ops.toOps scalar left)
          (FixedPolynomial.scale ops.toOps scalar right) := by
    apply PiCCSPolynomialRange.coefficient_ext
    intro slot
    simp only [PiCCSPolynomialRange.coefficient_scale, PiCCSPolynomialRange.coefficient_add]
    exact laws.left_distrib _ _ _
  rw [← PiCCSFirstRoundPair.pairPolynomialWithTotals_eq ops laws,
    PiCCSFirstRoundPair.pairPolynomialWithTotals,
    pairPolynomialWithNorm_split ops laws, scale_add]
  simp only [PiCCSPolynomialRange.coefficient_add, padValue, matrixValue]
  rfl

end NightstreamFPrime.Export.Stage1.PiCCSCarriedMoments
