import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ProtocolPolynomialDegree.Sparse
import NightstreamFPrime.Export.Stage1.PiCCSGammaPowers

/-!
One adjacent Boolean-pair contribution to the PiCCS polynomial. Inputs are
underlying endpoint images and two affine equality selectors. Every image
is interpolated before the nonlinear CCS/norm formulas are applied.
Owns the executable pair kernel and its equality to the existing terminal formula.
Source-image construction and completion summation belong to their consumers.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSFirstRoundPair

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.SumCheck.Finite
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open ProtocolPolynomialDegree.Support (polynomialLaws)

universe uField uIndex
variable {Field : Type uField} {shape : Shape}

/-- The same affine interpolation in all four existing image families. -/
def affineMessage (ops : InterpolationOps Field)
    (low high : ProtocolPolynomial.OutputMessage Field shape)
    (point : Field) : ProtocolPolynomial.OutputMessage Field shape where
  freshMatrixImage := fun source matrix =>
    ops.add (low.freshMatrixImage source matrix)
      (ops.mul point (ops.sub (high.freshMatrixImage source matrix)
        (low.freshMatrixImage source matrix)))
  sourceAssignment := fun source =>
    ops.add (low.sourceAssignment source)
      (ops.mul point (ops.sub (high.sourceAssignment source) (low.sourceAssignment source)))
  padImage := fun coordinate =>
    ops.add (low.padImage coordinate)
      (ops.mul point (ops.sub (high.padImage coordinate) (low.padImage coordinate)))
  matrixImage := fun coordinate =>
    ops.add (low.matrixImage coordinate)
      (ops.mul point (ops.sub (high.matrixImage coordinate) (low.matrixImage coordinate)))

private def affine (ops : InterpolationOps Field) (low high : Field) :
    FixedPolynomial Field 1 :=
  FixedPolynomial.affine low (ops.sub high low)

private theorem evaluate_affine (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) (low high point : Field) :
    (affine ops low high).evaluate ops.toOps point =
      ops.add low (ops.mul point (ops.sub high low)) :=
  FixedPolynomial.evaluate_affine ops.toOps (polynomialLaws laws) low (ops.sub high low) point

private theorem evaluate_sum (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) {Index : Type uIndex} {degree : Nat}
    (indices : List Index) (values : Index → FixedPolynomial Field degree) (point : Field) :
    (FixedPolynomial.sum ops.toOps indices values).evaluate ops.toOps point =
      FiniteSumAlgebra.sumMap ops indices (fun index => (values index).evaluate ops.toOps point) := by
  induction indices with
  | nil => exact FixedPolynomial.evaluate_zero ops.toOps (polynomialLaws laws) degree point
  | cons index indices ih =>
      rw [FixedPolynomial.sum, FixedPolynomial.evaluate_add ops.toOps (polynomialLaws laws), ih]
      rfl

private def weightedSum (ops : InterpolationOps Field)
    {Index : Type uIndex} {degree : Nat} (indices : List Index)
    (weight : Index → Field) (values : Index → FixedPolynomial Field degree) :
    FixedPolynomial Field degree :=
  FixedPolynomial.sum ops.toOps indices fun index =>
    FixedPolynomial.scale ops.toOps (weight index) (values index)

private theorem evaluate_weightedSum (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) {Index : Type uIndex} {degree : Nat}
    (indices : List Index) (weight : Index → Field)
    (values : Index → FixedPolynomial Field degree) (point : Field) :
    (weightedSum ops indices weight values).evaluate ops.toOps point =
      FiniteSumAlgebra.sumMap ops indices (fun index =>
        ops.mul (weight index) ((values index).evaluate ops.toOps point)) := by
  rw [weightedSum, evaluate_sum ops laws]
  simp only [FixedPolynomial.evaluate_scale ops.toOps (polynomialLaws laws)]

private theorem sumMap_attach (ops : InterpolationOps Field)
    {Index : Type uIndex} (indices : List Index) (value : Index → Field) :
    FiniteSumAlgebra.sumMap ops indices.attach (fun index => value index.val) =
      FiniteSumAlgebra.sumMap ops indices value := by
  simp only [FiniteSumAlgebra.sumMap, List.attach_map_val]

private theorem mul_left_comm (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) (left middle right : Field) :
    ops.mul left (ops.mul middle right) = ops.mul middle (ops.mul left right) := by
  rw [← laws.mul_assoc, laws.mul_comm left middle, laws.mul_assoc]

/-- Each syntax term is multiplied by the selector before its proved widening. -/
private def gatedConstraint (ops : InterpolationOps Field) {matrixCount : Nat}
    (polynomial : CCSResidualTable.ConstraintPolynomial Field matrixCount)
    (selector : FixedPolynomial Field 1)
    (images : Fin matrixCount → FixedPolynomial Field 1) :
    FixedPolynomial Field polynomial.canonicalEqualityGatedDegreeBound :=
  FixedPolynomial.sum ops.toOps polynomial.terms.attach fun term =>
    FixedPolynomial.widen ops.toOps (by
      have bound := polynomial.term_totalDegree_succ_le_canonicalEqualityGatedDegreeBound
        term.val term.property
      omega)
      (FixedPolynomial.mul ops.toOps selector
        (ProtocolPolynomialDegree.Sparse.monomialPolynomial ops term.val images))

private theorem evaluate_gatedConstraint (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) {matrixCount : Nat}
    (polynomial : CCSResidualTable.ConstraintPolynomial Field matrixCount)
    (selector : FixedPolynomial Field 1)
    (images : Fin matrixCount → FixedPolynomial Field 1) (point : Field) :
    (gatedConstraint ops polynomial selector images).evaluate ops.toOps point =
      ops.mul (selector.evaluate ops.toOps point)
        (CCSResidualTable.evaluatePolynomial ops polynomial
          (fun matrix => (images matrix).evaluate ops.toOps point)) := by
  rw [gatedConstraint, evaluate_sum ops laws]
  simp only [FixedPolynomial.evaluate_widen ops.toOps (polynomialLaws laws),
    FixedPolynomial.evaluate_mul ops.toOps (polynomialLaws laws),
    ProtocolPolynomialDegree.Sparse.evaluate_monomialPolynomial laws]
  rw [sumMap_attach ops polynomial.terms (fun term =>
    ops.mul (selector.evaluate ops.toOps point)
      (CCSResidualTable.evaluateMonomial ops term
        (fun matrix => (images matrix).evaluate ops.toOps point))),
    FiniteSumAlgebra.sumMap_mul_left ops laws,
    ← CCSResidualTable.evaluatePolynomial_eq_sumMap ops laws]

/-- The fresh CCS branch, with each matrix image interpolated before the
constraint polynomial. The common gamma shift remains with the pair caller. -/
def ccsPolynomialWithPowers (ops : InterpolationOps Field)
    (input : ProtocolPolynomial.VerifierInput Field shape) (powers : Nat → Field)
    (selector : FixedPolynomial Field 1)
    (low high : ProtocolPolynomial.OutputMessage Field shape) :
    FixedPolynomial Field input.constraintPolynomial.canonicalEqualityGatedDegreeBound :=
  weightedSum ops (canonicalFinIndices shape.freshCount)
    (fun source => powers source.val)
    (fun source => gatedConstraint ops input.constraintPolynomial selector
      (fun matrix => affine ops (low.freshMatrixImage source matrix)
        (high.freshMatrixImage source matrix)))

private def ccsPolynomial (ops : InterpolationOps Field)
    (input : ProtocolPolynomial.VerifierInput Field shape) (gamma : Field)
    (selector : FixedPolynomial Field 1)
    (low high : ProtocolPolynomial.OutputMessage Field shape) :
    FixedPolynomial Field input.constraintPolynomial.canonicalEqualityGatedDegreeBound :=
  ccsPolynomialWithPowers ops input (TargetPolynomial.power ops.toOps gamma) selector low high

private theorem ccsPolynomialWithPowers_power (ops : InterpolationOps Field)
    (input : ProtocolPolynomial.VerifierInput Field shape) (gamma : Field)
    (selector : FixedPolynomial Field 1)
    (low high : ProtocolPolynomial.OutputMessage Field shape) :
    ccsPolynomialWithPowers ops input (TargetPolynomial.power ops.toOps gamma) selector low high =
      ccsPolynomial ops input gamma selector low high := rfl

private theorem evaluate_ccsPolynomial (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (input : ProtocolPolynomial.VerifierInput Field shape) (gamma : Field)
    (selector : FixedPolynomial Field 1)
    (low high : ProtocolPolynomial.OutputMessage Field shape) (point : Field) :
    (ccsPolynomial ops input gamma selector low high).evaluate ops.toOps point =
      ops.mul (selector.evaluate ops.toOps point)
        (ProtocolPolynomial.ccsAtMessage ops input gamma (affineMessage ops low high point)) := by
  rw [ccsPolynomial, ccsPolynomialWithPowers, evaluate_weightedSum ops laws]
  simp only [evaluate_gatedConstraint ops laws, evaluate_affine ops laws]
  calc
    _ = FiniteSumAlgebra.sumMap ops (canonicalFinIndices shape.freshCount)
        (fun source => ops.mul (selector.evaluate ops.toOps point)
          (SignedJointIdentity.gammaTerm ops gamma source.val
            (CCSResidualTable.evaluatePolynomial ops input.constraintPolynomial
              ((affineMessage ops low high point).freshMatrixImage source)))) := by
      apply FiniteSumAlgebra.sumMap_congr
      intro source _
      exact mul_left_comm ops laws _ _ _
    _ = _ := FiniteSumAlgebra.sumMap_mul_left ops laws _ _ _

/-- The b=2 norm cubic is applied after interpolation of the two source values. -/
def normPair (ops : InterpolationOps Field) (low high : Field) :
    FixedPolynomial Field 3 :=
  let value := affine ops low high
  FixedPolynomial.mul ops.toOps
    (FixedPolynomial.mul ops.toOps
      (FixedPolynomial.add ops.toOps value (FixedPolynomial.affine ops.one ops.zero)) value)
    (FixedPolynomial.add ops.toOps value (FixedPolynomial.affine (ops.neg ops.one) ops.zero))

/-- The coefficient construction is the existing norm residual after interpolation. -/
theorem evaluate_normPair (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) (low high point : Field) :
    (normPair ops low high).evaluate ops.toOps point =
      ProtocolPolynomial.strictNormResidual ops
        (ops.add low (ops.mul point (ops.sub high low))) := by
  simp only [normPair, FixedPolynomial.evaluate_mul ops.toOps (polynomialLaws laws),
    FixedPolynomial.evaluate_add ops.toOps (polynomialLaws laws),
    FixedPolynomial.evaluate_affine ops.toOps (polynomialLaws laws),
    evaluate_affine ops laws, laws.mul_zero, laws.add_zero]
  rfl

/-- Keep one interpolated cubic for every original source, with its own gamma weight. -/
def normPolynomialWithPowers (ops : InterpolationOps Field) (powers : Nat → Field)
    (low high : ProtocolPolynomial.OutputMessage Field shape) : FixedPolynomial Field 3 :=
  weightedSum ops (canonicalFinIndices shape.sourceCount)
    (fun source => powers source.val)
    (fun source => normPair ops (low.sourceAssignment source) (high.sourceAssignment source))

private def normPolynomial (ops : InterpolationOps Field) (gamma : Field)
    (low high : ProtocolPolynomial.OutputMessage Field shape) : FixedPolynomial Field 3 :=
  normPolynomialWithPowers ops (TargetPolynomial.power ops.toOps gamma) low high

private theorem normPolynomialWithPowers_power (ops : InterpolationOps Field) (gamma : Field)
    (low high : ProtocolPolynomial.OutputMessage Field shape) :
    normPolynomialWithPowers ops (TargetPolynomial.power ops.toOps gamma) low high =
      normPolynomial ops gamma low high := rfl

private theorem evaluate_normPolynomial (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) (gamma : Field)
    (low high : ProtocolPolynomial.OutputMessage Field shape) (point : Field) :
    (normPolynomial ops gamma low high).evaluate ops.toOps point =
      ProtocolPolynomial.normAtMessage ops gamma (affineMessage ops low high point) := by
  rw [normPolynomial, normPolynomialWithPowers, evaluate_weightedSum ops laws]
  simp only [evaluate_normPair ops laws]
  rfl

private theorem ccsFits (input : ProtocolPolynomial.VerifierInput Field shape) :
    input.constraintPolynomial.canonicalEqualityGatedDegreeBound ≤ input.sumcheckDegreeBound :=
  Nat.le_max_left _ _

private theorem fourFits (input : ProtocolPolynomial.VerifierInput Field shape) :
    4 ≤ input.sumcheckDegreeBound := Nat.le_max_right _ _

private theorem twoFits (input : ProtocolPolynomial.VerifierInput Field shape) :
    2 ≤ input.sumcheckDegreeBound := Nat.le_trans (by decide) (fourFits input)

/-- The same coefficient construction with an explicit power lookup. -/
def pairPolynomialWithPowers (ops : InterpolationOps Field)
    (input : ProtocolPolynomial.VerifierInput Field shape) (powers : Nat → Field)
    (alphaSelector priorSelector : FixedPolynomial Field 1)
    (low high : ProtocolPolynomial.OutputMessage Field shape) :
    FixedPolynomial Field input.sumcheckDegreeBound :=
  let pad := FixedPolynomial.mul ops.toOps priorSelector
    (weightedSum ops (canonicalPadCoordinates shape)
      (fun coordinate => powers coordinate.localGammaExponent)
      (fun coordinate => affine ops (low.padImage coordinate) (high.padImage coordinate)))
  let matrix := FixedPolynomial.mul ops.toOps priorSelector
    (weightedSum ops (canonicalMatrixCoordinates shape)
      (fun coordinate => powers coordinate.localGammaExponent)
      (fun coordinate => affine ops (low.matrixImage coordinate) (high.matrixImage coordinate)))
  let ccs := ccsPolynomialWithPowers ops input powers alphaSelector low high
  let norm := FixedPolynomial.mul ops.toOps alphaSelector (normPolynomialWithPowers ops powers low high)
  let constraints := FixedPolynomial.add ops.toOps
    (FixedPolynomial.widen ops.toOps (ccsFits input) ccs)
    (FixedPolynomial.scale ops.toOps (powers shape.freshCount)
      (FixedPolynomial.widen ops.toOps (fourFits input) norm))
  FixedPolynomial.add ops.toOps (FixedPolynomial.widen ops.toOps (twoFits input) pad)
    (FixedPolynomial.add ops.toOps
      (FixedPolynomial.scale ops.toOps (powers shape.matrixEvaluationOffset)
        (FixedPolynomial.widen ops.toOps (twoFits input) matrix))
      (FixedPolynomial.scale ops.toOps (powers shape.constraintOffset)
        constraints))

/-- Exact verifier width; high zero coefficients are retained in constant-first order. -/
def pairPolynomial (ops : InterpolationOps Field)
    (input : ProtocolPolynomial.VerifierInput Field shape) (gamma : Field)
    (alphaSelector priorSelector : FixedPolynomial Field 1)
    (low high : ProtocolPolynomial.OutputMessage Field shape) :
    FixedPolynomial Field input.sumcheckDegreeBound :=
  pairPolynomialWithPowers ops input (TargetPolynomial.power ops.toOps gamma)
    alphaSelector priorSelector low high

/-- Cover every exponent used by the pair: local Pad/matrix ranges, source
indices, fresh-count shift, matrix shift, and constraint shift. The offsets
come from Shape; lookup remains correct outside this preparation extent. -/
def powerCount (shape : Shape) : Nat :=
  max (shape.constraintOffset + 1) (max (shape.freshCount + 1) shape.sourceCount)

/-- Preparing a power table preserves the entire fixed coefficient object,
including every high zero slot. No evaluation-only comparison is used. -/
theorem pairPolynomialWithPowers_prepared (ops : InterpolationOps Field)
    (input : ProtocolPolynomial.VerifierInput Field shape) (gamma : Field) (count : Nat)
    (alphaSelector priorSelector : FixedPolynomial Field 1)
    (low high : ProtocolPolynomial.OutputMessage Field shape) :
    pairPolynomialWithPowers ops input
      (PiCCSGammaPowers.lookup ops.toOps gamma (PiCCSGammaPowers.prepare ops.toOps gamma count))
      alphaSelector priorSelector low high =
      pairPolynomial ops input gamma alphaSelector priorSelector low high := by
  have same : PiCCSGammaPowers.lookup ops.toOps gamma (PiCCSGammaPowers.prepare ops.toOps gamma count) =
      TargetPolynomial.power ops.toOps gamma :=
    funext (PiCCSGammaPowers.lookup_prepare ops.toOps gamma count)
  rw [same]
  rfl

private theorem combine_gated (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) (selector ccs gamma norm : Field) :
    ops.add (ops.mul selector ccs) (ops.mul gamma (ops.mul selector norm)) =
      ops.mul selector (ops.add ccs (ops.mul gamma norm)) := by
  rw [mul_left_comm ops laws gamma selector norm, laws.left_distrib]

/-- Pair evaluation is exactly the existing terminal expression. Selector
correctness is stated at the same evaluation point, in alpha/prior order. -/
theorem pairPolynomial_evaluate (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (input : ProtocolPolynomial.VerifierInput Field shape) (gamma : Field)
    (alphaSelector priorSelector : FixedPolynomial Field 1)
    (low high : ProtocolPolynomial.OutputMessage Field shape)
    (alpha point : CubePoint Field shape.cubeVariables) (value : Field)
    (alphaMatches : alphaSelector.evaluate ops.toOps value =
      SumCheckTruthPath.pointEquality ops point alpha)
    (priorMatches : priorSelector.evaluate ops.toOps value =
      SumCheckTruthPath.pointEquality ops point input.priorPoint) :
    (pairPolynomial ops input gamma alphaSelector priorSelector low high).evaluate ops.toOps value =
      ProtocolPolynomial.terminalFromMessage ops input alpha gamma point
        (affineMessage ops low high value) := by
  simp only [pairPolynomial, pairPolynomialWithPowers,
    ccsPolynomialWithPowers_power, normPolynomialWithPowers_power,
    FixedPolynomial.evaluate_add ops.toOps (polynomialLaws laws),
    FixedPolynomial.evaluate_widen ops.toOps (polynomialLaws laws),
    FixedPolynomial.evaluate_scale ops.toOps (polynomialLaws laws),
    FixedPolynomial.evaluate_mul ops.toOps (polynomialLaws laws),
    evaluate_weightedSum ops laws, evaluate_affine ops laws,
    evaluate_ccsPolynomial ops laws, evaluate_normPolynomial ops laws,
    combine_gated ops laws, alphaMatches, priorMatches]
  rfl

/-- Direct equality of both coefficient slots, using the existing sum order. -/
private theorem weightedSum_affine_coefficients (ops : InterpolationOps Field)
    {Index : Type uIndex} (indices : List Index) (weight low high : Index → Field) :
    weightedSum ops indices weight (fun index => affine ops (low index) (high index)) =
      FixedPolynomial.affine
        (FiniteSumAlgebra.sumMap ops indices (fun index => ops.mul (weight index) (low index)))
        (FiniteSumAlgebra.sumMap ops indices
          (fun index => ops.mul (weight index) (ops.sub (high index) (low index)))) := by
  induction indices with
  | nil => rfl
  | cons index indices ih =>
      change FixedPolynomial.add ops.toOps
          (FixedPolynomial.scale ops.toOps (weight index) (affine ops (low index) (high index)))
          (weightedSum ops indices weight (fun next => affine ops (low next) (high next))) = _
      rw [ih]
      rfl

/-- Weighted interpolation is exactly the affine polynomial of weighted
endpoints. This is coefficient equality, not evaluation injectivity. -/
private theorem weightedSum_affine_totals (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) {Index : Type uIndex}
    (indices : List Index) (weight low high : Index → Field) :
    weightedSum ops indices weight (fun index => affine ops (low index) (high index)) =
      affine ops
        (FiniteSumAlgebra.sumMap ops indices (fun index => ops.mul (weight index) (low index)))
        (FiniteSumAlgebra.sumMap ops indices (fun index => ops.mul (weight index) (high index))) := by
  rw [weightedSum_affine_coefficients]
  unfold affine
  apply congrArg (FixedPolynomial.affine
    (FiniteSumAlgebra.sumMap ops indices (fun index => ops.mul (weight index) (low index))))
  calc
    _ = FiniteSumAlgebra.sumMap ops indices (fun index =>
        ops.sub (ops.mul (weight index) (high index)) (ops.mul (weight index) (low index))) := by
      apply FiniteSumAlgebra.sumMap_congr
      intro index _
      exact FiniteSumAlgebra.mul_sub ops laws _ _ _
    _ = _ := FiniteSumAlgebra.sumMap_sub ops laws indices _ _

/-- Reuse a computed source norm polynomial and four carried totals.
Only fresh matrix fields are read from the endpoint messages here. -/
def pairPolynomialWithNorm (ops : InterpolationOps Field)
    (input : ProtocolPolynomial.VerifierInput Field shape) (powers : Nat → Field)
    (alphaSelector priorSelector : FixedPolynomial Field 1)
    (low high : ProtocolPolynomial.OutputMessage Field shape)
    (padLow padHigh matrixLow matrixHigh : Field)
    (sourceNorm : FixedPolynomial Field 3) :
    FixedPolynomial Field input.sumcheckDegreeBound :=
  let pad := FixedPolynomial.mul ops.toOps priorSelector (affine ops padLow padHigh)
  let matrix := FixedPolynomial.mul ops.toOps priorSelector (affine ops matrixLow matrixHigh)
  let ccs := ccsPolynomialWithPowers ops input powers alphaSelector low high
  let norm := FixedPolynomial.mul ops.toOps alphaSelector sourceNorm
  let constraints := FixedPolynomial.add ops.toOps
    (FixedPolynomial.widen ops.toOps (ccsFits input) ccs)
    (FixedPolynomial.scale ops.toOps (powers shape.freshCount)
      (FixedPolynomial.widen ops.toOps (fourFits input) norm))
  FixedPolynomial.add ops.toOps (FixedPolynomial.widen ops.toOps (twoFits input) pad)
    (FixedPolynomial.add ops.toOps
      (FixedPolynomial.scale ops.toOps (powers shape.matrixEvaluationOffset)
        (FixedPolynomial.widen ops.toOps (twoFits input) matrix))
      (FixedPolynomial.scale ops.toOps (powers shape.constraintOffset) constraints))

/-- The four totals replace only carried-field enumeration. Matrix totals
are local: the existing matrix gamma shift is still applied exactly once. -/
def pairPolynomialWithTotals (ops : InterpolationOps Field)
    (input : ProtocolPolynomial.VerifierInput Field shape) (powers : Nat → Field)
    (alphaSelector priorSelector : FixedPolynomial Field 1)
    (low high : ProtocolPolynomial.OutputMessage Field shape)
    (padLow padHigh matrixLow matrixHigh : Field) :
    FixedPolynomial Field input.sumcheckDegreeBound :=
  pairPolynomialWithNorm ops input powers alphaSelector priorSelector low high
    padLow padHigh matrixLow matrixHigh (normPolynomialWithPowers ops powers low high)

/-- Supplying the original source norm preserves every pair coefficient. -/
theorem pairPolynomialWithNorm_eq (ops : InterpolationOps Field)
    (input : ProtocolPolynomial.VerifierInput Field shape) (powers : Nat → Field)
    (alphaSelector priorSelector : FixedPolynomial Field 1)
    (low high : ProtocolPolynomial.OutputMessage Field shape)
    (padLow padHigh matrixLow matrixHigh : Field) :
    pairPolynomialWithNorm ops input powers alphaSelector priorSelector low high
        padLow padHigh matrixLow matrixHigh (normPolynomialWithPowers ops powers low high) =
      pairPolynomialWithTotals ops input powers alphaSelector priorSelector low high
        padLow padHigh matrixLow matrixHigh := rfl

/-- Supplying the four original weighted endpoint sums preserves every
coefficient and the fixed width. The power callback is arbitrary and shared. -/
theorem pairPolynomialWithTotals_eq (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (input : ProtocolPolynomial.VerifierInput Field shape) (powers : Nat → Field)
    (alphaSelector priorSelector : FixedPolynomial Field 1)
    (low high : ProtocolPolynomial.OutputMessage Field shape) :
    pairPolynomialWithTotals ops input powers alphaSelector priorSelector low high
      (FiniteSumAlgebra.sumMap ops (canonicalPadCoordinates shape)
        (fun coordinate => ops.mul (powers coordinate.localGammaExponent) (low.padImage coordinate)))
      (FiniteSumAlgebra.sumMap ops (canonicalPadCoordinates shape)
        (fun coordinate => ops.mul (powers coordinate.localGammaExponent) (high.padImage coordinate)))
      (FiniteSumAlgebra.sumMap ops (canonicalMatrixCoordinates shape)
        (fun coordinate => ops.mul (powers coordinate.localGammaExponent) (low.matrixImage coordinate)))
      (FiniteSumAlgebra.sumMap ops (canonicalMatrixCoordinates shape)
        (fun coordinate => ops.mul (powers coordinate.localGammaExponent) (high.matrixImage coordinate))) =
      pairPolynomialWithPowers ops input powers alphaSelector priorSelector low high := by
  unfold pairPolynomialWithTotals pairPolynomialWithNorm pairPolynomialWithPowers
  rw [weightedSum_affine_totals ops laws (canonicalPadCoordinates shape)
      (fun coordinate => powers coordinate.localGammaExponent) low.padImage high.padImage,
    weightedSum_affine_totals ops laws (canonicalMatrixCoordinates shape)
      (fun coordinate => powers coordinate.localGammaExponent) low.matrixImage high.matrixImage]

/-- Carried message fields do not affect this constructor. Only the four
nonlinear endpoint projections and the separately supplied totals are read. -/
theorem pairPolynomialWithTotals_congr (ops : InterpolationOps Field)
    (input : ProtocolPolynomial.VerifierInput Field shape) (powers : Nat → Field)
    (alphaSelector priorSelector : FixedPolynomial Field 1)
    (low high replacementLow replacementHigh : ProtocolPolynomial.OutputMessage Field shape)
    (padLow padHigh matrixLow matrixHigh : Field)
    (lowFresh : low.freshMatrixImage = replacementLow.freshMatrixImage)
    (highFresh : high.freshMatrixImage = replacementHigh.freshMatrixImage)
    (lowAssignment : low.sourceAssignment = replacementLow.sourceAssignment)
    (highAssignment : high.sourceAssignment = replacementHigh.sourceAssignment) :
    pairPolynomialWithTotals ops input powers alphaSelector priorSelector low high
        padLow padHigh matrixLow matrixHigh =
      pairPolynomialWithTotals ops input powers alphaSelector priorSelector replacementLow replacementHigh
        padLow padHigh matrixLow matrixHigh := by
  simp only [pairPolynomialWithTotals, pairPolynomialWithNorm,
    ccsPolynomialWithPowers, normPolynomialWithPowers,
    lowFresh, highFresh, lowAssignment, highAssignment]

end NightstreamFPrime.Export.Stage1.PiCCSFirstRoundPair
