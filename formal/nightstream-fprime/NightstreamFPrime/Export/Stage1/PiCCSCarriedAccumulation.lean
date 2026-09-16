import NightstreamFPrime.Export.Stage1.PiCCSCarriedMoments
import NightstreamFPrime.Export.Stage1.PiCCSNumericSumOrder
import NightstreamFPrime.Export.Stage1.PiCCSPadBlockMoment

/-! Proof-only scalar accumulation from exact source callbacks to the two
complete carried moments. Callback equalities are explicit; the selected
source-value module discharges them. This file makes no IO/cache-loop claim. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSCarriedAccumulation

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open PiRLCPartialTrace
open NumericCompletionSum (numericSum)
open PiCCSNumericSumOrder
open FiniteSumAlgebra (sumMap)

universe uField
variable {Field : Type uField}

private theorem numericSum_congr (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) (count : Nat) (left right : Nat → Field)
    (equal : ∀ index, index < count → left index = right index) :
    numericSum ops count left = numericSum ops count right := by
  rw [numericSum_eq_finSum ops laws, numericSum_eq_finSum ops laws]
  apply FiniteSumAlgebra.sumMap_congr
  intro index _
  exact equal index.val index.isLt

/-- Flatten the actual complete-block kernel. Its whole-block zero shortcut
is already covered by blockMoment_value. The only callback premise identifies
its exact dot products with the same flat column values. -/
theorem blockMoments_eq_flat_sum
    (basis : FixedArray (Vector K ringDegree) ringDegree)
    (weight : Nat → K) (blocks : Nat → Vector K ringDegree)
    (count : Nat) (bit : Fin 2) (read : Nat → K)
    (read_value : ∀ block, block < count → ∀ lane : Fin ringDegree,
      PiCCSWeightedBasis.dotK (basis.get lane) (blocks block).get =
        read (block * ringDegree + lane.val)) :
    numericSum extensionOps count (fun block =>
      let value := PiCCSPadBlockMoment.blockMoment basis weight block (blocks block)
      if bit.val = 0 then value.1 else value.2) =
      numericSum extensionOps (count * ringDegree) (fun column =>
        if column % 2 = bit.val then extensionOps.mul (weight (column / 2)) (read column)
        else extensionOps.zero) := by
  rw [numericSum_group extensionOps extensionLaws]
  apply numericSum_congr extensionOps extensionLaws
  intro block live
  rw [PiCCSPadBlockMoment.blockMoment_value]
  have bitCases : bit.val = 0 ∨ bit.val = 1 := by
    have bounded := bit.isLt
    omega
  rcases bitCases with low | high
  · simp only [low, if_pos rfl]
    apply numericSum_congr extensionOps extensionLaws
    intro lane bounded
    have parity : (block * ringDegree + lane) % 2 = lane % 2 := by
      change (block * 54 + lane) % 2 = lane % 2
      omega
    simp only [dif_pos bounded, parity, read_value block live ⟨lane, bounded⟩]
  · simp only [high, show ¬ (1 : Nat) = 0 by decide, if_false]
    apply numericSum_congr extensionOps extensionLaws
    intro lane bounded
    have parity : (block * ringDegree + lane) % 2 = lane % 2 := by
      change (block * 54 + lane) % 2 = lane % 2
      omega
    simp only [dif_pos bounded, parity, read_value block live ⟨lane, bounded⟩]
    by_cases even : lane % 2 = 0
    · simp only [even, show ¬ (0 : Nat) = 1 by decide, if_false, if_true]
    · have odd : lane % 2 = 1 := by omega
      simp only [odd, show ¬ (1 : Nat) = 0 by decide, if_false, if_true]

private def sourcePad (ops : InterpolationOps Field) {shape : Shape}
    (data : ProtocolPolynomial.Data Field shape) (gamma : Field)
    (vertex : BooleanVertex shape.cubeVariables) : Field :=
  sumMap ops (canonicalPadCoordinates shape) fun coordinate =>
    ops.mul (TargetPolynomial.power ops.toOps gamma coordinate.localGammaExponent)
      ((ProtocolPolynomial.vertexMessage data vertex).padImage coordinate)

private def sourceMatrix (ops : InterpolationOps Field) {shape : Shape}
    (data : ProtocolPolynomial.Data Field shape) (gamma : Field)
    (vertex : BooleanVertex shape.cubeVariables) : Field :=
  sumMap ops (canonicalMatrixCoordinates shape) fun coordinate =>
    ops.mul (TargetPolynomial.power ops.toOps gamma coordinate.localGammaExponent)
      ((ProtocolPolynomial.vertexMessage data vertex).matrixImage coordinate)

/-- Complete scalar carried accumulation. All callback/source and omitted-row
conditions are stated explicitly. They concern exact values, not witness
validity, commitments, or expected output artifacts. The global matrix shift
is applied once when the two independently accumulated families are joined. -/
theorem moment_of_exact_row_callbacks (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) {shape : Shape}
    (data : ProtocolPolynomial.Data Field shape) (gamma : Field) {remaining : Nat}
    (dimension : shape.cubeVariables = remaining + 1)
    (pad matrix : Nat → Field) (padCount matrixCount : Nat) (bit : Fin 2)
    (padFits : padCount ≤ 2 ^ (remaining + 1))
    (matrixFits : matrixCount ≤ 2 ^ (remaining + 1))
    (padZero : ∀ row, padCount ≤ row → row < 2 ^ (remaining + 1) → pad row = ops.zero)
    (matrixZero : ∀ row, matrixCount ≤ row → row < 2 ^ (remaining + 1) → matrix row = ops.zero)
    (padSource : ∀ pair : Fin (2 ^ remaining),
      pad (2 * pair.val + bit.val) = sourcePad ops data gamma
        (PiCCSFirstRound.endpointVertex dimension (bit.val == 1) (NumericBooleanDomain.vertex remaining pair)))
    (matrixSource : ∀ pair : Fin (2 ^ remaining),
      matrix (2 * pair.val + bit.val) = sourceMatrix ops data gamma
        (PiCCSFirstRound.endpointVertex dimension (bit.val == 1) (NumericBooleanDomain.vertex remaining pair))) :
    let weight := NumericBooleanDomain.tensorWeightCoordinates ops data.priorPoint.coordinates.tail
    ops.add
      (numericSum ops padCount (fun row => if row % 2 = bit.val then
        ops.mul (weight (row / 2)) (pad row) else ops.zero))
      (ops.mul (TargetPolynomial.power ops.toOps gamma shape.matrixEvaluationOffset)
        (numericSum ops matrixCount (fun row => if row % 2 = bit.val then
          ops.mul (weight (row / 2)) (matrix row) else ops.zero))) =
      PiCCSCarriedMoments.moment ops data gamma dimension (bit.val == 1) := by
  let weight := NumericBooleanDomain.tensorWeightCoordinates ops data.priorPoint.coordinates.tail
  dsimp only
  have padSum := numericSum_prefix_parity_eq_finSum ops laws remaining padCount bit
    (fun row => ops.mul (weight (row / 2)) (pad row)) padFits (by
      intro row lower upper
      dsimp only
      rw [padZero row lower upper, laws.mul_zero])
  have matrixSum := numericSum_prefix_parity_eq_finSum ops laws remaining matrixCount bit
    (fun row => ops.mul (weight (row / 2)) (matrix row)) matrixFits (by
      intro row lower upper
      dsimp only
      rw [matrixZero row lower upper, laws.mul_zero])
  change ops.add
    (numericSum ops padCount (fun row => if row % 2 = bit.val then
      ops.mul (weight (row / 2)) (pad row) else ops.zero))
    (ops.mul (TargetPolynomial.power ops.toOps gamma shape.matrixEvaluationOffset)
      (numericSum ops matrixCount (fun row => if row % 2 = bit.val then
        ops.mul (weight (row / 2)) (matrix row) else ops.zero))) = _
  rw [padSum, matrixSum, ← FiniteSumAlgebra.sumMap_mul_left ops laws,
    ← FiniteSumAlgebra.sumMap_add ops laws]
  unfold PiCCSCarriedMoments.moment
  apply FiniteSumAlgebra.sumMap_congr
  intro pair _
  have quotient : (2 * pair.val + bit.val) / 2 = pair.val := by
    have bounded := bit.isLt
    omega
  dsimp only
  rw [quotient, padSource pair, matrixSource pair]
  change ops.add
      (ops.mul (weight pair.val) (sourcePad ops data gamma _))
      (ops.mul (TargetPolynomial.power ops.toOps gamma shape.matrixEvaluationOffset)
        (ops.mul (weight pair.val) (sourceMatrix ops data gamma _))) =
    ops.mul (weight pair.val) (ops.add (sourcePad ops data gamma _)
      (ops.mul (TargetPolynomial.power ops.toOps gamma shape.matrixEvaluationOffset)
        (sourceMatrix ops data gamma _)))
  rw [laws.left_distrib]
  apply congrArg (ops.add (ops.mul (weight pair.val) (sourcePad ops data gamma _)))
  rw [← laws.mul_assoc,
    laws.mul_comm (TargetPolynomial.power ops.toOps gamma shape.matrixEvaluationOffset) (weight pair.val),
    laws.mul_assoc]

end NightstreamFPrime.Export.Stage1.PiCCSCarriedAccumulation
