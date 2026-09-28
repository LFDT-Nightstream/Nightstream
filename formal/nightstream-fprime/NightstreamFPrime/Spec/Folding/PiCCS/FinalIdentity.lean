import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ProtocolPolynomial
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.SignedCoefficientPolynomial

/-!
Paper authority: SuperNeo v1.1, Section 7.3, Steps 2 and 4.
Obligation: Use the exact joint identity
`Eval_K + γ^(k*d) Eval_A + γ^(k*d*(t+1)) eq(F + γ^K NC)`.

Inputs:
- verifier-owned `α`, `γ`, prior point, and public prior claims;
- the verifier-derived SumCheck point;
- the complete typed PiCCS output message.

Outputs:
- the exact verifier-computed terminal value.

Parent coverage:
- `ProtocolPolynomial.VerifierInput.initial`;
- `ProtocolPolynomial.terminalFromMessage`.

This module owns the canonical named final-identity contract. It emits no
circuit constraints.
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.FinalIdentity

open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

universe uField

/-- The exact terminal-equality predicate used by a circuit leaf. -/
abbrev Holds
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (input : ProtocolPolynomial.VerifierInput Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field)
    (point : CubePoint Field shape.cubeVariables)
    (message : ProtocolPolynomial.OutputMessage Field shape)
    (claimedTerminal : Field) : Prop :=
  claimedTerminal =
    ProtocolPolynomial.terminalFromMessage
      ops input alpha gamma point message

/-- Constant-first verifier target coefficients: all `Eval_K` claims, then
all genuine-matrix `Eval_A` claims. -/
def targetCoefficientList
    {Field : Type uField}
    {shape : Shape}
    (input : ProtocolPolynomial.VerifierInput Field shape) : List Field :=
  (canonicalPadCoordinates shape).map input.claimedPadCoefficient ++
    (canonicalMatrixCoordinates shape).map input.claimedMatrixCoefficient

/-- Constant-first output `Eval_K` coefficients in exact paper exponent
order. This list contains no CCS-matrix value. -/
def outputPadCoefficientList
    {Field : Type uField}
    {shape : Shape}
    (message : ProtocolPolynomial.OutputMessage Field shape) : List Field :=
  (canonicalPadCoordinates shape).map message.padImage

/-- Constant-first output `Eval_A` coefficients in exact paper-local exponent
order. This list contains all genuine CCS matrices and no Pad value. -/
def outputMatrixCoefficientList
    {Field : Type uField}
    {shape : Shape}
    (message : ProtocolPolynomial.OutputMessage Field shape) : List Field :=
  (canonicalMatrixCoordinates shape).map message.matrixImage

theorem outputPadCoefficientList_length
    {Field : Type uField}
    {shape : Shape}
    (message : ProtocolPolynomial.OutputMessage Field shape) :
    (outputPadCoefficientList message).length = shape.padEvaluationCount := by
  simp [outputPadCoefficientList, canonicalPadCoordinates_length]

theorem outputMatrixCoefficientList_length
    {Field : Type uField}
    {shape : Shape}
    (message : ProtocolPolynomial.OutputMessage Field shape) :
    (outputMatrixCoefficientList message).length =
      shape.matrixEvaluationCount := by
  simp [outputMatrixCoefficientList, canonicalMatrixCoordinates_length]

theorem targetCoefficientList_length
    {Field : Type uField}
    {shape : Shape}
    (input : ProtocolPolynomial.VerifierInput Field shape) :
    (targetCoefficientList input).length = shape.constraintOffset := by
  simp [targetCoefficientList, canonicalPadCoordinates_length,
    canonicalMatrixCoordinates_length, Shape.constraintOffset]

private theorem finiteSum_eq_foldr
    {Field : Type uField}
    (ops : InterpolationOps Field) : ∀ values : List Field,
    BooleanTable.finiteSum ops values = values.foldr ops.add ops.zero
  | [] => rfl
  | value :: values => by
      simp only [BooleanTable.finiteSum, List.foldr]
      rw [finiteSum_eq_foldr ops values]

/-- The public initial claim is exactly `T_K + γ^(k*d) * T_A`. -/
theorem initial_eq_eval_K_add_shifted_eval_A
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (input : ProtocolPolynomial.VerifierInput Field shape)
    (gamma : Field) :
    input.initial ops gamma =
      ops.add
        (TargetPolynomial.evaluatePad
          ops.toOps input.targetCoefficients gamma)
        (ops.mul
          (TargetPolynomial.power ops.toOps gamma shape.padEvaluationCount)
          (TargetPolynomial.evaluateMatrixLocal
            ops.toOps input.targetCoefficients gamma)) := by
  let shiftLaws : TargetPolynomial.ShiftLaws ops.toOps := {
    one_mul := laws.one_mul
    mul_assoc := laws.mul_assoc
    mul_zero := laws.mul_zero
    mul_add := laws.left_distrib
  }
  unfold ProtocolPolynomial.VerifierInput.initial TargetPolynomial.evaluate
  rw [TargetPolynomial.evaluateMatrix_eq_shift_mul_evaluateMatrixLocal
    ops.toOps shiftLaws input.targetCoefficients gamma]

/-- The executable constant-first Horner builder over `Eval_K ++ Eval_A`
computes the production verifier's exact initial claim. -/
theorem evaluateTargetCoefficients_eq_initial
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (input : ProtocolPolynomial.VerifierInput Field shape)
    (gamma : Field) :
    SumCheck.Finite.Message.evaluateCoefficients ops.toOps gamma
        (targetCoefficientList input) =
      input.initial ops gamma := by
  let padValues := (canonicalPadCoordinates shape).map
    input.claimedPadCoefficient
  let matrixValues := (canonicalMatrixCoordinates shape).map
    input.claimedMatrixCoefficient
  have padPositions :
      (canonicalPadCoordinates shape).map
          PadCoordinate.localGammaExponent =
        List.range' 0 (canonicalPadCoordinates shape).length := by
    simpa [List.range_eq_range', canonicalPadCoordinates_length] using
      canonicalPadCoordinates_localGammaExponents shape
  have matrixPositions :
      (canonicalMatrixCoordinates shape).map
          MatrixCoordinate.localGammaExponent =
        List.range' 0 (canonicalMatrixCoordinates shape).length := by
    simpa [List.range_eq_range', canonicalMatrixCoordinates_length] using
      canonicalMatrixCoordinates_localGammaExponents shape
  have padEvaluate :
      SumCheck.Finite.Message.evaluateCoefficients ops.toOps gamma padValues =
        TargetPolynomial.evaluatePad ops.toOps input.targetCoefficients
          gamma := by
    have indexed := SignedCoefficientPolynomial.evaluate_map_eq_indexed
      ops laws gamma (canonicalPadCoordinates shape)
      PadCoordinate.localGammaExponent input.claimedPadCoefficient
      padPositions
    rw [TargetPolynomial.evaluatePad_eq_foldr]
    rw [← finiteSum_eq_foldr ops]
    simpa [padValues, TargetPolynomial.padTerm,
      SignedJointIdentity.gammaTerm,
      ProtocolPolynomial.VerifierInput.targetCoefficients] using indexed
  have matrixEvaluate :
      SumCheck.Finite.Message.evaluateCoefficients ops.toOps gamma
          matrixValues =
        TargetPolynomial.evaluateMatrixLocal ops.toOps
          input.targetCoefficients gamma := by
    have indexed := SignedCoefficientPolynomial.evaluate_map_eq_indexed
      ops laws gamma (canonicalMatrixCoordinates shape)
      MatrixCoordinate.localGammaExponent input.claimedMatrixCoefficient
      matrixPositions
    rw [TargetPolynomial.evaluateMatrixLocal_eq_foldr]
    rw [← finiteSum_eq_foldr ops]
    simpa [matrixValues, TargetPolynomial.matrixLocalTerm,
      SignedJointIdentity.gammaTerm,
      ProtocolPolynomial.VerifierInput.targetCoefficients] using indexed
  let shiftLaws : TargetPolynomial.ShiftLaws ops.toOps := {
    one_mul := laws.one_mul
    mul_assoc := laws.mul_assoc
    mul_zero := laws.mul_zero
    mul_add := laws.left_distrib
  }
  unfold targetCoefficientList
  rw [SignedCoefficientPolynomial.evaluate_append ops laws,
    padEvaluate, matrixEvaluate, List.length_map,
    canonicalPadCoordinates_length]
  unfold ProtocolPolynomial.VerifierInput.initial TargetPolynomial.evaluate
  rw [TargetPolynomial.evaluateMatrix_eq_shift_mul_evaluateMatrixLocal
    ops.toOps shiftLaws input.targetCoefficients gamma]

/-- Horner evaluation of the output Pad family is exactly the unshifted
paper `Eval_K` gamma sum. -/
theorem evaluateOutputPadCoefficients_eq_sum
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (gamma : Field)
    (message : ProtocolPolynomial.OutputMessage Field shape) :
    SumCheck.Finite.Message.evaluateCoefficients ops.toOps gamma
        (outputPadCoefficientList message) =
      SignedJointIdentity.sumMap ops
        (canonicalPadCoordinates shape) fun coordinate =>
          SignedJointIdentity.gammaTerm ops gamma
            coordinate.localGammaExponent (message.padImage coordinate) := by
  unfold outputPadCoefficientList SignedJointIdentity.sumMap
  apply SignedCoefficientPolynomial.evaluate_map_eq_indexed
    ops laws gamma (canonicalPadCoordinates shape)
      PadCoordinate.localGammaExponent message.padImage
  simpa [List.range_eq_range', canonicalPadCoordinates_length] using
    canonicalPadCoordinates_localGammaExponents shape

/-- Horner evaluation of the output matrix family is exactly the unshifted
paper-local `Eval_A` gamma sum. -/
theorem evaluateOutputMatrixCoefficients_eq_sum
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (gamma : Field)
    (message : ProtocolPolynomial.OutputMessage Field shape) :
    SumCheck.Finite.Message.evaluateCoefficients ops.toOps gamma
        (outputMatrixCoefficientList message) =
      SignedJointIdentity.sumMap ops
        (canonicalMatrixCoordinates shape) fun coordinate =>
          SignedJointIdentity.gammaTerm ops gamma
            coordinate.localGammaExponent (message.matrixImage coordinate) := by
  unfold outputMatrixCoefficientList SignedJointIdentity.sumMap
  apply SignedCoefficientPolynomial.evaluate_map_eq_indexed
    ops laws gamma (canonicalMatrixCoordinates shape)
      MatrixCoordinate.localGammaExponent message.matrixImage
  simpa [List.range_eq_range', canonicalMatrixCoordinates_length] using
    canonicalMatrixCoordinates_localGammaExponents shape

/-- Circuit-facing factorization of the exact production `Eval_K` terminal
term. -/
theorem padAtMessage_eq_pointEquality_mul_horner
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (input : ProtocolPolynomial.VerifierInput Field shape)
    (gamma : Field)
    (point : CubePoint Field shape.cubeVariables)
    (message : ProtocolPolynomial.OutputMessage Field shape) :
    ProtocolPolynomial.padAtMessage ops input gamma point message =
      ops.mul (SumCheckTruthPath.pointEquality ops point input.priorPoint)
        (SumCheck.Finite.Message.evaluateCoefficients ops.toOps gamma
          (outputPadCoefficientList message)) := by
  unfold ProtocolPolynomial.padAtMessage
  rw [evaluateOutputPadCoefficients_eq_sum ops laws]

/-- Circuit-facing factorization of the exact production-local `Eval_A`
terminal term. The global `k*d` shift belongs to the final identity. -/
theorem matrixAtMessage_eq_pointEquality_mul_horner
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (input : ProtocolPolynomial.VerifierInput Field shape)
    (gamma : Field)
    (point : CubePoint Field shape.cubeVariables)
    (message : ProtocolPolynomial.OutputMessage Field shape) :
    ProtocolPolynomial.matrixAtMessage ops input gamma point message =
      ops.mul (SumCheckTruthPath.pointEquality ops point input.priorPoint)
        (SumCheck.Finite.Message.evaluateCoefficients ops.toOps gamma
          (outputMatrixCoefficientList message)) := by
  unfold ProtocolPolynomial.matrixAtMessage
  rw [evaluateOutputMatrixCoefficients_eq_sum ops laws]

/-- The exact v1.1 terminal formula has distinct `Eval_K` and `Eval_A` terms.
The matrix term starts only after all `k*d` Pad coefficients. -/
theorem terminal_eq_eval_K_add_shifted_eval_A_add_constraints
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (input : ProtocolPolynomial.VerifierInput Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field)
    (point : CubePoint Field shape.cubeVariables)
    (message : ProtocolPolynomial.OutputMessage Field shape) :
    ProtocolPolynomial.terminalFromMessage
        ops input alpha gamma point message =
      ops.add
        (ProtocolPolynomial.padAtMessage ops input gamma point message)
        (ops.add
          (SignedJointIdentity.gammaTerm ops gamma
            shape.matrixEvaluationOffset
            (ProtocolPolynomial.matrixAtMessage
              ops input gamma point message))
          (SignedJointIdentity.gammaTerm ops gamma shape.constraintOffset
            (ops.mul
              (SumCheckTruthPath.pointEquality ops point alpha)
              (ops.add
                (ProtocolPolynomial.ccsAtMessage ops input gamma message)
                (SignedJointIdentity.gammaTerm ops gamma shape.freshCount
                  (ProtocolPolynomial.normAtMessage ops gamma message)))))) := by
  rfl

/-- The distinct Pad and matrix families occupy exactly the prefix before the
CCS and norm terms. -/
theorem evaluationCoordinateCount_eq_constraintOffset (shape : Shape) :
    shape.padEvaluationCount + shape.matrixEvaluationCount =
      shape.constraintOffset := by
  rfl

end NightstreamFPrime.Spec.Folding.PiCCS.FinalIdentity
