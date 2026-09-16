import NightstreamFPrime.Export.Stage1.PiCCSPrefixSelector
import NightstreamFPrime.Export.Stage1.PiCCSPolynomialRange

/-!
One sum-check round after a challenge prefix, using the original protocol
shape and output-message type. Endpoints are actual messageAt values of the
same data. This semantic kernel owns no stored-table, source-reader or IO
correspondence and makes no executed replay claim.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSPrefixRound

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.SumCheck.Finite
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open ProtocolPolynomialDegree.Support (polynomialLaws)

universe uField
variable {Field : Type uField} {shape : Shape}

/-- Insert the current coordinate between the consumed challenges and the
Boolean suffix. The original arity is retained. -/
def point (ops : InterpolationOps Field) {arity remaining : Nat}
    (challenges : List Field) (dimension : arity = challenges.length + remaining + 1)
    (value : Field) (suffix : BooleanVertex remaining) : CubePoint Field arity where
  coordinates := challenges ++ value :: SumCheckTruthPath.VertexEncoding.fieldCoordinates ops suffix
  dimension := by
    rw [List.length_append, List.length_cons, SumCheckTruthPath.VertexEncoding.fieldCoordinates_length]
    omega

private theorem affine_endpoints (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) (polynomial : FixedPolynomial Field 1)
    (value : Field) :
    polynomial.evaluate ops.toOps value =
      ops.add (polynomial.evaluate ops.toOps ops.zero)
        (ops.mul value (ops.sub (polynomial.evaluate ops.toOps ops.one)
          (polynomial.evaluate ops.toOps ops.zero))) := by
  rcases polynomial with ⟨coefficients, length⟩
  obtain ⟨constant, linear, rfl⟩ := List.length_eq_two.mp length
  change (FixedPolynomial.affine constant linear).evaluate ops.toOps value =
    ops.add ((FixedPolynomial.affine constant linear).evaluate ops.toOps ops.zero)
      (ops.mul value (ops.sub ((FixedPolynomial.affine constant linear).evaluate ops.toOps ops.one)
        ((FixedPolynomial.affine constant linear).evaluate ops.toOps ops.zero)))
  have zeroMul : ops.mul ops.zero linear = ops.zero := by
    rw [laws.mul_comm, laws.mul_zero]
  have subtract : ops.sub (ops.add constant linear) constant = linear := by
    unfold InterpolationOps.sub
    rw [laws.add_comm constant linear, laws.add_assoc, laws.add_neg, laws.add_zero]
  simp only [FixedPolynomial.evaluate_affine ops.toOps (polynomialLaws laws),
    zeroMul, laws.add_zero, laws.one_mul, subtract]

/-- Reuse the proved coordinate-slice representation only within the proof.
No existential representation is chosen by any executable definition. -/
private theorem table_affine (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) {arity remaining : Nat}
    (table : BooleanTable Field arity) (challenges : List Field)
    (dimension : arity = challenges.length + remaining + 1)
    (suffix : BooleanVertex remaining) (value : Field) :
    ops.add (table.evaluate ops (point ops challenges dimension ops.zero suffix))
      (ops.mul value (ops.sub
        (table.evaluate ops (point ops challenges dimension ops.one suffix))
        (table.evaluate ops (point ops challenges dimension ops.zero suffix)))) =
      table.evaluate ops (point ops challenges dimension value suffix) := by
  have length : challenges.length + 1 +
      (SumCheckTruthPath.VertexEncoding.fieldCoordinates ops suffix).length = arity := by
    rw [SumCheckTruthPath.VertexEncoding.fieldCoordinates_length]
    omega
  rcases ProtocolPolynomialDegree.Support.evaluateCoordinates_affine ops laws table
      challenges (SumCheckTruthPath.VertexEncoding.fieldCoordinates ops suffix) length with
    ⟨polynomial, represents⟩
  simpa only [represents, BooleanTable.evaluate, point] using
    (affine_endpoints ops laws polynomial value).symm

/-- Every existing message field interpolates at the same prefixed point.
There is no endpoint-image correctness hypothesis. -/
theorem affine_messageAt_eq (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : ProtocolPolynomial.Data Field shape) (challenges : List Field) {remaining : Nat}
    (dimension : shape.cubeVariables = challenges.length + remaining + 1)
    (suffix : BooleanVertex remaining) (value : Field) :
    PiCCSFirstRoundPair.affineMessage ops
      (ProtocolPolynomial.messageAt ops data (point ops challenges dimension ops.zero suffix))
      (ProtocolPolynomial.messageAt ops data (point ops challenges dimension ops.one suffix)) value =
      ProtocolPolynomial.messageAt ops data (point ops challenges dimension value suffix) := by
  apply ProtocolPolynomial.OutputMessage.ext
  · funext source matrix
    exact table_affine ops laws (data.freshMatrixImages source matrix) challenges dimension suffix value
  · funext source
    exact table_affine ops laws (data.sourceAssignments source) challenges dimension suffix value
  · funext coordinate
    exact table_affine ops laws (data.padImages coordinate) challenges dimension suffix value
  · funext coordinate
    exact table_affine ops laws (data.matrixImages coordinate) challenges dimension suffix value

/-- The existing pair kernel, with actual messageAt endpoints and consumed
alpha/prior factors. Shape and verifier degree are unchanged. -/
def pairPolynomial (ops : InterpolationOps Field)
    (data : ProtocolPolynomial.Data Field shape)
    (alpha : CubePoint Field shape.cubeVariables) (gamma : Field)
    (challenges : List Field) {remaining : Nat}
    (dimension : shape.cubeVariables = challenges.length + remaining + 1)
    (suffix : BooleanVertex remaining) : FixedPolynomial Field data.toVerifierInput.sumcheckDegreeBound :=
  PiCCSFirstRoundPair.pairPolynomial ops data.toVerifierInput gamma
    (PiCCSPrefixSelector.selector ops challenges suffix alpha)
    (PiCCSPrefixSelector.selector ops challenges suffix data.priorPoint)
    (ProtocolPolynomial.messageAt ops data (point ops challenges dimension ops.zero suffix))
    (ProtocolPolynomial.messageAt ops data (point ops challenges dimension ops.one suffix))

/-- This pair evaluates to the original Q at the complete prefixed point. -/
theorem pairPolynomial_evaluate (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : ProtocolPolynomial.Data Field shape)
    (alpha : CubePoint Field shape.cubeVariables) (gamma : Field)
    (challenges : List Field) {remaining : Nat}
    (dimension : shape.cubeVariables = challenges.length + remaining + 1)
    (suffix : BooleanVertex remaining) (value : Field) :
    (pairPolynomial ops data alpha gamma challenges dimension suffix).evaluate ops.toOps value =
      ProtocolPolynomial.polynomial ops data alpha gamma
        (challenges ++ value :: SumCheckTruthPath.VertexEncoding.fieldCoordinates ops suffix) := by
  calc
    _ = ProtocolPolynomial.terminalFromMessage ops data.toVerifierInput alpha gamma
        (point ops challenges dimension value suffix)
        (PiCCSFirstRoundPair.affineMessage ops
          (ProtocolPolynomial.messageAt ops data (point ops challenges dimension ops.zero suffix))
          (ProtocolPolynomial.messageAt ops data (point ops challenges dimension ops.one suffix)) value) :=
      PiCCSFirstRoundPair.pairPolynomial_evaluate ops laws data.toVerifierInput gamma
        (PiCCSPrefixSelector.selector ops challenges suffix alpha)
        (PiCCSPrefixSelector.selector ops challenges suffix data.priorPoint)
        (ProtocolPolynomial.messageAt ops data (point ops challenges dimension ops.zero suffix))
        (ProtocolPolynomial.messageAt ops data (point ops challenges dimension ops.one suffix))
        alpha (point ops challenges dimension value suffix) value
        (PiCCSPrefixSelector.selector_evaluate ops laws challenges suffix alpha dimension value)
        (PiCCSPrefixSelector.selector_evaluate ops laws challenges suffix data.priorPoint dimension value)
    _ = _ := by
      rw [affine_messageAt_eq ops laws]
      have length :
          (challenges ++ value :: SumCheckTruthPath.VertexEncoding.fieldCoordinates ops suffix).length =
            shape.cubeVariables := (point ops challenges dimension value suffix).dimension
      simp only [ProtocolPolynomial.polynomial, dif_pos length]
      rfl

/-- Total numeric access with the same zero convention as the first round. -/
def numericPair (ops : InterpolationOps Field)
    (data : ProtocolPolynomial.Data Field shape)
    (alpha : CubePoint Field shape.cubeVariables) (gamma : Field)
    (challenges : List Field) {remaining : Nat}
    (dimension : shape.cubeVariables = challenges.length + remaining + 1) (index : Nat) :
    FixedPolynomial Field data.toVerifierInput.sumcheckDegreeBound :=
  if inside : index < 2 ^ remaining then
    pairPolynomial ops data alpha gamma challenges dimension
      (NumericBooleanDomain.vertex remaining ⟨index, inside⟩)
  else FixedPolynomial.zero ops.toOps data.toVerifierInput.sumcheckDegreeBound

private theorem numericPair_index (ops : InterpolationOps Field)
    (data : ProtocolPolynomial.Data Field shape)
    (alpha : CubePoint Field shape.cubeVariables) (gamma : Field)
    (challenges : List Field) {remaining : Nat}
    (dimension : shape.cubeVariables = challenges.length + remaining + 1)
    (suffix : BooleanVertex remaining) :
    numericPair ops data alpha gamma challenges dimension (NumericBooleanDomain.index suffix) =
      pairPolynomial ops data alpha gamma challenges dimension suffix := by
  simp only [numericPair, dif_pos (NumericBooleanDomain.index_lt_twoPow suffix),
    NumericBooleanDomain.vertex_index]

private theorem range_evaluate (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) {degree : Nat}
    (start count : Nat) (term : Nat → FixedPolynomial Field degree) (value : Field) :
    (PiCCSPolynomialRange.range ops start count term).evaluate ops.toOps value =
      NumericCompletionSum.numericSum ops count
        (fun index => (term (start + index)).evaluate ops.toOps value) := by
  induction count with
  | zero => exact FixedPolynomial.evaluate_zero ops.toOps (polynomialLaws laws) degree value
  | succ count ih =>
      simpa only [PiCCSPolynomialRange.range, NumericCompletionSum.numericSum, Nat.fold_succ,
        FixedPolynomial.evaluate_add ops.toOps (polynomialLaws laws)] using
        congrArg (fun accumulated => ops.add accumulated ((term (start + count)).evaluate ops.toOps value)) ih

/-- Full completion sum at the original degree, with no new shape or table
representation. The executable accumulation reuses the existing numeric range. -/
def roundPolynomial (ops : InterpolationOps Field)
    (data : ProtocolPolynomial.Data Field shape)
    (alpha : CubePoint Field shape.cubeVariables) (gamma : Field)
    (challenges : List Field) {remaining : Nat}
    (dimension : shape.cubeVariables = challenges.length + remaining + 1) :
    FixedPolynomial Field data.toVerifierInput.sumcheckDegreeBound :=
  PiCCSPolynomialRange.range ops 0 (2 ^ remaining)
    (numericPair ops data alpha gamma challenges dimension)

/-- The generic next round is the existing completion-sum specification.
Only interpolation laws and the exact prefix/suffix dimension are premises. -/
theorem roundPolynomial_evaluate (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : ProtocolPolynomial.Data Field shape)
    (alpha : CubePoint Field shape.cubeVariables) (gamma : Field)
    (challenges : List Field) {remaining : Nat}
    (dimension : shape.cubeVariables = challenges.length + remaining + 1) (value : Field) :
    (roundPolynomial ops data alpha gamma challenges dimension).evaluate ops.toOps value =
      HypercubeTruth.sumCompletions ops.toOps
        (ProtocolPolynomial.polynomial ops data alpha gamma) (challenges ++ [value]) remaining := by
  rw [roundPolynomial, range_evaluate ops laws, NumericCompletionSum.numericSum_eq_vertexSum ops laws]
  simp only [Nat.zero_add]
  calc
    _ = FiniteSumAlgebra.sumMap ops (BooleanVertex.all remaining) (fun suffix =>
        ProtocolPolynomial.polynomial ops data alpha gamma
          ((challenges ++ [value]) ++ SumCheckTruthPath.VertexEncoding.fieldCoordinates ops suffix)) := by
      apply FiniteSumAlgebra.sumMap_congr
      intro suffix _
      rw [numericPair_index, pairPolynomial_evaluate ops laws]
      simp only [List.append_assoc, List.singleton_append]
    _ = _ := (SumCheckTruthPath.sumCompletions_eq_vertexSum ops laws
      (ProtocolPolynomial.polynomial ops data alpha gamma) (challenges ++ [value]) remaining).symm

end NightstreamFPrime.Export.Stage1.PiCCSPrefixRound
