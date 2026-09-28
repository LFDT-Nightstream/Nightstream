import NightstreamFPrime.Export.Stage1.PiCCSFirstRoundPair
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.NumericCompletionSum

/-!
Complete first-round coefficient construction from the existing protocol data.
Each numeric Boolean suffix selects its exact two endpoint image messages.
The finite polynomial sum uses no artifact-sized list of indices. This file
owns no source loading, selected matrix executor, or transcript operation.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSFirstRound

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.SumCheck.Finite
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open ProtocolPolynomialDegree.Support (polynomialLaws)

universe uField
variable {Field : Type uField} {shape : Shape}

/-- Prepend the first Boolean coordinate in the existing vertex type. -/
def endpointVertex {arity remaining : Nat}
    (dimension : arity = remaining + 1) (bit : Bool)
    (suffix : BooleanVertex remaining) : BooleanVertex arity :=
  dimension.symm ▸ BooleanVertex.cons bit suffix

/-- The exposed first coordinate followed by a complete Boolean suffix. -/
def pairPoint (ops : InterpolationOps Field) {arity remaining : Nat}
    (dimension : arity = remaining + 1) (value : Field)
    (suffix : BooleanVertex remaining) : CubePoint Field arity where
  coordinates := value :: SumCheckTruthPath.VertexEncoding.fieldCoordinates ops suffix
  dimension := by
    rw [List.length_cons, SumCheckTruthPath.VertexEncoding.fieldCoordinates_length]
    exact dimension.symm

private theorem table_pair (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) {arity remaining : Nat}
    (dimension : arity = remaining + 1) (table : BooleanTable Field arity)
    (suffix : BooleanVertex remaining) (value : Field) :
    table.evaluate ops (pairPoint ops dimension value suffix) =
      ops.add (table.valueAt (endpointVertex dimension false suffix))
        (ops.mul value
          (ops.sub (table.valueAt (endpointVertex dimension true suffix))
            (table.valueAt (endpointVertex dimension false suffix)))) := by
  subst arity
  cases table with
  | branch low high =>
      change ops.add
          (low.evaluateCoordinates ops (SumCheckTruthPath.VertexEncoding.fieldCoordinates ops suffix))
          (ops.mul value (ops.sub
            (high.evaluateCoordinates ops (SumCheckTruthPath.VertexEncoding.fieldCoordinates ops suffix))
            (low.evaluateCoordinates ops (SumCheckTruthPath.VertexEncoding.fieldCoordinates ops suffix)))) =
        ops.add (low.valueAt suffix) (ops.mul value (ops.sub (high.valueAt suffix) (low.valueAt suffix)))
      simp only [SumCheckTruthPath.evaluateCoordinates_fieldCoordinates_eq_valueAt ops laws]

/-- Actual endpoint leaves interpolate to the actual protocol image message.
No caller-supplied endpoint-image equality is used. -/
theorem affine_vertexMessage_eq (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : ProtocolPolynomial.Data Field shape) {remaining : Nat}
    (dimension : shape.cubeVariables = remaining + 1)
    (suffix : BooleanVertex remaining) (value : Field) :
    PiCCSFirstRoundPair.affineMessage ops
      (ProtocolPolynomial.vertexMessage data (endpointVertex dimension false suffix))
      (ProtocolPolynomial.vertexMessage data (endpointVertex dimension true suffix)) value =
      ProtocolPolynomial.messageAt ops data (pairPoint ops dimension value suffix) := by
  apply ProtocolPolynomial.OutputMessage.ext
  · funext source matrix
    exact (table_pair ops laws dimension (data.freshMatrixImages source matrix) suffix value).symm
  · funext source
    exact (table_pair ops laws dimension (data.sourceAssignments source) suffix value).symm
  · funext coordinate
    exact (table_pair ops laws dimension (data.padImages coordinate) suffix value).symm
  · funext coordinate
    exact (table_pair ops laws dimension (data.matrixImages coordinate) suffix value).symm

/-- The suffix equality is constant; only the first equality factor is affine.
The empty-target case is total and is excluded by the dimension theorem. -/
def equalitySelector (ops : InterpolationOps Field) {arity remaining : Nat}
    (suffix : BooleanVertex remaining) (target : CubePoint Field arity) :
    FixedPolynomial Field 1 :=
  match target.coordinates with
  | [] => FixedPolynomial.zero ops.toOps 1
  | head :: tail =>
      FixedPolynomial.scale ops.toOps
        (SumCheckTruthPath.pointEqualityCoordinates ops
          (SumCheckTruthPath.VertexEncoding.fieldCoordinates ops suffix) tail)
        (FixedPolynomial.affine (ops.sub ops.one head)
          (ops.sub head (ops.sub ops.one head)))

/-- This computable selector equals the exact equality polynomial at the
exposed coordinate and the supplied Boolean suffix. -/
theorem equalitySelector_evaluate (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) {arity remaining : Nat}
    (dimension : arity = remaining + 1) (suffix : BooleanVertex remaining)
    (target : CubePoint Field arity) (value : Field) :
    (equalitySelector ops suffix target).evaluate ops.toOps value =
      SumCheckTruthPath.pointEquality ops (pairPoint ops dimension value suffix) target := by
  rcases target with ⟨coordinates, length⟩
  cases coordinates with
  | nil =>
      simp only [List.length_nil] at length
      omega
  | cons head tail =>
      let tailEquality := SumCheckTruthPath.pointEqualityCoordinates ops
        (SumCheckTruthPath.VertexEncoding.fieldCoordinates ops suffix) tail
      change (FixedPolynomial.scale ops.toOps tailEquality
          (FixedPolynomial.affine (ops.sub ops.one head)
            (ops.sub head (ops.sub ops.one head)))).evaluate ops.toOps value =
        ops.mul (SumCheckTruthPath.equalityFactor ops value head) tailEquality
      rw [FixedPolynomial.evaluate_scale ops.toOps (polynomialLaws laws),
        FixedPolynomial.evaluate_affine ops.toOps (polynomialLaws laws),
        laws.mul_comm tailEquality]
      exact congrArg (fun factor => ops.mul factor tailEquality)
        (SumCheckTruthPath.equalityFactor_eq_affine laws value head).symm

/-- One full protocol pair, with both messages and both selectors constructed
from the same protocol data and verifier coins. -/
def vertexPolynomial (ops : InterpolationOps Field)
    (data : ProtocolPolynomial.Data Field shape)
    (alpha : CubePoint Field shape.cubeVariables) (gamma : Field) {remaining : Nat}
    (dimension : shape.cubeVariables = remaining + 1) (suffix : BooleanVertex remaining) :
    FixedPolynomial Field data.toVerifierInput.sumcheckDegreeBound :=
  PiCCSFirstRoundPair.pairPolynomial ops data.toVerifierInput gamma
    (equalitySelector ops suffix alpha) (equalitySelector ops suffix data.priorPoint)
    (ProtocolPolynomial.vertexMessage data (endpointVertex dimension false suffix))
    (ProtocolPolynomial.vertexMessage data (endpointVertex dimension true suffix))

/-- The pair coefficient computation evaluates to the actual off-cube Q,
with a free first coordinate and every remaining coordinate Boolean. -/
theorem vertexPolynomial_evaluate (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : ProtocolPolynomial.Data Field shape)
    (alpha : CubePoint Field shape.cubeVariables) (gamma : Field) {remaining : Nat}
    (dimension : shape.cubeVariables = remaining + 1)
    (suffix : BooleanVertex remaining) (value : Field) :
    (vertexPolynomial ops data alpha gamma dimension suffix).evaluate ops.toOps value =
      ProtocolPolynomial.polynomial ops data alpha gamma
        (value :: SumCheckTruthPath.VertexEncoding.fieldCoordinates ops suffix) := by
  calc
    _ = ProtocolPolynomial.terminalFromMessage ops data.toVerifierInput alpha gamma
        (pairPoint ops dimension value suffix)
        (PiCCSFirstRoundPair.affineMessage ops
          (ProtocolPolynomial.vertexMessage data (endpointVertex dimension false suffix))
          (ProtocolPolynomial.vertexMessage data (endpointVertex dimension true suffix)) value) :=
      PiCCSFirstRoundPair.pairPolynomial_evaluate ops laws data.toVerifierInput gamma
        (equalitySelector ops suffix alpha) (equalitySelector ops suffix data.priorPoint)
        (ProtocolPolynomial.vertexMessage data (endpointVertex dimension false suffix))
        (ProtocolPolynomial.vertexMessage data (endpointVertex dimension true suffix))
        alpha (pairPoint ops dimension value suffix) value
        (equalitySelector_evaluate ops laws dimension suffix alpha value)
        (equalitySelector_evaluate ops laws dimension suffix data.priorPoint value)
    _ = _ := by
      rw [affine_vertexMessage_eq ops laws]
      have length :
          (value :: SumCheckTruthPath.VertexEncoding.fieldCoordinates ops suffix).length =
            shape.cubeVariables := (pairPoint ops dimension value suffix).dimension
      simp only [ProtocolPolynomial.polynomial, dif_pos length]
      rfl

private def polynomialSum (ops : InterpolationOps Field) {degree : Nat}
    (count : Nat) (term : Nat → FixedPolynomial Field degree) : FixedPolynomial Field degree :=
  Nat.fold count (fun index _ accumulated => FixedPolynomial.add ops.toOps accumulated (term index))
    (FixedPolynomial.zero ops.toOps degree)

private theorem polynomialSum_evaluate (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) {degree : Nat}
    (count : Nat) (term : Nat → FixedPolynomial Field degree) (value : Field) :
    (polynomialSum ops count term).evaluate ops.toOps value =
      NumericCompletionSum.numericSum ops count (fun index => (term index).evaluate ops.toOps value) := by
  induction count with
  | zero => exact FixedPolynomial.evaluate_zero ops.toOps (polynomialLaws laws) degree value
  | succ count ih =>
      simpa only [polynomialSum, NumericCompletionSum.numericSum, Nat.fold_succ,
        FixedPolynomial.evaluate_add ops.toOps (polynomialLaws laws)] using
        congrArg (fun accumulated => ops.add accumulated ((term count).evaluate ops.toOps value)) ih

/-- Total Nat-indexed access; the full sum only uses indices inside its bound. -/
def numericPair (ops : InterpolationOps Field)
    (data : ProtocolPolynomial.Data Field shape)
    (alpha : CubePoint Field shape.cubeVariables) (gamma : Field) {remaining : Nat}
    (dimension : shape.cubeVariables = remaining + 1) (index : Nat) :
    FixedPolynomial Field data.toVerifierInput.sumcheckDegreeBound :=
  if inside : index < 2 ^ remaining then
    vertexPolynomial ops data alpha gamma dimension (NumericBooleanDomain.vertex remaining ⟨index, inside⟩)
  else FixedPolynomial.zero ops.toOps data.toVerifierInput.sumcheckDegreeBound

private theorem numericPair_index (ops : InterpolationOps Field)
    (data : ProtocolPolynomial.Data Field shape)
    (alpha : CubePoint Field shape.cubeVariables) (gamma : Field) {remaining : Nat}
    (dimension : shape.cubeVariables = remaining + 1) (suffix : BooleanVertex remaining) :
    numericPair ops data alpha gamma dimension (NumericBooleanDomain.index suffix) =
      vertexPolynomial ops data alpha gamma dimension suffix := by
  simp only [numericPair, dif_pos (NumericBooleanDomain.index_lt_twoPow suffix),
    NumericBooleanDomain.vertex_index]

/-- Sum every first-round pair, with the existing fixed coefficient width.
No proper prefix or zero-suffix assumption is used for this full constructor. -/
def firstRound (ops : InterpolationOps Field)
    (data : ProtocolPolynomial.Data Field shape)
    (alpha : CubePoint Field shape.cubeVariables) (gamma : Field) {remaining : Nat}
    (dimension : shape.cubeVariables = remaining + 1) :
    FixedPolynomial Field data.toVerifierInput.sumcheckDegreeBound :=
  polynomialSum ops (2 ^ remaining) (numericPair ops data alpha gamma dimension)

/-- The complete computable first-round polynomial is the existing Boolean
completion sum of the actual protocol polynomial, at every field value. -/
theorem firstRound_evaluate (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : ProtocolPolynomial.Data Field shape)
    (alpha : CubePoint Field shape.cubeVariables) (gamma : Field) {remaining : Nat}
    (dimension : shape.cubeVariables = remaining + 1) (value : Field) :
    (firstRound ops data alpha gamma dimension).evaluate ops.toOps value =
      HypercubeTruth.sumCompletions ops.toOps
        (ProtocolPolynomial.polynomial ops data alpha gamma) [value] remaining := by
  rw [firstRound, polynomialSum_evaluate ops laws,
    NumericCompletionSum.numericSum_eq_vertexSum ops laws]
  calc
    _ = FiniteSumAlgebra.sumMap ops (BooleanVertex.all remaining) (fun suffix =>
        ProtocolPolynomial.polynomial ops data alpha gamma
          (value :: SumCheckTruthPath.VertexEncoding.fieldCoordinates ops suffix)) := by
      apply FiniteSumAlgebra.sumMap_congr
      intro suffix _
      rw [numericPair_index, vertexPolynomial_evaluate ops laws]
    _ = _ := (SumCheckTruthPath.sumCompletions_eq_vertexSum ops laws
      (ProtocolPolynomial.polynomial ops data alpha gamma) [value] remaining).symm

end NightstreamFPrime.Export.Stage1.PiCCSFirstRound
