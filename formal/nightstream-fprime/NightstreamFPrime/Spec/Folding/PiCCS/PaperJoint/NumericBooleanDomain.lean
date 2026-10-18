import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.BooleanHypercubeSum

/-!
Canonical little-endian bridge between numeric indices and Boolean cubes.

Protocol: shared infrastructure for `Pi_CCS` row, lane, and column domains.
Phase: numeric-domain decoding before semantic polynomial evaluation.
Constraint family: none; this file emits no rows.

Owns: one little-endian numeric index for typed Boolean vertices, its typed
inverse on a `2^variables` domain, and the numeric tensor-weight formula with
its exact equality to the independent Boolean equality weight.

Does not own: a particular row, lane, or column domain; matrix contents;
residual formulas; transcript byte order; Rust; R1CS; or constraint counts.

Emits constraints: no.
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.NumericBooleanDomain

open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

universe uField

private def bitValue : Bool -> Nat
  | false => 0
  | true => 1

def index : {variables : Nat} -> BooleanVertex variables -> Nat
  | 0, .nil => 0
  | _ + 1, .cons bit tail => bitValue bit + 2 * index tail

theorem index_lt_twoPow :
    {variables : Nat} ->
      (point : BooleanVertex variables) ->
        index point < 2 ^ variables
  | 0, .nil => by decide
  | variables + 1, .cons bit tail => by
      have tailBound := index_lt_twoPow tail
      cases bit <;>
        simp only [index, bitValue, Nat.pow_succ] <;>
        omega

private def parityBit (value : Nat) : Bool :=
  value % 2 == 1

private theorem parityBit_value (value : Nat) :
    bitValue (parityBit value) = value % 2 := by
  have remainderBound : value % 2 < 2 := Nat.mod_lt _ (by decide)
  by_cases isOne : value % 2 = 1
  · simp [parityBit, bitValue, isOne]
  · have isZero : value % 2 = 0 := by omega
    simp [parityBit, bitValue, isZero]

private theorem divTwo_lt_twoPow
    {variables : Nat} (value : Fin (2 ^ (variables + 1))) :
    value.val / 2 < 2 ^ variables := by
  have valueBound : value.val < 2 ^ variables * 2 := by
    simpa only [Nat.pow_succ] using value.isLt
  omega

def vertex :
    (variables : Nat) -> Fin (2 ^ variables) -> BooleanVertex variables
  | 0, _ => .nil
  | variables + 1, value =>
      .cons (parityBit value.val)
        (vertex variables
          ⟨value.val / 2, divTwo_lt_twoPow value⟩)

theorem index_vertex
    (variables : Nat) (value : Fin (2 ^ variables)) :
    index (vertex variables value) = value.val := by
  induction variables with
  | zero =>
      have valueZero : value.val = 0 := by
        have valueBound := value.isLt
        simp at valueBound
        omega
      simp [vertex, index]
  | succ variables inductionHypothesis =>
      simp only [vertex, index]
      rw [inductionHypothesis, parityBit_value]
      exact Nat.mod_add_div value.val 2

theorem vertex_index
    {variables : Nat} (point : BooleanVertex variables) :
    vertex variables ⟨index point, index_lt_twoPow point⟩ = point := by
  induction point with
  | nil => rfl
  | @cons variables bit tail inductionHypothesis =>
      cases bit with
      | false =>
          have parity : parityBit (2 * index tail) = false := by
            simp [parityBit]
          simp only [index, bitValue, Nat.zero_add, vertex, parity]
          simpa using inductionHypothesis
      | true =>
          have parity : parityBit (1 + 2 * index tail) = true := by
            simp [parityBit, Nat.add_mod]
          have quotient : (1 + 2 * index tail) / 2 = index tail := by
            omega
          simp only [index, bitValue, vertex, parity]
          simpa [quotient] using inductionHypothesis

def tensorWeightCoordinates
    {Field : Type uField}
    (ops : InterpolationOps Field) : List Field -> Nat -> Field
  | [], _ => ops.one
  | coordinate :: coordinates, value =>
      ops.mul
        (if parityBit value then coordinate else ops.sub ops.one coordinate)
        (tensorWeightCoordinates ops coordinates (value / 2))

def tensorWeight
    {Field : Type uField}
    (ops : InterpolationOps Field)
    {variables : Nat}
    (value : Fin (2 ^ variables))
    (point : CubePoint Field variables) : Field :=
  tensorWeightCoordinates ops point.coordinates value.val

structure WeightProductLaws
    {Field : Type uField}
    (ops : InterpolationOps Field) : Prop where
  one_mul : forall value, ops.mul ops.one value = value
  mul_one : forall value, ops.mul value ops.one = value
  mul_assoc : forall left middle right,
    ops.mul (ops.mul left middle) right =
      ops.mul left (ops.mul middle right)

namespace WeightProductLaws

theorem ofInterpolationEvaluationLaws
    {Field : Type uField}
    {ops : InterpolationOps Field}
    (laws : InterpolationEvaluationLaws ops) : WeightProductLaws ops where
  one_mul := laws.one_mul
  mul_one := laws.mul_one
  mul_assoc := laws.mul_assoc

end WeightProductLaws

private theorem tensorWeightCoordinates_eq_equalityWeightCoordinates
    {Field : Type uField}
    (ops : InterpolationOps Field)
    {variables : Nat}
    (value : Fin (2 ^ variables))
    (coordinates : List Field)
    (dimension : coordinates.length = variables) :
    tensorWeightCoordinates ops coordinates value.val =
      BooleanVertex.equalityWeightCoordinates ops
        (vertex variables value) coordinates dimension := by
  induction variables generalizing coordinates with
  | zero =>
      cases coordinates with
      | nil => rfl
      | cons coordinate coordinates => simp at dimension
  | succ variables inductionHypothesis =>
      cases coordinates with
      | nil => simp at dimension
      | cons coordinate coordinates =>
          have tailDimension : coordinates.length = variables := by
            simpa using dimension
          simp only [tensorWeightCoordinates, vertex]
          cases parity : parityBit value.val <;>
            simp only [Bool.false_eq_true, if_false, if_true,
              BooleanVertex.equalityWeightCoordinates]
          · congr 1
            exact inductionHypothesis
              ⟨value.val / 2, divTwo_lt_twoPow value⟩
              coordinates tailDimension
          · congr 1
            exact inductionHypothesis
              ⟨value.val / 2, divTwo_lt_twoPow value⟩
              coordinates tailDimension

theorem tensorWeight_eq_equalityWeight
    {Field : Type uField}
    (ops : InterpolationOps Field)
    {variables : Nat}
    (value : Fin (2 ^ variables))
    (point : CubePoint Field variables) :
    tensorWeight ops value point =
      BooleanVertex.equalityWeight ops (vertex variables value) point := by
  exact tensorWeightCoordinates_eq_equalityWeightCoordinates
    ops value point.coordinates point.dimension

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.NumericBooleanDomain
