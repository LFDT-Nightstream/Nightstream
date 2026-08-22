import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.BooleanHypercubeSum

/-! Provenance: copied from `formal/nightstream-lean/Nightstream/SuperNeo/Folding/PiCCS/PaperJoint/NumericBooleanDomain.lean`
at commit `fb7a8a99aefbb8ebb5474681ecf80f1b95a1b7a2`; namespaces renamed, otherwise unchanged. -/

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

Authority boundary: the head Boolean coordinate is bit zero. Numeric users
must refine their serialization to this owner rather than defining another
bit-order convention locally.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `pi_ccs.domain.numeric.index` | `index(bit :: tail) = bit + 2 * index(tail)` | computed | `index` |
| `pi_ccs.domain.numeric.bound` | every typed vertex lies below `2^variables` | derived | `index_lt_twoPow` |
| `pi_ccs.domain.numeric.inverse` | parity/division and indexing are inverse | derived | `index_vertex`, `vertex_index` |
| `pi_ccs.domain.numeric.weight.recursive` | parity/division tensor weight equals Boolean `chi` | derived | `tensorWeight_eq_equalityWeight` |
| `pi_ccs.domain.numeric.weight.test_bit` | prior production-shaped `Nat.testBit` fold | computed | `testBitWeight` |
| `pi_ccs.domain.numeric.weight.bridge` | recursive and `Nat.testBit` products are identical | derived | `tensorWeight_eq_testBitWeight` |
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.NumericBooleanDomain

open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

universe uField

/-- Numeric value of one Boolean bit. -/
private def bitValue : Bool -> Nat
  | false => 0
  | true => 1

/-- Little-endian numeric index. The head coordinate is bit zero. -/
def index : {variables : Nat} -> BooleanVertex variables -> Nat
  | 0, .nil => 0
  | _ + 1, .cons bit tail => bitValue bit + 2 * index tail

/-- Every typed Boolean vertex maps into the exact `2^variables` domain. -/
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

/-- Parity bit used by the independent numeric decoder. -/
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

/-- Decode a bounded numeric index by repeatedly reading its least-significant
bit and dividing by two. -/
def vertex :
    (variables : Nat) -> Fin (2 ^ variables) -> BooleanVertex variables
  | 0, _ => .nil
  | variables + 1, value =>
      .cons (parityBit value.val)
        (vertex variables
          ⟨value.val / 2, divTwo_lt_twoPow value⟩)

/-- Encoding a decoded bounded index returns that exact number. -/
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

/-- Decoding a typed vertex's little-endian index returns that vertex. -/
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

/-- Numeric tensor-weight recursion. Coordinates are consumed from bit zero
upward while the numeric index is consumed least-significant bit first. -/
def tensorWeightCoordinates
    {Field : Type uField}
    (ops : InterpolationOps Field) : List Field -> Nat -> Field
  | [], _ => ops.one
  | coordinate :: coordinates, value =>
      ops.mul
        (if parityBit value then coordinate else ops.sub ops.one coordinate)
        (tensorWeightCoordinates ops coordinates (value / 2))

/-- Tensor weight of one bounded numeric index at a dimension-checked point. -/
def tensorWeight
    {Field : Type uField}
    (ops : InterpolationOps Field)
    {variables : Nat}
    (value : Fin (2 ^ variables))
    (point : CubePoint Field variables) : Field :=
  tensorWeightCoordinates ops point.coordinates value.val

/-- Little-endian `Nat.testBit` fold used by the pre-existing output-claim
semantics and the current Rust table construction. It remains a separate
definition so `tensorWeight_eq_testBitWeight` records the exact proof boundary
rather than silently substituting one formula for the other. -/
def testBitWeight
    {Field : Type uField}
    (ops : InterpolationOps Field)
    {variables : Nat}
    (point : CubePoint Field variables)
    (value : Fin (2 ^ variables)) : Field :=
  (canonicalFinIndices variables).foldl
    (fun accumulated bit =>
      let coordinate := point.coordinates.getD bit.val ops.zero
      let factor := if Nat.testBit value.val bit.val then
        coordinate
      else
        ops.sub ops.one coordinate
      ops.mul accumulated factor)
    ops.one

/-- Exactly the multiplicative laws needed to compare the right-associated
recursive tensor product with the left-associated production-shaped fold.
No additive, distributive, commutative, or inverse law is assumed. -/
structure WeightProductLaws
    {Field : Type uField}
    (ops : InterpolationOps Field) : Prop where
  one_mul : forall value, ops.mul ops.one value = value
  mul_one : forall value, ops.mul value ops.one = value
  mul_assoc : forall left middle right,
    ops.mul (ops.mul left middle) right =
      ops.mul left (ops.mul middle right)

namespace WeightProductLaws

/-- Project the three laws used by the numeric-weight bridge from the existing
full interpolation law package. This keeps concrete arithmetic ownership with
its existing provider while avoiding a stronger assumption in the bridge. -/
def ofInterpolationEvaluationLaws
    {Field : Type uField}
    {ops : InterpolationOps Field}
    (laws : InterpolationEvaluationLaws ops) : WeightProductLaws ops where
  one_mul := laws.one_mul
  mul_one := laws.mul_one
  mul_assoc := laws.mul_assoc

end WeightProductLaws

private theorem foldl_factor_eq_mul_foldl_one
    {Field : Type uField}
    {Index : Type}
    (ops : InterpolationOps Field)
    (laws : WeightProductLaws ops)
    (factor : Index -> Field)
    (accumulated : Field)
    (indices : List Index) :
    indices.foldl (fun product index => ops.mul product (factor index))
        accumulated =
      ops.mul accumulated
        (indices.foldl (fun product index => ops.mul product (factor index))
          ops.one) := by
  letI : Std.Associative ops.mul := ⟨laws.mul_assoc⟩
  calc
    indices.foldl (fun product index => ops.mul product (factor index))
        accumulated =
      (indices.map factor).foldl ops.mul accumulated :=
        List.foldl_map.symm
    _ = (indices.map factor).foldl ops.mul
        (ops.mul accumulated ops.one) := by rw [laws.mul_one]
    _ = ops.mul accumulated
        ((indices.map factor).foldl ops.mul ops.one) := List.foldl_assoc
    _ = ops.mul accumulated
        (indices.foldl (fun product index => ops.mul product (factor index))
          ops.one) := by rw [List.foldl_map]

private theorem testBit_zero_eq_parityBit (value : Nat) :
    Nat.testBit value 0 = parityBit value := by
  have remainderBound : value % 2 < 2 := Nat.mod_lt _ (by decide)
  by_cases isOne : value % 2 = 1
  · simp [Nat.testBit_zero, parityBit, isOne]
  · have isZero : value % 2 = 0 := by omega
    simp [Nat.testBit_zero, parityBit, isZero]

private theorem testBitWeightCoordinates_eq_tensorWeightCoordinates
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : WeightProductLaws ops) :
    {variables : Nat} ->
      (value : Fin (2 ^ variables)) ->
      (coordinates : List Field) ->
      coordinates.length = variables ->
      (canonicalFinIndices variables).foldl
        (fun accumulated bit =>
          let coordinate := coordinates.getD bit.val ops.zero
          let factor := if Nat.testBit value.val bit.val then
            coordinate
          else
            ops.sub ops.one coordinate
          ops.mul accumulated factor)
        ops.one =
        tensorWeightCoordinates ops coordinates value.val
  | 0, value, coordinates, dimension => by
      cases coordinates with
      | nil => simp [canonicalFinIndices, tensorWeightCoordinates]
      | cons coordinate coordinates => simp at dimension
  | variables + 1, value, coordinates, dimension => by
      cases coordinates with
      | nil => simp at dimension
      | cons coordinate coordinates =>
          have tailDimension : coordinates.length = variables := by
            simpa using dimension
          let tailValue : Fin (2 ^ variables) :=
            ⟨value.val / 2, divTwo_lt_twoPow value⟩
          have tailBridge :=
            testBitWeightCoordinates_eq_tensorWeightCoordinates
              ops laws tailValue coordinates tailDimension
          rw [canonicalFinIndices, List.ofFn_succ, List.foldl_cons]
          simp only [id_eq, Fin.val_zero, List.getD_cons_zero]
          rw [testBit_zero_eq_parityBit]
          rw [laws.one_mul]
          rw [foldl_factor_eq_mul_foldl_one ops laws]
          simp only [tensorWeightCoordinates]
          congr 1
          have tailIndices :
              List.ofFn (fun index : Fin variables => Fin.succ index) =
                (canonicalFinIndices variables).map Fin.succ := by
            simp [canonicalFinIndices, Function.comp_def]
          rw [tailIndices, List.foldl_map]
          simpa [tailValue, Nat.testBit_succ] using tailBridge

/-- The shared recursive tensor weight is exactly the preserved
production-shaped `Nat.testBit` fold for every bounded index and
dimension-checked point. The theorem is generic over the carrier and assumes
only the three laws needed to reassociate the product. -/
theorem tensorWeight_eq_testBitWeight
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : WeightProductLaws ops)
    {variables : Nat}
    (value : Fin (2 ^ variables))
    (point : CubePoint Field variables) :
    tensorWeight ops value point = testBitWeight ops point value := by
  exact (testBitWeightCoordinates_eq_tensorWeightCoordinates
    ops laws value point.coordinates point.dimension).symm

private theorem tensorWeightCoordinates_eq_equalityWeightCoordinates
    {Field : Type uField}
    (ops : InterpolationOps Field)
    {variables : Nat}
    (value : Fin (2 ^ variables))
    (coordinates : List Field)
    (dimension : coordinates.length = variables) :
    tensorWeightCoordinates ops coordinates value.val =
      BooleanVertex.equalityWeightCoordinates ops
        (vertex variables value) coordinates := by
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

/-- The numeric tensor formula is exactly the independent Boolean equality
weight at the decoded typed vertex. -/
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
