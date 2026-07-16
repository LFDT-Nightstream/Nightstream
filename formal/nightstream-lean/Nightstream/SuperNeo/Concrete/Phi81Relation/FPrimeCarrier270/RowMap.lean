import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanHypercubeSum

/-!
Typed row-number bridge for the five-ring F' CCS source.

Protocol: SuperNeo CCS/CE relation specialized to the F' source matrices.
Phase: production numeric matrix row to the independent Boolean row domain.
Constraint family: semantic row ownership only; this file emits no rows.

Owns: the explicit little-endian row number in which the head Boolean
coordinate is bit zero; its typed inverse on exactly `2^rowVariables` rows;
both inverse theorems; and equality between the production tensor-weight
recursion and the independent paper Boolean equality weight.

Does not own: matrix contents, row padding beyond a production row count,
column alignment, CCS satisfaction, CE coefficients, commitments, PiCCS,
PiRLC, PiDEC, NIFS, generated artifacts, R1CS, or constraint counts.

Emits constraints: no.

Authority boundary: `rowIndex` and `rowVertex` are independent finite
definitions. `productionTensorWeight` models the numeric-row formula by
reading the least-significant bit and dividing the row by two at each point
coordinate. It does not trust `BooleanVertex.all` positions or an artifact.

| Protocol | Phase | Family | Mathematical obligation | Lean owner |
|---|---|---|---|---|
| F' / CCS | row decoding | little-endian index | `index(bit :: tail) = bit + 2 * index(tail)` | `rowIndex` |
| F' / CCS | row decoding | finite bound | every typed vertex indexes a row below `2^ell` | `rowIndex_lt_twoPow` |
| F' / CCS | row decoding | typed inverse | repeated parity/division reconstructs every numeric row | `rowIndex_rowVertex`, `rowVertex_rowIndex` |
| F' / CCS | matrix evaluation | numeric tensor weight | coordinate `i` selects `r_i` exactly when numeric bit `i` is one | `productionTensorWeight` |
| F' / CCS | matrix evaluation | semantic refinement | numeric tensor weight equals `BooleanVertex.equalityWeight` | `productionTensorWeight_eq_equalityWeight` |
-/

namespace Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

universe uField

/-- Numeric value of one Boolean bit. This local definition avoids making an
external integer serializer part of the semantic row interface. -/
private def bitValue : Bool -> Nat
  | false => 0
  | true => 1

/-- Little-endian numeric row index. The head coordinate is bit zero. -/
def rowIndex : {rowVariables : Nat} -> BooleanVertex rowVariables -> Nat
  | 0, .nil => 0
  | _ + 1, .cons bit tail => bitValue bit + 2 * rowIndex tail

/-- Every typed Boolean row maps into the exact `2^rowVariables` row domain. -/
theorem rowIndex_lt_twoPow :
    {rowVariables : Nat} ->
      (vertex : BooleanVertex rowVariables) ->
        rowIndex vertex < 2 ^ rowVariables
  | 0, .nil => by decide
  | rowVariables + 1, .cons bit tail => by
      have tailBound := rowIndex_lt_twoPow tail
      cases bit <;>
        simp only [rowIndex, bitValue, Nat.pow_succ] <;>
        omega

/-- Parity bit used by the independent numeric-row decoder. -/
private def parityBit (row : Nat) : Bool :=
  row % 2 == 1

private theorem parityBit_value (row : Nat) :
    bitValue (parityBit row) = row % 2 := by
  have remainderBound : row % 2 < 2 := Nat.mod_lt _ (by decide)
  by_cases isOne : row % 2 = 1
  · simp [parityBit, bitValue, isOne]
  · have isZero : row % 2 = 0 := by omega
    simp [parityBit, bitValue, isZero]

private theorem divTwo_lt_twoPow
    {rowVariables : Nat} (row : Fin (2 ^ (rowVariables + 1))) :
    row.val / 2 < 2 ^ rowVariables := by
  have rowBound : row.val < 2 ^ rowVariables * 2 := by
    simpa only [Nat.pow_succ] using row.isLt
  omega

/-- Decode a bounded numeric row by repeatedly reading its least-significant
bit and dividing by two. -/
def rowVertex :
    (rowVariables : Nat) -> Fin (2 ^ rowVariables) ->
      BooleanVertex rowVariables
  | 0, _ => .nil
  | rowVariables + 1, row =>
      .cons (parityBit row.val)
        (rowVertex rowVariables
          ⟨row.val / 2, divTwo_lt_twoPow row⟩)

/-- Encoding a decoded bounded numeric row returns that exact row number. -/
theorem rowIndex_rowVertex
    (rowVariables : Nat) (row : Fin (2 ^ rowVariables)) :
    rowIndex (rowVertex rowVariables row) = row.val := by
  induction rowVariables with
  | zero =>
      have rowZero : row.val = 0 := by
        have rowBound := row.isLt
        simp at rowBound
        omega
      simp [rowVertex, rowIndex]
  | succ rowVariables inductionHypothesis =>
      simp only [rowVertex, rowIndex]
      rw [inductionHypothesis, parityBit_value]
      exact Nat.mod_add_div row.val 2

/-- Decoding the little-endian index of a typed vertex returns that vertex. -/
theorem rowVertex_rowIndex
    {rowVariables : Nat} (vertex : BooleanVertex rowVariables) :
    rowVertex rowVariables
        ⟨rowIndex vertex, rowIndex_lt_twoPow vertex⟩ = vertex := by
  induction vertex with
  | nil => rfl
  | @cons rowVariables bit tail inductionHypothesis =>
      cases bit with
      | false =>
          have parity : parityBit (2 * rowIndex tail) = false := by
            simp [parityBit]
          simp only [rowIndex, bitValue, Nat.zero_add, rowVertex, parity]
          simpa using inductionHypothesis
      | true =>
          have parity : parityBit (1 + 2 * rowIndex tail) = true := by
            simp [parityBit, Nat.add_mod]
          have quotient : (1 + 2 * rowIndex tail) / 2 = rowIndex tail := by
            omega
          simp only [rowIndex, bitValue, rowVertex, parity]
          simpa [quotient] using inductionHypothesis

/-- Numeric-row tensor-weight recursion used by production matrix evaluation.
The coordinate list is consumed from `r_0` upward while the numeric row is
consumed from its least-significant bit upward. The raw recursion consumes the
supplied list; `CubePoint` fixes its intended dimension in
`productionTensorWeight`. -/
def productionTensorWeightCoordinates
    {Field : Type uField}
    (ops : InterpolationOps Field) : List Field -> Nat -> Field
  | [], _ => ops.one
  | coordinate :: coordinates, row =>
      ops.mul
        (if parityBit row then coordinate else ops.sub ops.one coordinate)
        (productionTensorWeightCoordinates ops coordinates (row / 2))

/-- Tensor weight of a bounded production row at a dimension-checked point. -/
def productionTensorWeight
    {Field : Type uField}
    (ops : InterpolationOps Field)
    {rowVariables : Nat}
    (row : Fin (2 ^ rowVariables))
    (point : CubePoint Field rowVariables) : Field :=
  productionTensorWeightCoordinates ops point.coordinates row.val

private theorem productionTensorWeightCoordinates_eq_equalityWeightCoordinates
    {Field : Type uField}
    (ops : InterpolationOps Field)
    {rowVariables : Nat}
    (row : Fin (2 ^ rowVariables))
    (coordinates : List Field)
    (dimension : coordinates.length = rowVariables) :
    productionTensorWeightCoordinates ops coordinates row.val =
      BooleanVertex.equalityWeightCoordinates ops
        (rowVertex rowVariables row) coordinates := by
  induction rowVariables generalizing coordinates with
  | zero =>
      cases coordinates with
      | nil => rfl
      | cons coordinate coordinates => simp at dimension
  | succ rowVariables inductionHypothesis =>
      cases coordinates with
      | nil => simp at dimension
      | cons coordinate coordinates =>
          have tailDimension : coordinates.length = rowVariables := by
            simpa using dimension
          simp only [productionTensorWeightCoordinates, rowVertex]
          cases parity : parityBit row.val <;>
            simp only [Bool.false_eq_true, if_false, if_true,
              BooleanVertex.equalityWeightCoordinates]
          · congr 1
            exact inductionHypothesis
              ⟨row.val / 2, divTwo_lt_twoPow row⟩
              coordinates tailDimension
          · congr 1
            exact inductionHypothesis
              ⟨row.val / 2, divTwo_lt_twoPow row⟩
              coordinates tailDimension

/-- The production numeric-row tensor formula is exactly the independent
paper Boolean equality weight at the decoded typed vertex. -/
theorem productionTensorWeight_eq_equalityWeight
    {Field : Type uField}
    (ops : InterpolationOps Field)
    {rowVariables : Nat}
    (row : Fin (2 ^ rowVariables))
    (point : CubePoint Field rowVariables) :
    productionTensorWeight ops row point =
      BooleanVertex.equalityWeight ops
        (rowVertex rowVariables row) point := by
  exact productionTensorWeightCoordinates_eq_equalityWeightCoordinates
    ops row point.coordinates point.dimension

end Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
