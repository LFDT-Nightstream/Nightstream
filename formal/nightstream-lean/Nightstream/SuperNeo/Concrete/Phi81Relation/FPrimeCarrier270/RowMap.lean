import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.NumericBooleanDomain

/-!
F'-specific names for the shared numeric/Boolean row bridge.

Protocol: SuperNeo CCS/CE relation specialized to the F' source matrices.
Phase: production numeric matrix row to the independent Boolean row domain.
Constraint family: semantic row ownership only; this file emits no rows.

Owns: only the F'-specific row names used by the carrier refinement.

Does not own: the little-endian convention or its proofs; those belong to
`PaperJoint.NumericBooleanDomain`. It also does not own matrix contents, row
padding, CCS satisfaction, CE coefficients, Rust, R1CS, or constraint counts.

Emits constraints: no.

Authority boundary: F' rows delegate to the one shared numeric-domain owner.
The specialization cannot choose a different bit order or tensor formula.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.ccs.rows.index` | little-endian typed row index | computed | `rowIndex` delegates to `NumericBooleanDomain.index` |
| `fprime.ccs.rows.inverse` | numeric and typed rows are inverse | derived | `rowIndex_rowVertex`, `rowVertex_rowIndex` |
| `fprime.ccs.rows.weight` | production tensor weight equals Boolean `chi` | derived | `productionTensorWeight_eq_equalityWeight` |
-/

namespace Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

universe uField

/-- Little-endian numeric row index, delegated to the shared domain owner. -/
def rowIndex : {rowVariables : Nat} -> BooleanVertex rowVariables -> Nat :=
  NumericBooleanDomain.index

/-- Every typed Boolean row lies in the exact numeric row domain. -/
theorem rowIndex_lt_twoPow
    {rowVariables : Nat} (vertex : BooleanVertex rowVariables) :
    rowIndex vertex < 2 ^ rowVariables := by
  exact NumericBooleanDomain.index_lt_twoPow vertex

/-- Decode a bounded numeric row with the shared little-endian convention. -/
def rowVertex
    (rowVariables : Nat) (row : Fin (2 ^ rowVariables)) :
    BooleanVertex rowVariables :=
  NumericBooleanDomain.vertex rowVariables row

/-- Encoding a decoded bounded row returns that exact row number. -/
theorem rowIndex_rowVertex
    (rowVariables : Nat) (row : Fin (2 ^ rowVariables)) :
    rowIndex (rowVertex rowVariables row) = row.val := by
  exact NumericBooleanDomain.index_vertex rowVariables row

/-- Decoding the little-endian index of a typed row returns that row. -/
theorem rowVertex_rowIndex
    {rowVariables : Nat} (vertex : BooleanVertex rowVariables) :
    rowVertex rowVariables
        ⟨rowIndex vertex, rowIndex_lt_twoPow vertex⟩ = vertex := by
  exact NumericBooleanDomain.vertex_index vertex

/-- Numeric-row tensor recursion delegated to the shared domain owner. -/
def productionTensorWeightCoordinates
    {Field : Type uField}
    (ops : InterpolationOps Field) : List Field -> Nat -> Field :=
  NumericBooleanDomain.tensorWeightCoordinates ops

/-- Tensor weight of a bounded production row. -/
def productionTensorWeight
    {Field : Type uField}
    (ops : InterpolationOps Field)
    {rowVariables : Nat}
    (row : Fin (2 ^ rowVariables))
    (point : CubePoint Field rowVariables) : Field :=
  NumericBooleanDomain.tensorWeight ops row point

/-- The production numeric-row tensor formula is the independent Boolean
equality weight at the decoded row. -/
theorem productionTensorWeight_eq_equalityWeight
    {Field : Type uField}
    (ops : InterpolationOps Field)
    {rowVariables : Nat}
    (row : Fin (2 ^ rowVariables))
    (point : CubePoint Field rowVariables) :
    productionTensorWeight ops row point =
      BooleanVertex.equalityWeight ops (rowVertex rowVariables row) point := by
  exact NumericBooleanDomain.tensorWeight_eq_equalityWeight ops row point

end Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
