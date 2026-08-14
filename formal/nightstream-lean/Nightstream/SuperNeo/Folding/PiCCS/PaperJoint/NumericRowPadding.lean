import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.NumericRowMap
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

/-!
Zero-padding of finite numeric matrix rows into the Boolean CCS row domain.

Protocol: SuperNeo CCS/CE relation specialized to the F' source matrices.
Phase: finite numeric matrix storage to the independent Boolean row domain.
Constraint family: semantic row ownership only; this file emits no rows.

Owns: the model-level adapter that preserves every declared numeric row and
sets every unused Boolean-domain row to zero.

Does not own: selection of `rowVariables` from a Rust row count, the native
`next_power_of_two().max(2)` policy, concrete matrix contents, sparse-matrix
serialization, artifact conformance, CCS satisfaction, or constraint counts.

Emits constraints: no.

Authority boundary: `rowCount` and `rowVariables` are separate inputs. A
coverage proof permits embedding numeric rows, but this model-level adapter
does not infer or certify the production domain policy.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.ccs.rows.padding.decode` | numeric rows use the shared little-endian decoder | computed | `numericRowVertex` |
| `fprime.ccs.rows.padding.preserve` | every declared numeric row is preserved exactly | derived | `padRows_at_numericRow` |
| `fprime.ccs.rows.padding.zero` | every unused Boolean-domain row is fixed zero | checked | `padRows_atPadding` |
| `fprime.ccs.rows.padding.one_row` | the explicit `1 -> 2` specialization preserves row zero and zeros row one | derived | `padRows_oneRow_actual`, `padRows_oneRow_padding` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.RowPadding

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

universe uField

/-- A finite numeric-row matrix before embedding into a Boolean row cube. -/
abbrev NumericMatrix
    (Field : Type uField) (rowCount columns : Nat) :=
  Fin rowCount -> Fin columns -> Field

/-- Decode one declared numeric row into the shared little-endian Boolean
domain. The coverage proof is explicit because selecting the domain size is a
separate obligation. -/
def numericRowVertex
    {rowCount rowVariables : Nat}
    (covers : rowCount <= 2 ^ rowVariables)
    (row : Fin rowCount) : BooleanVertex rowVariables :=
  rowVertex rowVariables
    ⟨row.val, Nat.lt_of_lt_of_le row.isLt covers⟩

/-- The numeric row recovered from its Boolean-domain embedding is unchanged. -/
@[simp] theorem rowIndex_numericRowVertex
    {rowCount rowVariables : Nat}
    (covers : rowCount <= 2 ^ rowVariables)
    (row : Fin rowCount) :
    rowIndex (numericRowVertex covers row) = row.val := by
  exact rowIndex_rowVertex rowVariables
    ⟨row.val, Nat.lt_of_lt_of_le row.isLt covers⟩

/-- Embed a finite numeric-row matrix into a Boolean row cube and define every
unused row as zero. -/
def padRows
    {Field : Type uField} [Zero Field]
    {rowCount rowVariables columns : Nat}
    (matrix : NumericMatrix Field rowCount columns) :
    BooleanMatrix Field rowVariables columns :=
  fun vertex column =>
    if inRange : rowIndex vertex < rowCount then
      matrix ⟨rowIndex vertex, inRange⟩ column
    else
      0

/-- Row padding preserves every declared numeric matrix row exactly. -/
@[simp] theorem padRows_at_numericRow
    {Field : Type uField} [Zero Field]
    {rowCount rowVariables columns : Nat}
    (matrix : NumericMatrix Field rowCount columns)
    (covers : rowCount <= 2 ^ rowVariables)
    (row : Fin rowCount)
    (column : Fin columns) :
    padRows matrix (numericRowVertex covers row) column = matrix row column := by
  simp [padRows, row.isLt]

/-- Every Boolean-domain row at or above the declared numeric row count is a
derived zero row. -/
theorem padRows_atPadding
    {Field : Type uField} [Zero Field]
    {rowCount rowVariables columns : Nat}
    (matrix : NumericMatrix Field rowCount columns)
    (vertex : BooleanVertex rowVariables)
    (column : Fin columns)
    (isPadding : rowCount <= rowIndex vertex) :
    padRows matrix vertex column = 0 := by
  simp [padRows, Nat.not_lt.mpr isPadding]

/-- In the explicit one-row/two-row-domain specialization, row zero is the
actual matrix row. This theorem does not claim that production selected one
row variable; it records the consequence once that profile is supplied. -/
@[simp] theorem padRows_oneRow_actual
    {Field : Type uField} [Zero Field]
    {columns : Nat}
    (matrix : NumericMatrix Field 1 columns)
    (column : Fin columns) :
    padRows (rowVariables := 1) matrix (rowVertex 1 ⟨0, by decide⟩) column =
      matrix ⟨0, by decide⟩ column := by
  simpa [numericRowVertex] using
    (padRows_at_numericRow matrix (by decide)
      (⟨0, by decide⟩ : Fin 1) column)

/-- In the explicit one-row/two-row-domain specialization, row one is the
unique padding row and is fixed to zero. -/
@[simp] theorem padRows_oneRow_padding
    {Field : Type uField} [Zero Field]
    {columns : Nat}
    (matrix : NumericMatrix Field 1 columns)
    (column : Fin columns) :
    padRows (rowVariables := 1) matrix (rowVertex 1 ⟨1, by decide⟩) column =
      0 := by
  apply padRows_atPadding
  rw [rowIndex_rowVertex]
  decide

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.RowPadding
