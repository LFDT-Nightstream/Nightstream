import NightstreamFPrime.Export.Stage1.PiCCSSparseEvaluation
import NightstreamFPrime.Export.Stage1.PiDECMatrixNumericRows

/-! Evaluate existing matrix rows on aggregated extension-field reads. The
numeric interpreter runs once per scalar component. A stored Poseidon
invocation shares both evaluations across its existing 94 rows. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSLinearRows

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.ProductionRelation
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.MatrixProgram
open NightstreamFPrime.Layout.ProductionRelation

private theorem get_ofFn {Alpha : Type} {count : Nat}
    (value : Fin count → Alpha) (index : Fin count) :
    (Vector.ofFn value).get index = value index := by
  change (Vector.ofFn value)[index.val] = value index
  rw [Vector.getElem_ofFn]

private def pack (first second : Vector F matrixCount) : Vector K matrixCount :=
  Vector.ofFn fun port => ⟨first.get port, second.get port⟩

/-- Both scalar calls retain the original program's guards and failures. -/
def row? (program : MatrixProgram.Program) {columns : Nat}
    (sourceRow : Nat → Option R1CS.Row) (read : Fin columns → K)
    (ordinal : Nat) : Option (Vector K matrixCount) := do
  let first ← PiDECMatrixNumericRows.row? program sourceRow
    (fun column => (read column).c0) ordinal
  let second ← PiDECMatrixNumericRows.row? program sourceRow
    (fun column => (read column).c1) ordinal
  return pack first second

/-- Every optional matrix port equals the existing sparse extension
evaluation. No successful row lookup or valid assignment is assumed. -/
theorem row?_value (program : MatrixProgram.Program) {columns : Nat}
    (sourceRow : Nat → Option R1CS.Row) (read : Fin columns → K)
    (ordinal : Nat) (port : Fin matrixCount) :
    (row? program sourceRow read ordinal).map (fun values => values.get port) =
      (program.row? columns sourceRow ordinal).map (fun forms =>
        PiCCSSparseEvaluation.evaluateK
          (match meaningfulPort? port with
            | some meaningful => forms meaningful
            | none => SparseForm.empty) read) := by
  rw [row?, PiDECMatrixNumericRows.row?_eq, PiDECMatrixNumericRows.row?_eq]
  cases program.row? columns sourceRow ordinal with
  | none => rfl
  | some forms =>
      change some ((pack
        (PiDECMatrixNumericRows.sparseValues (fun column => (read column).c0) forms)
        (PiDECMatrixNumericRows.sparseValues (fun column => (read column).c1) forms)).get port) = _
      simp only [pack, get_ofFn, PiDECMatrixNumericRows.sparseValues]
      rfl

/-- Compute each scalar component once for all rows of one checked
invocation. The returned arrays preserve the canonical port and row order. -/
def invocation {columns : Nat} (read : Fin columns → K)
    (interface : PoseidonSboxPlan.Interface columns) :
    Vector (Vector K matrixCount) 94 :=
  let first := PiDECPoseidonNumericRows.stored (fun column => (read column).c0) interface
  let second := PiDECPoseidonNumericRows.stored (fun column => (read column).c1) interface
  Vector.ofFn fun row => Vector.ofFn fun port =>
    ⟨((first.get row).get port), ((second.get row).get port)⟩

/-- Every stored invocation port is the same sparse row evaluated on the
aggregated read. Constant columns and all eight output pins are included. -/
theorem invocation_value {columns : Nat} (read : Fin columns → K)
    (interface : PoseidonSboxPlan.Interface columns) (row : Fin 94)
    (port : Fin matrixCount) :
    ((invocation read interface).get row).get port =
      PiCCSSparseEvaluation.evaluateK
        (((PoseidonSboxPlan.rows interface).get
          ⟨row.val, by rw [PoseidonSboxPlan.rows_length]; exact row.isLt⟩).portForm port)
        read := by
  simp only [invocation, get_ofFn, PiDECPoseidonNumericRows.stored_value]
  rfl

/-- Reusing the stored invocation preserves both extension components of
all 94 rows and 14 ports, at the original Fin product encoding. -/
theorem invocation_loaded_value (block : Poseidon.Block) {columns : Nat}
    (invocationIndex : Fin block.invocationCount)
    (interface : PoseidonSboxPlan.Interface columns)
    (loaded : PiDECPoseidonNumericBlock.loadInvocation? block columns invocationIndex = some interface)
    (read : Fin columns → K) (row : Fin 94) (port : Fin matrixCount) :
    some (((invocation read interface).get row).get port) =
      (block.row? columns (Fin.encodeProd (invocationIndex, row)).val).map (fun forms =>
        PiCCSSparseEvaluation.evaluateK
          (match meaningfulPort? port with
            | some meaningful => forms meaningful
            | none => SparseForm.empty) read) := by
  have scalar (values : Fin columns → F) :
      some ((((PoseidonSboxPlan.rows interface).get
        ⟨row.val, by rw [PoseidonSboxPlan.rows_length]; exact row.isLt⟩).portForm port).evalSparse values) =
        (block.row? columns (Fin.encodeProd (invocationIndex, row)).val).map (fun forms =>
          (match meaningfulPort? port with
            | some meaningful => forms meaningful
            | none => SparseForm.empty).evalSparse values) := by
    have value := PiDECPoseidonNumericBlock.row?_value block values
      (Fin.encodeProd (invocationIndex, row)).val port
    simpa only [PiDECPoseidonNumericBlock.row?, PiDECPoseidonNumericBlock.loadRow?_encodeProd,
      loaded, Option.map_some, PiDECPoseidonNumericRows.stored_value] using! value
  rw [invocation_value]
  have first := scalar (fun column => (read column).c0)
  have second := scalar (fun column => (read column).c1)
  cases result : block.row? columns (Fin.encodeProd (invocationIndex, row)).val with
  | none =>
      rw [result] at first
      simp only [Option.map_none, reduceCtorEq] at first
  | some forms =>
      simp only [result, Option.map_some] at first second ⊢
      exact congrArg some (congrArg₂ K.mk (Option.some.inj first) (Option.some.inj second))

end NightstreamFPrime.Export.Stage1.PiCCSLinearRows
