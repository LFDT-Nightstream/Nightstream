import NightstreamFPrime.Export.Stage1.PiDECPoseidonNumericBlock
import NightstreamFPrime.Layout.MatrixProgram.Program

/-!
Evaluate one existing matrix-program row under an arbitrary scalar read.
Poseidon uses the proved numeric invocation evaluator. Every other block
uses its existing sparse row. Block order and all optional failures are
preserved. This interface does not select phases, witnesses, or source rows.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECMatrixNumericRows

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.ProductionRelation
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.MatrixProgram
open NightstreamFPrime.Layout.ProductionRelation

private theorem ofFn_get {count : Nat} (values : Fin count → F)
    (port : Fin count) :
    (Vector.ofFn values).get port = values port := by
  change (Vector.ofFn values)[port.val] = values port
  rw [Vector.getElem_ofFn]

/-- Materialize the exact fourteen sparse port evaluations. Matrix slot 13
uses the existing empty-form convention. -/
def sparseValues {columns : Nat} (read : Fin columns → F)
    (forms : RowForms columns) : Vector F matrixCount :=
  Vector.ofFn fun port =>
    (match meaningfulPort? port with
      | some meaningful => forms meaningful
      | none => SparseForm.empty).evalSparse read

private theorem sparseValues_get {columns : Nat} (read : Fin columns → F)
    (forms : RowForms columns) (port : Fin matrixCount) :
    (sparseValues read forms).get port =
      (match meaningfulPort? port with
        | some meaningful => forms meaningful
        | none => SparseForm.empty).evalSparse read := by
  exact ofFn_get _ port

private theorem option_ports_ext
    {left right : Option (Vector F matrixCount)}
    (equal : ∀ port : Fin matrixCount,
      left.map (fun values => values.get port) =
        right.map (fun values => values.get port)) :
    left = right := by
  cases left with
  | none =>
      cases right with
      | none => rfl
      | some right => cases equal ⟨0, by decide⟩
  | some left =>
      cases right with
      | none => cases equal ⟨0, by decide⟩
      | some right =>
          apply congrArg some
          apply Vector.ext
          intro lane bounded
          exact Option.some.inj (equal ⟨lane, bounded⟩)

/-- Numeric interpretation of the existing block constructors. A complete
runner can share loaded invocation and packet values across its row calls. -/
def blockRow? (block : MatrixProgram.Block) {columns : Nat}
    (sourceRow : Nat → Option R1CS.Row) (read : Fin columns → F)
    (ordinal : Nat) : Option (Vector F matrixCount) :=
  match block with
  | .poseidon poseidon =>
      (PiDECPoseidonNumericBlock.row? poseidon read ordinal).map fun values =>
        Vector.ofFn values.get
  | other => (other.row? columns sourceRow ordinal).map (sparseValues read)

/-- Complete optional-output equality at the block boundary. No successful
lookup, selector value, or assignment-validity premise is required. -/
theorem blockRow?_eq (block : MatrixProgram.Block) {columns : Nat}
    (sourceRow : Nat → Option R1CS.Row) (read : Fin columns → F)
    (ordinal : Nat) :
    blockRow? block sourceRow read ordinal =
      (block.row? columns sourceRow ordinal).map (sparseValues read) := by
  cases block with
  | ordinary block => rfl
  | multiplicationGrid block => rfl
  | phi81Product block => rfl
  | pin block => rfl
  | poseidon block =>
      apply option_ports_ext
      intro port
      simpa only [blockRow?, MatrixProgram.Block.row?, Option.map_map,
        Function.comp_def, ofFn_get, sparseValues_get] using
        PiDECPoseidonNumericBlock.row?_value block read ordinal port

/-- Use the same ordered block-count selection as Program.row?. No block
is expanded during selection. -/
def row? (program : MatrixProgram.Program) {columns : Nat}
    (sourceRow : Nat → Option R1CS.Row) (read : Fin columns → F) :
    Nat → Option (Vector F matrixCount)
  | ordinal => select program.blocks ordinal
where
  select : List MatrixProgram.Block → Nat → Option (Vector F matrixCount)
    | [], _ => none
    | block :: rest, ordinal =>
        if ordinal < block.rowCount then
          blockRow? block sourceRow read ordinal
        else
          select rest (ordinal - block.rowCount)

private theorem select_eq {columns : Nat}
    (sourceRow : Nat → Option R1CS.Row) (read : Fin columns → F)
    (blocks : List MatrixProgram.Block) (ordinal : Nat) :
    row?.select sourceRow read blocks ordinal =
      (MatrixProgram.Program.row?.select columns sourceRow blocks ordinal).map
        (sparseValues read) := by
  induction blocks generalizing ordinal with
  | nil => rfl
  | cons block rest inductionHypothesis =>
      by_cases selected : ordinal < block.rowCount
      · simpa only [row?.select, MatrixProgram.Program.row?.select,
          if_pos selected] using blockRow?_eq block sourceRow read ordinal
      · simpa only [row?.select, MatrixProgram.Program.row?.select,
          if_neg selected] using inductionHypothesis (ordinal - block.rowCount)

/-- The numeric program returns exactly the sparse reference evaluations,
including rejection of missing rows, invalid geometry, and missing sources. -/
theorem row?_eq (program : MatrixProgram.Program) {columns : Nat}
    (sourceRow : Nat → Option R1CS.Row) (read : Fin columns → F)
    (ordinal : Nat) :
    row? program sourceRow read ordinal =
      (program.row? columns sourceRow ordinal).map (sparseValues read) := by
  exact select_eq sourceRow read program.blocks ordinal

/-- Every numeric port is the evaluation of the original program's sparse
port. The read and source accessor are arbitrary and shared by both sides. -/
theorem row?_value (program : MatrixProgram.Program) {columns : Nat}
    (sourceRow : Nat → Option R1CS.Row) (read : Fin columns → F)
    (ordinal : Nat) (port : Fin matrixCount) :
    (row? program sourceRow read ordinal).map (fun values => values.get port) =
      (program.row? columns sourceRow ordinal).map (fun forms =>
        (match meaningfulPort? port with
          | some meaningful => forms meaningful
          | none => SparseForm.empty).evalSparse read) := by
  rw [row?_eq]
  cases program.row? columns sourceRow ordinal with
  | none => rfl
  | some forms => exact congrArg some (sparseValues_get read forms port)

/-- Numeric evaluation rejects exactly the rows rejected by the source program. -/
theorem row?_eq_none_iff (program : MatrixProgram.Program) {columns : Nat}
    (sourceRow : Nat → Option R1CS.Row) (read : Fin columns → F)
    (ordinal : Nat) :
    row? program sourceRow read ordinal = none ↔
      program.row? columns sourceRow ordinal = none := by
  rw [row?_eq, Option.map_eq_none_iff]

end NightstreamFPrime.Export.Stage1.PiDECMatrixNumericRows
