import NightstreamFPrime.Export.Stage1.PiDECPoseidonNumericRows
import NightstreamFPrime.Layout.MatrixProgram.Poseidon

/-!
Load the existing Poseidon invocation interface from a package block, then
evaluate its rows numerically. Geometry guards, input decoding and product
indexing remain those of the existing block interpreter. The optional value
theorem includes malformed inputs and all fourteen matrix ports.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECPoseidonNumericBlock

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.ProductionRelation
open NightstreamFPrime.Spec.ProductionRelation.RowSemantics (PortValues)
open NightstreamFPrime.Layout.MatrixProgram
open NightstreamFPrime.Layout.ProductionRelation

private theorem vector_get_ofFn {Alpha : Type} {count : Nat}
    (values : Fin count → Alpha) : (Vector.ofFn values).get = values := by
  funext index
  change (Vector.ofFn values)[index.val] = values index
  rw [Vector.getElem_ofFn]

/-- Store the existing sparse forms once per loaded interface. The cache is
independent of the child and output lane used to evaluate those forms. -/
private def cachedInterface {columns : Nat}
    (interface : PoseidonSboxPlan.Interface columns) :
    PoseidonSboxPlan.Interface columns :=
  let input := Vector.ofFn interface.input
  let sboxes := Vector.ofFn interface.sboxOutput
  let output := Vector.ofFn interface.output
  { oneColumn := interface.oneColumn
    input := input.get
    sboxOutput := sboxes.get
    output := output.get }

private theorem cachedInterface_eq {columns : Nat}
    (interface : PoseidonSboxPlan.Interface columns) :
    cachedInterface interface = interface := by
  cases interface
  simp only [cachedInterface, vector_get_ofFn]

/-- Reuse one checked interface for all rows of an invocation. The runner
can call NumericRows.stored once after this succeeds. -/
def loadInvocation? (block : Poseidon.Block) (columns : Nat)
    (invocation : Fin block.invocationCount) :
    Option (PoseidonSboxPlan.Interface columns) :=
  if oneBound : block.oneColumn < columns then
    if block.retained.kind = .field then
      if slotCountEq : block.retained.slotCount = block.invocationCount * 86 then
        if retainedFits : block.retained.start + block.retained.coordinateCount ≤ columns then
          (block.input.state? columns block.oneColumn invocation.val).map fun input =>
            cachedInterface (block.invocationInterface columns oneBound slotCountEq
              retainedFits input invocation)
        else none
      else none
    else none
  else none

/-- Decode the package ordinal with the existing Fin product convention.
The result contains only the existing interface and existing local row index. -/
def loadRow? (block : Poseidon.Block) (columns ordinal : Nat) :
    Option (PoseidonSboxPlan.Interface columns × Fin 86) :=
  if rowBound : ordinal < block.rowCount then
    let decoded : Fin block.invocationCount × Fin 86 :=
      Fin.decodeProd ⟨ordinal, rowBound⟩
    (loadInvocation? block columns decoded.1).map fun interface => (interface, decoded.2)
  else none

/-- Every encoded local row uses the same checked invocation interface.
No division formula or independent ordering convention is introduced. -/
theorem loadRow?_encodeProd (block : Poseidon.Block) (columns : Nat)
    (invocation : Fin block.invocationCount) (row : Fin 86) :
    loadRow? block columns (Fin.encodeProd (invocation, row)).val =
      (loadInvocation? block columns invocation).map fun interface => (interface, row) := by
  have bounded : (Fin.encodeProd (invocation, row)).val < block.rowCount :=
    (Fin.encodeProd (invocation, row)).isLt
  have encoded :
      (⟨(Fin.encodeProd (invocation, row)).val, bounded⟩ :
        Fin (block.invocationCount * 86)) = Fin.encodeProd (invocation, row) :=
    Fin.ext rfl
  simp only [loadRow?, dif_pos bounded, encoded, Fin.decodeProd_encodeProd]

private def referenceForms {columns : Nat}
    (loaded : PoseidonSboxPlan.Interface columns × Fin 86) : RowForms columns :=
  ((PoseidonRetainedRows.rows loaded.1).get
    ⟨loaded.2.val, by rw [PoseidonRetainedRows.rows_length]; exact loaded.2.isLt⟩).meaningfulForm

/-- Decoding the interface and then viewing its sparse row is exactly the
existing block interpreter, including each failed guard and missing input. -/
private theorem loadRow?_forms (block : Poseidon.Block) (columns ordinal : Nat) :
    (loadRow? block columns ordinal).map referenceForms = block.row? columns ordinal := by
  by_cases rowBound : ordinal < block.rowCount
  · simp only [loadRow?, Poseidon.Block.row?, Poseidon.Block.rowWithInput?, dif_pos rowBound]
    unfold loadInvocation?
    simp only [cachedInterface_eq]
    split_ifs <;> simp_all only [Option.map_none, Option.map_map]
    all_goals
      cases block.input.state? columns block.oneColumn
        (Fin.decodeProd (⟨ordinal, rowBound⟩ : Fin (block.invocationCount * 86))).1.val <;> rfl
  · simp only [loadRow?, Poseidon.Block.row?, Poseidon.Block.rowWithInput?,
      dif_neg rowBound, Option.map_none]

/-- Evaluate the selected row from a stored complete invocation. A runner
processing all rows can use loadInvocation? and retain that stored vector. -/
def row? (block : Poseidon.Block) {columns : Nat} (read : Fin columns → F)
    (ordinal : Nat) : Option PortValues :=
  (loadRow? block columns ordinal).map fun loaded =>
    (PiDECPoseidonNumericRows.stored read loaded.1).get loaded.2

/-- All fourteen numeric ports equal the corresponding sparse evaluations.
The zero port is the existing empty form. Equality of optional values also
retains rejection of malformed geometry or unavailable package input states. -/
theorem row?_value (block : Poseidon.Block) {columns : Nat} (read : Fin columns → F)
    (ordinal : Nat) (port : Fin matrixCount) :
    (row? block read ordinal).map (fun values => values.get port) =
      (block.row? columns ordinal).map (fun forms =>
        (match meaningfulPort? port with
          | some meaningful => forms meaningful
          | none => SparseForm.empty).evalSparse read) := by
  rw [row?, ← loadRow?_forms]
  cases loadRow? block columns ordinal with
  | none => rfl
  | some loaded =>
      exact congrArg some
        (PiDECPoseidonNumericRows.stored_value read loaded.1 loaded.2 port)

/-- The numeric path rejects exactly the same rows as Block.row?. -/
theorem row?_eq_none_iff (block : Poseidon.Block) {columns : Nat}
    (read : Fin columns → F) (ordinal : Nat) :
    row? block read ordinal = none ↔ block.row? columns ordinal = none := by
  rw [row?, Option.map_eq_none_iff, ← loadRow?_forms, Option.map_eq_none_iff]

end NightstreamFPrime.Export.Stage1.PiDECPoseidonNumericBlock
