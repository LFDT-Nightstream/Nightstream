import NightstreamFPrime.Export.Stage1.PiDECPoseidonNumericRows
import NightstreamFPrime.Layout.MatrixProgram.Poseidon

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

def loadInvocation? (block : Poseidon.Block) (columns : Nat)
    (invocation : Fin block.invocationCount) :
    Option (PoseidonSboxPlan.Interface columns) :=
  if oneBound : block.oneColumn < columns then
    if block.retained.kind = .field then
      if slotCountEq : block.retained.slotCount = block.invocationCount * 150 then
        if retainedFits : block.retained.start + block.retained.coordinateCount ≤ columns then
          (block.input.state? columns block.oneColumn invocation.val).map fun input =>
            cachedInterface (block.invocationInterface columns oneBound slotCountEq
              retainedFits input invocation)
        else none
      else none
    else none
  else none

def loadRow? (block : Poseidon.Block) (columns ordinal : Nat) :
    Option (PoseidonSboxPlan.Interface columns × Fin 150) :=
  if rowBound : ordinal < block.rowCount then
    let decoded : Fin block.invocationCount × Fin 150 :=
      Fin.decodeProd ⟨ordinal, rowBound⟩
    (loadInvocation? block columns decoded.1).map fun interface => (interface, decoded.2)
  else none

theorem loadRow?_encodeProd (block : Poseidon.Block) (columns : Nat)
    (invocation : Fin block.invocationCount) (row : Fin 150) :
    loadRow? block columns (Fin.encodeProd (invocation, row)).val =
      (loadInvocation? block columns invocation).map fun interface => (interface, row) := by
  have bounded : (Fin.encodeProd (invocation, row)).val < block.rowCount :=
    (Fin.encodeProd (invocation, row)).isLt
  have encoded :
      (⟨(Fin.encodeProd (invocation, row)).val, bounded⟩ :
        Fin (block.invocationCount * 150)) = Fin.encodeProd (invocation, row) :=
    Fin.ext rfl
  simp only [loadRow?, dite_eq_left bounded, encoded, Fin.decodeProd_encodeProd]

private def referenceForms {columns : Nat}
    (loaded : PoseidonSboxPlan.Interface columns × Fin 150) : RowForms columns :=
  ((PoseidonRetainedRows.rows loaded.1).get
    ⟨loaded.2.val, by rw [PoseidonRetainedRows.rows_length]; exact loaded.2.isLt⟩).meaningfulForm

private theorem loadRow?_forms (block : Poseidon.Block) (columns ordinal : Nat) :
    (loadRow? block columns ordinal).map referenceForms = block.row? columns ordinal := by
  by_cases rowBound : ordinal < block.rowCount
  · simp only [loadRow?, Poseidon.Block.row?, Poseidon.Block.rowWithInput?, dite_eq_left rowBound]
    unfold loadInvocation?
    simp only [cachedInterface_eq]
    split_ifs <;> simp_all only [Option.map_none, Option.map_map]
    all_goals
      cases block.input.state? columns block.oneColumn
        (Fin.decodeProd (⟨ordinal, rowBound⟩ : Fin (block.invocationCount * 150))).1.val <;> rfl
  · simp only [loadRow?, Poseidon.Block.row?, Poseidon.Block.rowWithInput?,
      dite_eq_right rowBound, Option.map_none]

def row? (block : Poseidon.Block) {columns : Nat} (read : Fin columns → F)
    (ordinal : Nat) : Option PortValues :=
  (loadRow? block columns ordinal).map fun loaded =>
    (PiDECPoseidonNumericRows.stored read loaded.1).get loaded.2

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

theorem row?_eq_none_iff (block : Poseidon.Block) {columns : Nat}
    (read : Fin columns → F) (ordinal : Nat) :
    row? block read ordinal = none ↔ block.row? columns ordinal = none := by
  rw [row?, Option.map_eq_none_iff, ← loadRow?_forms, Option.map_eq_none_iff]

end NightstreamFPrime.Export.Stage1.PiDECPoseidonNumericBlock
