import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Poseidon23ApplicationProfile
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls.Equality

/-!
Contract: one owner-local physical occurrence of the total fixed-23 binding
hash.

The occurrence owns the alignment equality, optional-result tag, canonical
fixed-23 Poseidon2 core, rejecting zero payload, and iteration normalization.
Its operands and visible output are supplied by the typed call adapter.

Does not own: a typed `CallFrame`, application serialization facts, Rust,
generated rows, collision resistance, or either whole verifier.
-/

set_option autoImplicit false
set_option maxRecDepth 32768

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls

namespace Poseidon23HashOccurrence

abbrev auxiliary (width : Nat) : Layout := auxiliaryLayout width

/-- Flattened physical occurrence.  Every bundle after `output` is a
nonoptional temporary of this call. -/
structure Frame (sourceWidth alignmentWidth : Nat) where
  owner : PhysicalOwner
  one : ColumnId
  active : ColumnId
  next : Bool
  iteration : OwnedColumn
  sourceTail : List OwnedColumn
  sourceTailLength : sourceTail.length + 1 = sourceWidth
  output : List OwnedColumn
  outputLength : output.length = 5
  normalized : ColumnBundle (auxiliary 1)
  preimage : ColumnBundle (auxiliary 23)
  inverses : ColumnBundle (auxiliary alignmentWidth)
  equals : ColumnBundle (auxiliary alignmentWidth)
  products : ColumnBundle (auxiliary alignmentWidth.pred)
  equalityOutput : ColumnBundle (auxiliary 1)
  selected : ColumnBundle (auxiliary 1)
  coreOutput : ColumnBundle (auxiliary 4)
  coreTemporaries : ColumnBundle (auxiliary 2464)
  plan : Poseidon23Hash.CoordinatePlan sourceWidth alignmentWidth

namespace Frame

def normalizedColumn
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth) : OwnedColumn :=
  frame.normalized.columns.get
    ⟨0, by
      rw [frame.normalized.length_eq]
      simp [auxiliary, auxiliaryLayout, ownedLayout]⟩

def equalityOutputColumn
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth) : OwnedColumn :=
  frame.equalityOutput.columns.get
    ⟨0, by
      rw [frame.equalityOutput.length_eq]
      simp [auxiliary, auxiliaryLayout, ownedLayout]⟩

def selectedColumn
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth) : OwnedColumn :=
  frame.selected.columns.get
    ⟨0, by
      rw [frame.selected.length_eq]
      simp [auxiliary, auxiliaryLayout, ownedLayout]⟩

def source
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth) : List OwnedColumn :=
  frame.normalizedColumn :: frame.sourceTail

@[simp] theorem source_length
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth) :
    frame.source.length = sourceWidth := by
  simp [source, frame.sourceTailLength]

def sourceAt
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (index : Fin sourceWidth) : OwnedColumn :=
  frame.source.get
    ⟨index.val, by
      rw [frame.source_length]
      exact index.isLt⟩

def projected
    {sourceWidth alignmentWidth targetWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (projection : Fin targetWidth -> Fin sourceWidth) :
    List OwnedColumn :=
  List.ofFn fun index => frame.sourceAt (projection index)

@[simp] theorem projected_length
    {sourceWidth alignmentWidth targetWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (projection : Fin targetWidth -> Fin sourceWidth) :
    (frame.projected projection).length = targetWidth := by
  simp [projected]

def outputAt
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (index : Fin 5) : OwnedColumn :=
  frame.output.get
    ⟨index.val, by simpa [frame.outputLength] using index.isLt⟩

def coreOutputAt
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (lane : Fin 4) : OwnedColumn :=
  frame.coreOutput.columns.get
    ⟨lane.val, by
      rw [frame.coreOutput.length_eq]
      simpa [auxiliary, auxiliaryLayout, ownedLayout] using lane.isLt⟩

/-- Visible coordinates supplied by the typed call adapter. -/
def visibleIds
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth) : List ColumnId :=
  [frame.one, frame.active, frame.iteration.id] ++
    frame.sourceTail.map (fun column => column.id) ++
    frame.output.map (fun column => column.id)

/-- Every non-core temporary in exact footprint order. -/
def prefixTemporaryIds
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth) : List ColumnId :=
  frame.normalized.ids ++
    frame.preimage.ids ++
    frame.inverses.ids ++
    frame.equals.ids ++
    frame.products.ids ++
    frame.equalityOutput.ids ++
    frame.selected.ids ++
    frame.coreOutput.ids

/-- Every temporary in exact footprint order. -/
def temporaryIds
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth) : List ColumnId :=
  frame.prefixTemporaryIds ++ frame.coreTemporaries.ids

@[simp] theorem temporaryIds_length
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth) :
    frame.temporaryIds.length =
      2494 + 2 * alignmentWidth + alignmentWidth.pred := by
  simp [temporaryIds, prefixTemporaryIds, ColumnBundle.ids,
    frame.normalized.length_eq, frame.preimage.length_eq,
    frame.inverses.length_eq, frame.equals.length_eq,
    frame.products.length_eq, frame.equalityOutput.length_eq,
    frame.selected.length_eq, frame.coreOutput.length_eq,
    frame.coreTemporaries.length_eq, auxiliary, auxiliaryLayout,
    ownedLayout]
  omega

def equality
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth) : EqualityRecipe where
  owner := frame.owner
  one := frame.one
  active := frame.active
  left := frame.projected frame.plan.alignmentLeft
  right := frame.projected frame.plan.alignmentRight
  output := frame.equalityOutputColumn
  inverses := frame.inverses.columns
  equals := frame.equals.columns
  products := frame.products.columns
  rightLength := by simp
  inverseLength := by
    rw [frame.inverses.length_eq]
    simp [auxiliary, auxiliaryLayout, ownedLayout]
  equalLength := by
    rw [frame.equals.length_eq]
    simp [auxiliary, auxiliaryLayout, ownedLayout]
  productLength := by
    rw [frame.products.length_eq]
    simp [auxiliary, auxiliaryLayout, ownedLayout]

end Frame

/-! The core constructor needs the four exact allocation facts.  Keeping
them outside `Frame` prevents desired semantic conclusions from becoming
fields while still making the physical ownership boundary explicit. -/
structure CoreAllocationFacts
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth) : Prop where
  allocationsNodup :
    (frame.coreOutput.ids ++ frame.coreTemporaries.ids).Nodup
  temporariesDisjointVisible :
    IdsDisjoint frame.coreTemporaries.ids
      ([frame.one, frame.selectedColumn.id] ++
        frame.preimage.ids ++ frame.coreOutput.ids)
  outputsDisjointPreexisting :
    IdsDisjoint frame.coreOutput.ids
      ([frame.one, frame.selectedColumn.id] ++ frame.preimage.ids)
  allocationsOwned :
    ∀ column,
      column ∈ frame.coreOutput.columns ++
          frame.coreTemporaries.columns ->
        column.id.owner = frame.owner

def core
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (facts : CoreAllocationFacts frame) :
    CanonicalPoseidon2Sponge23Recipe.Frame where
  owner := frame.owner
  firstOrdinal := 0
  one := frame.one
  active := frame.selectedColumn.id
  input := frame.preimage
  output := frame.coreOutput
  temporaries := frame.coreTemporaries
  allocationsNodup := facts.allocationsNodup
  temporariesDisjointVisible := facts.temporariesDisjointVisible
  outputsDisjointPreexisting := facts.outputsDisjointPreexisting
  allocationsOwned := facts.allocationsOwned

def normalizationRow
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth) : Row where
  a := singleton frame.one 1
  b :=
    if frame.next then
      [ { column := frame.iteration.id, coefficient := 1 }
      , { column := frame.one, coefficient := 1 }
      , { column := frame.normalizedColumn.id, coefficient := -1 }
      ]
    else
      difference frame.iteration.id frame.normalizedColumn.id
  c := []

def preimageRow
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (index : Fin 23) : Row where
  a := singleton frame.one 1
  b := difference
    (frame.sourceAt (frame.plan.preimage index)).id
    (frame.preimage.columns.get
      ⟨index.val, by
        rw [frame.preimage.length_eq]
        simpa [auxiliary, auxiliaryLayout, ownedLayout] using index.isLt⟩).id
  c := []

def preimageRows
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth) : List Row :=
  List.ofFn (preimageRow frame)

def selectedRow
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth) : Row where
  a := singleton frame.active 1
  b := singleton frame.equalityOutputColumn.id 1
  c := singleton frame.selectedColumn.id 1

def tagRow
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth) : Row where
  a := singleton frame.active 1
  b := difference frame.selectedColumn.id (frame.outputAt 0).id
  c := []

def payloadSuccessRow
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (lane : Fin 4) : Row where
  a := singleton frame.selectedColumn.id 1
  b := difference (frame.coreOutputAt lane).id
    (frame.outputAt ⟨lane.val + 1, by omega⟩).id
  c := []

def payloadAbsentRow
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (lane : Fin 4) : Row where
  a := difference frame.active frame.selectedColumn.id
  b := singleton (frame.outputAt ⟨lane.val + 1, by omega⟩).id 1
  c := []

def payloadRows
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth) : List Row :=
  (List.ofFn (payloadSuccessRow frame)) ++
    List.ofFn (payloadAbsentRow frame)

def wrapperRawRows
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth) : List Row :=
  [normalizationRow frame] ++
    (preimageRows frame ++
      (frame.equality.rawRows ++
        ([selectedRow frame, tagRow frame] ++
          payloadRows frame)))

def rawRows
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (facts : CoreAllocationFacts frame) : List Row :=
  wrapperRawRows frame ++
    (CanonicalPoseidon2Sponge23Recipe.rows
      (core frame facts)).map (fun owned => owned.row)

theorem rawRows_eq_wrapper_append_core
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (facts : CoreAllocationFacts frame) :
    rawRows frame facts =
      wrapperRawRows frame ++
        (CanonicalPoseidon2Sponge23Recipe.rows
          (core frame facts)).map (fun owned => owned.row) :=
  rfl

def rows
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (facts : CoreAllocationFacts frame) : List OwnedRow :=
  ownRows frame.owner (rawRows frame facts)

@[simp] theorem preimageRows_length
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth) :
    (preimageRows frame).length = 23 := by
  simp [preimageRows]

@[simp] theorem payloadRows_length
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth) :
    (payloadRows frame).length = 8 := by
  simp [payloadRows]

theorem rawRows_length
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (facts : CoreAllocationFacts frame) :
    (rawRows frame facts).length =
      (2 * alignmentWidth + alignmentWidth.pred + 1) + 2502 := by
  have leftLength :
      frame.equality.left.length = alignmentWidth := by
    simp [Frame.equality]
  unfold rawRows wrapperRawRows
  simp only [List.length_append, List.length_singleton,
    List.length_cons, List.length_nil, preimageRows_length,
    payloadRows_length, List.length_map]
  rw [frame.equality.raw_row_count,
    CanonicalPoseidon2Sponge23Recipe.rows_length, leftLength]
  simp only [CanonicalPoseidon2Sponge23Recipe.recurringRows,
    CanonicalPoseidon2Sponge23Recipe.coreRowCount,
    CanonicalPoseidon2Sponge23Recipe.gateRowCount]
  omega

theorem rows_length
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (facts : CoreAllocationFacts frame) :
    (rows frame facts).length =
      (2 * alignmentWidth + alignmentWidth.pred + 1) + 2502 := by
  rw [rows, ownRows_length, rawRows_length]

theorem rows_owned
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (facts : CoreAllocationFacts frame)
    (row : OwnedRow)
    (member : row ∈ rows frame facts) :
    row.id.owner = frame.owner :=
  ownRows_owner frame.owner (rawRows frame facts) row member

theorem row_ids_nodup
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (facts : CoreAllocationFacts frame) :
    ((rows frame facts).map fun row => row.id).Nodup :=
  ownRows_ids_nodup frame.owner (rawRows frame facts)

end Poseidon23HashOccurrence

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
