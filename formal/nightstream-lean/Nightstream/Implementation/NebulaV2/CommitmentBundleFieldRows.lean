import Nightstream.Implementation.NebulaV2.CanonicalFieldSchemaRows
import Nightstream.Implementation.NebulaV2.CommitmentBundleCodec

/-!
Contract: canonical bit-to-field R1CS bridge for the complete V2 commitment
bundle.

Assurance tier: implementation-to-protocol bridge.

Owns one canonical-u64 block for each of the 3,888 bundle coordinates, exact
component/coordinate order, rejection of noncanonical Goldilocks aliases,
and the theorem that every field wire equals the matching typed bundle
coordinate.

Does not own full-claim bit placement, Ajtai commitment evaluation, compact
token rows, absolute generated columns, or Rust conformance.

Emits constraints: yes.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.CommitmentBundleFieldRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.NebulaV2.CommitmentBundleCodec
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.CommitmentBundle

abbrev Slot := FieldTag

def Slot.all : List Slot := schema

theorem Slot.all_length_exact : Slot.all.length = 3888 := by
  rw [Slot.all, schema_length,
    MemoryWireGeometry.commitmentFieldCount_exact]
  rfl

theorem Slot.mem_all (slot : Slot) : slot ∈ Slot.all :=
  slot.mem_schema

private def componentFields (component : Component) : List Slot :=
  List.ofFn fun coordinate : Coordinate => { component, coordinate }

private theorem schema_eq_component_fields :
    Slot.all =
      componentFields .full ++
        (componentFields .operations ++
          (componentFields .initialSnapshot ++
            componentFields .finalSnapshot)) := by
  simp [Slot.all, schema, componentOrder, componentFields]

private theorem componentFields_length (component : Component) :
    (componentFields component).length = 972 := by
  unfold componentFields
  rw [List.length_ofFn]
  exact MemoryWireGeometry.commitmentFieldCount_exact

private theorem componentFields_nodup (component : Component) :
    (componentFields component).Nodup := by
  unfold componentFields
  rw [List.nodup_ofFn]
  intro left right equal
  injection equal

private theorem componentFields_mem
    (component : Component) (coordinate : Coordinate) :
    ({ component, coordinate } : Slot) ∈ componentFields component := by
  exact List.mem_ofFn.mpr ⟨coordinate, rfl⟩

private theorem componentFields_not_mem
    {left right : Component} (different : left ≠ right)
    (coordinate : Coordinate) :
    ({ component := left, coordinate := coordinate } : Slot) ∉
      componentFields right := by
  intro member
  rcases List.mem_ofFn.mp member with ⟨other, equal⟩
  exact different (congrArg FieldTag.component equal).symm

private theorem componentFields_idxOf
    (component : Component) (coordinate : Coordinate) :
    (componentFields component).idxOf
        { component := component, coordinate := coordinate } =
      coordinate.val := by
  have coordinateBound : coordinate.val < 972 := by
    rw [← MemoryWireGeometry.commitmentFieldCount_exact]
    exact coordinate.isLt
  have atCoordinate := (componentFields_nodup component).idxOf_getElem
    coordinate.val (by simpa [componentFields_length] using coordinateBound)
  simpa [componentFields] using atCoordinate

def componentIndex : Component → Nat
  | .full => 0
  | .operations => 1
  | .initialSnapshot => 2
  | .finalSnapshot => 3

def Slot.position (slot : Slot) : Nat :=
  componentIndex slot.component * MemoryWireGeometry.commitmentFieldCount +
    slot.coordinate.val

def Slot.bitOffset (slot : Slot) : Nat :=
  CanonicalFieldBits.bitCount * slot.position

theorem Slot.position_lt (slot : Slot) : slot.position < 3888 := by
  rcases slot with ⟨component, coordinate⟩
  have coordinateBound : coordinate.val < 972 := by
    rw [← MemoryWireGeometry.commitmentFieldCount_exact]
    exact coordinate.isLt
  cases component <;>
    norm_num [Slot.position, componentIndex,
      MemoryWireGeometry.commitmentFieldCount_exact] <;>
    omega

theorem Slot.position_eq_idxOf (slot : Slot) :
    slot.position = Slot.all.idxOf slot := by
  rcases slot with ⟨component, coordinate⟩
  rw [schema_eq_component_fields]
  cases component with
  | full =>
      rw [List.idxOf_append_of_mem
        (componentFields_mem .full coordinate)]
      simp [Slot.position, componentIndex,
        componentFields_idxOf]
  | operations =>
      rw [List.idxOf_append_of_notMem
        (componentFields_not_mem
          (left := .operations) (right := .full) (by decide) coordinate)]
      rw [List.idxOf_append_of_mem
        (componentFields_mem .operations coordinate)]
      simp [Slot.position, componentIndex, componentFields_length,
        componentFields_idxOf,
        MemoryWireGeometry.commitmentFieldCount_exact] <;> omega
  | initialSnapshot =>
      rw [List.idxOf_append_of_notMem
        (componentFields_not_mem
          (left := .initialSnapshot) (right := .full) (by decide) coordinate)]
      rw [List.idxOf_append_of_notMem
        (componentFields_not_mem
          (left := .initialSnapshot) (right := .operations) (by decide)
          coordinate)]
      rw [List.idxOf_append_of_mem
        (componentFields_mem .initialSnapshot coordinate)]
      simp [Slot.position, componentIndex, componentFields_length,
        componentFields_idxOf,
        MemoryWireGeometry.commitmentFieldCount_exact] <;> omega
  | finalSnapshot =>
      rw [List.idxOf_append_of_notMem
        (componentFields_not_mem
          (left := .finalSnapshot) (right := .full) (by decide) coordinate)]
      rw [List.idxOf_append_of_notMem
        (componentFields_not_mem
          (left := .finalSnapshot) (right := .operations) (by decide)
          coordinate)]
      rw [List.idxOf_append_of_notMem
        (componentFields_not_mem
          (left := .finalSnapshot) (right := .initialSnapshot) (by decide)
          coordinate)]
      simp [Slot.position, componentIndex, componentFields_length,
        componentFields_idxOf,
        MemoryWireGeometry.commitmentFieldCount_exact] <;> omega

structure Layout where
  publicBitStart : Nat
  columnMap : Slot → List Nat
  mapsConstantOne : ∀ slot, Relabel.column (columnMap slot) 0 = 0

def Layout.rawColumns (layout : Layout) (slot : Slot) : List Nat :=
  (List.range CanonicalFieldBits.bitCount).map fun bit =>
    layout.publicBitStart + slot.bitOffset + bit

def Layout.schema (layout : Layout) : CanonicalFieldSchemaRows.Layout Slot where
  columnMap := layout.columnMap
  rawColumns := layout.rawColumns
  rawColumnsLength := by intro slot; simp [Layout.rawColumns]
  mapsConstantOne := layout.mapsConstantOne

def Layout.fieldColumn (layout : Layout)
    (component : Component) (coordinate : Coordinate) : Nat :=
  Relabel.column (layout.columnMap { component, coordinate }) CanonicalU64.varCol

def rows (layout : Layout) : List Row :=
  CanonicalFieldSchemaRows.schemaRows Slot.all layout.schema

abbrev RawWords := CanonicalFieldSchemaRows.RawWords Slot

def Places (layout : Layout) (assignment : Nat → Nat)
    (raw : RawWords) : Prop :=
  CanonicalFieldSchemaRows.Places layout.schema assignment raw

def expectedWord (bundle : Value) (slot : Slot) : CanonicalFieldBits.Word :=
  CanonicalFieldBits.encode (bundle slot.component slot.coordinate)

theorem encodeFields_eq_expectedWord (bundle : Value) (slot : Slot) :
    encodeFields bundle slot = (expectedWord bundle slot).val := by
  unfold encodeFields expectedWord CanonicalFieldBits.encode encodeWord
    WasmStateCodec.encodeWord FieldTag.bitWidth Value.fieldValue
  change
    Nat.digitsAppend 2 64
        ((bundle slot.component slot.coordinate).val % 2 ^ 64) =
      Nat.digitsAppend 2 64 (bundle slot.component slot.coordinate).val
  have bound := (bundle slot.component slot.coordinate).property.trans
    CanonicalFieldBits.modulus_lt_capacity
  rw [Nat.mod_eq_of_lt (by
    simpa [CanonicalFieldBits.bitCount] using bound)]

/-- Exact placement of the 248,832 verified bundle bits in their full-claim
window. This premise is discharged by the full-claim link composition. -/
def BitsPlaced (layout : Layout) (assignment : Nat → Nat)
    (bundle : Value) : Prop :=
  ∀ index : Fin MemoryWireGeometry.mandatoryBundleBits,
    assignment (layout.publicBitStart + index.val) =
      (encode bundle).get
        ⟨index.val, by simpa [encode_length] using index.isLt⟩

theorem encoded_word_exact (bundle : Value) (slot : Slot) :
    ((encode bundle).drop slot.bitOffset).take CanonicalFieldBits.bitCount =
      (expectedWord bundle slot).val := by
  rw [encode, Slot.bitOffset, Slot.position_eq_idxOf]
  exact (encodeFor_word_at bundle slot slot.mem_all).trans
    (encodeFields_eq_expectedWord bundle slot)

theorem expected_words_placed
    {layout : Layout} {assignment : Nat → Nat} {bundle : Value}
    (placed : BitsPlaced layout assignment bundle) :
    Places layout assignment (expectedWord bundle) := by
  intro slot
  rw [← encoded_word_exact bundle slot]
  apply List.ext_get
  · rw [List.length_take, List.length_drop]
    have wordFits :
        slot.bitOffset + CanonicalFieldBits.bitCount ≤
          (encode bundle).length := by
      rw [encode_length]
      have positionBound := slot.position_lt
      norm_num [Slot.bitOffset, CanonicalFieldBits.bitCount,
        MemoryWireGeometry.mandatoryBundleBits_exact] at *
      omega
    have enough :
        CanonicalFieldBits.bitCount ≤
          (encode bundle).length - slot.bitOffset := by omega
    rw [Nat.min_eq_left enough]
    simp [CanonicalFieldSchemaRows.rawDigits, CanonicalFieldBits.bitCount]
  · intro bit leftBound rightBound
    have bitBound : bit < CanonicalFieldBits.bitCount := by
      simpa [CanonicalFieldSchemaRows.rawDigits, Layout.rawColumns] using
        rightBound
    have globalBound :
        slot.bitOffset + bit < MemoryWireGeometry.mandatoryBundleBits := by
      have positionBound := slot.position_lt
      norm_num [Slot.bitOffset, CanonicalFieldBits.bitCount,
        MemoryWireGeometry.mandatoryBundleBits_exact] at *
      omega
    let global : Fin MemoryWireGeometry.mandatoryBundleBits :=
      ⟨slot.bitOffset + bit, globalBound⟩
    have placedBit := placed global
    have globalEncodeBound :
        slot.bitOffset + bit < (encode bundle).length := by
      simpa [encode_length] using globalBound
    have leftBit :
        (((encode bundle).drop slot.bitOffset).take
            CanonicalFieldBits.bitCount).get ⟨bit, leftBound⟩ =
          (encode bundle).get
            ⟨slot.bitOffset + bit, globalEncodeBound⟩ := by
      simp [bitBound]
    have rightBit :
        (CanonicalFieldSchemaRows.rawDigits layout.schema assignment slot).get
            ⟨bit, rightBound⟩ =
          assignment (layout.publicBitStart + slot.bitOffset + bit) := by
      simp [CanonicalFieldSchemaRows.rawDigits, Layout.schema,
        Layout.rawColumns, bitBound]
    calc
      (((encode bundle).drop slot.bitOffset).take
            CanonicalFieldBits.bitCount).get ⟨bit, leftBound⟩ =
          (encode bundle).get
            ⟨slot.bitOffset + bit, globalEncodeBound⟩ := leftBit
      _ = assignment (layout.publicBitStart + slot.bitOffset + bit) := by
        simpa [global, Nat.add_assoc] using placedBit.symm
      _ = (CanonicalFieldSchemaRows.rawDigits layout.schema assignment slot).get
            ⟨bit, rightBound⟩ := rightBit.symm

def NativeParses (raw : RawWords) (bundle : Value) : Prop :=
  ∀ slot, FieldCodec.nativeDecode (raw slot) =
    some (bundle slot.component slot.coordinate)

theorem expected_words_parse (bundle : Value) :
    NativeParses (expectedWord bundle) bundle := by
  intro slot
  apply (FieldCodec.nativeDecode_some_iff _ _).2
  exact
    ⟨CanonicalFieldBits.encode_is_canonical _,
      (CanonicalFieldBits.decode_encode _).symm⟩

theorem rows_force_native_acceptance
    {layout : Layout} {assignment : Nat → Nat} {raw : RawWords}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (rows layout) assignment)
    (placed : Places layout assignment raw) :
    ∀ slot, ∃ value,
      FieldCodec.nativeDecode (raw slot) = some value ∧
        value.val = assignment
          (Relabel.column (layout.columnMap slot) CanonicalU64.varCol) := by
  intro slot
  exact CanonicalFieldSchemaRows.slot_sound canonical one satisfies placed
    slot.mem_all

theorem modulus_alias_impossible
    {layout : Layout} {assignment : Nat → Nat} {raw : RawWords}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (rows layout) assignment)
    (placed : Places layout assignment raw)
    (slot : Slot)
    (aliasEq : raw slot = CanonicalFieldBits.modulusWord) : False := by
  rcases rows_force_native_acceptance canonical one satisfies placed slot with
    ⟨value, decoded, _⟩
  rw [aliasEq, FieldCodec.rejects_zero_modulus_alias.2] at decoded
  simp at decoded

/-- The equality consumed by compact-token rows. It is derived from exact
bit placement, canonical field rows, and native parsing of the same typed
bundle. -/
theorem typed_columns_of_rows
    {layout : Layout} {assignment : Nat → Nat} {raw : RawWords}
    {bundle : Value}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (rows layout) assignment)
    (placed : Places layout assignment raw)
    (parsed : NativeParses raw bundle) :
    ∀ component coordinate,
      assignment (layout.fieldColumn component coordinate) =
        (bundle component coordinate).val := by
  intro component coordinate
  let slot : Slot := { component, coordinate }
  rcases rows_force_native_acceptance canonical one satisfies placed slot with
    ⟨decoded, decodedRaw, decodedWire⟩
  have decodedEqual : decoded = bundle component coordinate :=
    FieldCodec.nativeDecode_unique decodedRaw (parsed slot)
  calc
    assignment (layout.fieldColumn component coordinate) =
        decoded.val := decodedWire.symm
    _ = (bundle component coordinate).val := congrArg Subtype.val decodedEqual

/-- End-to-end local bundle-field bridge. All 3,888 field coordinates are
derived from the verified bit window; no native-parser or field-placement
assumption remains. -/
theorem typed_columns_of_bits_and_rows
    {layout : Layout} {assignment : Nat → Nat} {bundle : Value}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (rows layout) assignment)
    (bitsPlaced : BitsPlaced layout assignment bundle) :
    ∀ component coordinate,
      assignment (layout.fieldColumn component coordinate) =
        (bundle component coordinate).val := by
  exact typed_columns_of_rows canonical one satisfies
    (expected_words_placed bitsPlaced) (expected_words_parse bundle)

theorem rows_length_exact (layout : Layout) :
    (rows layout).length = 517104 := by
  rw [rows, CanonicalFieldSchemaRows.schemaRows_length,
    Slot.all_length_exact]

end Nightstream.Implementation.NebulaV2.CommitmentBundleFieldRows
