import Nightstream.Implementation.Nebula.Core.CanonicalFieldSchemaRows
import Nightstream.Implementation.Nebula.NIFS.Running.RunningParser

/-!
Contract: canonical bit-to-field R1CS bridge for the complete V2 paper-NIFS
running claim.

Assurance tier: implementation-to-protocol bridge.

Owns one canonical-u64 block for each of the 95,090 running-claim fields,
exact field-vector order, strict rejection of noncanonical Goldilocks aliases,
and equality between every parsed field and its generated field wire.

Does not own full-claim section placement, the paper-NIFS verifier arithmetic,
absolute generated columns, Rust conformance, or cryptographic soundness.

Emits constraints: yes.
-/

set_option autoImplicit false
set_option maxRecDepth 10000
set_option maxHeartbeats 500000

namespace Nightstream.Implementation.Nebula.ProductNifsRunningFieldRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.ProductNifsCodec
open Nightstream.Implementation.Nebula.ProductNifsFieldParser
open Nightstream.Protocol.Nebula

abbrev Slot := Fin runningFieldCount

def Slot.all : List Slot := List.ofFn id

theorem Slot.all_length_exact : Slot.all.length = runningFieldCount := by
  simp [Slot.all]

theorem Slot.mem_all (slot : Slot) : slot ∈ Slot.all := by
  simp [Slot.all]

def Slot.bitOffset (slot : Slot) : Nat := slot.val * fieldBitWidth

theorem Slot.bitOffset_fits (slot : Slot) :
    slot.bitOffset + CanonicalFieldBits.bitCount ≤ runningBitCount := by
  have bounded := slot.isLt
  change slot.val * 64 + 64 ≤ 6085760
  change slot.val < 95090 at bounded
  omega

structure Layout where
  publicBitStart : Nat
  columnMap : Slot → List Nat
  mapsConstantOne : ∀ slot, Relabel.column (columnMap slot) 0 = 0

def Layout.rawColumns (layout : Layout) (slot : Slot) : List Nat :=
  (List.range CanonicalFieldBits.bitCount).map fun bit =>
    layout.publicBitStart + slot.bitOffset + bit

def Layout.schema (layout : Layout) :
    CanonicalFieldSchemaRows.Layout Slot where
  columnMap := layout.columnMap
  rawColumns := layout.rawColumns
  rawColumnsLength := by intro slot; simp [Layout.rawColumns]
  mapsConstantOne := layout.mapsConstantOne

def Layout.fieldColumn (layout : Layout) (slot : Slot) : Nat :=
  Relabel.column (layout.columnMap slot) CanonicalU64.varCol

def rows (layout : Layout) : List Row :=
  CanonicalFieldSchemaRows.schemaRows Slot.all layout.schema

theorem rows_length_exact (layout : Layout) :
    (rows layout).length = 12646970 := by
  rw [rows, CanonicalFieldSchemaRows.schemaRows_length,
    Slot.all_length_exact]
  rfl

abbrev Block := ProductNifsFieldParser.Block runningFieldCount

def rawWords (block : Block) :
    CanonicalFieldSchemaRows.RawWords Slot :=
  fun slot => ProductNifsFieldParser.fieldWord block slot

def Places (layout : Layout) (assignment : Nat → Nat)
    (block : Block) : Prop :=
  CanonicalFieldSchemaRows.Places layout.schema assignment (rawWords block)

/-- Exact placement of every recursive-state input bit. -/
def BitsPlaced (layout : Layout) (assignment : Nat → Nat)
    (block : Block) : Prop :=
  ∀ index : Fin runningBitCount,
    assignment (layout.publicBitStart + index.val) =
      block.val.get
        ⟨index.val, by
          rw [block.property.1]
          change index.val < runningBitCount
          exact index.isLt⟩

theorem raw_words_placed
    {layout : Layout} {assignment : Nat → Nat} {block : Block}
    (placed : BitsPlaced layout assignment block) :
    Places layout assignment block := by
  intro slot
  apply List.ext_get
  · change
      ((block.val.drop (slot.val * 64)).take 64).length =
        ((List.range 64).map fun bit =>
          assignment
            (layout.publicBitStart + slot.val * 64 + bit)).length
    rw [List.length_take, List.length_drop, block.property.1]
    simp only [List.length_map, List.length_range]
    apply Nat.min_eq_left
    have fits := slot.bitOffset_fits
    change slot.val * 64 + 64 ≤ 5325440 at fits
    have commuted : 64 + slot.val * 64 ≤ 5325440 := by omega
    simpa [runningFieldCount, fieldBitWidth] using
      (Nat.le_sub_of_add_le commuted)
  · intro bit leftBound rightBound
    have bitBound : bit < CanonicalFieldBits.bitCount := by
      change bit < 64
      simpa using rightBound
    have globalBound : slot.bitOffset + bit < runningBitCount := by
      have fits := slot.bitOffset_fits
      omega
    let global : Fin runningBitCount :=
      ⟨slot.bitOffset + bit, globalBound⟩
    have selected := placed global
    change
      ((block.val.drop (slot.val * 64)).take 64)[bit]'leftBound =
        ((List.range 64).map fun index =>
          assignment
            (layout.publicBitStart + slot.val * 64 + index))[bit]'rightBound
    rw [List.getElem_map, List.getElem_range]
    rw [show ((block.val.drop (slot.val * 64)).take 64)[bit]'leftBound =
        block.val[slot.val * 64 + bit]'(by
          rw [block.property.1]
          simpa [Slot.bitOffset, runningBitCount] using globalBound) by simp]
    simpa [global, Slot.bitOffset, fieldBitWidth, Nat.add_assoc] using
      selected.symm

def ParsedColumnsMatch (layout : Layout) (assignment : Nat → Nat)
    (fields : Fin runningFieldCount →
      Nightstream.SuperNeo.Concrete.F) : Prop :=
  ∀ slot, (fields slot).val = assignment (layout.fieldColumn slot)

/-- Every satisfying running-field row family agrees with the direct
executable parser at every one of the 83,210 coordinates. -/
theorem parsed_columns_match
    {layout : Layout} {assignment : Nat → Nat} {block : Block}
    {fields : Fin runningFieldCount → Nightstream.SuperNeo.Concrete.F}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : BitsPlaced layout assignment block)
    (satisfies : Satisfies (rows layout) assignment)
    (accepted : ProductNifsFieldParser.parse block = some fields) :
    ParsedColumnsMatch layout assignment fields := by
  have rawPlaced := raw_words_placed placed
  intro slot
  rcases CanonicalFieldSchemaRows.slot_sound canonical one satisfies rawPlaced
      (Slot.mem_all slot) with ⟨value, nativeDecoded, columnExact⟩
  have nativeValue :=
    (FieldCodec.nativeDecode_some_iff (rawWords block slot) value).1
      nativeDecoded
  calc
    (fields slot).val =
        CanonicalFieldBits.decode
          (ProductNifsFieldParser.fieldWord block slot) :=
      ProductNifsFieldParser.parse_success_fields accepted slot
    _ = value.val := by
      simpa [rawWords] using nativeValue.2.symm
    _ = assignment (layout.fieldColumn slot) := columnExact

/-- The generated canonical-field rows construct a successful strict parser
result. This direction is independent of paper-NIFS acceptance and prevents a
caller from using verifier acceptance to obtain the verifier's own inputs. -/
theorem parse_from_rows
    {layout : Layout} {assignment : Nat → Nat} {block : Block}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : BitsPlaced layout assignment block)
    (satisfies : Satisfies (rows layout) assignment) :
    ∃ fields : Fin runningFieldCount → Nightstream.SuperNeo.Concrete.F,
      ProductNifsFieldParser.parse block = some fields ∧
        ParsedColumnsMatch layout assignment fields := by
  have rawPlaced := raw_words_placed placed
  have allCanonical : ProductNifsFieldParser.AllCanonical block := by
    intro slot
    rcases CanonicalFieldSchemaRows.slot_sound canonical one satisfies rawPlaced
        (Slot.mem_all slot) with ⟨value, nativeDecoded, _columnExact⟩
    exact (FieldCodec.nativeDecode_some_iff
      (ProductNifsFieldParser.fieldWord block slot) value).1 nativeDecoded |>.1
  let fields : Fin runningFieldCount → Nightstream.SuperNeo.Concrete.F :=
    ProductNifsFieldParser.decodedFields block allCanonical
  have parsed : ProductNifsFieldParser.parse block = some fields := by
    simp [ProductNifsFieldParser.parse, allCanonical, fields]
  exact ⟨fields, parsed,
    parsed_columns_match canonical one placed satisfies parsed⟩

/-- A modulus word in any recursive-state field is incompatible with the
canonical row family. -/
theorem modulus_alias_impossible
    {layout : Layout} {assignment : Nat → Nat} {block : Block}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : BitsPlaced layout assignment block)
    (satisfies : Satisfies (rows layout) assignment)
    (slot : Slot)
    (aliasEq : ProductNifsFieldParser.fieldWord block slot =
      CanonicalFieldBits.modulusWord) : False := by
  have rawPlaced := raw_words_placed placed
  rcases CanonicalFieldSchemaRows.slot_sound canonical one satisfies rawPlaced
      (Slot.mem_all slot) with ⟨value, nativeDecoded, _columnExact⟩
  rw [rawWords, aliasEq] at nativeDecoded
  rw [FieldCodec.rejects_zero_modulus_alias.2] at nativeDecoded
  contradiction

end Nightstream.Implementation.Nebula.ProductNifsRunningFieldRows
