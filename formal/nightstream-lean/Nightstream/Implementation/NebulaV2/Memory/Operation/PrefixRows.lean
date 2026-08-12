import Nightstream.Implementation.NebulaV2.Memory.Product.OperationSlotBridge
import Nightstream.Implementation.NebulaV2.Memory.Product.ClaimBridge
import Nightstream.Implementation.NebulaV2.Core.UnsignedAdditionRows

/-!
Contract: exact cross-slot counter and timestamp relation for one V2
operation lane.

Assurance tier: implementation-to-protocol bridge.

Owns the 64 bounded prefix counters, the 63 pad-sensitive recurrences, the
claim active-count link, the 63 bounded write timestamps, the timestamp
addition rows, and all 63 local operation-slot row blocks. It derives each
independent `OperationSlot.ValidAt` and both product-record gates in physical
slot order.

Does not own the 3-by-21 application-port refinement, snapshot rows, product
accumulation, absolute column disjointness, or the generated artifact.

Emits constraints: yes.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.OperationPrefixRows

open Nightstream.Implementation.NebulaV2.MemoryClaimCodec
open Nightstream.Implementation.NebulaV2.MemoryClaimCounterRows
open Nightstream.Implementation.NebulaV2.MemoryProductSemanticBridge
open Nightstream.Implementation.NebulaV2.MemoryProductUpdateRows
open Nightstream.Implementation.NebulaV2.MemoryProductClaimBridge
open Nightstream.Implementation.NebulaV2.OperationSlotProductBridge
open Nightstream.Implementation.NebulaV2.OperationSlotRows
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.Canonical.GoldilocksField
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.ConcreteLaneGeometry
open Nightstream.Protocol.NebulaV2.Fingerprint
open Nightstream.Protocol.NebulaV2.OperationSlot
open Nightstream.Protocol.NebulaV2.MemoryWireGeometry

abbrev CountIndex := Fin (operationSlots + 1)

def beforeIndex (slot : Fin operationSlots) : CountIndex :=
  ⟨slot.val, by omega⟩

def afterIndex (slot : Fin operationSlots) : CountIndex :=
  ⟨slot.val + 1, by omega⟩

def firstCount : CountIndex := ⟨0, by decide⟩

def lastCount : CountIndex := ⟨operationSlots, by decide⟩

structure Layout where
  product : MemoryProductUpdateRows.Layout
  countColumn : CountIndex → Nat
  countBitStart : CountIndex → Nat
  writeTimestampBitStart : Fin operationSlots → Nat
  slotAux : Fin operationSlots → OperationSlotRows.AuxColumns
  writeTimestampLinked : ∀ slot,
    product.writeTimestamp slot = [((slotAux slot).writeTimestamp, 1)]

def Layout.countWord (layout : Layout) (index : CountIndex) :
    BoundedWordRows.Layout :=
  { width := stepActiveAccessCountBits
    valueColumn := layout.countColumn index
    bitStart := layout.countBitStart index }

def Layout.operationSlot (layout : Layout) (slot : Fin operationSlots) :
    OperationSlotRows.Layout :=
  { product := layout.product
    slot := slot
    aux := layout.slotAux slot }

def Layout.writeTimestampWord (layout : Layout)
    (slot : Fin operationSlots) : BoundedWordRows.Layout :=
  { width := Nightstream.Protocol.NebulaV2.timestampBits
    valueColumn := (layout.slotAux slot).writeTimestamp
    bitStart := layout.writeTimestampBitStart slot }

def Layout.timestampAddition (layout : Layout)
    (slot : Fin operationSlots) : UnsignedAdditionRows.Layout :=
  { leftWidth := Nightstream.Protocol.NebulaV2.timestampBits
    rightWidth := stepActiveAccessCountBits
    leftColumn := layout.product.claim.counterValueColumn .timestampIn
    rightColumn := layout.countColumn (afterIndex slot)
    outputColumn := (layout.slotAux slot).writeTimestamp }

def Layout.timestampAdditionValid (layout : Layout)
    (slot : Fin operationSlots) : (layout.timestampAddition slot).Valid where
  sumFits := by
    norm_num [Layout.timestampAddition,
      Nightstream.Protocol.NebulaV2.timestampBits,
      stepActiveAccessCountBits, goldilocksP]

def Layout.countZeroRow (layout : Layout) : Row :=
  builderLinearRow (layout.countColumn firstCount) []

/-- Exact integer equation `count[j+1] + pad[j] = count[j] + 1`. Both
sides are below 65, so the field row cannot hide wraparound. -/
def Layout.countRecurrenceRow (layout : Layout)
    (slot : Fin operationSlots) : Row :=
  { a := [(layout.countColumn (afterIndex slot), 1),
      ((layout.operationSlot slot).padColumn, 1)]
    b := [(0, 1)]
    c := [(layout.countColumn (beforeIndex slot), 1), (0, 1)] }

def Layout.finalCountRow (layout : Layout) : Row :=
  builderLinearRow
    (layout.product.claim.counterValueColumn .activeAccessCount)
    [(layout.countColumn lastCount, 1)]

def countWordRows (layout : Layout) : List Row :=
  (List.ofFn fun index : CountIndex =>
    BoundedWordRows.rows (layout.countWord index)).flatten

def recurrenceRows (layout : Layout) : List Row :=
  List.ofFn layout.countRecurrenceRow

def writeTimestampWordRows (layout : Layout) : List Row :=
  (List.ofFn fun slot : Fin operationSlots =>
    BoundedWordRows.rows (layout.writeTimestampWord slot)).flatten

def timestampAdditionRows (layout : Layout) : List Row :=
  (List.ofFn fun slot : Fin operationSlots =>
    UnsignedAdditionRows.rows (layout.timestampAddition slot)).flatten

def operationSlotRows (layout : Layout) : List Row :=
  (List.ofFn fun slot : Fin operationSlots =>
    OperationSlotRows.rows (layout.operationSlot slot)).flatten

def pieces (layout : Layout) : List (List Row) :=
  [countWordRows layout, [layout.countZeroRow], recurrenceRows layout,
    [layout.finalCountRow], writeTimestampWordRows layout,
    timestampAdditionRows layout, operationSlotRows layout]

def rows (layout : Layout) : List Row :=
  (pieces layout).flatten

private theorem flatten_ofFn_length
    {α : Type} {count width : Nat} (blocks : Fin count → List α)
    (each : ∀ index, (blocks index).length = width) :
    (List.ofFn blocks).flatten.length = count * width := by
  rw [List.length_flatten]
  have constant : ∀ value ∈ (List.ofFn blocks).map List.length,
      value = width := by
    intro value member
    rcases List.mem_map.mp member with ⟨block, blockMember, rfl⟩
    rcases List.mem_ofFn.mp blockMember with ⟨index, rfl⟩
    exact each index
  rw [List.sum_eq_card_nsmul _ width constant]
  simp

theorem countWordRows_length (layout : Layout) :
    (countWordRows layout).length = 448 := by
  have exactLength := flatten_ofFn_length (width := 7)
    (fun index : CountIndex =>
      BoundedWordRows.rows (layout.countWord index)) (fun index => by
        simpa [Layout.countWord, stepActiveAccessCountBits] using
          BoundedWordRows.rows_length (layout.countWord index))
  change (List.ofFn fun index : CountIndex =>
    BoundedWordRows.rows (layout.countWord index)).flatten.length = 448
  calc
    _ = (operationSlots + 1) * 7 := exactLength
    _ = 448 := by decide

theorem recurrenceRows_length (layout : Layout) :
    (recurrenceRows layout).length = 63 := by
  simp [recurrenceRows, operationSlots]

theorem writeTimestampWordRows_length (layout : Layout) :
    (writeTimestampWordRows layout).length = 1512 := by
  have exactLength := flatten_ofFn_length (width := 24)
    (fun slot : Fin operationSlots =>
      BoundedWordRows.rows (layout.writeTimestampWord slot)) (fun slot => by
        simpa [Layout.writeTimestampWord,
          Nightstream.Protocol.NebulaV2.timestampBits] using
          BoundedWordRows.rows_length (layout.writeTimestampWord slot))
  change (List.ofFn fun slot : Fin operationSlots =>
    BoundedWordRows.rows (layout.writeTimestampWord slot)).flatten.length = 1512
  calc
    _ = operationSlots * 24 := exactLength
    _ = 1512 := by decide

theorem timestampAdditionRows_length (layout : Layout) :
    (timestampAdditionRows layout).length = 63 := by
  simp [timestampAdditionRows, UnsignedAdditionRows.rows_length,
    operationSlots]

theorem operationSlotRows_length (layout : Layout) :
    (operationSlotRows layout).length = 10080 := by
  simp [operationSlotRows, OperationSlotRows.rows_length_exact,
    operationSlots]

theorem rows_length_exact (layout : Layout) :
    (rows layout).length = 12168 := by
  simp only [rows, pieces, List.flatten_cons, List.flatten_nil,
    List.length_append, List.length_singleton, List.length_nil,
    countWordRows_length, recurrenceRows_length,
    writeTimestampWordRows_length, timestampAdditionRows_length,
    operationSlotRows_length]

private theorem piece_holds
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment)
    {part : List Row} (member : part ∈ pieces layout) :
    Satisfies part assignment := by
  exact (satisfies_flatten_iff (pieces layout) assignment).mp holds part member

private theorem count_word_holds
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) (index : CountIndex) :
    Satisfies (BoundedWordRows.rows (layout.countWord index)) assignment := by
  have group := piece_holds holds
    (part := countWordRows layout) (by simp [pieces])
  exact (satisfies_flatten_iff _ _).mp group _
    (List.mem_ofFn.mpr ⟨index, rfl⟩)

private theorem timestamp_word_holds
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment)
    (slot : Fin operationSlots) :
    Satisfies
      (BoundedWordRows.rows (layout.writeTimestampWord slot)) assignment := by
  have group := piece_holds holds
    (part := writeTimestampWordRows layout) (by simp [pieces])
  exact (satisfies_flatten_iff _ _).mp group _
    (List.mem_ofFn.mpr ⟨slot, rfl⟩)

private theorem timestamp_addition_holds
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment)
    (slot : Fin operationSlots) :
    Satisfies
      (UnsignedAdditionRows.rows (layout.timestampAddition slot))
      assignment := by
  have group := piece_holds holds
    (part := timestampAdditionRows layout) (by simp [pieces])
  exact (satisfies_flatten_iff _ _).mp group _
    (List.mem_ofFn.mpr ⟨slot, rfl⟩)

private theorem operation_slot_holds
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment)
    (slot : Fin operationSlots) :
    Satisfies (OperationSlotRows.rows (layout.operationSlot slot))
      assignment := by
  have group := piece_holds holds
    (part := operationSlotRows layout) (by simp [pieces])
  exact (satisfies_flatten_iff _ _).mp group _
    (List.mem_ofFn.mpr ⟨slot, rfl⟩)

private theorem count_lt_sixtyFour
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment)
    (index : CountIndex) :
    assignment (layout.countColumn index) < 64 := by
  have bounded := BoundedWordRows.value_lt_twoPower
    (layout := layout.countWord index) (by
      norm_num [Layout.countWord, stepActiveAccessCountBits, goldilocksP])
    canonical one (count_word_holds holds index)
  simpa [Layout.countWord, stepActiveAccessCountBits] using bounded

private theorem write_timestamp_lt_limit
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment)
    (slot : Fin operationSlots) :
    assignment (layout.slotAux slot).writeTimestamp < timestampLimit := by
  have bounded := BoundedWordRows.value_lt_twoPower
    (layout := layout.writeTimestampWord slot) (by
      norm_num [Layout.writeTimestampWord,
        Nightstream.Protocol.NebulaV2.timestampBits, goldilocksP])
    canonical one (timestamp_word_holds holds slot)
  simpa [Layout.writeTimestampWord, timestampLimit] using bounded

private theorem pad_binary
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment)
    (slot : Fin operationSlots) :
    assignment (layout.operationSlot slot).padColumn = 0 ∨
      assignment (layout.operationSlot slot).padColumn = 1 := by
  have slotHolds := operation_slot_holds holds slot
  have bitHolds : RowHolds assignment
      (bitRow (layout.operationSlot slot).padColumn) :=
    slotHolds _ (by
      simp [OperationSlotRows.rows, OperationSlotRows.flagRows])
  have atMost := bitRow_le_one goldilocks_euclidPrime
    (canonical (layout.operationSlot slot).padColumn) one bitHolds
  omega

private theorem pair_eval
    {assignment : Nat → Nat} {left right : Nat}
    (sumBound : assignment left + assignment right < goldilocksP) :
    lcEval assignment [(left, 1), (right, 1)] =
      assignment left + assignment right := by
  simp only [lcEval, List.foldl_cons, List.foldl_nil, Nat.zero_add,
    Nat.one_mul]
  exact Nat.mod_eq_of_lt sumBound

private theorem count_step
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment)
    (slot : Fin operationSlots) :
    assignment (layout.countColumn (afterIndex slot)) =
      assignment (layout.countColumn (beforeIndex slot)) +
        (1 - assignment (layout.operationSlot slot).padColumn) := by
  have beforeBound := count_lt_sixtyFour canonical one holds (beforeIndex slot)
  have afterBound := count_lt_sixtyFour canonical one holds (afterIndex slot)
  have padBinary := pad_binary canonical one holds slot
  have recurrenceGroup := piece_holds holds
    (part := recurrenceRows layout) (by simp [pieces])
  have recurrenceHolds : RowHolds assignment
      (layout.countRecurrenceRow slot) :=
    recurrenceGroup _ (List.mem_ofFn.mpr ⟨slot, rfl⟩)
  have leftBound :
      assignment (layout.countColumn (afterIndex slot)) +
          assignment (layout.operationSlot slot).padColumn < goldilocksP := by
    rcases padBinary with pad | pad <;>
      norm_num [pad, goldilocksP] <;> omega
  have rightBound :
      assignment (layout.countColumn (beforeIndex slot)) +
          assignment 0 < goldilocksP := by
    rw [one]
    norm_num [goldilocksP]
    omega
  simp only [Layout.countRecurrenceRow, RowHolds] at recurrenceHolds
  rw [pair_eval leftBound] at recurrenceHolds
  have oneEval : lcEval assignment [(0, 1)] = 1 := by
    simp [lcEval, one, goldilocksP]
  rw [oneEval, Nat.mul_one, Nat.mod_eq_of_lt leftBound] at recurrenceHolds
  rw [pair_eval rightBound] at recurrenceHolds
  rcases padBinary with pad | pad <;> omega

private theorem count_zero
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    assignment (layout.countColumn firstCount) = 0 := by
  apply builderLinearRow_sound canonical one _ [] (by
    simp [CanonicalTerms])
  have group := piece_holds holds
    (part := [layout.countZeroRow]) (by simp [pieces])
  change RowHolds assignment layout.countZeroRow
  exact group _ (by simp)

private theorem final_count_eq_claim
    {layout : Layout} {assignment : Nat → Nat} {claim : Claim}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (parsed : MemoryClaimRows.ParsedColumnsMatch
      layout.product.claim assignment claim)
    (holds : Satisfies (rows layout) assignment) :
    assignment (layout.countColumn lastCount) = claim.activeAccessCount := by
  have linked := builderLinearRow_sound canonical one
    (layout.product.claim.counterValueColumn .activeAccessCount)
    [(layout.countColumn lastCount, 1)] (by
      simp [CanonicalTerms]; decide)
    (by
      have group := piece_holds holds
        (part := [layout.finalCountRow]) (by simp [pieces])
      change RowHolds assignment layout.finalCountRow
      exact group _ (by simp))
  have lastCanonical := canonical (layout.countColumn lastCount)
  have linked' :
      assignment (layout.product.claim.counterValueColumn .activeAccessCount) =
        assignment (layout.countColumn lastCount) := by
    simpa [lcEval, Nat.mod_eq_of_lt lastCanonical] using linked
  have claimPlaced := parsed.counters .activeAccessCount
  change assignment
      (layout.product.claim.counterValueColumn .activeAccessCount) =
    claim.activeAccessCount at claimPlaced
  exact linked'.symm.trans claimPlaced

private theorem write_timestamp_rule
    {layout : Layout} {assignment : Nat → Nat} {claim : Claim}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (parsed : MemoryClaimRows.ParsedColumnsMatch
      layout.product.claim assignment claim)
    (holds : Satisfies (rows layout) assignment)
    (slot : Fin operationSlots) :
    assignment (layout.slotAux slot).writeTimestamp =
      claim.timestampIn +
        assignment (layout.countColumn (afterIndex slot)) := by
  have timestampPlaced := parsed.counters .timestampIn
  change assignment
      (layout.product.claim.counterValueColumn .timestampIn) =
    claim.timestampIn at timestampPlaced
  have timestampBound : claim.timestampIn <
      2 ^ Nightstream.Protocol.NebulaV2.timestampBits :=
    parsed.canonical.timestampIn
  have timestampColumnBound : assignment
      (layout.product.claim.counterValueColumn .timestampIn) <
      2 ^ Nightstream.Protocol.NebulaV2.timestampBits := by
    rw [timestampPlaced]
    exact timestampBound
  have countBound := count_lt_sixtyFour canonical one holds (afterIndex slot)
  have added := UnsignedAdditionRows.output_eq_add
    (layout.timestampAdditionValid slot) (by
      simpa [Layout.timestampAddition] using timestampColumnBound)
    (by
      simpa [Layout.timestampAddition, stepActiveAccessCountBits] using
        countBound)
    canonical one (timestamp_addition_holds holds slot)
  simp only [Layout.timestampAddition] at added
  rw [timestampPlaced] at added
  exact added

/-- Complete row-derived source meaning for all 63 physical operation slots.
The result contains no fingerprint product endpoint. -/
structure Sound
    (layout : Layout) (assignment : Nat → Nat) (claim : Claim) where
  countZero : assignment (layout.countColumn firstCount) = 0
  countStep : ∀ slot,
    assignment (layout.countColumn (afterIndex slot)) =
      assignment (layout.countColumn (beforeIndex slot)) +
        (1 - assignment (layout.operationSlot slot).padColumn)
  countBound : ∀ index, assignment (layout.countColumn index) ≤ 63
  finalCount : assignment (layout.countColumn lastCount) =
    claim.activeAccessCount
  slotValid : ∀ slot, OperationSlot.ValidAt
    (OperationSlotRows.decoded (layout.operationSlot slot) assignment
      (assignment (layout.countColumn (beforeIndex slot)))
      (assignment (layout.countColumn (afterIndex slot))))
    claim.timestampIn

def Sound.records
    {layout : Layout} {assignment : Nat → Nat} {claim : Claim}
    (sound : Sound layout assignment claim) (role : OperationRole)
    (slot : Fin operationSlots) : Option BoundedTuple :=
  OperationSlotProductBridge.representedRecord (sound.slotValid slot) role

theorem sound
    {layout : Layout} {assignment : Nat → Nat} {claim : Claim}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (parsed : MemoryClaimRows.ParsedColumnsMatch
      layout.product.claim assignment claim)
    (holds : Satisfies (rows layout) assignment) :
    Sound layout assignment claim := by
  have eachCountBound : ∀ index,
      assignment (layout.countColumn index) ≤ 63 := by
    intro index
    have := count_lt_sixtyFour canonical one holds index
    omega
  refine
    { countZero := count_zero canonical one holds
      countStep := count_step canonical one holds
      countBound := eachCountBound
      finalCount := final_count_eq_claim canonical one parsed holds
      slotValid := ?_ }
  intro slot
  apply OperationSlotRows.rows_sound canonical one
    (operation_slot_holds holds slot) claim.timestampIn
    (assignment (layout.countColumn (beforeIndex slot)))
    (assignment (layout.countColumn (afterIndex slot)))
  · exact count_step canonical one holds slot
  · exact eachCountBound (beforeIndex slot)
  · exact eachCountBound (afterIndex slot)
  · exact write_timestamp_rule canonical one parsed holds slot
  · exact write_timestamp_lt_limit canonical one holds slot

/-- Both repetitions use the same row-derived operation records. -/
theorem operation_source_refines
    {layout : Layout} {assignment : Nat → Nat} {claim : Claim}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment)
    (derived : Sound layout assignment claim)
    (repetition : Fin 2) (role : OperationRole) :
    List.Forall₂ (GateRepresents assignment)
      (layout.product.operationChain repetition role).entries
      (operationRecords fun slot => derived.records role slot) := by
  simp only [Layout.operationChain, Layout.operationEntries, operationRecords]
  apply List.forall₂_of_length_eq_of_get
  · simp
  · intro index leftBound _rightBound
    have indexBound : index < operationSlots := by
      simpa using leftBound
    let slot : Fin operationSlots := ⟨index, indexBound⟩
    have represented := OperationSlotProductBridge.gate_represents
      (layout := layout.operationSlot slot)
      ⟨by simpa [Layout.operationSlot] using
        layout.writeTimestampLinked slot⟩ canonical one
      (operation_slot_holds holds slot) (derived.slotValid slot)
      repetition role
    simpa [List.get_ofFn, slot] using represented

end Nightstream.Implementation.NebulaV2.OperationPrefixRows
