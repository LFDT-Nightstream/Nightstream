import Nightstream.Implementation.NebulaV2.Core.BoundedWordRows
import Nightstream.Implementation.NebulaV2.Core.ConditionalEqualityOneRows
import Nightstream.Implementation.NebulaV2.Core.ConditionalEqualityRows
import Nightstream.Implementation.NebulaV2.Memory.Product.UpdateRows
import Nightstream.Implementation.NebulaV2.Core.UnsignedAdditionRows
import Nightstream.Implementation.R1CS.Canonical.KLinear
import Nightstream.Protocol.NebulaV2.OperationSlot

/-!
Contract: exact local R1CS source relation for one V2 operation slot.

Assurance tier: implementation model.

Owns all operation payload word ranges, flag bitness, canonical inactive
padding, read-value preservation, ROM address and write restrictions, and the
active strict read-before-write comparison.

Prefix-counter and write-timestamp equations are inputs to the local
soundness theorem. `OperationPrefixRows` owns their cross-slot derivation.
Product rows consume these same lane and derived columns.

Does not own product accumulation, 3-by-21 application routing, absolute
column ownership, or the generated artifact.

Emits constraints: yes.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.OperationSlotRows

open Nightstream.Implementation.NebulaV2.MemoryProductUpdateRows
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.GoldilocksField
open Nightstream.Implementation.R1CS.Canonical.KLinear
open Nightstream.Implementation.R1CS.Program
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.ConcreteLaneGeometry
open Nightstream.Protocol.NebulaV2.OperationSlot

structure AuxColumns where
  zero : Nat
  address : Nat
  lowRomAddress : Nat
  readValue : Nat
  writeValue : Nat
  readTimestamp : Nat
  writeTimestamp : Nat
  strictIncrement : Nat
  strictSlack : Nat
  strictTotal : Nat
deriving DecidableEq, Repr

structure Layout where
  product : MemoryProductUpdateRows.Layout
  slot : Fin operationSlots
  aux : AuxColumns

def Layout.padColumn (layout : Layout) : Nat :=
  layout.product.operationPadColumn layout.slot

def Layout.isWriteColumn (layout : Layout) : Nat :=
  layout.product.operationSlotStart layout.slot + 1

def Layout.isRamColumn (layout : Layout) : Nat :=
  layout.product.operationIsRamColumn layout.slot

def Layout.addressWord (layout : Layout) : BoundedWordRows.Layout :=
  { width := addressBits
    valueColumn := layout.aux.address
    bitStart := layout.product.operationAddressStart layout.slot }

def Layout.lowRomAddressWord (layout : Layout) : BoundedWordRows.Layout :=
  { width := 12
    valueColumn := layout.aux.lowRomAddress
    bitStart := layout.product.operationAddressStart layout.slot }

def Layout.readValueWord (layout : Layout) : BoundedWordRows.Layout :=
  { width := ConcreteLaneGeometry.valueBits
    valueColumn := layout.aux.readValue
    bitStart := layout.product.operationReadValueStart layout.slot }

def Layout.writeValueWord (layout : Layout) : BoundedWordRows.Layout :=
  { width := ConcreteLaneGeometry.valueBits
    valueColumn := layout.aux.writeValue
    bitStart := layout.product.operationWriteValueStart layout.slot }

def Layout.readTimestampWord (layout : Layout) : BoundedWordRows.Layout :=
  { width := ConcreteLaneGeometry.timestampBits
    valueColumn := layout.aux.readTimestamp
    bitStart := layout.product.operationReadTimestampStart layout.slot }

def Layout.strictSlackWord (layout : Layout) : BoundedWordRows.Layout :=
  { width := ConcreteLaneGeometry.timestampBits
    valueColumn := layout.aux.strictSlack
    bitStart := layout.aux.strictSlack + 1 }

def Layout.strictIncrementLayout (layout : Layout) :
    UnsignedAdditionRows.Layout :=
  { leftWidth := ConcreteLaneGeometry.timestampBits
    rightWidth := 1
    leftColumn := layout.aux.readTimestamp
    rightColumn := 0
    outputColumn := layout.aux.strictIncrement }

def Layout.strictTotalLayout (layout : Layout) :
    UnsignedAdditionRows.Layout :=
  { leftWidth := ConcreteLaneGeometry.timestampBits + 1
    rightWidth := ConcreteLaneGeometry.timestampBits
    leftColumn := layout.aux.strictIncrement
    rightColumn := layout.aux.strictSlack
    outputColumn := layout.aux.strictTotal }

def Layout.zeroRow (layout : Layout) : Row :=
  builderLinearRow layout.aux.zero []

def flagRows (layout : Layout) : List Row :=
  [bitRow layout.padColumn, bitRow layout.isWriteColumn,
    bitRow layout.isRamColumn]

def wordRows (layout : Layout) : List Row :=
  BoundedWordRows.rows layout.addressWord ++
    BoundedWordRows.rows layout.lowRomAddressWord ++
    BoundedWordRows.rows layout.readValueWord ++
    BoundedWordRows.rows layout.writeValueWord ++
    BoundedWordRows.rows layout.readTimestampWord ++
    BoundedWordRows.rows layout.strictSlackWord

def paddingPairs (layout : Layout) : List (Nat × Nat) :=
  [(layout.isWriteColumn, layout.aux.zero),
    (layout.isRamColumn, layout.aux.zero),
    (layout.aux.address, layout.aux.zero),
    (layout.aux.readValue, layout.aux.zero),
    (layout.aux.writeValue, layout.aux.zero),
    (layout.aux.readTimestamp, layout.aux.zero)]

def paddingRows (layout : Layout) : List Row :=
  ConditionalEqualityOneRows.rows layout.padColumn (paddingPairs layout)

def readRuleRows (layout : Layout) : List Row :=
  ConditionalEqualityRows.rows layout.isWriteColumn
    [(layout.aux.writeValue, layout.aux.readValue)]

def romAddressRows (layout : Layout) : List Row :=
  ConditionalEqualityRows.rows layout.isRamColumn
    [(layout.aux.address, layout.aux.lowRomAddress)]

/-- A write flag times the ROM selector must be zero. -/
def Layout.noRomWriteRow (layout : Layout) : Row :=
  ⟨[(layout.isWriteColumn, 1)],
    [(0, 1), (layout.isRamColumn, goldilocksP - 1)], []⟩

def strictRows (layout : Layout) : List Row :=
  UnsignedAdditionRows.rows layout.strictIncrementLayout ++
    UnsignedAdditionRows.rows layout.strictTotalLayout ++
    ConditionalEqualityRows.rows layout.padColumn
      [(layout.aux.strictTotal, layout.aux.writeTimestamp)]

def rows (layout : Layout) : List Row :=
  [layout.zeroRow] ++ flagRows layout ++ wordRows layout ++
    paddingRows layout ++ readRuleRows layout ++ romAddressRows layout ++
    [layout.noRomWriteRow] ++ strictRows layout

theorem rows_length_exact (layout : Layout) :
    (rows layout).length = 160 := by
  simp only [rows, List.length_append, List.length_singleton, flagRows,
    List.length_cons, List.length_nil, wordRows,
    BoundedWordRows.rows_length, Layout.addressWord,
    Layout.lowRomAddressWord, Layout.readValueWord, Layout.writeValueWord,
    Layout.readTimestampWord, Layout.strictSlackWord, paddingRows,
    ConditionalEqualityOneRows.rows_length, paddingPairs, readRuleRows,
    romAddressRows, ConditionalEqualityRows.rows_length, strictRows,
    UnsignedAdditionRows.rows_length]
  norm_num [ConcreteLaneGeometry.addressBits,
    ConcreteLaneGeometry.valueBits, ConcreteLaneGeometry.timestampBits]

def decoded (layout : Layout) (assignment : Nat → Nat)
    (countBefore countAfter : Nat) : OperationSlot.Value :=
  { pad := assignment layout.padColumn
    isWrite := assignment layout.isWriteColumn
    isRam := assignment layout.isRamColumn
    address := assignment layout.aux.address
    readValue := assignment layout.aux.readValue
    writeValue := assignment layout.aux.writeValue
    readTimestamp := assignment layout.aux.readTimestamp
    writeTimestamp := assignment layout.aux.writeTimestamp
    countBefore := countBefore
    countAfter := countAfter }

private theorem subrows
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment)
    {part : List Row} (included : ∀ row ∈ part, row ∈ rows layout) :
    Satisfies part assignment := by
  intro row member
  exact holds row (included row member)

private theorem zero_sound
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    assignment layout.aux.zero = 0 := by
  apply builderLinearRow_sound canonical one layout.aux.zero []
    (by simp [CanonicalTerms])
  exact holds _ (by simp [rows, Layout.zeroRow])

private theorem flag_binary
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment)
    (column : Nat)
    (member : bitRow column ∈ flagRows layout) :
    assignment column = 0 ∨ assignment column = 1 := by
  have atMost := bitRow_le_one goldilocks_euclidPrime (canonical column) one
    (holds _ (by simp [rows, member]))
  omega

private theorem word_holds
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment)
    (word : BoundedWordRows.Layout)
    (included : ∀ row ∈ BoundedWordRows.rows word,
      row ∈ wordRows layout) :
    Satisfies (BoundedWordRows.rows word) assignment := by
  exact subrows holds fun row member => by
    simp [rows, included row member]

private theorem word_bound
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment)
    (word : BoundedWordRows.Layout)
    (fits : 2 ^ word.width ≤ goldilocksP)
    (included : ∀ row ∈ BoundedWordRows.rows word,
      row ∈ wordRows layout) :
    assignment word.valueColumn < 2 ^ word.width :=
  BoundedWordRows.value_lt_twoPower fits canonical one
    (word_holds holds word included)

private theorem address_bound
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    assignment layout.aux.address < 2 ^ 16 := by
  apply word_bound canonical one holds layout.addressWord
    (by
      simp only [Layout.addressWord]
      norm_num [ConcreteLaneGeometry.addressBits, goldilocksP])
  intro row member
  simp [wordRows, member]

private theorem low_rom_bound
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    assignment layout.aux.lowRomAddress < romCells := by
  have bound := word_bound canonical one holds layout.lowRomAddressWord
    (by
      simp only [Layout.lowRomAddressWord]
      norm_num [goldilocksP]) (by
      intro row member
      simp [wordRows, member])
  simpa [Layout.lowRomAddressWord, romCells] using bound

private theorem read_value_bound
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    assignment layout.aux.readValue < valueLimit := by
  have bound := word_bound canonical one holds layout.readValueWord
    (by
      simp only [Layout.readValueWord]
      norm_num [ConcreteLaneGeometry.valueBits, goldilocksP]) (by
      intro row member
      simp [wordRows, member])
  simpa [Layout.readValueWord, valueLimit,
    Nightstream.Protocol.NebulaV2.valueBits,
    ConcreteLaneGeometry.valueBits] using bound

private theorem write_value_bound
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    assignment layout.aux.writeValue < valueLimit := by
  have bound := word_bound canonical one holds layout.writeValueWord
    (by
      simp only [Layout.writeValueWord]
      norm_num [ConcreteLaneGeometry.valueBits, goldilocksP]) (by
      intro row member
      simp [wordRows, member])
  simpa [Layout.writeValueWord, valueLimit,
    Nightstream.Protocol.NebulaV2.valueBits,
    ConcreteLaneGeometry.valueBits] using bound

private theorem read_timestamp_bound
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    assignment layout.aux.readTimestamp < timestampLimit := by
  have bound := word_bound canonical one holds layout.readTimestampWord
    (by
      simp only [Layout.readTimestampWord]
      norm_num [ConcreteLaneGeometry.timestampBits, goldilocksP]) (by
      intro row member
      simp [wordRows, member])
  simpa [Layout.readTimestampWord, timestampLimit,
    Nightstream.Protocol.NebulaV2.timestampBits,
    ConcreteLaneGeometry.timestampBits] using bound

private theorem strict_slack_bound
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    assignment layout.aux.strictSlack < timestampLimit := by
  have bound := word_bound canonical one holds layout.strictSlackWord
    (by
      simp only [Layout.strictSlackWord]
      norm_num [ConcreteLaneGeometry.timestampBits, goldilocksP]) (by
      intro row member
      simp [wordRows, member])
  simpa [Layout.strictSlackWord, timestampLimit,
    Nightstream.Protocol.NebulaV2.timestampBits,
    ConcreteLaneGeometry.timestampBits] using bound

private theorem padding_sound
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment)
    (padded : assignment layout.padColumn = 1) :
    assignment layout.isWriteColumn = 0 ∧
      assignment layout.isRamColumn = 0 ∧
      assignment layout.aux.address = 0 ∧
      assignment layout.aux.readValue = 0 ∧
      assignment layout.aux.writeValue = 0 ∧
      assignment layout.aux.readTimestamp = 0 := by
  have localHolds : Satisfies (paddingRows layout) assignment :=
    subrows holds fun row member => by simp [rows, member]
  have equalities := ConditionalEqualityOneRows.rows_sound_one canonical one
    padded localHolds
  have zero := zero_sound canonical one holds
  have each (pair : Nat × Nat) (member : pair ∈ paddingPairs layout) :=
    equalities pair member
  simp only [paddingPairs, List.mem_cons, List.mem_singleton] at each
  constructor
  · rw [each (layout.isWriteColumn, layout.aux.zero) (by simp), zero]
  constructor
  · rw [each (layout.isRamColumn, layout.aux.zero) (by simp), zero]
  constructor
  · rw [each (layout.aux.address, layout.aux.zero) (by simp), zero]
  constructor
  · rw [each (layout.aux.readValue, layout.aux.zero) (by simp), zero]
  constructor
  · rw [each (layout.aux.writeValue, layout.aux.zero) (by simp), zero]
  · rw [each (layout.aux.readTimestamp, layout.aux.zero) (by simp), zero]

private theorem read_rule_sound
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment)
    (read : assignment layout.isWriteColumn = 0) :
    assignment layout.aux.writeValue = assignment layout.aux.readValue := by
  have localHolds : Satisfies (readRuleRows layout) assignment :=
    subrows holds fun row member => by simp [rows, member]
  exact ConditionalEqualityRows.rows_sound_closed canonical one read localHolds
    (layout.aux.writeValue, layout.aux.readValue) (by simp [readRuleRows])

private theorem rom_address_sound
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment)
    (rom : assignment layout.isRamColumn = 0) :
    assignment layout.aux.address < romCells := by
  have localHolds : Satisfies (romAddressRows layout) assignment :=
    subrows holds fun row member => by simp [rows, member]
  have equal := ConditionalEqualityRows.rows_sound_closed canonical one rom
    localHolds (layout.aux.address, layout.aux.lowRomAddress)
      (by simp [romAddressRows])
  rw [equal]
  exact low_rom_bound canonical one holds

private theorem no_rom_write_sound
    {layout : Layout} {assignment : Nat → Nat}
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment)
    (ramBinary : assignment layout.isRamColumn = 0 ∨
      assignment layout.isRamColumn = 1)
    (write : assignment layout.isWriteColumn = 1) :
    assignment layout.isRamColumn = 1 := by
  rcases ramBinary with ram | ram
  · have row := holds layout.noRomWriteRow (by simp [rows])
    simp [Layout.noRomWriteRow, RowHolds, lcEval, one, write, ram,
      goldilocksP] at row
  · exact ram

private theorem strict_sound
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment)
    (active : assignment layout.padColumn = 0)
    (writeTimestampBound :
      assignment layout.aux.writeTimestamp < timestampLimit) :
    assignment layout.aux.readTimestamp <
      assignment layout.aux.writeTimestamp := by
  have incrementHolds : Satisfies
      (UnsignedAdditionRows.rows layout.strictIncrementLayout) assignment :=
    subrows holds fun row member => by simp [rows, strictRows, member]
  have totalHolds : Satisfies
      (UnsignedAdditionRows.rows layout.strictTotalLayout) assignment :=
    subrows holds fun row member => by simp [rows, strictRows, member]
  have incrementValid : layout.strictIncrementLayout.Valid := by
    constructor
    norm_num [Layout.strictIncrementLayout,
      ConcreteLaneGeometry.timestampBits, goldilocksP]
  have increment := UnsignedAdditionRows.output_eq_add
    (layout := layout.strictIncrementLayout) incrementValid
    (read_timestamp_bound canonical one holds)
    (by simpa [Layout.strictIncrementLayout, one])
    canonical one incrementHolds
  simp only [Layout.strictIncrementLayout] at increment
  rw [one] at increment
  have incrementBound :
      assignment layout.aux.strictIncrement <
        2 ^ (ConcreteLaneGeometry.timestampBits + 1) := by
    rw [increment]
    have readBound := read_timestamp_bound canonical one holds
    simp only [timestampLimit,
      Nightstream.Protocol.NebulaV2.timestampBits,
      ConcreteLaneGeometry.timestampBits] at readBound ⊢
    omega
  have totalValid : layout.strictTotalLayout.Valid := by
    constructor
    norm_num [Layout.strictTotalLayout,
      ConcreteLaneGeometry.timestampBits, goldilocksP]
  have total := UnsignedAdditionRows.output_eq_add
    (layout := layout.strictTotalLayout) totalValid incrementBound
    (strict_slack_bound canonical one holds) canonical one totalHolds
  simp only [Layout.strictTotalLayout] at total
  change assignment layout.aux.strictTotal =
    assignment layout.aux.strictIncrement +
      assignment layout.aux.strictSlack at total
  have equalityHolds : Satisfies
      (ConditionalEqualityRows.rows layout.padColumn
        [(layout.aux.strictTotal, layout.aux.writeTimestamp)]) assignment :=
    subrows holds fun row member => by simp [rows, strictRows, member]
  have equality := ConditionalEqualityRows.rows_sound_closed canonical one
    active equalityHolds
      (layout.aux.strictTotal, layout.aux.writeTimestamp) (by simp)
  change assignment layout.aux.strictTotal =
    assignment layout.aux.writeTimestamp at equality
  rw [total, increment] at equality
  have slackNonnegative : 0 ≤ assignment layout.aux.strictSlack := by omega
  omega

/-- Satisfying local rows prove exactly the independent slot relation. The
counter and write-timestamp premises are source arithmetic facts supplied by
the separate prefix program; no fingerprint conclusion is assumed. -/
theorem rows_sound
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment)
    (stepTimestampIn countBefore countAfter : Nat)
    (countStep : countAfter = countBefore +
      (1 - assignment layout.padColumn))
    (countBeforeBound : countBefore ≤ 63)
    (countAfterBound : countAfter ≤ 63)
    (writeTimestampRule : assignment layout.aux.writeTimestamp =
      stepTimestampIn + countAfter)
    (writeTimestampBound :
      assignment layout.aux.writeTimestamp < timestampLimit) :
    OperationSlot.ValidAt
      (decoded layout assignment countBefore countAfter) stepTimestampIn := by
  have padBinary := flag_binary canonical one holds layout.padColumn (by
    simp [flagRows])
  have writeBinary := flag_binary canonical one holds layout.isWriteColumn (by
    simp [flagRows])
  have ramBinary := flag_binary canonical one holds layout.isRamColumn (by
    simp [flagRows])
  refine
    { padBinary := padBinary
      isWriteBinary := writeBinary
      isRamBinary := ramBinary
      addressBound := address_bound canonical one holds
      readValueBound := read_value_bound canonical one holds
      writeValueBound := write_value_bound canonical one holds
      readTimestampBound := read_timestamp_bound canonical one holds
      countStep := countStep
      countBeforeBound := countBeforeBound
      countAfterBound := countAfterBound
      writeTimestampRule := writeTimestampRule
      writeTimestampBound := writeTimestampBound
      inactiveZero := ?_
      readRule := ?_
      romAddressBound := ?_
      noRomWrite := ?_
      readBeforeWrite := ?_ }
  · intro padded
    exact padding_sound canonical one holds padded
  · intro _active read
    exact read_rule_sound canonical one holds read
  · intro _active rom
    exact rom_address_sound canonical one holds rom
  · intro _active write
    exact no_rom_write_sound one holds ramBinary write
  · intro active
    exact strict_sound canonical one holds active writeTimestampBound

private theorem word_base_eval
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment)
    (word : BoundedWordRows.Layout)
    (fits : 2 ^ word.width ≤ goldilocksP)
    (included : ∀ row ∈ BoundedWordRows.rows word,
      row ∈ wordRows layout) :
    lcEval assignment word.terms = assignment word.valueColumn := by
  have wordHolds := word_holds holds word included
  exact (BoundedWordRows.lcEval_terms_eq_decoded fits canonical one
    wordHolds).trans
      (BoundedWordRows.recomposition_sound fits canonical one wordHolds).symm

theorem address_scaled_eval
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment)
    (coefficient : Nat)
    (productBound :
      coefficient * assignment layout.aux.address < goldilocksP) :
    lcEval assignment
        (bitWord (layout.product.operationAddressStart layout.slot)
          addressBits coefficient) =
      coefficient * assignment layout.aux.address := by
  rw [bitWord, lcEval_scaleTerms]
  have base := word_base_eval canonical one holds layout.addressWord
    (by
      simp only [Layout.addressWord]
      norm_num [ConcreteLaneGeometry.addressBits, goldilocksP]) (by
        intro row member
        simp [wordRows, member])
  have base' : lcEval assignment
      (bitWordBase (layout.product.operationAddressStart layout.slot)
        addressBits) = assignment layout.aux.address := by
    simpa [bitWordBase, BoundedWordRows.Layout.terms,
      BoundedWordRows.Layout.bitColumn, Layout.addressWord] using base
  rw [base', Nat.mod_eq_of_lt productBound]

theorem read_value_eval
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    lcEval assignment
        (bitWord (layout.product.operationReadValueStart layout.slot)
          ConcreteLaneGeometry.valueBits 1) =
      assignment layout.aux.readValue := by
  rw [bitWord, lcEval_scaleTerms]
  have base := word_base_eval canonical one holds layout.readValueWord
    (by
      simp only [Layout.readValueWord]
      norm_num [ConcreteLaneGeometry.valueBits, goldilocksP]) (by
        intro row member
        simp [wordRows, member])
  have base' : lcEval assignment
      (bitWordBase (layout.product.operationReadValueStart layout.slot)
        ConcreteLaneGeometry.valueBits) =
      assignment layout.aux.readValue := by
    simpa [bitWordBase, BoundedWordRows.Layout.terms,
      BoundedWordRows.Layout.bitColumn, Layout.readValueWord] using base
  rw [base']
  simpa using Nat.mod_eq_of_lt (canonical layout.aux.readValue)

theorem write_value_eval
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    lcEval assignment
        (bitWord (layout.product.operationWriteValueStart layout.slot)
          ConcreteLaneGeometry.valueBits 1) =
      assignment layout.aux.writeValue := by
  rw [bitWord, lcEval_scaleTerms]
  have base := word_base_eval canonical one holds layout.writeValueWord
    (by
      simp only [Layout.writeValueWord]
      norm_num [ConcreteLaneGeometry.valueBits, goldilocksP]) (by
        intro row member
        simp [wordRows, member])
  have base' : lcEval assignment
      (bitWordBase (layout.product.operationWriteValueStart layout.slot)
        ConcreteLaneGeometry.valueBits) =
      assignment layout.aux.writeValue := by
    simpa [bitWordBase, BoundedWordRows.Layout.terms,
      BoundedWordRows.Layout.bitColumn, Layout.writeValueWord] using base
  rw [base']
  simpa using Nat.mod_eq_of_lt (canonical layout.aux.writeValue)

theorem read_timestamp_scaled_eval
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment)
    (coefficient : Nat)
    (productBound :
      coefficient * assignment layout.aux.readTimestamp < goldilocksP) :
    lcEval assignment
        (bitWord (layout.product.operationReadTimestampStart layout.slot)
          ConcreteLaneGeometry.timestampBits coefficient) =
      coefficient * assignment layout.aux.readTimestamp := by
  rw [bitWord, lcEval_scaleTerms]
  have base := word_base_eval canonical one holds layout.readTimestampWord
    (by
      simp only [Layout.readTimestampWord]
      norm_num [ConcreteLaneGeometry.timestampBits, goldilocksP]) (by
        intro row member
        simp [wordRows, member])
  have base' : lcEval assignment
      (bitWordBase (layout.product.operationReadTimestampStart layout.slot)
        ConcreteLaneGeometry.timestampBits) =
      assignment layout.aux.readTimestamp := by
    simpa [bitWordBase, BoundedWordRows.Layout.terms,
      BoundedWordRows.Layout.bitColumn, Layout.readTimestampWord] using base
  rw [base', Nat.mod_eq_of_lt productBound]

end Nightstream.Implementation.NebulaV2.OperationSlotRows
