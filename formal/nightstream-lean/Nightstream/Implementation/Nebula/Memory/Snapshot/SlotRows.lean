import Nightstream.Implementation.Nebula.Memory.Product.SemanticBridge
import Nightstream.Implementation.Nebula.Memory.Product.UpdateRows
import Nightstream.Implementation.Nebula.Core.UnsignedAdditionRows
import Nightstream.Protocol.Nebula.SnapshotSlot

/-!
Contract: exact R1CS source relation for one V2 snapshot role and scan slot.

Assurance tier: implementation-to-protocol bridge.

Owns the 32-bit value word, 23-bit timestamp and slack words, the exact
segment-boundary addition, the structural scan address, and the product-entry
source meaning.

Does not own all 128 snapshot slots, step scheduling across a segment,
product accumulation, absolute column disjointness, or the generated artifact.

Emits constraints: yes.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.SnapshotSlotRows

open Nightstream.Implementation.Nebula.MemoryClaimCounterRows
open Nightstream.Implementation.Nebula.MemoryClaimCodec
open Nightstream.Implementation.Nebula.MemoryProductSemanticBridge
open Nightstream.Implementation.Nebula.MemoryProductUpdateRows
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.KHorner
open Nightstream.Implementation.R1CS.Canonical.KLinear
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.ConcreteLaneGeometry
open Nightstream.Protocol.Nebula.Fingerprint
open Nightstream.Protocol.Nebula.SnapshotSlot

structure AuxColumns where
  value : Nat
  timestamp : Nat
  slack : Nat
  slackBitStart : Nat
deriving DecidableEq, Repr

structure Layout where
  product : MemoryProductUpdateRows.Layout
  role : SnapshotRole
  slot : Fin scanSlots
  aux : AuxColumns

def Layout.valueWord (layout : Layout) : BoundedWordRows.Layout :=
  { width := Nightstream.Protocol.Nebula.valueBits
    valueColumn := layout.aux.value
    bitStart := layout.product.snapshotValueStart layout.role layout.slot }

def Layout.timestampWord (layout : Layout) : BoundedWordRows.Layout :=
  { width := Nightstream.Protocol.Nebula.timestampBits
    valueColumn := layout.aux.timestamp
    bitStart := layout.product.snapshotTimestampStart layout.role layout.slot }

def Layout.slackWord (layout : Layout) : BoundedWordRows.Layout :=
  { width := Nightstream.Protocol.Nebula.timestampBits
    valueColumn := layout.aux.slack
    bitStart := layout.aux.slackBitStart }

def boundaryCounter : SnapshotRole → Counter
  | .initialSnapshot => .segmentStartTimestamp
  | .finalSnapshot => .segmentEndTimestamp

def boundaryValue (claim : Claim) : SnapshotRole → Nat
  | .initialSnapshot => claim.segmentStartTimestamp
  | .finalSnapshot => claim.segmentEndTimestamp

def Layout.boundaryColumn (layout : Layout) : Nat :=
  layout.product.claim.counterValueColumn (boundaryCounter layout.role)

def Layout.boundaryAddition (layout : Layout) : UnsignedAdditionRows.Layout :=
  { leftWidth := Nightstream.Protocol.Nebula.timestampBits
    rightWidth := Nightstream.Protocol.Nebula.timestampBits
    leftColumn := layout.aux.timestamp
    rightColumn := layout.aux.slack
    outputColumn := layout.boundaryColumn }

def Layout.boundaryAdditionValid (layout : Layout) :
    layout.boundaryAddition.Valid where
  sumFits := by
    norm_num [Layout.boundaryAddition,
      Nightstream.Protocol.Nebula.timestampBits, goldilocksP]

def rows (layout : Layout) : List Row :=
  BoundedWordRows.rows layout.valueWord ++
    BoundedWordRows.rows layout.timestampWord ++
    BoundedWordRows.rows layout.slackWord ++
    UnsignedAdditionRows.rows layout.boundaryAddition

theorem rows_length_exact (layout : Layout) :
    (rows layout).length = 82 := by
  simp [rows, BoundedWordRows.rows_length,
    UnsignedAdditionRows.rows_length, Layout.valueWord,
    Layout.timestampWord, Layout.slackWord,
    Nightstream.Protocol.Nebula.valueBits,
    Nightstream.Protocol.Nebula.timestampBits]

private theorem subrows
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment)
    {part : List Row} (included : ∀ row ∈ part, row ∈ rows layout) :
    Satisfies part assignment := by
  intro row member
  exact holds row (included row member)

private theorem value_word_holds
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies (BoundedWordRows.rows layout.valueWord) assignment :=
  subrows holds fun row member => by simp [rows, member]

private theorem timestamp_word_holds
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies (BoundedWordRows.rows layout.timestampWord) assignment :=
  subrows holds fun row member => by simp [rows, member]

private theorem slack_word_holds
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies (BoundedWordRows.rows layout.slackWord) assignment :=
  subrows holds fun row member => by simp [rows, member]

private theorem addition_holds
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies (UnsignedAdditionRows.rows layout.boundaryAddition)
      assignment :=
  subrows holds fun row member => by simp [rows, member]

private theorem value_bound
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    assignment layout.aux.value < valueLimit := by
  have bounded := BoundedWordRows.value_lt_twoPower
    (layout := layout.valueWord) (by
      norm_num [Layout.valueWord,
        Nightstream.Protocol.Nebula.valueBits, goldilocksP])
    canonical one (value_word_holds holds)
  simpa [Layout.valueWord, valueLimit] using bounded

private theorem timestamp_bound
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    assignment layout.aux.timestamp < timestampLimit := by
  have bounded := BoundedWordRows.value_lt_twoPower
    (layout := layout.timestampWord) (by
      norm_num [Layout.timestampWord,
        Nightstream.Protocol.Nebula.timestampBits, goldilocksP])
    canonical one (timestamp_word_holds holds)
  simpa [Layout.timestampWord, timestampLimit] using bounded

private theorem slack_bound
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    assignment layout.aux.slack < timestampLimit := by
  have bounded := BoundedWordRows.value_lt_twoPower
    (layout := layout.slackWord) (by
      norm_num [Layout.slackWord,
        Nightstream.Protocol.Nebula.timestampBits, goldilocksP])
    canonical one (slack_word_holds holds)
  simpa [Layout.slackWord, timestampLimit] using bounded

def decoded (layout : Layout) (assignment : Nat → Nat) :
    SnapshotSlot.Value :=
  { value := assignment layout.aux.value
    timestamp := assignment layout.aux.timestamp }

private theorem boundary_placed
    {layout : Layout} {assignment : Nat → Nat} {claim : Claim}
    (parsed : MemoryClaimRows.ParsedColumnsMatch
      layout.product.claim assignment claim) :
    assignment layout.boundaryColumn = boundaryValue claim layout.role := by
  cases roleEq : layout.role with
  | initialSnapshot =>
      have placed := parsed.counters .segmentStartTimestamp
      change assignment layout.boundaryColumn = claim.segmentStartTimestamp
      simpa [Layout.boundaryColumn, boundaryCounter, boundaryValue, roleEq]
        using placed
  | finalSnapshot =>
      have placed := parsed.counters .segmentEndTimestamp
      change assignment layout.boundaryColumn = claim.segmentEndTimestamp
      simpa [Layout.boundaryColumn, boundaryCounter, boundaryValue, roleEq]
        using placed

private theorem boundary_bound
    {layout : Layout} {assignment : Nat → Nat} {claim : Claim}
    (parsed : MemoryClaimRows.ParsedColumnsMatch
      layout.product.claim assignment claim) :
    boundaryValue claim layout.role < timestampLimit := by
  cases roleEq : layout.role with
  | initialSnapshot =>
      simpa [boundaryValue, roleEq, timestampLimit] using
        parsed.canonical.segmentStartTimestamp
  | finalSnapshot =>
      simpa [boundaryValue, roleEq, timestampLimit] using
        parsed.canonical.segmentEndTimestamp

private theorem timestamp_le_boundary
    {layout : Layout} {assignment : Nat → Nat} {claim : Claim}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (parsed : MemoryClaimRows.ParsedColumnsMatch
      layout.product.claim assignment claim)
    (holds : Satisfies (rows layout) assignment) :
    assignment layout.aux.timestamp ≤ boundaryValue claim layout.role := by
  have addition := UnsignedAdditionRows.output_eq_add
    layout.boundaryAdditionValid
    (by
      simpa [Layout.boundaryAddition, timestampLimit] using
        timestamp_bound canonical one holds)
    (by
      simpa [Layout.boundaryAddition, timestampLimit] using
        slack_bound canonical one holds)
    canonical one (addition_holds holds)
  simp only [Layout.boundaryAddition] at addition
  rw [boundary_placed parsed] at addition
  omega

/-- The slot rows derive the exact independent snapshot source relation. -/
theorem sound
    {layout : Layout} {assignment : Nat → Nat} {claim : Claim}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (parsed : MemoryClaimRows.ParsedColumnsMatch
      layout.product.claim assignment claim)
    (holds : Satisfies (rows layout) assignment) :
    SnapshotSlot.ValidAt (decoded layout assignment) claim.stepIndex.val
      (boundaryValue claim layout.role) where
  stepIndexBound := parsed.stepStrict
  valueBound := value_bound canonical one holds
  timestampBound := timestamp_bound canonical one holds
  boundaryBound := boundary_bound parsed
  timestampLeBoundary := timestamp_le_boundary canonical one parsed holds

private theorem word_base_eval
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (word : BoundedWordRows.Layout)
    (fits : 2 ^ word.width ≤ goldilocksP)
    (wordHolds : Satisfies (BoundedWordRows.rows word) assignment) :
    lcEval assignment word.terms = assignment word.valueColumn := by
  exact (BoundedWordRows.lcEval_terms_eq_decoded fits canonical one
    wordHolds).trans
      (BoundedWordRows.recomposition_sound fits canonical one wordHolds).symm

private theorem value_eval
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    lcEval assignment
        (layout.product.snapshotValue layout.role layout.slot) =
      assignment layout.aux.value := by
  rw [Layout.snapshotValue, bitWord, lcEval_scaleTerms]
  have base := word_base_eval canonical one layout.valueWord
    (by norm_num [Layout.valueWord,
      Nightstream.Protocol.Nebula.valueBits, goldilocksP])
    (value_word_holds holds)
  have base' : lcEval assignment
      (bitWordBase
        (layout.product.snapshotValueStart layout.role layout.slot)
        ConcreteLaneGeometry.valueBits) = assignment layout.aux.value := by
    simpa [bitWordBase, BoundedWordRows.Layout.terms,
      BoundedWordRows.Layout.bitColumn, Layout.valueWord,
      ConcreteLaneGeometry.valueBits,
      Nightstream.Protocol.Nebula.valueBits] using base
  rw [base']
  simpa using Nat.mod_eq_of_lt (canonical layout.aux.value)

private theorem timestamp_eval
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    lcEval assignment
        (bitWord (layout.product.snapshotTimestampStart layout.role layout.slot)
          ConcreteLaneGeometry.timestampBits 1) =
      assignment layout.aux.timestamp := by
  rw [bitWord, lcEval_scaleTerms]
  have base := word_base_eval canonical one layout.timestampWord
    (by
      norm_num [Layout.timestampWord,
        Nightstream.Protocol.Nebula.timestampBits, goldilocksP])
    (timestamp_word_holds holds)
  have base' : lcEval assignment
      (bitWordBase
        (layout.product.snapshotTimestampStart layout.role layout.slot)
        ConcreteLaneGeometry.timestampBits) =
      assignment layout.aux.timestamp := by
    simpa [bitWordBase, BoundedWordRows.Layout.terms,
      BoundedWordRows.Layout.bitColumn, Layout.timestampWord,
      ConcreteLaneGeometry.timestampBits,
      Nightstream.Protocol.Nebula.timestampBits] using base
  rw [base']
  simpa using Nat.mod_eq_of_lt (canonical layout.aux.timestamp)

private theorem singleton_eval
    {assignment : Nat → Nat} (column coefficient : Nat)
    (bound : coefficient * assignment column < goldilocksP) :
    lcEval assignment [(column, coefficient)] =
      coefficient * assignment column := by
  simp only [lcEval, List.foldl_cons, List.foldl_nil, Nat.zero_add]
  exact Nat.mod_eq_of_lt bound

private theorem constant_eval
    {assignment : Nat → Nat} (one : assignment 0 = 1)
    (value : Nat) (bound : value < goldilocksP) :
    lcEval assignment (constantWord value) = value := by
  by_cases zero : value = 0
  · simp [constantWord, zero, lcEval]
  · simp only [constantWord, if_neg zero]
    have reduced : value % goldilocksP = value := Nat.mod_eq_of_lt bound
    simp [lcEval, one, reduced]

private theorem packed_eval
    {layout : Layout} {assignment : Nat → Nat} {claim : Claim}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (parsed : MemoryClaimRows.ParsedColumnsMatch
      layout.product.claim assignment claim)
    (holds : Satisfies (rows layout) assignment)
    (valid : SnapshotSlot.ValidAt (decoded layout assignment)
      claim.stepIndex.val (boundaryValue claim layout.role)) :
    lcEval assignment
        (layout.product.snapshotPacked layout.role layout.slot) =
      packedNat ((valid.boundedTuple layout.slot).1) := by
  have stepPlaced := parsed.counters .stepIndex
  change assignment
      (layout.product.claim.counterValueColumn .stepIndex) =
    claim.stepIndex.val at stepPlaced
  have stepTermBound : timestampLimit * scanSlots *
      assignment (layout.product.claim.counterValueColumn .stepIndex) <
      goldilocksP := by
    rw [stepPlaced]
    have stepBound := valid.stepIndexBound
    norm_num [timestampLimit,
      Nightstream.Protocol.Nebula.timestampBits, scanSlots,
      Lifecycle.claimsPerSegment, goldilocksP] at stepBound ⊢
    omega
  have stepEval := singleton_eval
    (layout.product.claim.counterValueColumn .stepIndex)
    (timestampLimit * scanSlots) stepTermBound
  have slotBound : timestampLimit * layout.slot.val < goldilocksP := by
    have slotLt := layout.slot.isLt
    norm_num [timestampLimit,
      Nightstream.Protocol.Nebula.timestampBits, scanSlots,
      goldilocksP] at slotLt ⊢
    omega
  have slotEval := constant_eval one
    (timestampLimit * layout.slot.val) slotBound
  simp only [Layout.snapshotPacked]
  rw [lcEval_append, lcEval_append, timestamp_eval canonical one holds,
    stepEval, slotEval]
  have tupleBound := packedNat_lt_goldilocks
    (valid.tuple_in_range layout.slot)
  simp only [packedNat, timestampRadix, SnapshotSlot.ValidAt.boundedTuple,
    SnapshotSlot.Value.tuple, decoded, SnapshotSlot.globalIndex] at tupleBound ⊢
  rw [stepPlaced]
  change assignment layout.aux.timestamp +
      timestampLimit * (claim.stepIndex.val * 64 + layout.slot.val) <
    goldilocksP at tupleBound
  have firstBound : assignment layout.aux.timestamp +
      timestampLimit * scanSlots * claim.stepIndex.val < goldilocksP := by
    norm_num [scanSlots, timestampLimit,
      Nightstream.Protocol.Nebula.timestampBits] at tupleBound ⊢
    omega
  rw [Nat.mod_eq_of_lt firstBound]
  have finalBound : assignment layout.aux.timestamp +
      timestampLimit * scanSlots * claim.stepIndex.val +
        timestampLimit * layout.slot.val < goldilocksP := by
    norm_num [scanSlots, timestampLimit,
      Nightstream.Protocol.Nebula.timestampBits] at tupleBound ⊢
    omega
  rw [Nat.mod_eq_of_lt finalBound]
  simp only [scanSlots]
  ring

/-- One snapshot product entry is the exact structural typed record. -/
theorem gate_represents
    {layout : Layout} {assignment : Nat → Nat} {claim : Claim}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (parsed : MemoryClaimRows.ParsedColumnsMatch
      layout.product.claim assignment claim)
    (holds : Satisfies (rows layout) assignment)
    (valid : SnapshotSlot.ValidAt (decoded layout assignment)
      claim.stepIndex.val (boundaryValue claim layout.role))
    (repetition : Fin 2) :
    GateRepresents assignment
      (snapshotEntry layout.product repetition layout.role layout.slot)
      (some (valid.boundedTuple layout.slot)) := by
  apply GateRepresents.always
  · rfl
  · simpa using packed_eval canonical one parsed holds valid
  · simpa [snapshotEntry, SnapshotSlot.ValidAt.boundedTuple,
      SnapshotSlot.Value.tuple, decoded] using
      value_eval canonical one holds

end Nightstream.Implementation.Nebula.SnapshotSlotRows
