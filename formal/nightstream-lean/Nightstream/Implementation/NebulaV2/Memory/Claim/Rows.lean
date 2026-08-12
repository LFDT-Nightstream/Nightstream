import Nightstream.Implementation.NebulaV2.Core.LessThanConstantLinkedRows
import Nightstream.Implementation.NebulaV2.Memory.Claim.Parser
import Nightstream.Implementation.NebulaV2.Core.PublicBitBlock

/-!
Contract: complete public-bit row bridge for one exact V2 fresh-claim memory
block.

Assurance tier: implementation model.

Owns the 4,980 consecutive authority-bearing public bits, all seven narrow
counter blocks, the strict `step_index < 1088` check without duplicate value
rows, all 76 canonical Goldilocks limb blocks, and derivation of every typed
circuit value from the fail-closed parser result.

Does not own the enclosing full CCS claim, absolute generated columns,
state-transition rows, or Rust/container refinement.

Emits constraints: yes.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.MemoryClaimRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.NebulaV2.MemoryClaimCodec
open Nightstream.Implementation.NebulaV2.MemoryClaimCounterRows
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.Protocol.NebulaV2.MemoryWireGeometry

structure Layout where
  publicBitStart : Nat
  counterValueColumn : Counter → Nat
  stepSlackColumn : Nat
  stepSlackBitStart : Nat
  fieldColumnMap : MemoryClaimFieldRows.Slot → List Nat
  fieldMapsConstantOne : ∀ slot,
    Relabel.column (fieldColumnMap slot) 0 = 0

def Layout.publicBits (layout : Layout) : PublicBitBlock.Layout :=
  { publicBitStart := layout.publicBitStart }

def Layout.counters (layout : Layout) : MemoryClaimCounterRows.Layout :=
  { publicBitStart := layout.publicBitStart
    valueColumn := layout.counterValueColumn }

def Layout.stepLimit (layout : Layout) : LessThanConstantLinkedRows.Layout :=
  { width := stepIndexBits
    limit := Lifecycle.claimsPerSegment
    valueColumn := layout.counterValueColumn .stepIndex
    slackColumn := layout.stepSlackColumn
    slackBitStart := layout.stepSlackBitStart }

theorem Layout.stepLimit_valid (layout : Layout) : layout.stepLimit.Valid where
  limitPositive := by simp [Layout.stepLimit]; decide
  limitFits := by simp [Layout.stepLimit]; decide
  sumFits := by simp [Layout.stepLimit]; decide

def Layout.fields (layout : Layout) : MemoryClaimFieldRows.Layout :=
  { publicBitStart := layout.publicBitStart
    columnMap := layout.fieldColumnMap
    mapsConstantOne := layout.fieldMapsConstantOne }

def rows (layout : Layout) : List Row :=
  MemoryClaimCounterRows.rows layout.counters ++
    LessThanConstantLinkedRows.rows layout.stepLimit ++
    MemoryClaimFieldRows.rows layout.fields

theorem rows_length_exact (layout : Layout) :
    (rows layout).length = 10244 := by
  rw [rows, List.length_append, List.length_append,
    MemoryClaimCounterRows.rows_length_exact,
    LessThanConstantLinkedRows.rows_length,
    MemoryClaimFieldRows.rows_length_exact]
  simp [Layout.stepLimit, stepIndexBits]

private theorem counter_rows_hold
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies (MemoryClaimCounterRows.rows layout.counters) assignment := by
  intro row member
  exact holds row (by simp [rows, member])

private theorem step_rows_hold
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies (LessThanConstantLinkedRows.rows layout.stepLimit)
      assignment := by
  intro row member
  exact holds row (by simp [rows, member])

private theorem field_rows_hold
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies (MemoryClaimFieldRows.rows layout.fields) assignment := by
  intro row member
  exact holds row (by simp [rows, member])

private theorem one_counter_rows_hold
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) (counter : Counter) :
    Satisfies
      (BoundedWordRows.rows (layout.counters.word counter)) assignment := by
  have allCounters := counter_rows_hold holds
  rw [MemoryClaimCounterRows.rows] at allCounters
  exact (satisfies_flatten_iff _ _).mp allCounters _
    (List.mem_map.mpr ⟨counter, counter.mem_all, rfl⟩)

theorem counter_digits_eq_block
    {layout : Layout} {assignment : Nat → Nat}
    {block : MemoryClaimParser.Block}
    (placed : PublicBitBlock.Placed layout.publicBits assignment block)
    (counter : Counter) :
    (layout.counters.word counter).digits assignment =
      (MemoryClaimParser.counterWord block counter).val := by
  symm
  change
    (FixedBits.slice block counter.bitOffset counter.width _).val =
      (List.range counter.width).map fun index =>
        assignment
          (layout.publicBitStart + counter.bitOffset + index)
  exact PublicBitBlock.slice_eq_columns placed counter.bitOffset counter.width _

/-- Counter value columns are derived from exact public bits and satisfying
rows. No typed claim-placement premise is used. -/
theorem counter_column_eq_parsed_word
    {layout : Layout} {assignment : Nat → Nat}
    {block : MemoryClaimParser.Block}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : PublicBitBlock.Placed layout.publicBits assignment block)
    (holds : Satisfies (rows layout) assignment)
    (counter : Counter) :
    assignment (layout.counterValueColumn counter) =
      MemoryClaimParser.counterValue block counter := by
  have recomposed := BoundedWordRows.recomposition_sound
    counter.fitsGoldilocks canonical one
    (one_counter_rows_hold holds counter)
  simp only [Layout.counters, MemoryClaimCounterRows.Layout.word] at recomposed
  calc
    assignment (layout.counterValueColumn counter) =
        BoundedWordRows.decoded
          { width := counter.width
            valueColumn := layout.counterValueColumn counter
            bitStart := layout.publicBitStart + counter.bitOffset }
          assignment := recomposed
    _ = Nat.ofDigits 2
        ((layout.counters.word counter).digits assignment) := rfl
    _ = Nat.ofDigits 2 (MemoryClaimParser.counterWord block counter).val :=
      congrArg (Nat.ofDigits 2) (counter_digits_eq_block placed counter)
    _ = MemoryClaimParser.counterValue block counter := rfl

theorem strict_step_bound_of_rows
    {layout : Layout} {assignment : Nat → Nat}
    {block : MemoryClaimParser.Block}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : PublicBitBlock.Placed layout.publicBits assignment block)
    (holds : Satisfies (rows layout) assignment) :
    MemoryClaimParser.counterValue block .stepIndex <
      Lifecycle.claimsPerSegment := by
  have valueBound := BoundedWordRows.value_lt_twoPower
    (Counter.fitsGoldilocks .stepIndex) canonical one
    (one_counter_rows_hold holds .stepIndex)
  have strict := LessThanConstantLinkedRows.value_lt_limit
    layout.stepLimit_valid valueBound canonical one (step_rows_hold holds)
  simp only [Layout.stepLimit] at strict
  rw [counter_column_eq_parsed_word canonical one placed holds .stepIndex]
    at strict
  exact strict

theorem counters_place_parsed_claim
    {layout : Layout} {assignment : Nat → Nat}
    {block : MemoryClaimParser.Block} {claim : Claim}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : PublicBitBlock.Placed layout.publicBits assignment block)
    (holds : Satisfies (rows layout) assignment)
    (accepted : MemoryClaimParser.parse block = some claim) :
    MemoryClaimCounterRows.Placed layout.counters assignment claim := by
  rcases MemoryClaimParser.parse_some_bound_and_fields accepted with
    ⟨stepBound, allCanonical, claimEqual⟩
  subst claim
  intro counter
  change assignment (layout.counterValueColumn counter) =
    counter.claimValue (MemoryClaimParser.decodedClaim block stepBound)
  rw [counter_column_eq_parsed_word canonical one placed holds counter]
  cases counter <;> rfl

theorem fields_place_parser_words
    {layout : Layout} {assignment : Nat → Nat}
    {block : MemoryClaimParser.Block}
    (placed : PublicBitBlock.Placed layout.publicBits assignment block) :
    MemoryClaimFieldRows.Places layout.fields assignment
      (MemoryClaimParser.rawWords block) := by
  intro slot
  change
    (MemoryClaimParser.fieldWord block slot).val =
      CanonicalFieldSchemaRows.rawDigits layout.fields.schema assignment slot
  change
    (FixedBits.slice block slot.bitOffset CanonicalFieldBits.bitCount
      (MemoryClaimParser.field_slice_fits slot)).val =
      CanonicalFieldSchemaRows.rawDigits layout.fields.schema assignment slot
  rw [PublicBitBlock.slice_eq_columns placed slot.bitOffset
    CanonicalFieldBits.bitCount (MemoryClaimParser.field_slice_fits slot)]
  simp [PublicBitBlock.sliceColumns,
    CanonicalFieldSchemaRows.rawDigits,
    MemoryClaimFieldRows.Layout.schema,
    MemoryClaimFieldRows.Layout.rawColumns, Layout.publicBits, Layout.fields,
    List.getD]
  intro index bound
  have rangeBound : index < (List.range CanonicalFieldBits.bitCount).length := by
    simpa using bound
  rw [List.getElem?_eq_getElem rangeBound]
  simp

/-- Every typed field column is fixed by the accepted parser output and the
same authority-bearing public bits. -/
theorem field_columns_eq_parsed_claim
    {layout : Layout} {assignment : Nat → Nat}
    {block : MemoryClaimParser.Block} {claim : Claim}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : PublicBitBlock.Placed layout.publicBits assignment block)
    (holds : Satisfies (rows layout) assignment)
    (accepted : MemoryClaimParser.parse block = some claim) :
    ∀ slot,
      assignment
          (Relabel.column (layout.fieldColumnMap slot) CanonicalU64.varCol) =
        claim.fieldValue slot.tag := by
  exact MemoryClaimFieldRows.typed_columns_of_rows canonical one
    (field_rows_hold holds) (fields_place_parser_words placed)
    (MemoryClaimParser.parse_native_parses accepted)

structure ParsedColumnsMatch
    (layout : Layout) (assignment : Nat → Nat) (claim : Claim) : Prop where
  counters : MemoryClaimCounterRows.Placed layout.counters assignment claim
  fields : ∀ slot,
    assignment
        (Relabel.column (layout.fieldColumnMap slot) CanonicalU64.varCol) =
      claim.fieldValue slot.tag
  stepStrict : claim.stepIndex.val < Lifecycle.claimsPerSegment
  canonical : claim.Canonical

/-- Complete local soundness bridge from one raw public block. The only
non-row premise is the exact per-position placement of the parser bits. -/
theorem parsed_columns_match
    {layout : Layout} {assignment : Nat → Nat}
    {block : MemoryClaimParser.Block} {claim : Claim}
    (canonicalAssignment : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : PublicBitBlock.Placed layout.publicBits assignment block)
    (holds : Satisfies (rows layout) assignment)
    (accepted : MemoryClaimParser.parse block = some claim) :
    ParsedColumnsMatch layout assignment claim where
  counters := counters_place_parsed_claim canonicalAssignment one placed
    holds accepted
  fields := field_columns_eq_parsed_claim canonicalAssignment one placed
    holds accepted
  stepStrict := by
    have strict := strict_step_bound_of_rows canonicalAssignment one placed holds
    rcases MemoryClaimParser.parse_some_bound_and_fields accepted with
      ⟨stepBound, allCanonical, claimEqual⟩
    subst claim
    exact strict
  canonical := MemoryClaimParser.parse_claim_canonical accepted

end Nightstream.Implementation.NebulaV2.MemoryClaimRows
