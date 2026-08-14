import Nightstream.Implementation.Nebula.Memory.Claim.PoseidonBinding
import Nightstream.Implementation.Nebula.Memory.Claim.Rows
import Nightstream.Implementation.R1CS.Core.ConstantPins

/-!
Contract: exact 91-field input frame for the V2 memory-claim Poseidon2
digest.

Assurance tier: implementation model.

Owns the eight fixed domain/profile columns, the exact ordered selection of
all 83 typed memory-claim columns, row soundness to the lossless `frame`
definition, and honest local completeness.

Does not own the memory-claim parser rows, the Poseidon2 sponge trace,
absolute generated columns, or Rust conformance.

Emits constraints: yes, for the eight fixed prefix values only.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.MemoryClaimHashFrameRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.Nebula.MemoryClaimCodec
open Nightstream.Implementation.Nebula.MemoryClaimPoseidonBinding
open Nightstream.Protocol.Nebula

structure Layout where
  claim : MemoryClaimRows.Layout
  prefixStart : Nat

def Layout.prefixColumn (layout : Layout) (index : Nat) : Nat :=
  layout.prefixStart + index

def prefixPins (layout : Layout) : List (Nat × Nat) :=
  [ (layout.prefixColumn 0, domainTag)
  , (layout.prefixColumn 1, frameVersion)
  , (layout.prefixColumn 2, 2)
  , (layout.prefixColumn 3, 2)
  , (layout.prefixColumn 4, 1)
  , (layout.prefixColumn 5, 1)
  , (layout.prefixColumn 6, MemoryWireGeometry.stepPublicBits)
  , (layout.prefixColumn 7, MemoryClaimCodec.schema.length)
  ]

def prefixColumns (layout : Layout) : List Nat :=
  (prefixPins layout).map Prod.fst

def prefixValues (layout : Layout) : List Nat :=
  (prefixPins layout).map Prod.snd

theorem prefixValues_exact (layout : Layout) :
    prefixValues layout = fixedPrefix := by
  change
    [domainTag, frameVersion, 2, 2, 1, 1,
      MemoryWireGeometry.stepPublicBits, MemoryClaimCodec.schema.length] =
      fixedPrefix
  rw [fixedPrefix_exact, MemoryWireGeometry.stepPublicBits_exact,
    schema_length_exact]
  norm_num [domainTag, frameVersion]

theorem prefixColumns_length (layout : Layout) :
    (prefixColumns layout).length = 8 := by
  simp [prefixColumns, prefixPins]

theorem prefixPins_valuesCanonical (layout : Layout) :
    ConstantPins.ValuesCanonical (prefixPins layout) := by
  intro pin member
  simp only [prefixPins, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    norm_num [goldilocksP, domainTag, frameVersion,
      MemoryWireGeometry.stepPublicBits_exact, schema_length_exact]

def Layout.claimColumn (layout : Layout) : FieldTag → Nat
  | .segmentIndex => layout.claim.counterValueColumn .segmentIndex
  | .stepIndex => layout.claim.counterValueColumn .stepIndex
  | .timestampIn => layout.claim.counterValueColumn .timestampIn
  | .timestampOut => layout.claim.counterValueColumn .timestampOut
  | .segmentStartTimestamp =>
      layout.claim.counterValueColumn .segmentStartTimestamp
  | .segmentEndTimestamp =>
      layout.claim.counterValueColumn .segmentEndTimestamp
  | .activeAccessCount => layout.claim.counterValueColumn .activeAccessCount
  | .challenge repetition coordinate limb =>
      Relabel.column
        (layout.claim.fieldColumnMap
          (.challenge repetition coordinate limb)) CanonicalU64.varCol
  | .product side repetition role limb =>
      Relabel.column
        (layout.claim.fieldColumnMap
          (.product side repetition role limb)) CanonicalU64.varCol
  | .root stage role lane =>
      Relabel.column
        (layout.claim.fieldColumnMap (.root stage role lane))
        CanonicalU64.varCol

def claimColumns (layout : Layout) : List Nat :=
  MemoryClaimCodec.schema.map layout.claimColumn

theorem claimColumns_length (layout : Layout) :
    (claimColumns layout).length = 83 := by
  simp [claimColumns, schema_length_exact]

def inputColumns (layout : Layout) : List Nat :=
  prefixColumns layout ++ claimColumns layout

theorem inputColumns_length (layout : Layout) :
    (inputColumns layout).length = 91 := by
  rw [inputColumns, List.length_append, prefixColumns_length,
    claimColumns_length]

def rows (layout : Layout) : List Row :=
  ConstantPins.rows (prefixPins layout)

theorem rows_length_exact (layout : Layout) :
    (rows layout).length = 8 := by
  simp [rows, ConstantPins.rows, prefixPins]

private theorem selfIncluded (program : List Row) :
    rowsIncluded program program = true := by
  rw [rowsIncluded, List.all_eq_true]
  intro row member
  exact decide_eq_true member

private theorem prefix_facts
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    ∀ pin ∈ prefixPins layout, assignment pin.1 = pin.2 :=
  ConstantPins.sound (prefixPins_valuesCanonical layout)
    (selfIncluded (rows layout)) canonical one holds

theorem prefix_column_values
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    (prefixColumns layout).map assignment = fixedPrefix := by
  rw [← prefixValues_exact layout]
  simp only [prefixColumns, prefixValues, List.map_map]
  apply List.map_congr_left
  intro pin member
  exact prefix_facts canonical one holds pin member

theorem claimColumn_value
    {layout : Layout} {assignment : Nat → Nat}
    {claim : MemoryClaimCodec.Claim}
    (parsed : MemoryClaimRows.ParsedColumnsMatch layout.claim assignment claim) :
    ∀ tag, assignment (layout.claimColumn tag) = claim.fieldValue tag := by
  intro tag
  cases tag with
  | segmentIndex =>
      simpa [Layout.claimColumn,
        MemoryClaimCounterRows.Counter.claimValue_eq_tag] using
        parsed.counters .segmentIndex
  | stepIndex =>
      simpa [Layout.claimColumn,
        MemoryClaimCounterRows.Counter.claimValue_eq_tag] using
        parsed.counters .stepIndex
  | timestampIn =>
      simpa [Layout.claimColumn,
        MemoryClaimCounterRows.Counter.claimValue_eq_tag] using
        parsed.counters .timestampIn
  | timestampOut =>
      simpa [Layout.claimColumn,
        MemoryClaimCounterRows.Counter.claimValue_eq_tag] using
        parsed.counters .timestampOut
  | segmentStartTimestamp =>
      simpa [Layout.claimColumn,
        MemoryClaimCounterRows.Counter.claimValue_eq_tag] using
        parsed.counters .segmentStartTimestamp
  | segmentEndTimestamp =>
      simpa [Layout.claimColumn,
        MemoryClaimCounterRows.Counter.claimValue_eq_tag] using
        parsed.counters .segmentEndTimestamp
  | activeAccessCount =>
      simpa [Layout.claimColumn,
        MemoryClaimCounterRows.Counter.claimValue_eq_tag] using
        parsed.counters .activeAccessCount
  | challenge repetition coordinate limb =>
      simpa [Layout.claimColumn, MemoryClaimFieldRows.Slot.tag] using
        parsed.fields (.challenge repetition coordinate limb)
  | product side repetition role limb =>
      simpa [Layout.claimColumn, MemoryClaimFieldRows.Slot.tag] using
        parsed.fields (.product side repetition role limb)
  | root stage role lane =>
      simpa [Layout.claimColumn, MemoryClaimFieldRows.Slot.tag] using
        parsed.fields (.root stage role lane)

theorem claim_column_values
    {layout : Layout} {assignment : Nat → Nat}
    {claim : MemoryClaimCodec.Claim}
    (parsed : MemoryClaimRows.ParsedColumnsMatch layout.claim assignment claim) :
    (claimColumns layout).map assignment = claimFields claim := by
  simp only [claimColumns, claimFields, List.map_map]
  apply List.map_congr_left
  intro tag _member
  exact claimColumn_value parsed tag

/-- The ordered 91 assigned columns are the exact lossless frame of the same
typed memory suffix that the strict full-claim parser returned. -/
theorem input_column_values
    {layout : Layout} {assignment : Nat → Nat}
    {claim : MemoryClaimCodec.Claim}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (parsed : MemoryClaimRows.ParsedColumnsMatch layout.claim assignment claim)
    (holds : Satisfies (rows layout) assignment) :
    (inputColumns layout).map assignment = frame claim := by
  rw [inputColumns, List.map_append,
    prefix_column_values canonical one holds,
    claim_column_values parsed]
  rfl

structure Honest (layout : Layout) (assignment : Nat → Nat) : Prop where
  prefixPlaced : ∀ pin ∈ prefixPins layout, assignment pin.1 = pin.2

theorem rows_complete
    {layout : Layout} {assignment : Nat → Nat}
    (one : assignment 0 = 1)
    (honest : Honest layout assignment) :
    Satisfies (rows layout) assignment := by
  exact ConstantPins.complete (prefixPins_valuesCanonical layout) one
    honest.prefixPlaced

end Nightstream.Implementation.Nebula.MemoryClaimHashFrameRows
