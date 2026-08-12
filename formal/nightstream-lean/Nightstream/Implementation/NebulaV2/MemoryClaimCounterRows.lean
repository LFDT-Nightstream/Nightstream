import Nightstream.Implementation.NebulaV2.BoundedWordRows
import Nightstream.Implementation.NebulaV2.MemoryClaimCodec

/-!
Contract: exact narrow-counter rows for the V2 fresh-claim memory block.

Assurance tier: implementation model.

Owns the first 116 public-bit positions, one bounded-word block for each of
the seven counters, derivation of `Claim.Canonical` from row satisfaction,
and equality between every satisfying counter bit word and the independent
claim codec.

Does not own the 64-bit Goldilocks canonicality blocks, the exact
`step_index < 1088` decoder check, absolute generated columns, or full claim
verification.

Emits constraints: yes.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.MemoryClaimCounterRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.NebulaV2.MemoryClaimCodec
open Nightstream.Protocol.NebulaV2.MemoryWireGeometry

inductive Counter where
  | segmentIndex
  | stepIndex
  | timestampIn
  | timestampOut
  | segmentStartTimestamp
  | segmentEndTimestamp
  | activeAccessCount
deriving DecidableEq, Repr

def Counter.all : List Counter :=
  [.segmentIndex, .stepIndex, .timestampIn, .timestampOut,
    .segmentStartTimestamp, .segmentEndTimestamp, .activeAccessCount]

def Counter.width : Counter → Nat
  | .segmentIndex => segmentIndexBits
  | .stepIndex => stepIndexBits
  | .timestampIn => timestampBits
  | .timestampOut => timestampBits
  | .segmentStartTimestamp => timestampBits
  | .segmentEndTimestamp => timestampBits
  | .activeAccessCount => stepActiveAccessCountBits

/-- Exact offset inside the 4,980-bit claim block. -/
def Counter.bitOffset : Counter → Nat
  | .segmentIndex => 0
  | .stepIndex => 7
  | .timestampIn => 18
  | .timestampOut => 41
  | .segmentStartTimestamp => 64
  | .segmentEndTimestamp => 87
  | .activeAccessCount => 110

def Counter.claimValue (counter : Counter) (claim : Claim) : Nat :=
  match counter with
  | .segmentIndex => claim.segmentIndex
  | .stepIndex => claim.stepIndex.val
  | .timestampIn => claim.timestampIn
  | .timestampOut => claim.timestampOut
  | .segmentStartTimestamp => claim.segmentStartTimestamp
  | .segmentEndTimestamp => claim.segmentEndTimestamp
  | .activeAccessCount => claim.activeAccessCount

def Counter.tag : Counter → MemoryClaimCodec.FieldTag
  | .segmentIndex => .segmentIndex
  | .stepIndex => .stepIndex
  | .timestampIn => .timestampIn
  | .timestampOut => .timestampOut
  | .segmentStartTimestamp => .segmentStartTimestamp
  | .segmentEndTimestamp => .segmentEndTimestamp
  | .activeAccessCount => .activeAccessCount

theorem Counter.width_eq_tag (counter : Counter) :
    counter.width = counter.tag.bitWidth := by
  cases counter <;> rfl

theorem Counter.bitOffset_eq_tag (counter : Counter) :
    counter.bitOffset = counter.tag.bitOffset := by
  cases counter <;> decide

theorem Counter.claimValue_eq_tag (counter : Counter) (claim : Claim) :
    counter.claimValue claim = claim.fieldValue counter.tag := by
  cases counter <;> rfl

theorem Counter.mem_all (counter : Counter) : counter ∈ Counter.all := by
  cases counter <;> simp [Counter.all]

theorem Counter.fitsGoldilocks (counter : Counter) :
    2 ^ counter.width ≤ goldilocksP := by
  cases counter <;> decide

structure Layout where
  publicBitStart : Nat
  valueColumn : Counter → Nat

def Layout.word (layout : Layout) (counter : Counter) :
    BoundedWordRows.Layout :=
  { width := counter.width
    valueColumn := layout.valueColumn counter
    bitStart := layout.publicBitStart + counter.bitOffset }

def rows (layout : Layout) : List Row :=
  (Counter.all.map fun counter =>
    BoundedWordRows.rows (layout.word counter)).flatten

def Placed (layout : Layout) (assignment : Nat → Nat)
    (claim : Claim) : Prop :=
  ∀ counter, assignment (layout.valueColumn counter) =
    counter.claimValue claim

private theorem counter_rows_hold
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment)
    (counter : Counter) :
    Satisfies (BoundedWordRows.rows (layout.word counter)) assignment := by
  rw [rows] at holds
  exact (satisfies_flatten_iff _ _).mp holds _
    (List.mem_map.mpr
      ⟨counter, counter.mem_all, rfl⟩)

theorem counter_bound
    {layout : Layout} {assignment : Nat → Nat} {claim : Claim}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : Placed layout assignment claim)
    (holds : Satisfies (rows layout) assignment)
    (counter : Counter) :
    counter.claimValue claim < 2 ^ counter.width := by
  have bounded := BoundedWordRows.value_lt_twoPower
    (counter.fitsGoldilocks) canonical one
    (counter_rows_hold holds counter)
  simp only [Layout.word] at bounded
  rw [placed counter] at bounded
  exact bounded

/-- Counter canonicality is derived from the concrete rows. It is not a
field of the call-site certificate. -/
theorem claim_canonical_of_rows
    {layout : Layout} {assignment : Nat → Nat} {claim : Claim}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : Placed layout assignment claim)
    (holds : Satisfies (rows layout) assignment) :
    claim.Canonical where
  segmentIndex := counter_bound canonical one placed holds .segmentIndex
  timestampIn := counter_bound canonical one placed holds .timestampIn
  timestampOut := counter_bound canonical one placed holds .timestampOut
  segmentStartTimestamp :=
    counter_bound canonical one placed holds .segmentStartTimestamp
  segmentEndTimestamp :=
    counter_bound canonical one placed holds .segmentEndTimestamp
  activeAccessCount :=
    counter_bound canonical one placed holds .activeAccessCount

/-- Each satisfying counter block is the exact word used by the independent
claim codec. -/
theorem counter_digits_eq_codec
    {layout : Layout} {assignment : Nat → Nat} {claim : Claim}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : Placed layout assignment claim)
    (holds : Satisfies (rows layout) assignment)
    (counter : Counter) :
    (layout.word counter).digits assignment =
      WasmStateCodec.encodeWord counter.width (counter.claimValue claim) := by
  have exactDigits := BoundedWordRows.digits_eq_encodeWord
    (counter.fitsGoldilocks) canonical one
    (counter_rows_hold holds counter)
  simp only [Layout.word] at exactDigits
  rw [placed counter] at exactDigits
  exact exactDigits

theorem rows_length_exact (layout : Layout) :
    (rows layout).length = 123 := by
  simp [rows, Counter.all, BoundedWordRows.rows_length, Layout.word,
    Counter.width, segmentIndexBits, stepIndexBits, timestampBits,
    stepActiveAccessCountBits]

/-- Artifact-facing row-inclusion certificate. Canonical counter bounds are
conclusions of `sound`; they are not certificate fields. -/
structure CallSite (programRows : List Row) (assignment : Nat → Nat)
    (claim : Claim) where
  layout : Layout
  rowsIncluded : rowsIncluded (rows layout) programRows = true
  canonicalAssignment : ∀ column, assignment column < goldilocksP
  one : assignment 0 = 1
  placed : Placed layout assignment claim

theorem CallSite.sound
    {programRows : List Row} {assignment : Nat → Nat} {claim : Claim}
    (site : CallSite programRows assignment claim)
    (satisfies : Satisfies programRows assignment) :
    claim.Canonical := by
  apply claim_canonical_of_rows site.canonicalAssignment site.one site.placed
  intro row member
  exact satisfies row (rowsIncluded_sound site.rowsIncluded row member)

end Nightstream.Implementation.NebulaV2.MemoryClaimCounterRows
