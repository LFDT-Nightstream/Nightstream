import Nightstream.Implementation.Nebula.Core.BoundedWordRows
import Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.ObservedTrace

/-!
Contract: exact R1CS encoding of one unsigned 64-bit state word as the two
little-endian 32-bit field limbs used by the F-prime state-output frame.

Assurance tier: implementation model.

Owns 64 Boolean rows, two integer-safe 32-bit recomposition rows, extraction
of one bounded unsigned word, equality with the existing `u64Halves` encoder,
and honest local completeness.

Does not own the semantic source of the word, absolute generated columns, or
Rust conformance.

Emits constraints: yes.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.U64HalvesRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.Nebula.BoundedWordRows
open Nightstream.Implementation.Rust.CanonicalConformance.NativeStep

def limbBase : Nat := 2 ^ 32

structure Layout where
  lowColumn : Nat
  highColumn : Nat
  bitStart : Nat
deriving DecidableEq, Repr

def Layout.lowWord (layout : Layout) : BoundedWordRows.Layout where
  width := 32
  valueColumn := layout.lowColumn
  bitStart := layout.bitStart

def Layout.highWord (layout : Layout) : BoundedWordRows.Layout where
  width := 32
  valueColumn := layout.highColumn
  bitStart := layout.bitStart + 32

def rows (layout : Layout) : List Row :=
  BoundedWordRows.rows layout.lowWord ++
    BoundedWordRows.rows layout.highWord

theorem rows_length_exact (layout : Layout) :
    (rows layout).length = 66 := by
  simp [rows, BoundedWordRows.rows_length, Layout.lowWord, Layout.highWord]

def lowValue (layout : Layout) (assignment : Nat → Nat) : Nat :=
  BoundedWordRows.decoded layout.lowWord assignment

def highValue (layout : Layout) (assignment : Nat → Nat) : Nat :=
  BoundedWordRows.decoded layout.highWord assignment

/-- The authority-bearing integer reconstructed from all 64 Boolean bits. -/
def value (layout : Layout) (assignment : Nat → Nat) : Nat :=
  lowValue layout assignment + limbBase * highValue layout assignment

private theorem low_rows_hold
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies (BoundedWordRows.rows layout.lowWord) assignment := by
  intro row member
  exact holds row (by simp [rows, member])

private theorem high_rows_hold
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies (BoundedWordRows.rows layout.highWord) assignment := by
  intro row member
  exact holds row (by simp [rows, member])

private theorem limbBase_le_goldilocksP : limbBase ≤ goldilocksP := by
  norm_num [limbBase, goldilocksP]

private theorem lowWord_fits (layout : Layout) :
    2 ^ layout.lowWord.width ≤ goldilocksP := by
  simpa [Layout.lowWord, limbBase] using limbBase_le_goldilocksP

private theorem highWord_fits (layout : Layout) :
    2 ^ layout.highWord.width ≤ goldilocksP := by
  simpa [Layout.highWord, limbBase] using limbBase_le_goldilocksP

theorem lowValue_lt
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    lowValue layout assignment < limbBase := by
  exact BoundedWordRows.decoded_lt canonical one (low_rows_hold holds)

theorem highValue_lt
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    highValue layout assignment < limbBase := by
  exact BoundedWordRows.decoded_lt canonical one (high_rows_hold holds)

theorem value_lt_u64
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    value layout assignment < 2 ^ 64 := by
  have low := lowValue_lt canonical one holds
  have high := highValue_lt canonical one holds
  norm_num [value, limbBase] at low high ⊢
  omega

theorem low_column_eq
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    assignment layout.lowColumn = lowValue layout assignment := by
  exact BoundedWordRows.recomposition_sound (lowWord_fits layout)
    canonical one (low_rows_hold holds)

theorem high_column_eq
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    assignment layout.highColumn = highValue layout assignment := by
  exact BoundedWordRows.recomposition_sound (highWord_fits layout)
    canonical one (high_rows_hold holds)

private theorem value_mod_limbBase
    (low high : Nat) (lowBound : low < limbBase) :
    (low + limbBase * high) % limbBase = low := by
  rw [Nat.add_mul_mod_self_left]
  exact Nat.mod_eq_of_lt lowBound

private theorem value_div_limbBase
    (low high : Nat) (lowBound : low < limbBase) :
    (low + limbBase * high) / limbBase = high := by
  rw [Nat.add_mul_div_left low high (by norm_num [limbBase] : 0 < limbBase),
    Nat.div_eq_of_lt lowBound, Nat.zero_add]

/-- Satisfying rows make the two visible frame columns exactly the existing
Rust-source `u64Halves` encoding of the same 64-bit integer. -/
theorem half_column_values
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    [assignment layout.lowColumn, assignment layout.highColumn] =
      u64Halves (value layout assignment) := by
  rw [low_column_eq canonical one holds,
    high_column_eq canonical one holds]
  change
    [lowValue layout assignment, highValue layout assignment] =
      [value layout assignment % limbBase,
        value layout assignment / limbBase]
  unfold value
  rw [value_mod_limbBase _ _ (lowValue_lt canonical one holds),
    value_div_limbBase _ _ (lowValue_lt canonical one holds)]

/-- The two-limb unsigned encoding is injective. This is integer arithmetic,
not a field-binding assumption. -/
theorem u64Halves_injective : Function.Injective u64Halves := by
  intro left right equal
  change [left % limbBase, left / limbBase] =
    [right % limbBase, right / limbBase] at equal
  have lowEqual := congrArg (fun values => values.getD 0 0) equal
  have highEqual := congrArg (fun values => values.getD 1 0) equal
  simp only [List.getD_cons_zero] at lowEqual
  simp only [List.getD_cons_succ, List.getD_cons_zero] at highEqual
  calc
    left = left % limbBase + limbBase * (left / limbBase) :=
      (Nat.mod_add_div left limbBase).symm
    _ = right % limbBase + limbBase * (right / limbBase) := by
      rw [lowEqual, highEqual]
    _ = right := Nat.mod_add_div right limbBase

/-- A bounded unsigned word produces two canonical Goldilocks limbs. -/
theorem u64Halves_canonical
    {word : Nat} (wordBound : word < 2 ^ 64) :
    ∀ value ∈ u64Halves word, value < goldilocksP := by
  intro value member
  change value ∈ [word % limbBase, word / limbBase] at member
  simp only [List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl
  · exact (Nat.mod_lt word (by norm_num [limbBase])).trans_le
      limbBase_le_goldilocksP
  · have highBound : word / limbBase < limbBase := by
      apply Nat.div_lt_of_lt_mul
      simpa [limbBase] using wordBound
    exact highBound.trans_le limbBase_le_goldilocksP

structure Honest (layout : Layout) (assignment : Nat → Nat)
    (word : Nat) : Prop where
  wordBound : word < 2 ^ 64
  low : BoundedWordRows.Honest layout.lowWord assignment (word % limbBase)
  high : BoundedWordRows.Honest layout.highWord assignment (word / limbBase)

/-- Exact honest limb placement decodes to the same unsigned 64-bit word. -/
theorem value_eq_of_honest
    {layout : Layout} {assignment : Nat → Nat} {word : Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (honest : Honest layout assignment word) :
    value layout assignment = word := by
  have lowRows := BoundedWordRows.rows_complete (lowWord_fits layout) one
    honest.low
  have highRows := BoundedWordRows.rows_complete (highWord_fits layout) one
    honest.high
  have lowExact : lowValue layout assignment = word % limbBase := by
    calc
      lowValue layout assignment = assignment layout.lowColumn :=
        (BoundedWordRows.recomposition_sound (lowWord_fits layout)
          canonical one lowRows).symm
      _ = word % limbBase := honest.low.valuePlaced
  have highExact : highValue layout assignment = word / limbBase := by
    calc
      highValue layout assignment = assignment layout.highColumn :=
        (BoundedWordRows.recomposition_sound (highWord_fits layout)
          canonical one highRows).symm
      _ = word / limbBase := honest.high.valuePlaced
  rw [value, lowExact, highExact]
  exact Nat.mod_add_div word limbBase

theorem rows_complete
    {layout : Layout} {assignment : Nat → Nat} {word : Nat}
    (one : assignment 0 = 1)
    (honest : Honest layout assignment word) :
    Satisfies (rows layout) assignment := by
  intro row member
  rw [rows, List.mem_append] at member
  rcases member with lowMember | highMember
  · exact BoundedWordRows.rows_complete (lowWord_fits layout) one
      honest.low row lowMember
  · exact BoundedWordRows.rows_complete (highWord_fits layout) one
      honest.high row highMember

end Nightstream.Implementation.Nebula.U64HalvesRows
