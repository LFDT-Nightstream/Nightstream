import Nightstream.Implementation.R1CS.Core.LinearSubstitution

/-!
Contract: model-level equivalence for removing an exact duplicated Boolean row.

Owns: the one-column substitution theorem used when a source bit wire and its
encoded wire are the same singleton coordinate.

Does not own: detection of source row shape, production column layout, or the
decision to remove any concrete row.

Emits constraints: no.

Authority boundary: this theorem applies only after external correspondence
evidence establishes both the exact source `bitRow` and the singleton slot
map. Column metadata alone is not such evidence.

Assurance tier: model-level.

| Predicate/theorem | Rust stage | Guarantee | Assumptions | Permits row removal? |
|---|---|---|---|---|
| `substituted_bitRow_iff_slot_bitRow` | gadget-native fallback lowering | The substituted source row and common encoded bitness gate accept exactly the same assignments | exact `bitRow`; nonconstant source; singleton slot substitution | yes, only after concrete Rust correspondence |
| `substituted_swappedBitRow_iff_slot_bitRow` | gadget-native fallback lowering | The same equivalence holds when normalized source A/B factors are exchanged | exact factor exchange; nonconstant source; singleton slot substitution | yes, only after concrete Rust correspondence |
-/

namespace Nightstream.Implementation.R1CS.BooleanRowDedup

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.LinearSubstitution

/-- Map one source column to one encoded slot and leave every other column,
including the constant-one column, unchanged. -/
def singletonSlotExpansion (source slot : Nat) : ColumnExpansion :=
  fun column =>
    if column = source then [(slot, 1)] else [(column, 1)]

/-- Exchange the two multiplicative factors without changing the output LC. -/
def swapFactors (source : Row) : Row :=
  ⟨source.b, source.a, source.c⟩

theorem rowHolds_swapFactors_iff (encoded : Nat → Nat) (source : Row) :
    RowHolds encoded (swapFactors source) ↔ RowHolds encoded source := by
  simp [RowHolds, swapFactors, Nat.mul_comm]

theorem substitute_swapFactors
    (expansion : ColumnExpansion) (source : Row) :
    row expansion (swapFactors source) =
      swapFactors (row expansion source) := by
  rfl

@[simp] theorem assignment_source
    (source slot : Nat) (encoded : Nat → Nat) :
    assignment (singletonSlotExpansion source slot) encoded source =
      encoded slot % goldilocksP := by
  simp [assignment, singletonSlotExpansion, lcEval]

@[simp] theorem assignment_zero
    {source slot : Nat} (sourceNeZero : source ≠ 0) (encoded : Nat → Nat) :
    assignment (singletonSlotExpansion source slot) encoded 0 =
      encoded 0 % goldilocksP := by
  have zeroNeSource : 0 ≠ source := Ne.symm sourceNeZero
  simp [assignment, singletonSlotExpansion, lcEval, zeroNeSource]

/-- Exact singleton substitution turns the exact source bit row into the exact
common encoded bit row, not merely an algebraically similar predicate. -/
theorem substituted_bitRow_eq_slot_bitRow
    {source slot : Nat} (sourceNeZero : source ≠ 0) :
    row (singletonSlotExpansion source slot) (bitRow source) =
      bitRow slot := by
  have zeroNeSource : 0 ≠ source := Ne.symm sourceNeZero
  simp [row, bitRow, terms, scaleTerms, singletonSlotExpansion,
    zeroNeSource, goldilocksP]

/-- Once exact row-shape checking and exact singleton-slot checking have been
performed, the generic source row adds no condition beyond the common encoded
bitness gate. -/
theorem substituted_bitRow_iff_slot_bitRow
    {source slot : Nat} (sourceNeZero : source ≠ 0)
    (encoded : Nat → Nat) :
    RowHolds encoded
        (row (singletonSlotExpansion source slot) (bitRow source)) ↔
      RowHolds encoded (bitRow slot) := by
  rw [substituted_bitRow_eq_slot_bitRow sourceNeZero]

/-- The Rust matcher also accepts the exact same equation with its A/B
factors exchanged. Commutativity is explicit here rather than implicit in the
row classifier. -/
theorem substituted_swappedBitRow_iff_slot_bitRow
    {source slot : Nat} (sourceNeZero : source ≠ 0)
    (encoded : Nat → Nat) :
    RowHolds encoded
        (row (singletonSlotExpansion source slot)
          (swapFactors (bitRow source))) ↔
      RowHolds encoded (bitRow slot) := by
  rw [substitute_swapFactors,
    substituted_bitRow_eq_slot_bitRow sourceNeZero,
    rowHolds_swapFactors_iff]

end Nightstream.Implementation.R1CS.BooleanRowDedup
