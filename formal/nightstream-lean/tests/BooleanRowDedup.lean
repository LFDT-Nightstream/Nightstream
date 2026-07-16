import Nightstream.Implementation.R1CS.Core.BooleanRowDedup

namespace NightstreamTests.BooleanRowDedup

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.BooleanRowDedup

private def encodedTwo : Nat → Nat
  | 0 => 1
  | 9 => 2
  | _ => 0

/-- The common encoded bitness gate and the exactly substituted source row
reject the same non-Boolean slot value. -/
example :
    (¬ RowHolds encodedTwo (bitRow 9)) ∧
      (¬ RowHolds encodedTwo
        (LinearSubstitution.row (singletonSlotExpansion 4 9) (bitRow 4))) := by
  native_decide

/-- The reusable theorem is independent of the concrete source/slot indices. -/
example (encoded : Nat → Nat) :
    RowHolds encoded
        (LinearSubstitution.row (singletonSlotExpansion 4 9) (bitRow 4)) ↔
      RowHolds encoded (bitRow 9) :=
  substituted_bitRow_iff_slot_bitRow (by decide) encoded

/-- The exact A/B-exchanged row accepted by the Rust classifier has the same
acceptance predicate as the common slot bitness gate. -/
example (encoded : Nat → Nat) :
    RowHolds encoded
        (LinearSubstitution.row (singletonSlotExpansion 4 9)
          (swapFactors (bitRow 4))) ↔
      RowHolds encoded (bitRow 9) :=
  substituted_swappedBitRow_iff_slot_bitRow (by decide) encoded

private def nearBitRow (source : Nat) : Row :=
  ⟨[(source, 1)], [(source, 1), (0, goldilocksP - 2)], []⟩

/-- A same-column near miss is structurally different and therefore cannot
instantiate the exact-row theorem. -/
example : nearBitRow 4 ≠ bitRow 4 := by decide

end NightstreamTests.BooleanRowDedup
