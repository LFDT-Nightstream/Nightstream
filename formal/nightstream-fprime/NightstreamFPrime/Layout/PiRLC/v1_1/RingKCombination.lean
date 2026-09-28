import NightstreamFPrime.Layout.PiRLC.v1_1.CombinationFamily
import NightstreamFPrime.Lifecycle.PiRLC.v1_1.RingKCombination

/-!
Owns the physical two-cell `RingK` combination adapter.

This owner selects both extension-field cells for each block from the exact
generic family. `Eval_K` and `Eval_A` remain separate public owners above it.
-/

namespace NightstreamFPrime.Layout.PiRLC.v1_1.RingKCombination

open NightstreamFPrime.Circuit

namespace Logical

abbrev Interface (blockCount : Nat) :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.RingKCombination.Interface blockCount
abbrev familyInterface {blockCount : Nat}
    (interface : Interface blockCount) :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.RingKCombination.familyInterface
    interface
abbrev Assumptions {blockCount : Nat} (interface : Interface blockCount)
    (offset : Nat) (env : Env) :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.RingKCombination.Assumptions
    interface offset env
abbrev SpecHolds {blockCount : Nat} (interface : Interface blockCount)
    (offset : Nat) (env : Env) :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.RingKCombination.SpecHolds
    interface offset env

end Logical

abbrev logicalConstraints {blockCount : Nat}
    (interface : Logical.Interface blockCount) (offset : Nat) :=
  CombinationFamily.logicalConstraints (Logical.familyInterface interface) offset

abbrev physicalFreshColumnCount {blockCount : Nat}
    (interface : Logical.Interface blockCount) (offset : Nat) :=
  CombinationFamily.physicalFreshColumnCount
    (Logical.familyInterface interface) offset

abbrev physicalRowCount {blockCount : Nat}
    (interface : Logical.Interface blockCount) (offset : Nat) :=
  CombinationFamily.physicalRowCount (Logical.familyInterface interface) offset

abbrev physicalPrivateColumnCount {blockCount : Nat}
    (interface : Logical.Interface blockCount) (offset : Nat) :=
  CombinationFamily.physicalPrivateColumnCount
    (Logical.familyInterface interface) offset

abbrev plan {blockCount : Nat} (interface : Logical.Interface blockCount)
    (offset : Nat) :=
  CombinationFamily.plan (Logical.familyInterface interface) offset

abbrev PhysicalHolds {blockCount : Nat}
    (interface : Logical.Interface blockCount) (offset : Nat) (env : Env) :=
  CombinationFamily.PhysicalHolds (Logical.familyInterface interface) offset env

theorem totalFreshCount_eq {blockCount : Nat}
    (interface : Logical.Interface blockCount) (offset : Nat) :
    R1CS.totalFreshCount (logicalConstraints interface offset) =
      physicalFreshColumnCount interface offset :=
  CombinationFamily.totalFreshCount_eq_deltas
    (Logical.familyInterface interface) offset

theorem totalRowCount_eq {blockCount : Nat}
    (interface : Logical.Interface blockCount) (offset : Nat) :
    R1CS.totalRowCount (logicalConstraints interface offset) =
      physicalRowCount interface offset :=
  CombinationFamily.totalRowCount_eq_deltas
    (Logical.familyInterface interface) offset

theorem physicalRowCount_eq {blockCount : Nat}
    (interface : Logical.Interface blockCount) (offset : Nat) :
    (plan interface offset).rowCount = physicalRowCount interface offset :=
  CombinationFamily.physicalRowCount_eq
    (Logical.familyInterface interface) offset

theorem physical_implies_spec {blockCount : Nat}
    (interface : Logical.Interface blockCount) (offset : Nat) (env : Env)
    (assumptions : Logical.Assumptions interface offset env)
    (physical : PhysicalHolds interface offset env) :
    Logical.SpecHolds interface offset env :=
  CombinationFamily.physical_implies_canonical
    (Logical.familyInterface interface) offset env assumptions physical

theorem physical_complete {blockCount : Nat}
    (interface : Logical.Interface blockCount) (offset : Nat) (env : Env)
    (assumptions : Logical.Assumptions interface offset env)
    (specification : Logical.SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
          (physicalPrivateColumnCount interface offset) ∧
      PhysicalHolds interface offset completed :=
  CombinationFamily.physical_complete
    (Logical.familyInterface interface) offset env assumptions specification

end NightstreamFPrime.Layout.PiRLC.v1_1.RingKCombination
