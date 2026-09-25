import NightstreamFPrime.Layout.PiRLC.v1_1.CombinationFamily
import NightstreamFPrime.Lifecycle.PiRLC.v1_1.CommitmentCombination

/-!
Owns the physical PiRLC commitment-combination family.

This owner selects the exact 18-row logical family. It adds no transcript,
public-input, evaluation, copy, or boundary rows.
-/

namespace NightstreamFPrime.Layout.PiRLC.v1_1.CommitmentCombination

open NightstreamFPrime.Circuit

namespace Logical

abbrev Interface :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.CommitmentCombination.Interface
abbrev familyInterface :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.CommitmentCombination.familyInterface
abbrev Assumptions :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.CommitmentCombination.Assumptions
abbrev SpecHolds :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.CommitmentCombination.SpecHolds

end Logical

abbrev logicalConstraints (interface : Logical.Interface) (offset : Nat) :=
  CombinationFamily.logicalConstraints (Logical.familyInterface interface) offset

abbrev physicalFreshColumnCount (interface : Logical.Interface) (offset : Nat) :=
  CombinationFamily.physicalFreshColumnCount
    (Logical.familyInterface interface) offset

abbrev physicalRowCount (interface : Logical.Interface) (offset : Nat) :=
  CombinationFamily.physicalRowCount (Logical.familyInterface interface) offset

abbrev physicalPrivateColumnCount (interface : Logical.Interface)
    (offset : Nat) :=
  CombinationFamily.physicalPrivateColumnCount
    (Logical.familyInterface interface) offset

abbrev plan (interface : Logical.Interface) (offset : Nat) :=
  CombinationFamily.plan (Logical.familyInterface interface) offset

abbrev PhysicalHolds (interface : Logical.Interface) (offset : Nat)
    (env : Env) :=
  CombinationFamily.PhysicalHolds (Logical.familyInterface interface) offset env

theorem totalFreshCount_eq (interface : Logical.Interface) (offset : Nat) :
    R1CS.totalFreshCount (logicalConstraints interface offset) =
      physicalFreshColumnCount interface offset :=
  CombinationFamily.totalFreshCount_eq_deltas
    (Logical.familyInterface interface) offset

theorem totalRowCount_eq (interface : Logical.Interface) (offset : Nat) :
    R1CS.totalRowCount (logicalConstraints interface offset) =
      physicalRowCount interface offset :=
  CombinationFamily.totalRowCount_eq_deltas
    (Logical.familyInterface interface) offset

theorem physicalRowCount_eq (interface : Logical.Interface) (offset : Nat) :
    (plan interface offset).rowCount = physicalRowCount interface offset :=
  CombinationFamily.physicalRowCount_eq
    (Logical.familyInterface interface) offset

theorem physical_implies_spec (interface : Logical.Interface) (offset : Nat)
    (env : Env) (assumptions : Logical.Assumptions interface offset env)
    (physical : PhysicalHolds interface offset env) :
    Logical.SpecHolds interface offset env :=
  CombinationFamily.physical_implies_canonical
    (Logical.familyInterface interface) offset env assumptions physical

theorem physical_complete (interface : Logical.Interface) (offset : Nat)
    (env : Env) (assumptions : Logical.Assumptions interface offset env)
    (specification : Logical.SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
          (physicalPrivateColumnCount interface offset) ∧
      PhysicalHolds interface offset completed :=
  CombinationFamily.physical_complete
    (Logical.familyInterface interface) offset env assumptions specification

end NightstreamFPrime.Layout.PiRLC.v1_1.CommitmentCombination
