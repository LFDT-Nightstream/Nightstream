import NightstreamFPrime.Layout.PiRLC.v1_1.RingKCombination
import NightstreamFPrime.Lifecycle.PiRLC.v1_1.EvalACombination

/-!
Owns the physical PiRLC matrix evaluation combinations.

This is the separate 14-block `Eval_A` owner. It shares only the two-cell
`RingK` adapter with `Eval_K` and adds no cross-family row.
-/

namespace NightstreamFPrime.Layout.PiRLC.v1_1.EvalACombination

open NightstreamFPrime.Circuit

namespace Logical

abbrev Interface :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.EvalACombination.Interface
abbrev ringInterface :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.EvalACombination.ringInterface
abbrev Assumptions :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.EvalACombination.Assumptions
abbrev SpecHolds :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.EvalACombination.SpecHolds

end Logical

abbrev logicalConstraints (interface : Logical.Interface) (offset : Nat) :=
  RingKCombination.logicalConstraints (Logical.ringInterface interface) offset

abbrev physicalFreshColumnCount (interface : Logical.Interface) (offset : Nat) :=
  RingKCombination.physicalFreshColumnCount
    (Logical.ringInterface interface) offset

abbrev physicalRowCount (interface : Logical.Interface) (offset : Nat) :=
  RingKCombination.physicalRowCount (Logical.ringInterface interface) offset

abbrev physicalPrivateColumnCount (interface : Logical.Interface)
    (offset : Nat) :=
  RingKCombination.physicalPrivateColumnCount
    (Logical.ringInterface interface) offset

abbrev plan (interface : Logical.Interface) (offset : Nat) :=
  RingKCombination.plan (Logical.ringInterface interface) offset

abbrev PhysicalHolds (interface : Logical.Interface) (offset : Nat)
    (env : Env) :=
  RingKCombination.PhysicalHolds (Logical.ringInterface interface) offset env

theorem totalFreshCount_eq (interface : Logical.Interface) (offset : Nat) :
    R1CS.totalFreshCount (logicalConstraints interface offset) =
      physicalFreshColumnCount interface offset :=
  RingKCombination.totalFreshCount_eq (Logical.ringInterface interface) offset

theorem totalRowCount_eq (interface : Logical.Interface) (offset : Nat) :
    R1CS.totalRowCount (logicalConstraints interface offset) =
      physicalRowCount interface offset :=
  RingKCombination.totalRowCount_eq (Logical.ringInterface interface) offset

theorem physicalRowCount_eq (interface : Logical.Interface) (offset : Nat) :
    (plan interface offset).rowCount = physicalRowCount interface offset :=
  RingKCombination.physicalRowCount_eq
    (Logical.ringInterface interface) offset

theorem physical_implies_spec (interface : Logical.Interface) (offset : Nat)
    (env : Env) (assumptions : Logical.Assumptions interface offset env)
    (physical : PhysicalHolds interface offset env) :
    Logical.SpecHolds interface offset env :=
  RingKCombination.physical_implies_spec
    (Logical.ringInterface interface) offset env assumptions physical

theorem physical_complete (interface : Logical.Interface) (offset : Nat)
    (env : Env) (assumptions : Logical.Assumptions interface offset env)
    (specification : Logical.SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
          (physicalPrivateColumnCount interface offset) ∧
      PhysicalHolds interface offset completed :=
  RingKCombination.physical_complete
    (Logical.ringInterface interface) offset env assumptions specification

end NightstreamFPrime.Layout.PiRLC.v1_1.EvalACombination
