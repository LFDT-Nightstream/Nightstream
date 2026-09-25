import NightstreamFPrime.Layout.PiRLC.v1_1.CombinationFamily
import NightstreamFPrime.Lifecycle.PiRLC.v1_1.PublicInputCombination

/-!
Owns the physical PiRLC public-input combination family.

This owner selects the exact canonical public-column view. It adds no
commitment, evaluation, transcript, copy, or boundary rows.
-/

namespace NightstreamFPrime.Layout.PiRLC.v1_1.PublicInputCombination

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

namespace Logical

abbrev Interface (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.PublicInputCombination.Interface
    logicalWidth publicFits
abbrev familyInterface
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.PublicInputCombination.familyInterface
    interface
abbrev Assumptions
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) (env : Env) :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.PublicInputCombination.Assumptions
    interface offset env
abbrev SpecHolds
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) (env : Env) :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.PublicInputCombination.SpecHolds
    interface offset env

end Logical

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

abbrev logicalConstraints
    (interface : Logical.Interface logicalWidth publicFits) (offset : Nat) :=
  CombinationFamily.logicalConstraints (Logical.familyInterface interface) offset

abbrev physicalFreshColumnCount
    (interface : Logical.Interface logicalWidth publicFits) (offset : Nat) :=
  CombinationFamily.physicalFreshColumnCount
    (Logical.familyInterface interface) offset

abbrev physicalRowCount
    (interface : Logical.Interface logicalWidth publicFits) (offset : Nat) :=
  CombinationFamily.physicalRowCount (Logical.familyInterface interface) offset

abbrev physicalPrivateColumnCount
    (interface : Logical.Interface logicalWidth publicFits) (offset : Nat) :=
  CombinationFamily.physicalPrivateColumnCount
    (Logical.familyInterface interface) offset

abbrev plan (interface : Logical.Interface logicalWidth publicFits)
    (offset : Nat) :=
  CombinationFamily.plan (Logical.familyInterface interface) offset

abbrev PhysicalHolds (interface : Logical.Interface logicalWidth publicFits)
    (offset : Nat) (env : Env) :=
  CombinationFamily.PhysicalHolds (Logical.familyInterface interface) offset env

theorem totalFreshCount_eq
    (interface : Logical.Interface logicalWidth publicFits) (offset : Nat) :
    R1CS.totalFreshCount (logicalConstraints interface offset) =
      physicalFreshColumnCount interface offset :=
  CombinationFamily.totalFreshCount_eq_deltas
    (Logical.familyInterface interface) offset

theorem totalRowCount_eq
    (interface : Logical.Interface logicalWidth publicFits) (offset : Nat) :
    R1CS.totalRowCount (logicalConstraints interface offset) =
      physicalRowCount interface offset :=
  CombinationFamily.totalRowCount_eq_deltas
    (Logical.familyInterface interface) offset

theorem physicalRowCount_eq
    (interface : Logical.Interface logicalWidth publicFits) (offset : Nat) :
    (plan interface offset).rowCount = physicalRowCount interface offset :=
  CombinationFamily.physicalRowCount_eq
    (Logical.familyInterface interface) offset

theorem physical_implies_spec
    (interface : Logical.Interface logicalWidth publicFits) (offset : Nat)
    (env : Env) (assumptions : Logical.Assumptions interface offset env)
    (physical : PhysicalHolds interface offset env) :
    Logical.SpecHolds interface offset env :=
  CombinationFamily.physical_implies_canonical
    (Logical.familyInterface interface) offset env assumptions physical

theorem physical_complete
    (interface : Logical.Interface logicalWidth publicFits) (offset : Nat)
    (env : Env) (assumptions : Logical.Assumptions interface offset env)
    (specification : Logical.SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
          (physicalPrivateColumnCount interface offset) ∧
      PhysicalHolds interface offset completed :=
  CombinationFamily.physical_complete
    (Logical.familyInterface interface) offset env assumptions specification

end NightstreamFPrime.Layout.PiRLC.v1_1.PublicInputCombination
