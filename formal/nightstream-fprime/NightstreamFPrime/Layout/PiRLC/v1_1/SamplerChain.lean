import NightstreamFPrime.Layout.PiRLC.v1_1.SamplerChain.Lowering
import NightstreamFPrime.Layout.R1CS.Completeness

/-!
Owns physical lowering and preservation for the exact 17-sampler PiRLC chain.

The imported composition theorem fixes the logical row order and footprint.
This module lowers that certified list without unfolding it across the module
boundary. It adds no copy or boundary row.
-/

namespace NightstreamFPrime.Layout.PiRLC.v1_1.SamplerChain

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout

def PhysicalHolds (interface : Logical.Interface) (offset : Nat)
    (env : Env) : Prop :=
  R1CS.RowsHold env (physicalRows interface offset)

/-- Physical chain rows imply the exact logical 17-sampler relation. -/
theorem physical_implies_relation (interface : Logical.Interface)
    (offset : Nat) (env : Env)
    (assumptions : Logical.Assumptions interface offset)
    (physical : PhysicalHolds interface offset env) :
    Logical.SpecHolds interface offset env := by
  change R1CS.RowsHold env (physicalRows interface offset) at physical
  rw [physicalRows_eq] at physical
  have logicalRows :=
    R1CS.LoweringPlan.sound (plan interface offset) env physical
  rw [plan_constraints] at logicalRows
  apply Logical.soundness interface env offset assumptions
  apply holdsFlat_implies_holds
  simpa only [logicalConstraints] using! logicalRows

theorem physical_complete (interface : Logical.Interface) (offset : Nat)
    (env : Env) (inputs : InputsAffine interface offset)
    (assumptions : Logical.Assumptions interface offset) :
    ∃ completed,
      AgreesOutside env completed offset 74987 ∧
      PhysicalHolds interface offset completed := by
  rcases Logical.complete interface env offset assumptions with
    ⟨logicalEnv, logicalAgrees, logicalRows⟩
  have logicalAgreesFixed : AgreesOutside env logicalEnv offset
      Logical.logicalPrivateCount := by
    rw [Logical.localLength_eq] at logicalAgrees
    exact logicalAgrees
  have logicalScope : ∀ expression ∈ logicalConstraints interface offset,
      expression.VarsBelow (offset + Logical.logicalPrivateCount) :=
    Logical.scope interface offset assumptions
  have planScope : ∀ expression ∈ (plan interface offset).constraints,
      expression.VarsBelow (plan interface offset).firstFresh := by
    rw [plan_constraints, plan_firstFresh]
    exact logicalScope
  have planLogical : ConstraintsHold logicalEnv
      (plan interface offset).constraints := by
    rw [plan_constraints]
    simpa only [logicalConstraints] using! logicalRows
  rcases R1CS.LoweringPlan.complete (plan interface offset) logicalEnv
      planScope planLogical with
    ⟨completed, physicalAgrees, rows⟩
  have physicalAgreesFixed : AgreesOutside logicalEnv completed
      (offset + Logical.logicalPrivateCount) 2448 := by
    rw [← plan_firstFresh interface offset,
      ← freshColumnCount_eq interface offset inputs]
    exact physicalAgrees
  refine ⟨completed, ?_, ?_⟩
  · have combined := logicalAgreesFixed.append physicalAgreesFixed
    have logicalCount : Logical.logicalPrivateCount = 72539 :=
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerChain.counts.1
    rw [logicalCount] at combined
    simpa using combined
  · change R1CS.RowsHold completed (physicalRows interface offset)
    rw [physicalRows_eq]
    exact rows

end NightstreamFPrime.Layout.PiRLC.v1_1.SamplerChain
