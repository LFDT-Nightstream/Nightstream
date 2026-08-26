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
    (assumptions : Logical.Assumptions interface offset env)
    (physical : PhysicalHolds interface offset env) :
    Logical.RelationHolds interface offset env := by
  change R1CS.RowsHold env (physicalRows interface offset) at physical
  rw [physicalRows_eq] at physical
  have logicalRows :=
    R1CS.LoweringPlan.sound (plan interface offset) env physical
  rw [plan_constraints] at logicalRows
  apply Logical.soundness interface offset env assumptions
  apply holdsFlat_implies_holds
  simpa only [logicalConstraints] using logicalRows

set_option maxRecDepth 100000 in -- fixed-size: 17 scalar samplers
theorem physical_complete (interface : Logical.Interface) (offset : Nat)
    (env : Env) (inputs : InputsAffine interface offset)
    (assumptions : Logical.Assumptions interface offset env)
    (relation : Logical.RelationHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset 1007199 ∧
      PhysicalHolds interface offset completed := by
  rcases Logical.completeness interface offset env assumptions relation with
    ⟨logicalEnv, logicalAgrees, logicalRows⟩
  have logicalAgreesFixed : AgreesOutside env logicalEnv offset
      Logical.logicalPrivateCount := by
    rw [Logical.localLength_eq] at logicalAgrees
    exact logicalAgrees
  have logicalAssumptions : Logical.Assumptions interface offset logicalEnv :=
    ⟨assumptions.initialBelow⟩
  have logicalScope : ∀ expression ∈ logicalConstraints interface offset,
      expression.VarsBelow (offset + Logical.logicalPrivateCount) := by
    simpa only [logicalConstraints] using
      (Logical.flatConstraints_varsBelow_of_rows interface offset logicalEnv
        logicalAssumptions logicalRows)
  have planScope : ∀ expression ∈ (plan interface offset).constraints,
      expression.VarsBelow (plan interface offset).firstFresh := by
    rw [plan_constraints, plan_firstFresh]
    exact logicalScope
  have planLogical : ConstraintsHold logicalEnv
      (plan interface offset).constraints := by
    rw [plan_constraints]
    simpa only [logicalConstraints] using logicalRows
  rcases R1CS.LoweringPlan.complete (plan interface offset) logicalEnv
      planScope planLogical with
    ⟨completed, physicalAgrees, rows⟩
  have physicalAgreesFixed : AgreesOutside logicalEnv completed
      (offset + Logical.logicalPrivateCount) 743631 := by
    rw [← plan_firstFresh interface offset,
      ← freshColumnCount_eq interface offset inputs]
    exact physicalAgrees
  refine ⟨completed, ?_, ?_⟩
  · have combined := logicalAgreesFixed.append physicalAgreesFixed
    have logicalCount : Logical.logicalPrivateCount = 263568 :=
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerChain.logicalPrivateCount_eq
    rw [logicalCount] at combined
    simpa using combined
  · change R1CS.RowsHold completed (physicalRows interface offset)
    rw [physicalRows_eq]
    exact rows

end NightstreamFPrime.Layout.PiRLC.v1_1.SamplerChain
