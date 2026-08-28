import NightstreamFPrime.Layout.PiDEC.v1_1.Lowering
import NightstreamFPrime.Layout.R1CS.Completeness

/-!
Owns deterministic physical soundness and constructive completeness for the
sole PiDEC v1_1 phase layout. No cryptographic assumption occurs here.
-/

namespace NightstreamFPrime.Layout.PiDEC.v1_1

open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiDEC.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

def PhysicalHolds
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (env : Env) : Prop :=
  R1CS.RowsHold env (physicalRows relation interface offset)

theorem physical_implies_holdsFlat
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (env : Env)
    (physical : PhysicalHolds relation interface offset env) :
    holdsFlat env (Circuit.ops (Formal.main relation interface) offset) := by
  change R1CS.RowsHold env (physicalRows relation interface offset) at physical
  rw [physicalRows_eq] at physical
  have logicalRows :=
    R1CS.LoweringPlan.sound (plan relation interface offset) env physical
  rw [plan_constraints] at logicalRows
  simpa only [logicalConstraints] using logicalRows

theorem physical_implies_specHolds
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (env : Env)
    (assumptions : Formal.Assumptions relation interface offset env)
    (physical : PhysicalHolds relation interface offset env) :
    Formal.SpecHolds relation interface offset env := by
  apply Formal.soundness relation interface offset env assumptions
  exact holdsFlat_implies_holds env _
    (physical_implies_holdsFlat relation interface offset env physical)

/-- Physical rows imply the exact production PiDEC verifier result. -/
theorem physical_implies_phaseHolds
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (env : Env)
    (assumptions : Formal.Assumptions relation interface offset env)
    (physical : PhysicalHolds relation interface offset env) :
    Semantics.PhaseHolds relation ajtai interface offset env := by
  apply (Formal.circuit relation ajtai interface).soundness env offset assumptions
  exact holdsFlat_implies_holds env _
    (physical_implies_holdsFlat relation interface offset env physical)

/-- Constructive completion over adjacent logical and R1CS-fresh intervals. -/
theorem physical_complete
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (env : Env)
    (assumptions : Formal.Assumptions relation interface offset env)
    (phase : Semantics.PhaseHolds relation ajtai interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
          (localLength (Circuit.ops (Formal.main relation interface) offset) +
            physicalFreshColumnCount relation interface offset) ∧
      PhysicalHolds relation interface offset completed := by
  rcases Formal.completePrefix relation ajtai interface env offset assumptions
      phase with ⟨logical, operationsEq⟩
  have mainOpsEq : logical.operations =
      Circuit.ops (Formal.main relation interface) offset := by
    rw [Formal.main_ops]
    exact operationsEq
  have planScope : ∀ expression ∈
      (plan relation interface offset).constraints,
      expression.VarsBelow (plan relation interface offset).firstFresh := by
    rw [plan_constraints, plan_firstFresh]
    change ∀ expression ∈ flatConstraints
        (Circuit.ops (Formal.main relation interface) offset),
      expression.VarsBelow (logicalColumnCount relation interface offset)
    rw [logicalColumnCount_eq, ← mainOpsEq]
    exact logical.scope
  have planLogical : ConstraintsHold logical.current
      (plan relation interface offset).constraints := by
    rw [plan_constraints]
    change holdsFlat logical.current
      (Circuit.ops (Formal.main relation interface) offset)
    rw [← mainOpsEq]
    exact logical.rows
  rcases R1CS.LoweringPlan.complete (plan relation interface offset)
      logical.current planScope planLogical with
    ⟨completed, loweringAgrees, physicalRowsHold⟩
  have logicalAgrees : AgreesOutside env logical.current offset
      (localLength (Circuit.ops (Formal.main relation interface) offset)) := by
    rw [← mainOpsEq]
    exact logical.agrees
  have loweringAgreesAtEnd : AgreesOutside logical.current completed
      (offset + localLength
        (Circuit.ops (Formal.main relation interface) offset))
      (physicalFreshColumnCount relation interface offset) := by
    rw [← logicalColumnCount_eq relation interface offset]
    change AgreesOutside logical.current completed
      (plan relation interface offset).firstFresh
      (plan relation interface offset).freshColumnCount
    exact loweringAgrees
  refine ⟨completed, logicalAgrees.append loweringAgreesAtEnd, ?_⟩
  change R1CS.RowsHold completed (physicalRows relation interface offset)
  rw [physicalRows_eq]
  exact physicalRowsHold

theorem physical_complete_production
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (env : Env)
    (inputs : InputShapes relation interface offset)
    (assumptions : Formal.Assumptions relation interface offset env)
    (phase : Semantics.PhaseHolds relation ajtai interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset 18090 ∧
      PhysicalHolds relation interface offset completed := by
  rcases physical_complete relation ajtai interface offset env assumptions phase with
    ⟨completed, agrees, rows⟩
  refine ⟨completed, ?_, rows⟩
  rw [Formal.localLength_eq,
    physicalFreshColumnCount_eq_production relation interface offset inputs]
    at agrees
  exact agrees

end NightstreamFPrime.Layout.PiDEC.v1_1
