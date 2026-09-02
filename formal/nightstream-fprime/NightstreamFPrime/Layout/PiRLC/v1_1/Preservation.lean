import NightstreamFPrime.Layout.PiRLC.v1_1.Lowering
import NightstreamFPrime.Layout.R1CS.Completeness

/-!
Owns deterministic physical soundness and constructive completeness for the
sole PiRLC v1_1 phase layout.

No cryptographic assumption occurs here. Physical R1CS rows imply the exact
logical phase and therefore the canonical `PiRLC.Accepted` result and outgoing
transcript state already proved by the logical circuit.
-/

namespace NightstreamFPrime.Layout.PiRLC.v1_1

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
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

/-- Physical PiRLC rows imply the full semantic phase result. -/
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

/-- A valid semantic phase provides the static scope required by its lowering
plan. This theorem remains generic so production users do not reconstruct the
seven-child completion proof. -/
theorem plan_constraints_varsBelow_of_phase
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (env : Env)
    (assumptions : Formal.Assumptions relation interface offset env)
    (phase : Semantics.PhaseHolds relation ajtai interface offset env) :
    ∀ expression ∈ (plan relation interface offset).constraints,
      expression.VarsBelow (plan relation interface offset).firstFresh := by
  rcases Formal.completePrefix relation ajtai interface env offset assumptions
      phase with ⟨logical, operationsEq⟩
  have mainOpsEq : logical.operations =
      Circuit.ops (Formal.main relation interface) offset := by
    rw [Formal.main_ops]
    exact operationsEq
  rw [plan_constraints, plan_firstFresh]
  change ∀ expression ∈ flatConstraints
      (Circuit.ops (Formal.main relation interface) offset),
    expression.VarsBelow (logicalColumnCount relation interface offset)
  rw [logicalColumnCount_eq, ← mainOpsEq]
  exact logical.scope

/-- Every physical row of a valid phase uses only columns below the plan's
declared physical endpoint. -/
theorem physicalRows_varsBelow_of_phase
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (env : Env)
    (assumptions : Formal.Assumptions relation interface offset env)
    (phase : Semantics.PhaseHolds relation ajtai interface offset env) :
    ∀ row ∈ physicalRows relation interface offset,
      row.VarsBelow (physicalColumnCount relation interface offset) := by
  intro row member
  have scope := R1CS.lowerConstraints_rows_varsBelow
    (plan relation interface offset).constraints
    (plan relation interface offset).firstFresh
    (plan_constraints_varsBelow_of_phase relation ajtai interface offset env
      assumptions phase)
  change row ∈ (R1CS.lowerConstraints
    (plan relation interface offset).constraints
    (plan relation interface offset).firstFresh).rows at member
  have rowScope := scope row member
  simpa [physicalColumnCount, R1CS.LoweringPlan.next_eq,
    R1CS.LoweringPlan.freshColumnCount] using rowScope

/-- Constructive completion over the adjacent logical and R1CS-fresh
intervals. -/
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
      AgreesOutside env completed offset 8908425 ∧
      PhysicalHolds relation interface offset completed := by
  rcases physical_complete relation ajtai interface offset env assumptions phase with
    ⟨completed, agrees, rows⟩
  refine ⟨completed, ?_, rows⟩
  rw [Formal.localLength_eq,
    physicalFreshColumnCount_eq_production relation interface offset inputs]
    at agrees
  exact agrees

end NightstreamFPrime.Layout.PiRLC.v1_1
