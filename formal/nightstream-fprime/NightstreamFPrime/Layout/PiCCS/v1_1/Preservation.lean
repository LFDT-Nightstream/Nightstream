import NightstreamFPrime.Layout.PiCCS.v1_1.Lowering
import NightstreamFPrime.Layout.R1CS.Completeness

/-!
Paper authority: SuperNeo v1_1, section 7.3, PiCCS steps 1--5.
Obligation: Physical satisfaction implies the exact PiCCS phase relation.

Inputs:
- the one lowering from `Lowering.lean`;
- the exact logical phase assumptions;
- an arbitrary physical R1CS assignment.

Outputs:
- satisfaction of the logical rows;
- exact `Formal.PhaseHolds`, including `Accepted` and the outgoing transcript state.
- constructive completion of all logical and physical witness columns.

Parent coverage:
- the future Stage 1 assembler consumes this theorem as the PiCCS phase edge.
-/

namespace NightstreamFPrime.Layout.PiCCS.v1_1

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth degreeBound : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

def PhysicalHolds
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) (env : Env) : Prop :=
  R1CS.RowsHold env (physicalRows relation interface offset)

theorem physical_implies_holdsFlat
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) (env : Env)
    (physical : PhysicalHolds relation interface offset env) :
    holdsFlat env (Circuit.ops (Formal.main relation interface) offset) := by
  change ConstraintsHold env (logicalConstraints relation interface offset)
  exact R1CS.lowerConstraints_sound env _ _ physical

/-- Deterministic PiCCS layout-preservation theorem. It has no cryptographic
assumption and does not inspect any emitted row in the kernel. -/
theorem physical_implies_phaseHolds
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Formal.Interface logicalWidth
      (ProductionKey.degreeBound relation) publicFits)
    (template : Proof (ProductionKey.degreeBound relation))
    (offset : Nat) (env : Env)
    (assumptions : Formal.Assumptions relation interface offset env)
    (physical : PhysicalHolds relation interface offset env) :
    Formal.PhaseHolds relation ajtai interface offset env template := by
  apply (Formal.circuit relation ajtai interface template).soundness env offset
    assumptions
  change holds env (Circuit.ops (Formal.main relation interface) offset)
  exact holdsFlat_implies_holds env _
    (physical_implies_holdsFlat relation interface offset env physical)

/-- Constructive physical completeness for the sole PiCCS layout. The final
environment differs from the caller only inside the adjacent logical and R1CS
fresh-column intervals. -/
theorem physical_complete
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Formal.Interface logicalWidth
      (ProductionKey.degreeBound relation) publicFits)
    (template : Proof (ProductionKey.degreeBound relation))
    (offset : Nat) (env : Env)
    (assumptions : Formal.Assumptions relation interface offset env)
    (specification :
      Formal.PhaseHolds relation ajtai interface offset env template) :
    ∃ completed,
      AgreesOutside env completed offset
          (localLength (Circuit.ops (Formal.main relation interface) offset) +
            physicalFreshColumnCount relation interface offset) ∧
        PhysicalHolds relation interface offset completed := by
  rcases Formal.completePrefix relation ajtai interface template env offset
      assumptions specification with ⟨logical, operationsEq⟩
  have mainOpsEq : logical.operations =
      Circuit.ops (Formal.main relation interface) offset := by
    rw [Formal.main_ops]
    exact operationsEq
  have planScope : ∀ expression ∈ (plan relation interface offset).constraints,
      expression.VarsBelow (plan relation interface offset).firstFresh := by
    change ∀ expression ∈ flatConstraints
        (Circuit.ops (Formal.main relation interface) offset),
      expression.VarsBelow (logicalColumnCount relation interface offset)
    rw [logicalColumnCount_eq relation interface offset, ← mainOpsEq]
    exact logical.scope
  have planLogical : ConstraintsHold logical.current
      (plan relation interface offset).constraints := by
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
    exact loweringAgrees
  refine ⟨completed, logicalAgrees.append loweringAgreesAtEnd, ?_⟩
  exact physicalRowsHold

end NightstreamFPrime.Layout.PiCCS.v1_1
