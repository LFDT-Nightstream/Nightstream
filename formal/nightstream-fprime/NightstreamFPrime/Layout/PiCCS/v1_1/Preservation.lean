import NightstreamFPrime.Layout.PiCCS.v1_1.Lowering

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

end NightstreamFPrime.Layout.PiCCS.v1_1
