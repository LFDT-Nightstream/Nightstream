import NightstreamFPrime.Layout.PiRLC.v1_1.Lowering
import NightstreamFPrime.Layout.PiRLC.v1_1.PacketBoundaries
import NightstreamFPrime.Layout.R1CS.Segments

/-!
Owns the generic physical-row projection for the seven-child PiRLC phase.
It keeps the relation, interface, and offset symbolic and returns one held
lowering segment per opaque child.
-/

namespace NightstreamFPrime.Layout.PiRLC.v1_1.PacketProjection

open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

def PacketRowsHold
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat)
    (env : Env) : Prop :=
  R1CS.SegmentsHold env
    (PacketBoundaries.packetConstraintLists interface offset)
    (logicalColumnCount relation interface offset)

def samplerFreshStart
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat) : Nat :=
  logicalColumnCount relation interface offset

def commitmentFreshStart
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat) : Nat :=
  samplerFreshStart relation interface offset +
    R1CS.totalFreshCount
      (PacketBoundaries.samplerPacketConstraints interface offset)

def publicInputFreshStart
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat) : Nat :=
  commitmentFreshStart relation interface offset +
    R1CS.totalFreshCount
      (PacketBoundaries.commitmentPacketConstraints interface offset)

def evalKFreshStart
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat) : Nat :=
  publicInputFreshStart relation interface offset +
    R1CS.totalFreshCount
      (PacketBoundaries.publicInputPacketConstraints interface offset)

def evalAFreshStart
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat) : Nat :=
  evalKFreshStart relation interface offset +
    R1CS.totalFreshCount
      (PacketBoundaries.evalKPacketConstraints interface offset)

structure NonemptyPacketRowsHold
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat)
    (env : Env) : Prop where
  sampler : R1CS.RowsHold env
    (R1CS.lowerConstraints
      (PacketBoundaries.samplerPacketConstraints interface offset)
      (samplerFreshStart relation interface offset)).rows
  commitment : R1CS.RowsHold env
    (R1CS.lowerConstraints
      (PacketBoundaries.commitmentPacketConstraints interface offset)
      (commitmentFreshStart relation interface offset)).rows
  publicInput : R1CS.RowsHold env
    (R1CS.lowerConstraints
      (PacketBoundaries.publicInputPacketConstraints interface offset)
      (publicInputFreshStart relation interface offset)).rows
  evalK : R1CS.RowsHold env
    (R1CS.lowerConstraints
      (PacketBoundaries.evalKPacketConstraints interface offset)
      (evalKFreshStart relation interface offset)).rows
  evalA : R1CS.RowsHold env
    (R1CS.lowerConstraints
      (PacketBoundaries.evalAPacketConstraints interface offset)
      (evalAFreshStart relation interface offset)).rows

private theorem emptyFreshCount :
    R1CS.totalFreshCount ([] : List Expr) = 0 := by
  rfl

private theorem packetRowsHold_nonempty
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat)
    (env : Env) (packets : PacketRowsHold relation interface offset env) :
    NonemptyPacketRowsHold relation interface offset env := by
  unfold PacketRowsHold PacketBoundaries.packetConstraintLists
    R1CS.SegmentsHold at packets
  simp only [emptyFreshCount, Nat.add_zero] at packets
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · simpa [samplerFreshStart] using packets.2.1
  · simpa [commitmentFreshStart, samplerFreshStart] using packets.2.2.1
  · simpa [publicInputFreshStart, commitmentFreshStart, samplerFreshStart]
      using packets.2.2.2.1
  · simpa [evalKFreshStart, publicInputFreshStart, commitmentFreshStart,
      samplerFreshStart] using packets.2.2.2.2.1
  · simpa [evalAFreshStart, evalKFreshStart, publicInputFreshStart,
      commitmentFreshStart, samplerFreshStart] using packets.2.2.2.2.2.1

/-- Held full physical rows imply every exact opaque child lowering segment.
No production list is reduced by this theorem. -/
theorem physicalRows_imply_packets
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat)
    (env : Env)
    (physical : R1CS.RowsHold env (physicalRows relation interface offset)) :
    PacketRowsHold relation interface offset env := by
  rw [physicalRows_eq] at physical
  have constraintsEq :
      (plan relation interface offset).constraints =
        (childConstraintLists relation interface offset).flatten := by
    calc
      (plan relation interface offset).constraints =
          logicalConstraints relation interface offset :=
        plan_constraints relation interface offset
      _ = orderedConstraints relation interface offset :=
        logicalConstraints_eq_ordered relation interface offset
      _ = (childConstraintLists relation interface offset).flatten := by
        rfl
  have segmented :=
    R1CS.LoweringPlan.rowsHold_segments_of_constraints
      (plan relation interface offset) env
      (childConstraintLists relation interface offset) constraintsEq physical
  rw [PacketBoundaries.childConstraintLists_eq_packets,
    plan_firstFresh] at segmented
  exact segmented

/-- Held full physical rows imply the five nonempty packet lowerings with
symbolic, sequential fresh starts. -/
theorem physicalRows_imply_nonemptyPackets
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat)
    (env : Env)
    (physical : R1CS.RowsHold env (physicalRows relation interface offset)) :
    NonemptyPacketRowsHold relation interface offset env :=
  packetRowsHold_nonempty relation interface offset env
    (physicalRows_imply_packets relation interface offset env physical)

end NightstreamFPrime.Layout.PiRLC.v1_1.PacketProjection
