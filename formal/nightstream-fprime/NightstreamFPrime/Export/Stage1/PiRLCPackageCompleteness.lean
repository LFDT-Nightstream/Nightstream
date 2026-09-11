import NightstreamFPrime.Export.Stage1.PiRLCPhysicalCompleteness
import NightstreamFPrime.Layout.PiRLC.v1_1.PacketProjection

/-!
Owns the production specialization of the generic PiRLC packet projection.
It proves the exact Stage 1 fresh starts, maps every held child packet to the
final Spartan columns, and composes them with semantic physical completion.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCPackageCompleteness

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

def phaseInterface :
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.Interface
      Data.logicalWidth Data.publicFits :=
  PiRLCInputs.interface
    (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

private theorem samplerFreshCount
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    R1CS.totalFreshCount
        (NightstreamFPrime.Layout.PiRLC.v1_1.PacketBoundaries.samplerPacketConstraints
          (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits)) PiRLCInputs.phaseOffset) =
      743631 := by
  unfold NightstreamFPrime.Layout.PiRLC.v1_1.PacketBoundaries.samplerPacketConstraints
  exact NightstreamFPrime.Layout.PiRLC.v1_1.SamplerChain.totalFreshCount_eq
    (NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerInterface
      (NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.atOffset (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits))
        PiRLCInputs.phaseOffset))
    PiRLCStarts.samplerLogicalStart
    (PiRLCInputs.inputShapes relation).sampler

private theorem commitmentFreshCount
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    R1CS.totalFreshCount
        (NightstreamFPrime.Layout.PiRLC.v1_1.PacketBoundaries.commitmentPacketConstraints
          (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits)) PiRLCInputs.phaseOffset) =
      3029400 := by
  unfold NightstreamFPrime.Layout.PiRLC.v1_1.PacketBoundaries.commitmentPacketConstraints
  rw [NightstreamFPrime.Layout.PiRLC.v1_1.CommitmentCombination.totalFreshCount_eq]
  exact (PiRLCInputs.inputShapes relation).commitmentFresh

private theorem publicInputFreshCount
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    R1CS.totalFreshCount
        (NightstreamFPrime.Layout.PiRLC.v1_1.PacketBoundaries.publicInputPacketConstraints
          (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits)) PiRLCInputs.phaseOffset) =
      688500 := by
  unfold NightstreamFPrime.Layout.PiRLC.v1_1.PacketBoundaries.publicInputPacketConstraints
  rw [NightstreamFPrime.Layout.PiRLC.v1_1.PublicInputCombination.totalFreshCount_eq]
  exact (PiRLCInputs.inputShapes relation).publicInputFresh

private theorem evalKFreshCount
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    R1CS.totalFreshCount
        (NightstreamFPrime.Layout.PiRLC.v1_1.PacketBoundaries.evalKPacketConstraints
          (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits)) PiRLCInputs.phaseOffset) =
      275400 := by
  unfold NightstreamFPrime.Layout.PiRLC.v1_1.PacketBoundaries.evalKPacketConstraints
  rw [NightstreamFPrime.Layout.PiRLC.v1_1.EvalKCombination.totalFreshCount_eq]
  exact (PiRLCInputs.inputShapes relation).evalKFresh

private theorem samplerFreshStart_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    NightstreamFPrime.Layout.PiRLC.v1_1.PacketProjection.samplerFreshStart
        relation (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits)) PiRLCInputs.phaseOffset =
      PiRLCStarts.samplerFreshStart := by
  rfl

private theorem commitmentFreshStart_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    NightstreamFPrime.Layout.PiRLC.v1_1.PacketProjection.commitmentFreshStart
        relation (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits)) PiRLCInputs.phaseOffset =
      PiRLCStarts.commitmentFreshStart := by
  unfold NightstreamFPrime.Layout.PiRLC.v1_1.PacketProjection.commitmentFreshStart
  rw [samplerFreshStart_eq relation, samplerFreshCount relation]
  rfl

private theorem publicInputFreshStart_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    NightstreamFPrime.Layout.PiRLC.v1_1.PacketProjection.publicInputFreshStart
        relation (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits)) PiRLCInputs.phaseOffset =
      PiRLCStarts.publicInputFreshStart := by
  unfold NightstreamFPrime.Layout.PiRLC.v1_1.PacketProjection.publicInputFreshStart
  rw [commitmentFreshStart_eq relation, commitmentFreshCount relation]
  rfl

private theorem evalKFreshStart_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    NightstreamFPrime.Layout.PiRLC.v1_1.PacketProjection.evalKFreshStart
        relation (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits)) PiRLCInputs.phaseOffset =
      PiRLCStarts.evalKFreshStart := by
  unfold NightstreamFPrime.Layout.PiRLC.v1_1.PacketProjection.evalKFreshStart
  rw [publicInputFreshStart_eq relation, publicInputFreshCount relation]
  rfl

private theorem evalAFreshStart_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    NightstreamFPrime.Layout.PiRLC.v1_1.PacketProjection.evalAFreshStart
        relation (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits)) PiRLCInputs.phaseOffset =
      PiRLCStarts.evalAFreshStart := by
  unfold NightstreamFPrime.Layout.PiRLC.v1_1.PacketProjection.evalAFreshStart
  rw [evalKFreshStart_eq relation, evalKFreshCount relation]
  rfl

structure SourcePacketRowsHold (env : Env) : Prop where
  sampler : R1CS.RowsHold env
    (R1CS.lowerConstraints
      (NightstreamFPrime.Layout.PiRLC.v1_1.PacketBoundaries.samplerPacketConstraints
        phaseInterface PiRLCInputs.phaseOffset)
      PiRLCStarts.samplerFreshStart).rows
  commitment : R1CS.RowsHold env
    (R1CS.lowerConstraints
      (NightstreamFPrime.Layout.PiRLC.v1_1.PacketBoundaries.commitmentPacketConstraints
        phaseInterface PiRLCInputs.phaseOffset)
      PiRLCStarts.commitmentFreshStart).rows
  publicInput : R1CS.RowsHold env
    (R1CS.lowerConstraints
      (NightstreamFPrime.Layout.PiRLC.v1_1.PacketBoundaries.publicInputPacketConstraints
        phaseInterface PiRLCInputs.phaseOffset)
      PiRLCStarts.publicInputFreshStart).rows
  evalK : R1CS.RowsHold env
    (R1CS.lowerConstraints
      (NightstreamFPrime.Layout.PiRLC.v1_1.PacketBoundaries.evalKPacketConstraints
        phaseInterface PiRLCInputs.phaseOffset)
      PiRLCStarts.evalKFreshStart).rows
  evalA : R1CS.RowsHold env
    (R1CS.lowerConstraints
      (NightstreamFPrime.Layout.PiRLC.v1_1.PacketBoundaries.evalAPacketConstraints
        phaseInterface PiRLCInputs.phaseOffset)
      PiRLCStarts.evalAFreshStart).rows

private theorem commitmentPacket_eq :
    NightstreamFPrime.Layout.PiRLC.v1_1.PacketBoundaries.commitmentPacketConstraints
        (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits)) PiRLCInputs.phaseOffset =
      NightstreamFPrime.Layout.PiRLC.v1_1.PacketBoundaries.commitmentPacketConstraints
        phaseInterface PiRLCInputs.phaseOffset :=
  (PiRLCCombinationConformance.commitmentFamilyConstraints_eq_parent
    (logicalWidth := logicalWidth) (publicFits := publicFits)).symm.trans
      (PiRLCCombinationConformance.commitmentFamilyConstraints_eq_parent
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))

private theorem publicInputPacket_eq :
    NightstreamFPrime.Layout.PiRLC.v1_1.PacketBoundaries.publicInputPacketConstraints
        (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits)) PiRLCInputs.phaseOffset =
      NightstreamFPrime.Layout.PiRLC.v1_1.PacketBoundaries.publicInputPacketConstraints
        phaseInterface PiRLCInputs.phaseOffset :=
  (PiRLCCombinationConformance.publicInputFamilyConstraints_eq_parent
    (logicalWidth := logicalWidth) (publicFits := publicFits)).symm.trans
      (PiRLCCombinationConformance.publicInputFamilyConstraints_eq_parent
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))

private theorem evalKPacket_eq :
    NightstreamFPrime.Layout.PiRLC.v1_1.PacketBoundaries.evalKPacketConstraints
        (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits)) PiRLCInputs.phaseOffset =
      NightstreamFPrime.Layout.PiRLC.v1_1.PacketBoundaries.evalKPacketConstraints
        phaseInterface PiRLCInputs.phaseOffset :=
  (PiRLCCombinationConformance.evalKFamilyConstraints_eq_parent
    (logicalWidth := logicalWidth) (publicFits := publicFits)).symm.trans
      (PiRLCCombinationConformance.evalKFamilyConstraints_eq_parent
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))

private theorem evalAPacket_eq :
    NightstreamFPrime.Layout.PiRLC.v1_1.PacketBoundaries.evalAPacketConstraints
        (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits)) PiRLCInputs.phaseOffset =
      NightstreamFPrime.Layout.PiRLC.v1_1.PacketBoundaries.evalAPacketConstraints
        phaseInterface PiRLCInputs.phaseOffset :=
  (PiRLCCombinationConformance.evalAFamilyConstraints_eq_parent
    (logicalWidth := logicalWidth) (publicFits := publicFits)).symm.trans
      (PiRLCCombinationConformance.evalAFamilyConstraints_eq_parent
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))

/-- Full source-ordered physical rows project to the five exact production
packet lowerings. The relation width is arbitrary; these fixed-size packets
retain the existing width-erased package definitions. -/
theorem sourcePhysicalRows_imply_packets
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (env : Env)
    (physical : R1CS.RowsHold env
      (NightstreamFPrime.Layout.PiRLC.v1_1.physicalRows relation
        (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits)) PiRLCInputs.phaseOffset)) :
    SourcePacketRowsHold env := by
  have packets :=
    NightstreamFPrime.Layout.PiRLC.v1_1.PacketProjection.physicalRows_imply_nonemptyPackets
      relation (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits)) PiRLCInputs.phaseOffset env physical
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · simpa only [samplerFreshStart_eq relation] using packets.sampler
  · simpa only [commitmentFreshStart_eq relation, commitmentPacket_eq] using packets.commitment
  · simpa only [publicInputFreshStart_eq relation, publicInputPacket_eq] using packets.publicInput
  · simpa only [evalKFreshStart_eq relation, evalKPacket_eq] using packets.evalK
  · simpa only [evalAFreshStart_eq relation, evalAPacket_eq] using packets.evalA

structure RemappedPacketRowsHold (env : Env) : Prop where
  sampler : R1CS.RowsHold env
    (Spartan.remapRows (R1CS.lowerConstraints
      (NightstreamFPrime.Layout.PiRLC.v1_1.PacketBoundaries.samplerPacketConstraints
        phaseInterface PiRLCInputs.phaseOffset)
      PiRLCStarts.samplerFreshStart).rows)
  commitment : R1CS.RowsHold env
    (Spartan.remapRows (R1CS.lowerConstraints
      (NightstreamFPrime.Layout.PiRLC.v1_1.PacketBoundaries.commitmentPacketConstraints
        phaseInterface PiRLCInputs.phaseOffset)
      PiRLCStarts.commitmentFreshStart).rows)
  publicInput : R1CS.RowsHold env
    (Spartan.remapRows (R1CS.lowerConstraints
      (NightstreamFPrime.Layout.PiRLC.v1_1.PacketBoundaries.publicInputPacketConstraints
        phaseInterface PiRLCInputs.phaseOffset)
      PiRLCStarts.publicInputFreshStart).rows)
  evalK : R1CS.RowsHold env
    (Spartan.remapRows (R1CS.lowerConstraints
      (NightstreamFPrime.Layout.PiRLC.v1_1.PacketBoundaries.evalKPacketConstraints
        phaseInterface PiRLCInputs.phaseOffset)
      PiRLCStarts.evalKFreshStart).rows)
  evalA : R1CS.RowsHold env
    (Spartan.remapRows (R1CS.lowerConstraints
      (NightstreamFPrime.Layout.PiRLC.v1_1.PacketBoundaries.evalAPacketConstraints
        phaseInterface PiRLCInputs.phaseOffset)
      PiRLCStarts.evalAFreshStart).rows)

/-- Final-column full physical rows project to the five exact final-column
packet lowerings. -/
theorem remappedPhysicalRows_imply_packets
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (env : Env)
    (physical : R1CS.RowsHold env
      (Spartan.remapRows
        (NightstreamFPrime.Layout.PiRLC.v1_1.physicalRows relation
          (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits)) PiRLCInputs.phaseOffset))) :
    RemappedPacketRowsHold env := by
  have sourcePhysical :=
    (Spartan.remapRows_hold env
      (NightstreamFPrime.Layout.PiRLC.v1_1.physicalRows relation
        (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits)) PiRLCInputs.phaseOffset)).mp physical
  have packets := sourcePhysicalRows_imply_packets relation
    (Spartan.pullback env) sourcePhysical
  exact ⟨(Spartan.remapRows_hold env _).mpr packets.sampler,
    (Spartan.remapRows_hold env _).mpr packets.commitment,
    (Spartan.remapRows_hold env _).mpr packets.publicInput,
    (Spartan.remapRows_hold env _).mpr packets.evalK,
    (Spartan.remapRows_hold env _).mpr packets.evalA⟩

/-- A valid semantic production phase constructs one final-column assignment
that holds every exact nonempty PiRLC packet. -/
theorem completePackets
    (relation : ProductionKey.LogicalRelation Data.logicalWidth Data.publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
    (env : Env)
    (assumptions :
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.Assumptions relation
        phaseInterface PiRLCInputs.phaseOffset (Spartan.pullback env))
    (phase :
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.Semantics.PhaseHolds relation ajtai
        phaseInterface PiRLCInputs.phaseOffset (Spartan.pullback env)) :
    ∃ completed,
      AgreesOutside env completed
          (Spartan.sourceToSpartan PiRLCInputs.phaseOffset) 8908425 ∧
        RemappedPacketRowsHold completed := by
  rcases PiRLCPhysicalCompleteness.completePhysicalRows relation ajtai env
      assumptions phase with ⟨completed, agrees, physical⟩
  exact ⟨completed, agrees,
    remappedPhysicalRows_imply_packets relation completed physical⟩

end NightstreamFPrime.Export.Stage1.PiRLCPackageCompleteness
