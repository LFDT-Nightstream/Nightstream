import NightstreamFPrime.Export.Stage1.PiRLCCombinationOrdinaryReadSupport
import NightstreamFPrime.Export.Stage1.StoredPhysicalExecution

/-! Canonical batch and ordinary-instruction fields of stored-event support. -/

namespace NightstreamFPrime.Export.Stage1.PiRLCCombinationEventReadSupport

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Lifecycle
open PiRLCCombinationWitnessReadSupport (Outside)
open PiRLCCombinationReadSupport
open StoredPhysicalExecution (EventSupported)

local notation "selectedApplication" => Poseidon2HashChainV1Package.application
local notation "selectedShift" => PerApplicationCachedShift.Context.ofProgram selectedApplication
local notation "pilot" => PilotData.circuitPackage ()

/-- Only the relation shape enters the existing row-support proof. These zero
matrices are not a verifier relation or semantic authority. -/
private def shapeRelation : ProductionKey.LogicalRelation Data.logicalWidth Data.publicFits where
  matrices := fun _ _ _ => 0
  cubeFits := by decide

theorem pilotBatches_supported (templates : Array CompactRowTemplate)
    (batch : WitnessBatch)
    (member : batch ∈ Data.liftPilotBatches (PilotData.priorWordBatches ())) :
    EventSupported Outside pilot templates
      (.batch (PerApplicationCachedShift.shiftBatch selectedShift batch)) := by
  rw [PerApplicationCachedShift.shiftBatch_eq]
  apply PiRLCCombinationWitnessReadSupport.shiftBatch_readsSatisfy
  rcases List.mem_map.mp member with ⟨original, _, rfl⟩
  exact liftPilotBatch_supported original

theorem piCcsBatches_supported (templates : Array CompactRowTemplate)
    (batches : List WitnessBatch)
    (family : batches ∈ [WitnessProgram.initialClaimBatches Data.logicalWidth Data.publicFits,
      WitnessProgram.sumcheckBatches Data.logicalWidth Data.publicFits,
      WitnessProgram.evalKBatches Data.logicalWidth Data.publicFits,
      WitnessProgram.evalABatches Data.logicalWidth Data.publicFits,
      WitnessProgram.ccsBatches Data.logicalWidth Data.publicFits,
      WitnessProgram.normBatches Data.logicalWidth Data.publicFits,
      WitnessProgram.finalIdentityBatches Data.logicalWidth Data.publicFits])
    (batch : WitnessBatch) (member : batch ∈ batches) :
    EventSupported Outside pilot templates
      (.batch (PerApplicationCachedShift.shiftBatch selectedShift batch)) := by
  rw [PerApplicationCachedShift.shiftBatch_eq]
  exact PiRLCCombinationWitnessReadSupport.shiftBatch_readsSatisfy _ _
    (PiRLCCombinationPiCCSReadSupport.piCcsBatches_readsSatisfy
      Data.logicalWidth Data.publicFits batches family batch member)

theorem witnessBatches_supported (templates : Array CompactRowTemplate)
    (block : WitnessPlan.Block)
    (blockMember : block ∈ WitnessPlan.canonicalBlocks Data.logicalWidth Data.publicFits)
    (batch : WitnessBatch) (member : batch ∈ block.expand) :
    EventSupported Outside pilot templates
      (.batch (PerApplicationCachedShift.shiftBatch selectedShift batch)) := by
  rw [PerApplicationCachedShift.shiftBatch_eq]
  exact PiRLCCombinationWitnessReadSupport.shiftBatch_readsSatisfy _ _
    (PiRLCCombinationWitnessReadSupport.canonicalBlock_readsSatisfy
      Data.logicalWidth Data.publicFits block blockMember batch member)

theorem applicationBatches_supported (templates : Array CompactRowTemplate)
    (batch : WitnessBatch)
    (member : batch ∈ (PerApplicationPackage.directApplicationPlan selectedApplication).witnessBatches) :
    EventSupported Outside pilot templates (.batch batch) :=
  PiRLCCombinationWitnessReadSupport.selectedApplicationBatches_readsSatisfy batch member

theorem pilotInstructions_supported (templates : Array CompactRowTemplate)
    (instruction : WitnessInstruction)
    (member : instruction ∈ Data.liftPilotInstructions (PilotData.witnessInstructions ())) :
    EventSupported Outside pilot templates
      (.instruction (PerApplicationCachedShift.shiftWitnessInstruction selectedShift instruction)) := by
  apply shiftInstruction_supported
  rcases List.mem_map.mp member with ⟨original, _, rfl⟩
  exact ⟨liftPilotCombination_supported original.a, liftPilotCombination_supported original.b⟩

theorem ordinaryInstructions_supported (templates : Array CompactRowTemplate)
    (block : OrdinaryRowPlan.Block) (blockMember : block ∈ OrdinaryRowPlan.canonicalBlocks ())
    (instruction : WitnessInstruction)
    (member : instruction ∈ Rows.witnessInstructionsTR (block.rows Data.logicalWidth Data.publicFits)) :
    EventSupported Outside pilot templates
      (.instruction (PerApplicationCachedShift.shiftWitnessInstruction selectedShift instruction)) :=
  PiRLCCombinationOrdinaryReadSupport.canonicalInstructions_supported shapeRelation selectedShift
    block blockMember instruction member

theorem applicationInstructions_supported (templates : Array CompactRowTemplate)
    (instruction : WitnessInstruction)
    (member : instruction ∈
      (PerApplicationPackage.directApplicationPlan selectedApplication).witnessInstructions) :
    EventSupported Outside pilot templates (.instruction instruction) :=
  PiRLCCombinationOrdinaryReadSupport.applicationInstructions_supported selectedApplication
    instruction member

end NightstreamFPrime.Export.Stage1.PiRLCCombinationEventReadSupport
