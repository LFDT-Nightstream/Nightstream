import NightstreamFPrime.Export.Stage1.PiRLCRetainedCompleteness
import NightstreamFPrime.Export.Stage1.PiRLCSamplerRetainedCustody
import NightstreamFPrime.Export.PermutationOutput

/-!
Owns the exact retained sampler S-box and output readback from the existing
physical invocation list. The sampler slice follows the PiCCS slice in the
same retained block. Actual permutation rows establish the final linear layer.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiRLCSamplerPoseidonValues

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

private def laterIndex (current : Fin PiRLCSamplerPoseidonPlan.invocationCount) :
    Fin PoseidonRetainedBlock.laterInvocationCount :=
  ⟨LaterPoseidonRetainedBlocks.piCcsInvocationCount + current.val, by
    have bounded : current.val < 153 := by
      simpa only [PiRLCSamplerPoseidonPlan.invocationCount_eq] using current.isLt
    rw [PoseidonRetainedBlock.laterInvocationCount_eq]
    simp only [LaterPoseidonRetainedBlocks.piCcsInvocationCount]
    omega⟩

/-- Select the existing physical sampler invocation after the PiCCS prefix. -/
def physicalInvocation (current : Fin PiRLCSamplerPoseidonPlan.invocationCount) :
    PermutationInvocation :=
  PoseidonRetainedBlock.basePackage.permutationInvocations.get
    ⟨(laterIndex current).val, by
      rw [PoseidonRetainedBlock.basePackage_permutationInvocations_length]
      exact (laterIndex current).isLt⟩

private theorem physicalInvocation_mem
    (current : Fin PiRLCSamplerPoseidonPlan.invocationCount) :
    physicalInvocation current ∈ PiRLCSamplerInvocations.invocations
      (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits) := by
  unfold physicalInvocation PoseidonRetainedBlock.basePackage PerApplicationPackage.basePackage
  simp only [List.get_eq_getElem, Data.circuitPackage_permutationInvocations,
    Data.components_permutationInvocations, Data.permutationInvocations_eq]
  rw [List.getElem_append_right]
  · exact List.getElem_mem _
  · rw [PiCCSInvocations.invocations_length]
    change 7604 ≤ 7604 + current.val
    omega

private theorem physicalInvocation_witnessStart
    (current : Fin PiRLCSamplerPoseidonPlan.invocationCount) :
    (physicalInvocation current).witnessStart =
      PoseidonRetainedBlock.laterWitnessStart (laterIndex current) := by
  rfl

/-- The retained sampler and physical invocation use the same owned witness
start. The invocation list itself is not expanded by this consumer. -/
theorem physicalInvocation_witnessStart_sampler
    (current : Fin PiRLCSamplerPoseidonPlan.invocationCount) :
    (physicalInvocation current).witnessStart =
      PermutationPlan.samplerWitnessStartAt current := by
  rw [physicalInvocation_witnessStart]
  exact PiRLCSamplerRetainedCustody.laterWitnessStart_sampler current

private def physicalSboxColumn (current : Fin PiRLCSamplerPoseidonPlan.invocationCount)
    (row : Fin PoseidonRetainedSlots.rows.length) :
    Fin PoseidonRetainedBlock.basePackage.layout.constantColumn :=
  PoseidonRetainedBlock.laterBlock.source (Fin.encodeProd (laterIndex current, row))

private theorem physicalSboxColumn_val
    (current : Fin PiRLCSamplerPoseidonPlan.invocationCount)
    (row : Fin PoseidonRetainedSlots.rows.length) :
    (physicalSboxColumn current row).val = (physicalInvocation current).witnessStart +
      (PoseidonRetainedSlots.localOutput row).val := by
  have selected := Layout.ProductionRelation.PoseidonRetainedBlock.block_source
    PoseidonRetainedBlock.basePackage.layout.constantColumn
    PoseidonRetainedBlock.laterInvocationCount PoseidonRetainedBlock.laterWitnessStart
    PoseidonRetainedBlock.laterWitnessStart_bound (Fin.encodeProd (laterIndex current, row))
  simpa only [Fin.decodeProd_encodeProd, ← physicalInvocation_witnessStart] using selected

private theorem liftedSlice_source_val {sourceWidth middleWidth outputWidth : Nat}
    (block : LowNormBlock.Block sourceWidth) (firstFits : sourceWidth ≤ middleWidth)
    (offset count : Nat) (sliceFits : offset + count ≤ block.slotCount)
    (lastFits : middleWidth ≤ outputWidth) (selected : Fin count) :
    ((((block.lift firstFits).slice offset count sliceFits).lift lastFits).source selected).val =
      (block.source ⟨offset + selected.val,
        Nat.lt_of_lt_of_le (Nat.add_lt_add_left selected.isLt offset) sliceFits⟩).val := by
  simp only [LowNormBlock.Block.lift, LowNormBlock.Block.slice]

private theorem schedule_source_val (application : Lifecycle.Stage1.Application.Program)
    (current : Fin PiRLCSamplerPoseidonPlan.invocationCount)
    (row : Fin PoseidonRetainedSlots.rows.length) :
    ((PiRLCSamplerPoseidonPlan.schedule application).block.source
      (PoseidonRetainedFamily.slot (PiRLCSamplerPoseidonPlan.schedule application) current row)).val =
        (physicalSboxColumn current row).val := by
  let selected := PoseidonRetainedFamily.slot
    (PiRLCSamplerPoseidonPlan.schedule application) current row
  let parentSlot : Fin PoseidonRetainedBlock.laterBlock.slotCount :=
    ⟨LaterPoseidonRetainedBlocks.piCcsSlotCount + selected.val,
      Nat.lt_of_lt_of_le (Nat.add_lt_add_left selected.isLt _)
        (LaterPoseidonRetainedBlocks.samplerFits application)⟩
  have parentSlot_eq : parentSlot = Fin.encodeProd (laterIndex current, row) := by
    apply Fin.ext
    change 7604 * PoseidonRetainedSlots.rows.length +
      (PoseidonRetainedSlots.rows.length * current.val + row.val) =
      PoseidonRetainedSlots.rows.length * (7604 + current.val) + row.val
    simp only [Nat.mul_add,
      Nat.mul_comm PoseidonRetainedSlots.rows.length 7604, Nat.add_assoc]
  have sourceEq := liftedSlice_source_val PoseidonRetainedBlock.laterBlock
    (PiRLCRetainedGeometry.poseidonSourceFits application)
    LaterPoseidonRetainedBlocks.piCcsSlotCount LaterPoseidonRetainedBlocks.samplerSlotCount
    (LaterPoseidonRetainedBlocks.samplerFits application)
    (PiCCSPoseidonPlan.prefixSourceFits application) selected
  have sourceEq' : ((PiRLCSamplerPoseidonPlan.schedule application).block.source selected).val =
      (PoseidonRetainedBlock.laterBlock.source parentSlot).val := sourceEq
  change ((PiRLCSamplerPoseidonPlan.schedule application).block.source selected).val = _
  rw [sourceEq', parentSlot_eq]
  rfl

private def applicationSboxColumn (application : Lifecycle.Stage1.Application.Program)
    (current : Fin PiRLCSamplerPoseidonPlan.invocationCount)
    (row : Fin PoseidonRetainedSlots.rows.length) :
    Fin (PiRLCProductPlan.baseSourceWidth application) :=
  ⟨(physicalSboxColumn current row).val,
    Nat.lt_of_lt_of_le (physicalSboxColumn current row).isLt
      (PiRLCProductPlan.basePackage_fits application)⟩

/-- The sampler's retained source projection reads its actual private S-box
cell. It follows the existing slice and source lifts and requires no rows. -/
theorem source_sbox
    (application : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth application) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (current : Fin PiRLCSamplerPoseidonPlan.invocationCount)
    (row : Fin PoseidonRetainedSlots.rows.length) :
    PiRLCSamplerPoseidonPreservation.sourceAssignment application
      (PiRLCRetainedPreservation.sourceAssignment application base groupValue products)
      ((PiRLCSamplerPoseidonPlan.schedule application).block.source
        (PoseidonRetainedFamily.slot (PiRLCSamplerPoseidonPlan.schedule application) current row)) =
      RunningTransitionDirectPlan.packageEnv application base
        ((physicalInvocation current).witnessStart + (PoseidonRetainedSlots.localOutput row).val) := by
  have sourceEq : (PiRLCSamplerPoseidonPlan.schedule application).block.source
      (PoseidonRetainedFamily.slot (PiRLCSamplerPoseidonPlan.schedule application) current row) =
      PiCCSActionPayloadBlock.prefixColumn application
        (PiRLCRetainedPreservation.baseSourceColumn application
          (applicationSboxColumn application current row)) := by
    apply Fin.ext
    exact schedule_source_val application current row
  rw [sourceEq]
  unfold PiRLCSamplerPoseidonPreservation.sourceAssignment
  rw [PiCCSActionPayloadBlock.sourceAssignment_prefix]
  rw [PiRLCRetainedPreservation.sourceAssignment_base]
  rw [← physicalSboxColumn_val current row]
  unfold RunningTransitionDirectPlan.packageEnv PerApplicationPackage.baseEnv
  rw [PerApplicationPackage.shiftColumn_private application
    (physicalSboxColumn current row).val (physicalSboxColumn current row).isLt]
  exact (SourceCompiler.sourceEnv_at base (applicationSboxColumn application current row)).symm

/-- The exact sampler packets identify each retained output with the output
of the same actual physical invocation. Its final-layer rows supply the
linear output relation; no output equality is assumed. -/
theorem outputValue_of_packets
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth application) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (encoding : PiRLCSamplerPoseidonPreservation.Encoding geometry assignment
      (PiRLCRetainedPreservation.sourceAssignment application base groupValue products))
    (packets : PiRLCPackageCompleteness.RemappedPacketRowsHold
      (RunningTransitionDirectPlan.packageEnv application base))
    (current : Fin PiRLCSamplerPoseidonPlan.invocationCount) :
    PiRLCSamplerPoseidonPreservation.outputValue geometry assignment current =
      fun lane : Fin 8 => RunningTransitionDirectPlan.packageEnv application base
        ((physicalInvocation current).witnessStart + 584 + lane.val) := by
  rw [PiRLCSamplerPoseidonPreservation.outputValue_sourceAssignment geometry assignment
    (PiRLCRetainedPreservation.sourceAssignment application base groupValue products) encoding]
  have sourceEq : (fun lane : Fin 8 =>
      PiRLCSamplerPoseidonPreservation.sourceAssignment application
        (PiRLCRetainedPreservation.sourceAssignment application base groupValue products)
        ((PiRLCSamplerPoseidonPlan.schedule application).block.source
          (PoseidonRetainedFamily.slot (PiRLCSamplerPoseidonPlan.schedule application)
            current (PoseidonRetainedSlots.finalRow lane)))) =
      fun lane : Fin 8 => RunningTransitionDirectPlan.packageEnv application base
        ((physicalInvocation current).witnessStart +
          (PoseidonRetainedSlots.localOutput (PoseidonRetainedSlots.finalRow lane)).val) := by
    funext lane
    exact source_sbox application base groupValue products current _
  rw [sourceEq]
  exact (PermutationOutput.invocation_finalLayer (physicalInvocation current)
    (RunningTransitionDirectPlan.packageEnv application base)
    (PiRLCSamplerCompleteness.remappedPacket_implies_permutationInvocations _ packets
      _ (physicalInvocation_mem current))).symm

end NightstreamFPrime.Export.Stage1.PiRLCSamplerPoseidonValues
