import NightstreamFPrime.Export.Stage1.PiCCSPoseidonPlan.Retained
import NightstreamFPrime.Export.Stage1.PiRLCRetainedPreservation

/-!
Owns transport from retained PiCCS S-box coordinates to physical output
values. This contract is shared by the parent readers and phase preservation.
-/

namespace NightstreamFPrime.Export.Stage1.PiCCSPoseidonPreservation

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Poseidon2.Duplex
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Export.Package

def sourceAssignment (program : Lifecycle.Stage1.Application.Program)
    (prefixAssignment :
      Fin (PiCCSActionPayloadBlock.prefixSourceWidth program) → F) :
    Fin (PiCCSPoseidonPlan.sourceWidth program) → F :=
  PiCCSActionPayloadBlock.sourceAssignment program prefixAssignment

theorem sourceToSpartan_lt_basePackage (column : Nat)
    (bound : column < NightstreamFPrime.Layout.Stage1.Spartan.SourceColumnCount) :
    NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan column <
      PiRLCProductPlan.basePackage.layout.totalColumnCount := by
  have mapped := NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan_lt
    column bound
  rw [NightstreamFPrime.Layout.Stage1.Spartan.spartanColumnCount_eq] at mapped
  simpa [PiRLCProductPlan.basePackage] using mapped

def logicalPackageColumn (program : Lifecycle.Stage1.Application.Program)
    (column : Nat)
    (bound : column < NightstreamFPrime.Layout.Stage1.Spartan.SourceColumnCount) :
    Fin (PiRLCProductPlan.baseSourceWidth program) :=
  PiRLCProductPlan.shiftedPackageColumn program
    (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan column)
    (sourceToSpartan_lt_basePackage column bound)

/-- The retained source prefix preserves every column of the physical base
package, including its shifted public and constant columns. -/
theorem retainedPrefix_baseEnv
    (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (column : Nat)
    (bound : column < NightstreamFPrime.Layout.Stage1.Spartan.spartanColumnCount) :
    PerApplicationPackage.baseEnv program
        (SourceCompiler.sourceEnv (PiRLCRetainedPreservation.sourceAssignment
          program base groupValue products)) column =
      PerApplicationPackage.baseEnv program (SourceCompiler.sourceEnv base) column := by
  have packageTotal : PiRLCProductPlan.basePackage.layout.totalColumnCount = 29336725 := by
    change PerApplicationPackage.basePackage.layout.totalColumnCount = 29336725
    exact Package.circuitPackage_layout_values.2.2.2.2
  have packageBound : column < PiRLCProductPlan.basePackage.layout.totalColumnCount := by
    rw [packageTotal]
    simpa only [NightstreamFPrime.Layout.Stage1.Spartan.spartanColumnCount_eq] using bound
  have shiftedBase := PiRLCProductPlan.shiftColumn_lt_baseSourceWidth program column packageBound
  have shiftedRetained : PerApplicationPackage.shiftColumn program column <
      PiRLCRetainedGeometry.sourceWidth program := by
    unfold PiRLCRetainedGeometry.sourceWidth PiRLCFirst54DirectPlan.sourceWidth
      PiRLCFirst54DirectPlan.prefixSourceWidth PiRLCProductPlan.sourceWidth
      ProductRetainedBlock.sourceWidth FieldSuffixBlock.sourceWidth
    omega
  unfold PerApplicationPackage.baseEnv SourceCompiler.sourceEnv
  rw [dif_pos shiftedRetained, dif_pos shiftedBase]
  have sourceEq :
      (⟨PerApplicationPackage.shiftColumn program column, shiftedRetained⟩ :
        Fin (PiRLCRetainedGeometry.sourceWidth program)) =
      PiRLCRetainedPreservation.baseSourceColumn program
        (PiRLCProductPlan.shiftedPackageColumn program column packageBound) := by
    apply Fin.ext
    rfl
  rw [sourceEq, PiRLCRetainedPreservation.sourceAssignment_base]
  rfl

/-- The computed transcript view commutes with the retained-prefix embedding.
The proof checks all final S-box source bounds rather than assuming copied
transcript outputs agree. -/
theorem readout_sourceAssignment
    (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (column : Nat)
    (bound : column < NightstreamFPrime.Layout.Stage1.Spartan.spartanColumnCount) :
    PiCCSTranscriptReadout.env (PerApplicationPackage.baseEnv program
        (SourceCompiler.sourceEnv (PiRLCRetainedPreservation.sourceAssignment
          program base groupValue products))) column =
      PiCCSTranscriptReadout.env
        (PerApplicationPackage.baseEnv program (SourceCompiler.sourceEnv base)) column := by
  apply PermutationOutput.Readout.env_congr_at
  · exact retainedPrefix_baseEnv program base groupValue products column bound
  · intro index lane
    exact retainedPrefix_baseEnv program base groupValue products _
      (PiCCSTranscriptReadout.sboxColumn_lt_spartanColumnCount index lane)

/-- Actual PiCCS payload expressions read the computed transcript view of the
same shifted base assignment. -/
theorem packageEnv_sourceAssignment
    (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (column : Nat)
    (bound : column < NightstreamFPrime.Layout.Stage1.Spartan.SourceColumnCount) :
    PiCCSActionPayloadBlock.packageEnv program
        (PiRLCRetainedPreservation.sourceAssignment program base groupValue products) column =
      PiCCSTranscriptReadout.env
        (PerApplicationPackage.baseEnv program (SourceCompiler.sourceEnv base))
        (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan column) := by
  exact readout_sourceAssignment program base groupValue products _
    (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan_lt column bound)

abbrev InvocationIndex := Fin PiCCSPoseidonPlan.invocationCount

def laterIndex (index : InvocationIndex) : Fin PoseidonRetainedBlock.laterInvocationCount :=
  ⟨index.val, by
    have bound : index.val < 7604 := by
      simpa only [PiCCSPoseidonPlan.invocationCount_eq] using index.isLt
    change index.val < 7757
    omega⟩

/-- The exact package invocation selected by the PiCCS retained block. -/
def physicalInvocation (index : InvocationIndex) : PermutationInvocation :=
  PoseidonRetainedBlock.basePackage.permutationInvocations.get
    ⟨index.val, by
      rw [PoseidonRetainedBlock.basePackage_permutationInvocations_length]
      exact (laterIndex index).isLt⟩

private theorem physicalInvocation_witnessStart (index : InvocationIndex) :
    (physicalInvocation index).witnessStart =
      PoseidonRetainedBlock.laterWitnessStart (laterIndex index) := by
  rfl

private def physicalSboxColumn (index : InvocationIndex)
    (row : Fin PoseidonRetainedSlots.rows.length) :
    Fin PoseidonRetainedBlock.basePackage.layout.constantColumn :=
  PoseidonRetainedBlock.laterBlock.source
    (Fin.encodeProd (laterIndex index, row))

private theorem physicalSboxColumn_val (index : InvocationIndex)
    (row : Fin PoseidonRetainedSlots.rows.length) :
    (physicalSboxColumn index row).val =
      (physicalInvocation index).witnessStart +
        (PoseidonRetainedSlots.localOutput row).val := by
  have selected := Layout.ProductionRelation.PoseidonRetainedBlock.block_source
    PoseidonRetainedBlock.basePackage.layout.constantColumn
    PoseidonRetainedBlock.laterInvocationCount PoseidonRetainedBlock.laterWitnessStart
    PoseidonRetainedBlock.laterWitnessStart_bound
    (Fin.encodeProd (laterIndex index, row))
  simpa only [Fin.decodeProd_encodeProd, ← physicalInvocation_witnessStart]
    using selected

/-- Projection through two domain lifts and a zero-offset slice depends only
on the selected source slot. The underlying block remains opaque. -/
private theorem liftedSlice_source_val {sourceWidth middleWidth outputWidth : Nat}
    (block : LowNormBlock.Block sourceWidth)
    (firstFits : sourceWidth ≤ middleWidth) (count : Nat)
    (sliceFits : 0 + count ≤ block.slotCount) (lastFits : middleWidth ≤ outputWidth)
    (selected : Fin count) :
    ((((block.lift firstFits).slice 0 count sliceFits).lift lastFits).source selected).val =
      (block.source ⟨selected.val, by
        have fits : count ≤ block.slotCount := by simpa only [Nat.zero_add] using sliceFits
        exact Nat.lt_of_lt_of_le selected.isLt fits⟩).val := by
  simp only [LowNormBlock.Block.lift, LowNormBlock.Block.slice, Nat.zero_add]

/-- Slicing the first PiCCS invocations and lifting their source domain keep
the physical invocation-major S-box coordinate unchanged. -/
private theorem schedule_source_val (application : Lifecycle.Stage1.Application.Program)
    (index : InvocationIndex) (row : Fin PoseidonRetainedSlots.rows.length) :
    ((PiCCSPoseidonPlan.schedule application).block.source
        (PoseidonRetainedFamily.slot (PiCCSPoseidonPlan.schedule application)
          index row)).val =
      (physicalSboxColumn index row).val := by
  let selected := PoseidonRetainedFamily.slot
    (PiCCSPoseidonPlan.schedule application) index row
  let parentSlot : Fin PoseidonRetainedBlock.laterBlock.slotCount :=
    ⟨selected.val, by
      have fits : LaterPoseidonRetainedBlocks.piCcsSlotCount ≤
          PoseidonRetainedBlock.laterBlock.slotCount := by
        simpa only [Nat.zero_add] using LaterPoseidonRetainedBlocks.piCcsFits application
      exact Nat.lt_of_lt_of_le selected.isLt fits⟩
  have parentSlot_eq : parentSlot = Fin.encodeProd (laterIndex index, row) := by
    apply Fin.ext
    rfl
  have sourceEq := liftedSlice_source_val PoseidonRetainedBlock.laterBlock
    (PiRLCRetainedGeometry.poseidonSourceFits application)
    LaterPoseidonRetainedBlocks.piCcsSlotCount
    (LaterPoseidonRetainedBlocks.piCcsFits application)
    (PiCCSPoseidonPlan.prefixSourceFits application) selected
  have sourceEq' :
      ((PiCCSPoseidonPlan.schedule application).block.source selected).val =
        (PoseidonRetainedBlock.laterBlock.source parentSlot).val := by
    exact sourceEq
  change ((PiCCSPoseidonPlan.schedule application).block.source selected).val = _
  rw [sourceEq', parentSlot_eq]
  rfl

private def applicationSboxColumn (application : Lifecycle.Stage1.Application.Program)
    (index : InvocationIndex) (row : Fin PoseidonRetainedSlots.rows.length) :
    Fin (PiRLCProductPlan.baseSourceWidth application) :=
  ⟨(physicalSboxColumn index row).val,
    Nat.lt_of_lt_of_le (physicalSboxColumn index row).isLt
      (PiRLCProductPlan.basePackage_fits application)⟩

private theorem retainedSource_sbox
    (application : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth application) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (index : InvocationIndex) (row : Fin PoseidonRetainedSlots.rows.length) :
    PiCCSPoseidonPreservation.sourceAssignment application (PiRLCRetainedPreservation.sourceAssignment application base groupValue products)
        ((PiCCSPoseidonPlan.schedule application).block.source
          (PoseidonRetainedFamily.slot (PiCCSPoseidonPlan.schedule application)
            index row)) =
      PerApplicationPackage.baseEnv application (SourceCompiler.sourceEnv base)
        ((physicalInvocation index).witnessStart +
          (PoseidonRetainedSlots.localOutput row).val) := by
  have sourceEq : (PiCCSPoseidonPlan.schedule application).block.source
        (PoseidonRetainedFamily.slot (PiCCSPoseidonPlan.schedule application)
          index row) =
      PiCCSActionPayloadBlock.prefixColumn application
        (PiRLCRetainedPreservation.baseSourceColumn application
          (applicationSboxColumn application index row)) := by
    apply Fin.ext
    exact schedule_source_val application index row
  rw [sourceEq]
  unfold PiCCSPoseidonPreservation.sourceAssignment
  rw [PiCCSActionPayloadBlock.sourceAssignment_prefix]
  change PiRLCRetainedPreservation.sourceAssignment application base
      groupValue products
      (PiRLCRetainedPreservation.baseSourceColumn application
        (applicationSboxColumn application index row)) = _
  rw [PiRLCRetainedPreservation.sourceAssignment_base]
  rw [← physicalSboxColumn_val index row]
  unfold PerApplicationPackage.baseEnv
  rw [PerApplicationPackage.shiftColumn_private application
    (physicalSboxColumn index row).val (physicalSboxColumn index row).isLt]
  exact (SourceCompiler.sourceEnv_at base
    (applicationSboxColumn application index row)).symm

/-- Each output form reads the final retained S-boxes of its actual physical
invocation. This needs only the retained encoding, not accepted physical rows
or a canonical packet. -/
theorem outputState_baseEnv
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (sboxes : (PiCCSPoseidonPlan.retainedBlock program).EncodesAt
      (PiCCSPoseidonPlan.retainedStart program)
      (PiCCSPoseidonPlan.retainedFits geometry) assignment
      (sourceAssignment program (PiRLCRetainedPreservation.sourceAssignment
        program base groupValue products))) (index : InvocationIndex) :
    SparseLayer.evalState assignment (PiCCSPoseidonPlan.outputState geometry index) =
      Layer.externalF (fun lane =>
        PerApplicationPackage.baseEnv program (SourceCompiler.sourceEnv base)
          ((physicalInvocation index).witnessStart +
            (PoseidonRetainedSlots.localOutput (PoseidonRetainedSlots.finalRow lane)).val)) := by
  rw [PiCCSPoseidonPlan.outputState, PoseidonRetainedFamily.outputState_eval
    (PiCCSPoseidonPlan.schedule program)
    (PiCCSPoseidonPlan.retainedStart program)
    (PiCCSPoseidonPlan.retainedFits geometry) assignment
    (sourceAssignment program (PiRLCRetainedPreservation.sourceAssignment
      program base groupValue products)) sboxes index]
  apply congrArg Layer.externalF
  funext lane
  exact retainedSource_sbox program base groupValue products index
    (PoseidonRetainedSlots.finalRow lane)

end NightstreamFPrime.Export.Stage1.PiCCSPoseidonPreservation
