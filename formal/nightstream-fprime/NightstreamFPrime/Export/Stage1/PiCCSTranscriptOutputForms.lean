import NightstreamFPrime.Export.Stage1.PiCCSPoseidonPlan.RetainedValues
import NightstreamFPrime.Export.Stage1.PiCCSTranscriptReadout
import NightstreamFPrime.Export.MatrixProgram
import NightstreamFPrime.Layout.Stage1.PiCCSOrdinarySourceSupportData
import NightstreamFPrime.Layout.Stage1.RunningTransitionPointBoundsDirect

/-!
Owns the shared forms for PiCCS transcript output lanes and the running
evaluation point. Each form is the actual output of the retained Poseidon
plan. The compact grids reconstruct the same final external layer.

This module adds no allocation, copy row, or assumption about an assignment.
-/

namespace NightstreamFPrime.Export.Stage1.PiCCSTranscriptOutputForms

open NightstreamFPrime.Export.MatrixProgram
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

abbrev ApplicationProgram := Lifecycle.Stage1.Application.Program
abbrev Geometry := PiCCSPoseidonPlan.Geometry
abbrev TranscriptIndex := Fin PiCCSOrdinarySourceSupport.transcriptInvocationCount

def invocation (index : TranscriptIndex) : Fin PiCCSPoseidonPlan.invocationCount :=
  ⟨index.val, by
    have bound : index.val < 718 := by
      simpa only [PiCCSOrdinarySourceSupport.transcriptInvocationCount_eq]
        using index.isLt
    rw [PiCCSPoseidonPlan.invocationCount_eq]
    omega⟩

def transcriptForm {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (index : TranscriptIndex) (lane : Fin Poseidon2.width) :
    SparseForm logicalWidth :=
  PiCCSPoseidonPlan.outputState geometry (invocation index) lane

theorem transcriptForm_eq_outputState
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (index : TranscriptIndex) (lane : Fin Poseidon2.width) :
    transcriptForm geometry index lane =
      PiCCSPoseidonPlan.outputState geometry (invocation index) lane := by
  rfl

def pointInvocation (coordinate : Fin Lifecycle.productionShape.cubeVariables)
    (component : Fin 2) : TranscriptIndex :=
  ⟨472 + coordinate.val * 9 + component.val, by
    have coordinateBound := coordinate.isLt
    have componentBound := component.isLt
    change coordinate.val < 28 at coordinateBound
    rw [PiCCSOrdinarySourceSupport.transcriptInvocationCount_eq]
    omega⟩

def pointForm {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (coordinate : Fin Lifecycle.productionShape.cubeVariables) (component : Fin 2) :
    SparseForm logicalWidth :=
  transcriptForm geometry (pointInvocation coordinate component)
    ⟨0, by norm_num [Poseidon2.width]⟩

theorem pointForm_eq_outputState
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (coordinate : Fin Lifecycle.productionShape.cubeVariables) (component : Fin 2) :
    pointForm geometry coordinate component =
      PiCCSPoseidonPlan.outputState geometry
        (invocation (pointInvocation coordinate component)) 0 := by
  rfl

def transcriptSourceStart : Nat := PiCCSInputs.phaseOffset + 584

def transcriptSource (index : TranscriptIndex) (lane : Fin Poseidon2.width) : Nat :=
  transcriptSourceStart + index.val * 592 + lane.val

def pointSourceStart : Nat :=
  PiCCSStarts.roundTranscriptWitnessStart +
    RunningTransitionInputs.roundSampleC0Offset

def pointSource (coordinate : Fin Lifecycle.productionShape.cubeVariables)
    (component : Fin 2) : Nat :=
  pointSourceStart + coordinate.val * RunningTransitionInputs.roundStride +
    component.val * 592

theorem pointSource_eq_transcriptSource
    (coordinate : Fin Lifecycle.productionShape.cubeVariables) (component : Fin 2) :
    pointSource coordinate component =
      transcriptSource (pointInvocation coordinate component)
        ⟨0, by norm_num [Poseidon2.width]⟩ := by
  simp only [pointSource, pointSourceStart, transcriptSource,
    transcriptSourceStart, pointInvocation,
    PiCCSStarts.roundTranscriptWitnessStart_eq, PiCCSInputs.phaseOffset_eq]
  norm_num [RunningTransitionInputs.roundSampleC0Offset,
    RunningTransitionInputs.roundStride]
  omega

theorem pointSource_c0 (coordinate : Fin Lifecycle.productionShape.cubeVariables) :
    pointSource coordinate 0 =
      PiCCSStarts.roundTranscriptWitnessStart +
        coordinate.val * RunningTransitionInputs.roundStride +
        RunningTransitionInputs.roundSampleC0Offset := by
  unfold pointSource pointSourceStart
  simp only [Fin.val_zero, Nat.zero_mul, Nat.add_zero]
  omega

theorem pointSource_c1 (coordinate : Fin Lifecycle.productionShape.cubeVariables) :
    pointSource coordinate 1 =
      PiCCSStarts.roundTranscriptWitnessStart +
        coordinate.val * RunningTransitionInputs.roundStride +
        RunningTransitionInputs.roundSampleC1Offset := by
  unfold pointSource pointSourceStart
  norm_num [RunningTransitionInputs.roundSampleC0Offset,
    RunningTransitionInputs.roundSampleC1Offset]
  omega

def transcriptGrid (program : ApplicationProgram) : SourceGrid :=
  SourceGrid.externalOfSemantic
    (PiCCSPoseidonPlan.retainedBlock program)
    (PiCCSPoseidonPlan.retainedStart program)
    (Spartan.sourceToSpartan transcriptSourceStart)
    PiCCSOrdinarySourceSupport.transcriptInvocationCount 592 1 8 8 78 86 0

def pointGrid (program : ApplicationProgram) (component : Fin 2) : SourceGrid :=
  SourceGrid.externalOfSemantic
    (PiCCSPoseidonPlan.retainedBlock program)
    (PiCCSPoseidonPlan.retainedStart program)
    (Spartan.sourceToSpartan (pointSourceStart + component.val * 592))
    Lifecycle.productionShape.cubeVariables RunningTransitionInputs.roundStride
    1 1 1 (40670 + component.val * 86) 774 0

/-- Exact compact interpretation of all pre-ordinary transcript output lanes.
The final eight retained S-box slots supply each external-layer output. -/
theorem transcriptGrid_form?
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (index : TranscriptIndex) (lane : Fin Poseidon2.width) :
    (transcriptGrid program).form? logicalWidth
        (Spartan.sourceToSpartan (transcriptSource index lane)) =
      some (transcriptForm geometry index lane) := by
  have indexBound : index.val < 718 := by
    simpa only [PiCCSOrdinarySourceSupport.transcriptInvocationCount_eq]
      using index.isLt
  have laneBound := lane.isLt
  change lane.val < 8 at laneBound
  have sourceEq :
      Spartan.sourceToSpartan (transcriptSource index lane) =
        Spartan.sourceToSpartan transcriptSourceStart +
          index.val * 592 + lane.val := by
    unfold transcriptSource
    rw [Nat.add_assoc, Spartan.sourceToSpartan_add_of_piCcsLocal]
    · omega
    · norm_num [transcriptSourceStart, PiCCSInputs.phaseOffset_eq,
        Spartan.piCcsPhaseOffset]
  let minor : Fin 1 := ⟨0, by omega⟩
  have direct := SourceGrid.form?_externalOfSemantic
    (PiCCSPoseidonPlan.retainedBlock program)
    (PiCCSPoseidonPlan.retainedStart program)
    (Spartan.sourceToSpartan transcriptSourceStart)
    PiCCSOrdinarySourceSupport.transcriptInvocationCount 592 1 8 8 78 86 0
    (PiCCSPoseidonPlan.retainedFits geometry) (by omega) (by omega)
    index minor lane (by omega) laneBound laneBound (by
      intro selected
      have selectedBound := selected.isLt
      rw [PiCCSPoseidonPlan.retainedBlock_slotCount]
      omega)
  have outputEq :
      SparseLayer.external (fun selected : Fin 8 =>
        (PiCCSPoseidonPlan.retainedBlock program).form
          (PiCCSPoseidonPlan.retainedStart program)
          (PiCCSPoseidonPlan.retainedFits geometry)
          ⟨78 + index.val * 86 + minor.val * 0 + selected.val, by
            have selectedBound := selected.isLt
            rw [PiCCSPoseidonPlan.retainedBlock_slotCount]
            omega⟩) lane = transcriptForm geometry index lane := by
    unfold transcriptForm PiCCSPoseidonPlan.outputState
      PoseidonRetainedFamily.outputState
    apply congrArg (fun state => SparseLayer.external state lane)
    funext selected
    unfold PoseidonRetainedFamily.form
    apply congrArg ((PiCCSPoseidonPlan.retainedBlock program).form
      (PiCCSPoseidonPlan.retainedStart program)
      (PiCCSPoseidonPlan.retainedFits geometry))
    apply Fin.ext
    simp [PoseidonRetainedFamily.slot, Fin.encodeProd,
      PoseidonRetainedSlots.finalRow_val, invocation]
    omega
  have laneEq : (⟨lane.val, laneBound⟩ : Fin 8) = lane := by
    apply Fin.ext
    rfl
  rw [laneEq] at direct
  have result := direct.trans
    (congrArg (fun value : SparseForm logicalWidth => some value) outputEq)
  rw [sourceEq]
  simpa [transcriptGrid, minor] using result

/-- The same compact external-layer operation supplies both running-point
components from the exact indexed PiCCS outputs. -/
theorem pointGrid_form?
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (coordinate : Fin Lifecycle.productionShape.cubeVariables) (component : Fin 2) :
    (pointGrid program component).form? logicalWidth
        (Spartan.sourceToSpartan (pointSource coordinate component)) =
      some (pointForm geometry coordinate component) := by
  have coordinateBound := coordinate.isLt
  have componentBound := component.isLt
  change coordinate.val < 28 at coordinateBound
  have sourceEq :
      Spartan.sourceToSpartan (pointSource coordinate component) =
        Spartan.sourceToSpartan (pointSourceStart + component.val * 592) +
          coordinate.val * RunningTransitionInputs.roundStride := by
    have grouped : pointSource coordinate component =
        pointSourceStart + component.val * 592 +
          coordinate.val * RunningTransitionInputs.roundStride := by
      unfold pointSource
      omega
    rw [grouped, Spartan.sourceToSpartan_add_of_piCcsLocal]
    norm_num [pointSourceStart, PiCCSStarts.roundTranscriptWitnessStart_eq,
      RunningTransitionInputs.roundSampleC0Offset, Spartan.piCcsPhaseOffset] <;> omega
  let offset : Fin 1 := ⟨0, by omega⟩
  have direct := SourceGrid.form?_externalOfSemantic
    (PiCCSPoseidonPlan.retainedBlock program)
    (PiCCSPoseidonPlan.retainedStart program)
    (Spartan.sourceToSpartan (pointSourceStart + component.val * 592))
    Lifecycle.productionShape.cubeVariables RunningTransitionInputs.roundStride
    1 1 1 (40670 + component.val * 86) 774 0
    (PiCCSPoseidonPlan.retainedFits geometry)
    (by norm_num [RunningTransitionInputs.roundStride]) (by omega)
    coordinate offset offset
    (by norm_num [RunningTransitionInputs.roundStride] <;> omega)
    (by omega) (by omega) (by
      intro selected
      have selectedBound := selected.isLt
      rw [PiCCSPoseidonPlan.retainedBlock_slotCount]
      omega)
  have outputEq :
      SparseLayer.external (fun selected : Fin 8 =>
        (PiCCSPoseidonPlan.retainedBlock program).form
          (PiCCSPoseidonPlan.retainedStart program)
          (PiCCSPoseidonPlan.retainedFits geometry)
          ⟨40670 + component.val * 86 + coordinate.val * 774 +
              offset.val * 0 + selected.val, by
            have selectedBound := selected.isLt
            rw [PiCCSPoseidonPlan.retainedBlock_slotCount]
            omega⟩) 0 = pointForm geometry coordinate component := by
    unfold pointForm transcriptForm PiCCSPoseidonPlan.outputState
      PoseidonRetainedFamily.outputState
    apply congrArg (fun state => SparseLayer.external state (0 : Fin 8))
    funext selected
    unfold PoseidonRetainedFamily.form
    apply congrArg ((PiCCSPoseidonPlan.retainedBlock program).form
      (PiCCSPoseidonPlan.retainedStart program)
      (PiCCSPoseidonPlan.retainedFits geometry))
    apply Fin.ext
    simp [PoseidonRetainedFamily.slot, Fin.encodeProd,
      PoseidonRetainedSlots.finalRow_val, invocation, pointInvocation]
    omega
  have laneEq : (⟨offset.val, by omega⟩ : Fin 8) = 0 := by
    apply Fin.ext
    rfl
  rw [laneEq] at direct
  have result := direct.trans
    (congrArg (fun value : SparseForm logicalWidth => some value) outputEq)
  rw [sourceEq]
  simpa [pointGrid, offset] using result

/-- The logical transcript source and the physical readout use one address. -/
theorem transcriptSource_column (index : TranscriptIndex) (lane : Fin 8) :
    PermutationOutput.Readout.outputColumn PiCCSTranscriptReadout.phaseStart
        index lane = Spartan.sourceToSpartan (transcriptSource index lane) := by
  have sourceEq : transcriptSource index lane =
      PiCCSInputs.phaseOffset + (index.val * 592 + 584 + lane.val) := by
    unfold transcriptSource transcriptSourceStart
    omega
  rw [sourceEq, Spartan.sourceToSpartan_add_of_piCcsLocal
    PiCCSInputs.phaseOffset (index.val * 592 + 584 + lane.val) (by
      norm_num [PiCCSInputs.phaseOffset_eq, Spartan.piCcsPhaseOffset])]
  unfold PermutationOutput.Readout.outputColumn
    PermutationOutput.Readout.witnessStart PiCCSTranscriptReadout.phaseStart
  omega

/-- Retained S-box encoding is enough to evaluate the shared transcript form.
The target is the computed readout, so no equality to an arbitrary copied
output word is assumed. -/
theorem transcriptForm_eval
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (sboxes : (PiCCSPoseidonPlan.retainedBlock program).EncodesAt
      (PiCCSPoseidonPlan.retainedStart program)
      (PiCCSPoseidonPlan.retainedFits geometry) assignment
      (PiCCSPoseidonPreservation.sourceAssignment program
        (PiRLCRetainedPreservation.sourceAssignment program base groupValue products)))
    (index : TranscriptIndex) (lane : Fin 8) :
    (transcriptForm geometry index lane).eval assignment =
      PiCCSTranscriptReadout.env
        (PerApplicationPackage.baseEnv program (SourceCompiler.sourceEnv base))
        (Spartan.sourceToSpartan (transcriptSource index lane)) := by
  rw [← transcriptSource_column]
  change _ = PermutationOutput.Readout.env PiCCSTranscriptReadout.phaseStart
    PiCCSOrdinarySourceSupport.transcriptInvocationCount
    (PerApplicationPackage.baseEnv program (SourceCompiler.sourceEnv base))
    (PermutationOutput.Readout.outputColumn PiCCSTranscriptReadout.phaseStart index lane)
  rw [PermutationOutput.Readout.env_outputColumn]
  have values := congrFun (PiCCSPoseidonPreservation.outputState_baseEnv geometry
    assignment base groupValue products sboxes (invocation index)) lane
  have startEq :
      (PiCCSPoseidonPreservation.physicalInvocation (invocation index)).witnessStart =
        PermutationOutput.Readout.witnessStart PiCCSTranscriptReadout.phaseStart index :=
    PiCCSTranscriptReadout.invocation_witnessStart index
  rw [startEq] at values
  exact values

/-- The running point is read from the same computed transcript output. -/
theorem pointForm_eval
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (sboxes : (PiCCSPoseidonPlan.retainedBlock program).EncodesAt
      (PiCCSPoseidonPlan.retainedStart program)
      (PiCCSPoseidonPlan.retainedFits geometry) assignment
      (PiCCSPoseidonPreservation.sourceAssignment program
        (PiRLCRetainedPreservation.sourceAssignment program base groupValue products)))
    (coordinate : Fin Lifecycle.productionShape.cubeVariables) (component : Fin 2) :
    (pointForm geometry coordinate component).eval assignment =
      PiCCSTranscriptReadout.env
        (PerApplicationPackage.baseEnv program (SourceCompiler.sourceEnv base))
        (Spartan.sourceToSpartan (pointSource coordinate component)) := by
  rw [pointSource_eq_transcriptSource]
  exact transcriptForm_eval geometry assignment base groupValue products sboxes
    (pointInvocation coordinate component) 0

end NightstreamFPrime.Export.Stage1.PiCCSTranscriptOutputForms
