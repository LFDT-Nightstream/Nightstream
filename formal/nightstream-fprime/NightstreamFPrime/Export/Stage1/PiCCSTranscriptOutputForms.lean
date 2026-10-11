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

open NightstreamFPrime.Layout.MatrixProgram
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
    have bound : index.val < 183 := by
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

/-- Round `r` reads its coin from the output of the second permutation of its
message absorb. -/
def pointInvocation (coordinate : Fin Lifecycle.productionShape.cubeVariables) :
    TranscriptIndex :=
  ⟨128 + coordinate.val * 2, by
    have coordinateBound := coordinate.isLt
    change coordinate.val < 28 at coordinateBound
    rw [PiCCSOrdinarySourceSupport.transcriptInvocationCount_eq]
    omega⟩

/-- The two coin components are rate lanes 0 and 1. -/
def pointLane (component : Fin 2) : Fin Poseidon2.width :=
  ⟨component.val, by
    have componentBound := component.isLt
    norm_num [Poseidon2.width]
    omega⟩

def pointForm {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (coordinate : Fin Lifecycle.productionShape.cubeVariables) (component : Fin 2) :
    SparseForm logicalWidth :=
  transcriptForm geometry (pointInvocation coordinate) (pointLane component)

theorem pointForm_eq_outputState
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (coordinate : Fin Lifecycle.productionShape.cubeVariables) (component : Fin 2) :
    pointForm geometry coordinate component =
      PiCCSPoseidonPlan.outputState geometry
        (invocation (pointInvocation coordinate)) (pointLane component) := by
  rfl

def transcriptSourceStart : Nat := PiCCSStarts.statementWitnessStart + 1080

def transcriptSource (index : TranscriptIndex) (lane : Fin Poseidon2.width) : Nat :=
  transcriptSourceStart + index.val * 1096 + lane.val

def pointSourceStart : Nat :=
  PiCCSStarts.roundTranscriptWitnessStart +
    RunningTransitionInputs.roundSampleC0Offset

def pointSource (coordinate : Fin Lifecycle.productionShape.cubeVariables)
    (component : Fin 2) : Nat :=
  pointSourceStart + coordinate.val * RunningTransitionInputs.roundStride +
    component.val

theorem pointSource_eq_transcriptSource
    (coordinate : Fin Lifecycle.productionShape.cubeVariables) (component : Fin 2) :
    pointSource coordinate component =
      transcriptSource (pointInvocation coordinate) (pointLane component) := by
  unfold pointSource pointSourceStart transcriptSource transcriptSourceStart
    pointInvocation pointLane
  rw [PiCCSStarts.roundTranscriptWitnessStart_eq, PiCCSStarts.statementWitnessStart_eq]
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
    PiCCSOrdinarySourceSupport.transcriptInvocationCount 1096 1 16 16 134 150 0

/-- One grid reads both coin lanes of every round. -/
def pointGrid (program : ApplicationProgram) : SourceGrid :=
  SourceGrid.externalOfSemantic
    (PiCCSPoseidonPlan.retainedBlock program)
    (PiCCSPoseidonPlan.retainedStart program)
    (Spartan.sourceToSpartan pointSourceStart)
    Lifecycle.productionShape.cubeVariables RunningTransitionInputs.roundStride
    1 2 2 19334 300 0

/-- Exact compact interpretation of all pre-ordinary transcript output lanes.
The final sixteen retained S-box slots supply each external-layer output. -/
theorem transcriptGrid_form?
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (index : TranscriptIndex) (lane : Fin Poseidon2.width) :
    (transcriptGrid program).form? logicalWidth
        (Spartan.sourceToSpartan (transcriptSource index lane)) =
      some (transcriptForm geometry index lane) := by
  have indexBound : index.val < 183 := by
    simpa only [PiCCSOrdinarySourceSupport.transcriptInvocationCount_eq]
      using index.isLt
  have laneBound := lane.isLt
  change lane.val < 16 at laneBound
  have sourceEq :
      Spartan.sourceToSpartan (transcriptSource index lane) =
        Spartan.sourceToSpartan transcriptSourceStart +
          index.val * 1096 + lane.val := by
    unfold transcriptSource
    rw [Nat.add_assoc, Spartan.sourceToSpartan_add_of_piCcsLocal]
    · omega
    · unfold transcriptSourceStart
      rw [PiCCSStarts.statementWitnessStart_eq]
      norm_num [Spartan.piCcsPhaseOffset]
  let minor : Fin 1 := ⟨0, by omega⟩
  have direct := SourceGrid.form?_externalOfSemantic
    (PiCCSPoseidonPlan.retainedBlock program)
    (PiCCSPoseidonPlan.retainedStart program)
    (Spartan.sourceToSpartan transcriptSourceStart)
    PiCCSOrdinarySourceSupport.transcriptInvocationCount 1096 1 16 16 134 150 0
    (PiCCSPoseidonPlan.retainedFits geometry) (by omega) (by omega)
    index minor lane (by omega) laneBound laneBound (by
      intro selected
      have selectedBound := selected.isLt
      rw [PiCCSPoseidonPlan.retainedBlock_slotCount]
      omega)
  have outputEq :
      SparseLayer.external (fun selected : Fin 16 =>
        (PiCCSPoseidonPlan.retainedBlock program).form
          (PiCCSPoseidonPlan.retainedStart program)
          (PiCCSPoseidonPlan.retainedFits geometry)
          ⟨134 + index.val * 150 + minor.val * 0 + selected.val, by
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
    simp only [PoseidonRetainedFamily.slot_val, PoseidonRetainedSlots.rows_length,
      PoseidonRetainedSlots.finalRow_val, invocation]
    omega
  have laneEq : (⟨lane.val, laneBound⟩ : Fin 16) = lane := by
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
    (pointGrid program).form? logicalWidth
        (Spartan.sourceToSpartan (pointSource coordinate component)) =
      some (pointForm geometry coordinate component) := by
  have coordinateBound := coordinate.isLt
  have componentBound := component.isLt
  change coordinate.val < 28 at coordinateBound
  have sourceEq :
      Spartan.sourceToSpartan (pointSource coordinate component) =
        Spartan.sourceToSpartan pointSourceStart +
          coordinate.val * RunningTransitionInputs.roundStride + 0 * 2 +
            component.val := by
    have grouped : pointSource coordinate component =
        pointSourceStart +
          (coordinate.val * RunningTransitionInputs.roundStride + component.val) := by
      unfold pointSource
      omega
    rw [grouped, Spartan.sourceToSpartan_add_of_piCcsLocal]
    · omega
    · norm_num [pointSourceStart, PiCCSStarts.roundTranscriptWitnessStart_eq,
        RunningTransitionInputs.roundSampleC0Offset, Spartan.piCcsPhaseOffset]
  let minor : Fin 1 := ⟨0, by omega⟩
  let offset : Fin 2 := component
  have direct := SourceGrid.form?_externalOfSemantic
    (PiCCSPoseidonPlan.retainedBlock program)
    (PiCCSPoseidonPlan.retainedStart program)
    (Spartan.sourceToSpartan pointSourceStart)
    Lifecycle.productionShape.cubeVariables RunningTransitionInputs.roundStride
    1 2 2 19334 300 0
    (PiCCSPoseidonPlan.retainedFits geometry)
    (by norm_num [RunningTransitionInputs.roundStride]) (by omega)
    coordinate minor offset
    (by norm_num [RunningTransitionInputs.roundStride, minor, offset]; omega)
    (by simp only [offset]; omega) (by simp only [offset]; omega) (by
      intro selected
      have selectedBound := selected.isLt
      rw [PiCCSPoseidonPlan.retainedBlock_slotCount]
      omega)
  have outputEq :
      SparseLayer.external (fun selected : Fin 16 =>
        (PiCCSPoseidonPlan.retainedBlock program).form
          (PiCCSPoseidonPlan.retainedStart program)
          (PiCCSPoseidonPlan.retainedFits geometry)
          ⟨19334 + coordinate.val * 300 + minor.val * 0 + selected.val, by
            have selectedBound := selected.isLt
            rw [PiCCSPoseidonPlan.retainedBlock_slotCount]
            omega⟩) ⟨offset.val, by simp only [offset]; omega⟩ =
        pointForm geometry coordinate component := by
    unfold pointForm transcriptForm PiCCSPoseidonPlan.outputState
      PoseidonRetainedFamily.outputState
    have laneEq : (⟨offset.val, by simp only [offset]; omega⟩ : Fin 16) =
        pointLane component := by
      apply Fin.ext
      rfl
    rw [laneEq]
    apply congrArg (fun state => SparseLayer.external state (pointLane component))
    funext selected
    unfold PoseidonRetainedFamily.form
    apply congrArg ((PiCCSPoseidonPlan.retainedBlock program).form
      (PiCCSPoseidonPlan.retainedStart program)
      (PiCCSPoseidonPlan.retainedFits geometry))
    apply Fin.ext
    simp only [PoseidonRetainedFamily.slot_val, PoseidonRetainedSlots.rows_length,
      PoseidonRetainedSlots.finalRow_val, invocation, pointInvocation, minor]
    omega
  have result := direct.trans
    (congrArg (fun value : SparseForm logicalWidth => some value) outputEq)
  rw [sourceEq]
  simpa [pointGrid, offset, minor] using result

/-- The logical transcript source and the physical readout use one address. -/
theorem transcriptSource_column (index : TranscriptIndex) (lane : Fin 16) :
    PermutationOutput.Readout.outputColumn PiCCSTranscriptReadout.transcriptStart
        index lane = Spartan.sourceToSpartan (transcriptSource index lane) := by
  have sourceEq : transcriptSource index lane =
      PiCCSStarts.statementWitnessStart + (index.val * 1096 + 1080 + lane.val) := by
    unfold transcriptSource transcriptSourceStart
    omega
  rw [sourceEq, Spartan.sourceToSpartan_add_of_piCcsLocal
    PiCCSStarts.statementWitnessStart (index.val * 1096 + 1080 + lane.val) (by
      norm_num [PiCCSStarts.statementWitnessStart_eq, Spartan.piCcsPhaseOffset])]
  unfold PermutationOutput.Readout.outputColumn
    PermutationOutput.Readout.witnessStart PiCCSTranscriptReadout.transcriptStart
  omega

/-- Retained S-box encoding is enough to evaluate the shared transcript form.
The target is the computed readout, so no equality to an arbitrary copied
output word is assumed. -/
theorem transcriptForm_eval
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 1 → F)
    (sboxes : (PiCCSPoseidonPlan.retainedBlock program).EncodesAt
      (PiCCSPoseidonPlan.retainedStart program)
      (PiCCSPoseidonPlan.retainedFits geometry) assignment
      (PiCCSPoseidonPreservation.sourceAssignment program
        (PiRLCRetainedPreservation.sourceAssignment program base groupValue)))
    (index : TranscriptIndex) (lane : Fin 16) :
    (transcriptForm geometry index lane).eval assignment =
      PiCCSTranscriptReadout.env
        (PerApplicationPackage.baseEnv program (SourceCompiler.sourceEnv base))
        (Spartan.sourceToSpartan (transcriptSource index lane)) := by
  rw [← transcriptSource_column]
  change _ = PermutationOutput.Readout.env PiCCSTranscriptReadout.transcriptStart
    PiCCSOrdinarySourceSupport.transcriptInvocationCount
    (PerApplicationPackage.baseEnv program (SourceCompiler.sourceEnv base))
    (PermutationOutput.Readout.outputColumn PiCCSTranscriptReadout.transcriptStart index lane)
  rw [PermutationOutput.Readout.env_outputColumn]
  have values := congrFun (PiCCSPoseidonPreservation.outputState_baseEnv geometry
    assignment base groupValue sboxes (invocation index)) lane
  have startEq :
      (PiCCSPoseidonPreservation.physicalInvocation (invocation index)).witnessStart =
        PermutationOutput.Readout.witnessStart PiCCSTranscriptReadout.transcriptStart index :=
    PiCCSTranscriptReadout.invocation_witnessStart index
  rw [startEq] at values
  exact values

/-- The running point is read from the same computed transcript output. -/
theorem pointForm_eval
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 1 → F)
    (sboxes : (PiCCSPoseidonPlan.retainedBlock program).EncodesAt
      (PiCCSPoseidonPlan.retainedStart program)
      (PiCCSPoseidonPlan.retainedFits geometry) assignment
      (PiCCSPoseidonPreservation.sourceAssignment program
        (PiRLCRetainedPreservation.sourceAssignment program base groupValue)))
    (coordinate : Fin Lifecycle.productionShape.cubeVariables) (component : Fin 2) :
    (pointForm geometry coordinate component).eval assignment =
      PiCCSTranscriptReadout.env
        (PerApplicationPackage.baseEnv program (SourceCompiler.sourceEnv base))
        (Spartan.sourceToSpartan (pointSource coordinate component)) := by
  rw [pointSource_eq_transcriptSource]
  exact transcriptForm_eval geometry assignment base groupValue sboxes
    (pointInvocation coordinate) (pointLane component)

end NightstreamFPrime.Export.Stage1.PiCCSTranscriptOutputForms
