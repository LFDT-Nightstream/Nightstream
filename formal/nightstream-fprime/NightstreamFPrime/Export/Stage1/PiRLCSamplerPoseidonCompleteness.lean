import NightstreamFPrime.Export.Stage1.PiRLCSamplerPoseidonValues
import NightstreamFPrime.Export.Stage1.PiRLCSamplerDirectSemantics
import NightstreamFPrime.Export.Stage1.PiCCSEndpointCompleteness
import NightstreamFPrime.Layout.ProductionRelation.PoseidonSboxSourceCompleteness

/-!
Owns the direct sampler permutation plan from actual cumulative physical
rows. Entry inputs use the preceding physical sampler output, with the first
entry using the actual C endpoint. Each advance uses its scalar entry output.
The canonical template rows supply every retained S-box equation.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiRLCSamplerPoseidonCompleteness

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Sampling
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open PerApplicationAssignmentTransportExecution
open PiRLCSamplerDirectSemantics

private theorem sourceWitnessStart
    (source : Fin PiRLCSamplerPoseidonPlan.sourceCount)
    (step : Fin PiRLCSamplerPoseidonPlan.invocationsPerSource) :
    (PiRLCSamplerPoseidonValues.physicalInvocation
      (PiRLCSamplerPoseidonPlan.invocation source step)).witnessStart =
      PermutationPlan.samplerSourceWitnessStartAt source.val step := by
  rw [PiRLCSamplerPoseidonValues.physicalInvocation_witnessStart_sampler]
  exact congrArg
    (fun pair : Fin PiRLCSamplerInvocations.sourceCount ×
        Fin PermutationPlan.samplerStepsPerSource =>
      PermutationPlan.samplerSourceWitnessStartAt pair.1.val pair.2)
    (Fin.decodeProd_encodeProd (source, step))

private theorem sourceStart_local (source step : Nat) :
    Spartan.piCcsPhaseOffset ≤
      (if step = 0 then PiRLCStarts.samplerSourceLogicalStart source
       else PiRLCStarts.advanceLogicalStart source) := by
  have initial : Spartan.piCcsPhaseOffset ≤ PiRLCStarts.samplerLogicalStart := by
    norm_num [Spartan.piCcsPhaseOffset, PiRLCStarts.samplerLogicalStart,
      PiRLCStarts.phaseLogicalStart, PiRLCInputs.phaseOffset, Formal.samplerOffset]
  split <;>
    simp only [PiRLCStarts.advanceLogicalStart, Sampler.advanceOffset, Sampler.rangeOffset,
      PiRLCStarts.samplerSourceLogicalStart, SamplerChain.sourceOffset] <;> omega

private theorem invocation_input (phase rowStart start : Nat) (state : Layer.EState)
    (affine : NightstreamFPrime.Layout.Poseidon2.StateAffine state) (env : Env) :
    Layer.evalState (Pilot.canonicalInvocationEnv
      (Invocations.invocation phase rowStart start state) env)
      PoseidonScheduleTrace.canonicalState = Layer.evalState (Spartan.pullback env) state := by
  funext lane
  change Pilot.canonicalInvocationEnv (Invocations.invocation phase rowStart start state)
    env lane.val = _
  rw [Pilot.canonicalInvocationEnv_input]
  have selected : invocationInputCombination
      (Invocations.invocation phase rowStart start state) lane.val =
      Invocations.inputCombination (state lane) := by
    change (List.ofFn (fun current : Fin 16 => Invocations.inputCombination (state current))).getD
      lane.val zeroSparseCombination = _
    exact PriorStateHash.ofFn_getD (fun current : Fin 16 =>
      Invocations.inputCombination (state current)) lane zeroSparseCombination
  rw [selected]
  exact Invocations.inputCombination_eval (affine lane) env

private theorem entry_affine (source : Nat) :
    NightstreamFPrime.Layout.Poseidon2.StateAffine
      (PiRLCSamplerCompleteness.entryPermutationState source) := by
  apply NightstreamFPrime.Layout.Poseidon2.absorbE_affine
  · exact PiRLCSamplerInvocations.entryState_affine source
  · intro expression member
    rcases List.mem_map.mp member with ⟨word, _, rfl⟩
    exact R1CS.isAffine_const word

private theorem entry_absorb (source : Fin PiRLCSamplerPoseidonPlan.sourceCount)
    (state : Layer.EState) (env : Env) :
    Layer.evalState env (Hash.absorbE state (PiRLCSamplerCompleteness.entryWords source.val)) =
      fun lane => Layer.evalState env state lane + PiRLCSamplerPoseidonPlan.entryWord source lane := by
  funext lane
  fin_cases lane <;>
    simp [Layer.evalState, Hash.absorbE, PiRLCSamplerCompleteness.entryWords,
      TranscriptAbsorption.constantWords, TranscriptAbsorption.frameWords,
      PiRLCSamplerPoseidonPlan.entryWord, Expr.eval]

section Values

variable {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
  (geometry : PiCCSPoseidonPlan.Geometry application logicalWidth)
  (assignment : Assignment F logicalWidth)
  (base : Fin (PiRLCProductPlan.baseSourceWidth application) → F)
  (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 1 → F)
  (encoding : PiRLCSamplerPoseidonPreservation.Encoding geometry assignment
    (PiRLCRetainedPreservation.sourceAssignment application base groupValue))
  (packets : PiRLCPackageCompleteness.RemappedPacketRowsHold
    (RunningTransitionDirectPlan.packageEnv application base))

include groupValue encoding packets

private theorem output_state
    (source : Fin PiRLCSamplerPoseidonPlan.sourceCount)
    (step : Fin PiRLCSamplerPoseidonPlan.invocationsPerSource) :
    PiRLCSamplerPoseidonPreservation.outputValue geometry assignment
      (PiRLCSamplerPoseidonPlan.invocation source step) =
      Layer.evalState (Spartan.pullback (RunningTransitionDirectPlan.packageEnv application base))
        (Permutation.scheduleOutput
          (if step.val = 0 then PiRLCStarts.samplerSourceLogicalStart source.val
           else PiRLCStarts.advanceLogicalStart source.val)) := by
  rw [PiRLCSamplerPoseidonValues.outputValue_of_packets geometry assignment base
    groupValue encoding packets]
  funext lane
  rw [sourceWitnessStart]
  have mapped := Spartan.sourceToSpartan_add_of_piCcsLocal
    (if step.val = 0 then PiRLCStarts.samplerSourceLogicalStart source.val
     else PiRLCStarts.advanceLogicalStart source.val)
    (1080 + lane.val) (sourceStart_local source.val step.val)
  by_cases first : step.val = 0
  · simp only [PermutationPlan.samplerSourceWitnessStartAt, first, if_pos,
      PiRLCSamplerInvocations.sourceLogicalStart] at mapped ⊢
    change _ = RunningTransitionDirectPlan.packageEnv application base
      (Spartan.sourceToSpartan (PiRLCStarts.samplerSourceLogicalStart source.val + 1080 + lane.val))
    apply congrArg (RunningTransitionDirectPlan.packageEnv application base)
    simpa only [Nat.add_assoc] using mapped.symm
  · simp only [PermutationPlan.samplerSourceWitnessStartAt, first, if_false] at mapped ⊢
    change _ = RunningTransitionDirectPlan.packageEnv application base
      (Spartan.sourceToSpartan
        (PiRLCStarts.advanceLogicalStart source.val + 1080 + lane.val))
    apply congrArg (RunningTransitionDirectPlan.packageEnv application base)
    simpa only [Nat.add_assoc] using mapped.symm

private theorem previous_advance (source : Fin PiRLCSamplerPoseidonPlan.sourceCount) :
    PiRLCSamplerPoseidonPreservation.previousValue geometry assignment
      (PiRLCSamplerPoseidonPlan.invocation source ⟨1, by decide⟩) =
      Layer.evalState (Spartan.pullback (RunningTransitionDirectPlan.packageEnv application base))
        (PiRLCSamplerInvocations.advanceState
          (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits) source.val) := by
  rw [previousValue_advance, output_state geometry assignment base groupValue encoding packets]
  rw [← PiRLCSamplerInvocations.fastAdvanceState_eq]
  unfold PiRLCSamplerInvocations.fastAdvanceState
  rw [PiRLCSamplerProjection.fastProductionEntryOutput_eq_scheduleOutput]
  rfl

omit packets in
private theorem sboxes_of_invocation
    (one : assignment (PiRLCSamplerPoseidonPlan.oneColumn geometry) = 1)
    (current : Fin PiRLCSamplerPoseidonPlan.invocationCount)
    (actual : PermutationInvocation)
    (witness : actual.witnessStart =
      (PiRLCSamplerPoseidonValues.physicalInvocation current).witnessStart)
    (input : SparseLayer.evalState assignment
      ((PiRLCSamplerPoseidonPlan.interface geometry).input current) =
      Layer.evalState (Pilot.canonicalInvocationEnv actual
        (RunningTransitionDirectPlan.packageEnv application base)) PoseidonScheduleTrace.canonicalState)
    (rows : PermutationInvocationHolds (PilotData.circuitPackage ()) actual
      (RunningTransitionDirectPlan.packageEnv application base)) :
    PoseidonSboxPlan.SboxEquations
      (PoseidonSboxFamilyPlan.invocationInterface (PiRLCSamplerPoseidonPlan.interface geometry) current)
      assignment := by
  apply PoseidonSboxSourceCompleteness.equations_of_sourceRows
    (PoseidonSboxFamilyPlan.invocationInterface (PiRLCSamplerPoseidonPlan.interface geometry) current)
    assignment
    (Pilot.canonicalInvocationEnv actual (RunningTransitionDirectPlan.packageEnv application base))
    one input _ (Pilot.canonicalPermutationInvocation_implies_constraints actual _ rows)
  intro row
  change (PoseidonRetainedFamily.form (PiRLCSamplerPoseidonPlan.schedule application)
    (PiRLCSamplerPoseidonPlan.retainedStart application)
    (PiRLCSamplerPoseidonPlan.retainedFits geometry) current row).eval assignment = _
  rw [PoseidonRetainedFamily.form_eval (PiRLCSamplerPoseidonPlan.schedule application)
    (PiRLCSamplerPoseidonPlan.retainedStart application)
    (PiRLCSamplerPoseidonPlan.retainedFits geometry) assignment
    (PiRLCSamplerPoseidonPreservation.sourceAssignment application
      (PiRLCRetainedPreservation.sourceAssignment application base groupValue))
    encoding.sboxes current row]
  rw [PiRLCSamplerPoseidonValues.source_sbox,
    PoseidonRetainedSlots.output_eq_input_add_local]
  change RunningTransitionDirectPlan.packageEnv application base
      ((PiRLCSamplerPoseidonValues.physicalInvocation current).witnessStart +
        (PoseidonRetainedSlots.localOutput row).val) =
    Pilot.canonicalInvocationEnv actual (RunningTransitionDirectPlan.packageEnv application base)
      (16 + (PoseidonRetainedSlots.localOutput row).val)
  rw [Pilot.canonicalInvocationEnv_local, witness]

end Values

section Inputs

variable {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
  (geometry : PiCCSPoseidonPlan.Geometry application logicalWidth)
  (assignment : Assignment F logicalWidth)
  (base : Fin (PiRLCProductPlan.baseSourceWidth application) → F)
  (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 1 → F)
  (encoding : PiRLCSamplerPoseidonPreservation.Encoding geometry assignment
    (PiRLCRetainedPreservation.sourceAssignment application base groupValue))
  (packets : PiRLCPackageCompleteness.RemappedPacketRowsHold
    (RunningTransitionDirectPlan.packageEnv application base))
  (initial : PiRLCSamplerPoseidonPreservation.piCcsFinalValue geometry assignment =
    Layer.evalState (Spartan.pullback (RunningTransitionDirectPlan.packageEnv application base))
      (PiRLCSamplerProjection.productionInitialState
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)))

include groupValue encoding packets initial

private theorem previous_entry (source : Fin PiRLCSamplerPoseidonPlan.sourceCount) :
    PiRLCSamplerPoseidonPreservation.previousValue geometry assignment
      (PiRLCSamplerPoseidonPlan.invocation source ⟨0, by decide⟩) =
      Layer.evalState (Spartan.pullback (RunningTransitionDirectPlan.packageEnv application base))
        (PiRLCSamplerInvocations.entryState
          (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits) source.val) := by
  rw [← PiRLCSamplerInvocations.fastEntryState_eq_entryState]
  unfold PiRLCSamplerInvocations.fastEntryState PiRLCSamplerProjection.fastProductionEntryState
  rcases source with ⟨source, bounded⟩
  cases source with
  | zero => exact initial
  | succ previous =>
      rw [previousValue_entrySucc]
      have output := output_state geometry assignment base groupValue encoding packets
        ⟨previous, by omega⟩ ⟨1, by decide⟩
      simpa only [PiRLCSamplerProjection.fastChainedEntryStateFrom,
        PiRLCStarts.advanceLogicalStart, PiRLCStarts.samplerSourceLogicalStart,
        Nat.one_ne_zero, if_false] using! output

omit geometry assignment groupValue encoding initial in
private theorem entry_rows (source : Fin PiRLCSamplerPoseidonPlan.sourceCount) :
    PermutationInvocationHolds (PilotData.circuitPackage ())
      (Invocations.invocation PiRLCSamplerInvocations.phase
        (PiRLCStarts.entryRowStart source.val)
        (PiRLCSamplerInvocations.sourceLogicalStart source.val)
        (PiRLCSamplerCompleteness.entryPermutationState source.val))
      (RunningTransitionDirectPlan.packageEnv application base) := by
  exact PiRLCSamplerCompleteness.remappedPacket_implies_entryPermutations
    (RunningTransitionDirectPlan.packageEnv application base) packets source _ (by
      rw [PiRLCSamplerCompleteness.entryInvocations_eq_singleton]
      exact List.mem_singleton.mpr rfl)

private theorem entry_inputValue (source : Fin PiRLCSamplerPoseidonPlan.sourceCount) :
    PiRLCSamplerPoseidonPreservation.canonicalInput geometry assignment
      (PiRLCSamplerPoseidonPlan.invocation source ⟨0, by decide⟩) =
      Layer.evalState (Spartan.pullback (RunningTransitionDirectPlan.packageEnv application base))
        (PiRLCSamplerCompleteness.entryPermutationState source.val) := by
  let current := PiRLCSamplerPoseidonPlan.invocation source ⟨0, by decide⟩
  have previous := previous_entry geometry assignment base groupValue encoding packets initial source
  have decodedInput : PiRLCSamplerPoseidonPreservation.canonicalInput geometry assignment current =
      fun lane => PiRLCSamplerPoseidonPreservation.previousValue geometry assignment current lane +
        PiRLCSamplerPoseidonPlan.entryWord source lane := by
    simp only [current, PiRLCSamplerPoseidonPreservation.canonicalInput,
      PiRLCSamplerPoseidonPlan.descriptor_invocation, if_pos]
  have added := congrArg
    (fun value : Layer.FState => fun lane => value lane + PiRLCSamplerPoseidonPlan.entryWord source lane)
    previous
  exact decodedInput.trans (added.trans (entry_absorb source
    (PiRLCSamplerInvocations.entryState
      (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits) source.val)
    (Spartan.pullback (RunningTransitionDirectPlan.packageEnv application base))).symm)

private theorem entry_input
    (one : assignment (PiRLCSamplerPoseidonPlan.oneColumn geometry) = 1)
    (source : Fin PiRLCSamplerPoseidonPlan.sourceCount) :
    SparseLayer.evalState assignment
      ((PiRLCSamplerPoseidonPlan.interface geometry).input
        (PiRLCSamplerPoseidonPlan.invocation source ⟨0, by decide⟩)) =
      Layer.evalState (Pilot.canonicalInvocationEnv
        (Invocations.invocation PiRLCSamplerInvocations.phase
          (PiRLCStarts.entryRowStart source.val)
          (PiRLCSamplerInvocations.sourceLogicalStart source.val)
          (PiRLCSamplerCompleteness.entryPermutationState source.val))
        (RunningTransitionDirectPlan.packageEnv application base))
        PoseidonScheduleTrace.canonicalState := by
  exact (PiRLCSamplerPoseidonPreservation.inputState_eval geometry assignment one
    (PiRLCSamplerPoseidonPlan.invocation source ⟨0, by decide⟩)).trans
      ((entry_inputValue geometry assignment base groupValue encoding packets initial source).trans
        (invocation_input PiRLCSamplerInvocations.phase
          (PiRLCStarts.entryRowStart source.val)
          (PiRLCSamplerInvocations.sourceLogicalStart source.val)
          (PiRLCSamplerCompleteness.entryPermutationState source.val)
          (entry_affine source.val)
          (RunningTransitionDirectPlan.packageEnv application base)).symm)

private theorem entry_sboxes
    (one : assignment (PiRLCSamplerPoseidonPlan.oneColumn geometry) = 1)
    (current : Fin PiRLCSamplerPoseidonPlan.invocationCount)
    (source : Fin PiRLCSamplerPoseidonPlan.sourceCount)
    (atEntry : current = PiRLCSamplerPoseidonPlan.invocation source ⟨0, by decide⟩) :
    PoseidonSboxPlan.SboxEquations
      (PoseidonSboxFamilyPlan.invocationInterface (PiRLCSamplerPoseidonPlan.interface geometry)
        current) assignment := by
  have witnessEq : (Invocations.invocation PiRLCSamplerInvocations.phase
      (PiRLCStarts.entryRowStart source.val)
      (PiRLCSamplerInvocations.sourceLogicalStart source.val)
      (PiRLCSamplerCompleteness.entryPermutationState source.val)).witnessStart =
      (PiRLCSamplerPoseidonValues.physicalInvocation current).witnessStart := by
    calc
      (Invocations.invocation PiRLCSamplerInvocations.phase
      (PiRLCStarts.entryRowStart source.val)
      (PiRLCSamplerInvocations.sourceLogicalStart source.val)
      (PiRLCSamplerCompleteness.entryPermutationState source.val)).witnessStart = Spartan.sourceToSpartan
          (PiRLCSamplerInvocations.sourceLogicalStart source.val) :=
        Invocations.invocation_witnessStart PiRLCSamplerInvocations.phase
          (PiRLCStarts.entryRowStart source.val)
          (PiRLCSamplerInvocations.sourceLogicalStart source.val)
          (PiRLCSamplerCompleteness.entryPermutationState source.val)
      _ = PermutationPlan.samplerSourceWitnessStartAt source.val ⟨0, by decide⟩ := by
        simp only [PermutationPlan.samplerSourceWitnessStartAt, PiRLCSamplerInvocations.sourceLogicalStart,
          if_pos]
      _ = (PiRLCSamplerPoseidonValues.physicalInvocation
          (PiRLCSamplerPoseidonPlan.invocation source ⟨0, by decide⟩)).witnessStart :=
        (sourceWitnessStart source ⟨0, by decide⟩).symm
      _ = (PiRLCSamplerPoseidonValues.physicalInvocation current).witnessStart :=
        (congrArg (fun index => (PiRLCSamplerPoseidonValues.physicalInvocation index).witnessStart)
          atEntry).symm
  have inputEq : SparseLayer.evalState assignment
      ((PiRLCSamplerPoseidonPlan.interface geometry).input current) =
      Layer.evalState (Pilot.canonicalInvocationEnv (Invocations.invocation PiRLCSamplerInvocations.phase
      (PiRLCStarts.entryRowStart source.val)
      (PiRLCSamplerInvocations.sourceLogicalStart source.val)
      (PiRLCSamplerCompleteness.entryPermutationState source.val))
        (RunningTransitionDirectPlan.packageEnv application base))
        PoseidonScheduleTrace.canonicalState :=
    (congrArg (fun index => SparseLayer.evalState assignment
      ((PiRLCSamplerPoseidonPlan.interface geometry).input index)) atEntry).trans
      (entry_input geometry assignment base groupValue encoding packets initial one source)
  exact sboxes_of_invocation geometry assignment base groupValue encoding
    one current (Invocations.invocation PiRLCSamplerInvocations.phase
      (PiRLCStarts.entryRowStart source.val)
      (PiRLCSamplerInvocations.sourceLogicalStart source.val)
      (PiRLCSamplerCompleteness.entryPermutationState source.val)) witnessEq inputEq (entry_rows base packets source)

omit initial in
private theorem advance_sboxes
    (one : assignment (PiRLCSamplerPoseidonPlan.oneColumn geometry) = 1)
    (source : Fin PiRLCSamplerPoseidonPlan.sourceCount) :
    PoseidonSboxPlan.SboxEquations
      (PoseidonSboxFamilyPlan.invocationInterface (PiRLCSamplerPoseidonPlan.interface geometry)
        (PiRLCSamplerPoseidonPlan.invocation source ⟨1, by decide⟩)) assignment := by
  let env := RunningTransitionDirectPlan.packageEnv application base
  let state := PiRLCSamplerInvocations.advanceState
    (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits) source.val
  let actual := Invocations.invocation PiRLCSamplerInvocations.phase
    (PiRLCStarts.advanceRowStart source.val) (PiRLCStarts.advanceLogicalStart source.val) state
  have rows : PermutationInvocationHolds (PilotData.circuitPackage ()) actual env := by
    simpa only [PiRLCSamplerInvocations.advanceInvocation, PiRLCSamplerInvocations.fastAdvanceState_eq] using
      PiRLCSamplerCompleteness.remappedPacket_implies_advancePermutation env packets source
  apply sboxes_of_invocation geometry assignment base groupValue encoding
    one (PiRLCSamplerPoseidonPlan.invocation source ⟨1, by decide⟩) actual _ _ rows
  · rw [sourceWitnessStart]
    simp only [PermutationPlan.samplerSourceWitnessStartAt, Nat.one_ne_zero, if_false,
      actual, Invocations.invocation_witnessStart]
  · rw [invocation_input _ _ _ _ (PiRLCSamplerInvocations.advanceState_affine source.val) env]
    change SparseLayer.evalState assignment
      (PiRLCSamplerPoseidonPlan.inputState geometry
        (PiRLCSamplerPoseidonPlan.invocation source ⟨1, by decide⟩)) = _
    rw [PiRLCSamplerPoseidonPreservation.inputState_eval geometry assignment one]
    unfold PiRLCSamplerPoseidonPreservation.canonicalInput
    rw [PiRLCSamplerPoseidonPlan.descriptor_invocation]
    simp only [Nat.one_ne_zero, if_false]
    exact previous_advance geometry assignment base groupValue encoding packets source

end Inputs

variable {relationLogicalWidth : Nat}
  {relationPublicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth relationLogicalWidth}

private theorem initial_value
    (application : Lifecycle.Stage1.Application.Program)
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (target : Env)
    (suffix : Fin (PerApplicationPackage.addedPrivateColumnCount application) → F)
    (physical : NightstreamFPrime.Layout.PiCCS.v1_1.PhysicalHolds relation
      (PiCCSInputs.interface relationLogicalWidth relationPublicFits) PiCCSInputs.phaseOffset
      (Spartan.pullback target)) :
    let raw := canonicalRawValues application (PerApplicationSourceAssignment.ofCompleted application target suffix)
    PiRLCSamplerPoseidonPreservation.piCcsFinalValue
      (PerApplicationCanonicalEncodes.poseidonGeometry application) raw.assignment =
      Layer.evalState (Spartan.pullback (RunningTransitionDirectPlan.packageEnv application raw.base))
        (PiRLCSamplerProjection.productionInitialState
          (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)) := by
  intro raw
  funext lane
  have state : PiRLCSamplerProjection.productionInitialState
      (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits) =
      Lifecycle.PiCCS.v1_1.OutputBinding.finalState
        (PiCCSInvocations.outputInterface Data.logicalWidth Data.publicFits)
        PiCCSInvocations.outputWitnessStart := by
    change PiCCSProjection.fastOutputState Data.logicalWidth Data.publicFits = _
    exact (PiCCSProjection.fastOutputState_eq Data.logicalWidth Data.publicFits).trans
      ((congrArg Invocations.Trace.state
        (PiCCSInvocations.outputTrace_eq_semantic Data.logicalWidth Data.publicFits)).trans
          (PiCCSInvocations.outputSemanticTrace_state_matches Data.logicalWidth Data.publicFits))
  have stateColumn := (congrFun state lane).trans
    (PiCCSTranscriptEndpointPlan.outputFinalState_endpoint_of_shape Data.logicalWidth Data.publicFits lane)
  change PiCCSPoseidonPreservation.outputValue
      (PerApplicationCanonicalEncodes.poseidonGeometry application) raw.assignment
      (PiCCSTranscriptEndpointPlan.endpointInvocation PiCCSTranscriptEndpointPlan.outputFamily) lane =
    (PiRLCSamplerProjection.productionInitialState
      (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits) lane).eval
      (Spartan.pullback (RunningTransitionDirectPlan.packageEnv application raw.base))
  rw [stateColumn]
  exact (PiCCSEndpointCompleteness.endpointValue_of_completed application relation target suffix
    physical PiCCSTranscriptEndpointPlan.outputFamily lane).trans
    (PerApplicationSourceAssignment.source_ofCompleted application target suffix _
      (PiCCSTranscriptEndpointPlan.endpointColumn_lt_source PiCCSTranscriptEndpointPlan.outputFamily lane)).symm

/-- Actual cumulative rows imply the complete direct R permutation plan on
its canonical assignment. The first entry is tied to the actual C output;
every other input comes from its physical predecessor. The source templates
supply every S-box equation, so no output, endpoint, or equation assumption
is part of this selected consumer. -/
theorem rowsZero_of_completed
    (application : Lifecycle.Stage1.Application.Program)
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (ajtai : AjtaiKey (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits))
    (target : Env)
    (suffix : Fin (PerApplicationPackage.addedPrivateColumnCount application) → F)
    (physical : R1CS.RowsHold target (Spartan.remappedRows relation)) :
    (PiRLCSamplerPoseidonPlan.plan (PerApplicationCanonicalEncodes.poseidonGeometry application)).RowsZero
      (canonicalRawValues application (PerApplicationSourceAssignment.ofCompleted application target suffix)).assignment := by
  let raw := canonicalRawValues application (PerApplicationSourceAssignment.ofCompleted application target suffix)
  let geometry := PerApplicationCanonicalEncodes.poseidonGeometry application
  have packets := PiRLCRetainedCompleteness.packets_of_completed application relation ajtai target suffix physical
  have allRows := (Spartan.remappedRows_hold relation target).mp physical
  have throughD := ((PilotPiCCSPiRLCPiDECRunningTransition.physicalHolds_iff relation
    (Spartan.pullback target)).mp allRows).1
  have throughR := ((PilotPiCCSPiRLCPiDEC.physicalHolds_iff relation
    (Spartan.pullback target)).mp throughD).1
  have throughC := ((PilotPiCCSPiRLC.physicalHolds_iff relation
    (Spartan.pullback target)).mp throughR).1
  have cRows := ((PilotPiCCS.physicalHolds_iff relation (Spartan.pullback target)).mp throughC).2
  have initial := initial_value application relation target suffix cRows
  have encoding := PiRLCSamplerPoseidonPreservation.encodingOfRetained geometry raw.assignment
    raw.retainedSource _ (PerApplicationCanonicalEncodes.retainedEncodes raw).laterPoseidon
  have one := PerApplicationCanonicalAssignment.assignment_one raw
  apply PiRLCSamplerPoseidonPlan.equations_imply_rowsZero geometry raw.assignment one
  intro current
  let decoded := PiRLCSamplerPoseidonPlan.descriptor current
  have same : PiRLCSamplerPoseidonPlan.invocation decoded.1 decoded.2 = current :=
    Fin.encodeProd_decodeProd current
  rcases decoded with ⟨source, step⟩
  rcases step with ⟨step, bounded⟩
  cases step with
  | zero =>
      exact entry_sboxes geometry raw.assignment raw.base raw.groupValue
        encoding packets initial one current source same.symm
  | succ previous =>
      have previousZero : previous = 0 := by change previous + 1 < 2 at bounded; omega
      subst previous
      rw [← same]
      exact advance_sboxes geometry raw.assignment raw.base raw.groupValue encoding packets one source

end NightstreamFPrime.Export.Stage1.PiRLCSamplerPoseidonCompleteness
