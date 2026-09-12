import NightstreamFPrime.Export.Stage1.PiCCSPhysicalPackets
import NightstreamFPrime.Export.Stage1.PiCCSInvocationSchedule
import NightstreamFPrime.Export.Stage1.PerApplicationSourceAssignment
import NightstreamFPrime.Export.Stage1.PiCCSTranscriptEndpointPlan
import NightstreamFPrime.Export.Stage1.PiCCSTranscriptOutputCoherence

/-!
Owns the source agreement derived from completed PiCCS permutation rows.
The actual final layers determine the computed transcript readout and the
canonical retained output forms, including the final state consumed by R.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSCompletedReadout

open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Poseidon2.Duplex
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Export.Package
open PiCCSInvocations
open PiCCSPoseidonPreservation
open PerApplicationAssignmentTransportExecution

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

private theorem invocations_eq_selected :
    PiCCSInvocations.invocations logicalWidth publicFits =
      PiCCSInvocations.invocations Data.logicalWidth Data.publicFits := by
  have statementActionsEq := (PiCCSTranscriptEndpointPlan.statementActions_eq_of_shape
    logicalWidth publicFits).trans
      (PiCCSTranscriptEndpointPlan.statementActions_eq_of_shape Data.logicalWidth Data.publicFits).symm
  have statementEq : statementTrace logicalWidth publicFits = statementTrace Data.logicalWidth Data.publicFits := by
    unfold statementTrace
    rw [statementActionsEq]
  have challengeEq : challengeTrace logicalWidth publicFits = challengeTrace Data.logicalWidth Data.publicFits := by
    unfold challengeTrace
    rw [statementEq]
    rfl
  have roundShapes := (roundActions_shape_matches logicalWidth publicFits).trans
    ((congrArg (List.map Formal.Action.shape)
      ((PiCCSTranscriptEndpointPlan.roundActions_eq_of_shape logicalWidth publicFits).trans
        (PiCCSTranscriptEndpointPlan.roundActions_eq_of_shape Data.logicalWidth Data.publicFits).symm)).trans
      (roundActions_shape_matches Data.logicalWidth Data.publicFits).symm)
  have roundEq : roundTrace logicalWidth publicFits = roundTrace Data.logicalWidth Data.publicFits := by
    unfold roundTrace
    rw [challengeEq]
    exact Invocations.compileActions_eq_of_shapes roundPhase roundRowStart roundWitnessStart
      (challengeTrace Data.logicalWidth Data.publicFits).state
      (roundActions logicalWidth publicFits) (roundActions Data.logicalWidth Data.publicFits) roundShapes
  have outputActionsEq := (PiCCSTranscriptEndpointPlan.outputActions_eq_of_shape
    logicalWidth publicFits).trans
      (PiCCSTranscriptEndpointPlan.outputActions_eq_of_shape Data.logicalWidth Data.publicFits).symm
  have outputEq : outputTrace logicalWidth publicFits = outputTrace Data.logicalWidth Data.publicFits := by
    unfold outputTrace
    rw [roundEq, outputActionsEq]
  unfold PiCCSInvocations.invocations
  rw [statementEq, challengeEq, roundEq, outputEq]

private theorem physicalInvocation_mem (index : InvocationIndex) :
    physicalInvocation index ∈ PiCCSInvocations.invocations logicalWidth publicFits := by
  rw [invocations_eq_selected]
  have bounded : index.val < (PiCCSInvocations.invocations Data.logicalWidth Data.publicFits).length := by
    rw [PiCCSInvocations.invocations_length]
    simpa only [PiCCSPoseidonPlan.invocationCount_eq] using index.isLt
  have packageBound : index.val < PoseidonRetainedBlock.basePackage.permutationInvocations.length := by
    rw [PoseidonRetainedBlock.basePackage_permutationInvocations_length]
    exact (laterIndex index).isLt
  have same : physicalInvocation index =
      (PiCCSInvocations.invocations Data.logicalWidth Data.publicFits).get ⟨index.val, bounded⟩ := by
    apply Option.some.inj
    calc
      some (physicalInvocation index) =
          PoseidonRetainedBlock.basePackage.permutationInvocations[index.val]? :=
        (List.getElem?_eq_getElem packageBound).symm
      _ = (Data.permutationInvocations ())[index.val]? :=
        congrArg (fun entries : List PermutationInvocation => entries[index.val]?)
          PoseidonRetainedBlock.basePackage_permutationInvocations_eq
      _ = (PiCCSInvocations.invocations Data.logicalWidth Data.publicFits)[index.val]? := by
        rw [Data.permutationInvocations_eq, List.getElem?_append_left bounded]
      _ = some ((PiCCSInvocations.invocations Data.logicalWidth Data.publicFits).get ⟨index.val, bounded⟩) :=
        List.getElem?_eq_getElem bounded
  rw [same]
  exact List.get_mem _ ⟨index.val, bounded⟩

/-- Every retained C selector denotes a permutation constrained by the actual
completed C rows. Width erasure is derived from the existing action owners. -/
theorem invocation_holds
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (target : Env)
    (physical : NightstreamFPrime.Layout.PiCCS.v1_1.PhysicalHolds relation
      (PiCCSInputs.interface logicalWidth publicFits) PiCCSInputs.phaseOffset (Spartan.pullback target))
    (index : InvocationIndex) :
    PermutationInvocationHolds (PilotData.circuitPackage ()) (physicalInvocation index) target :=
  PiCCSPhysicalPackets.permutations_of_physical relation target physical
    (physicalInvocation index) (physicalInvocation_mem index)

private theorem readout_eq_target
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (target : Env)
    (physical : NightstreamFPrime.Layout.PiCCS.v1_1.PhysicalHolds relation
      (PiCCSInputs.interface logicalWidth publicFits) PiCCSInputs.phaseOffset (Spartan.pullback target)) :
    PiCCSTranscriptReadout.env target = target := by
  apply PiCCSTranscriptReadout.env_eq_of_invocations target
  intro index
  let selected : InvocationIndex := ⟨index.val, by
    have bounded : index.val < 718 := by
      simpa only [PiCCSOrdinarySourceSupport.transcriptInvocationCount_eq] using index.isLt
    rw [PiCCSPoseidonPlan.invocationCount_eq]
    omega⟩
  rw [show PiCCSTranscriptReadout.invocation index = physicalInvocation selected from rfl]
  exact invocation_holds relation target physical selected

/-- The copied completed assignment has the same transcript source values on
the existing Spartan source domain. Final-layer rows prove the agreement. -/
theorem transitionEnv_of_completed
    (application : Lifecycle.Stage1.Application.Program)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (target : Env)
    (suffix : Fin (PerApplicationPackage.addedPrivateColumnCount application) → F)
    (physical : NightstreamFPrime.Layout.PiCCS.v1_1.PhysicalHolds relation
      (PiCCSInputs.interface logicalWidth publicFits) PiCCSInputs.phaseOffset (Spartan.pullback target))
    (source : Nat) (bounded : source < Spartan.SourceColumnCount) :
    Spartan.pullback (RunningTransitionDirectPlan.transitionEnv application
      (PerApplicationSourceAssignment.ofCompleted application target suffix)) source =
      Spartan.pullback target source := by
  have copied (column : Nat) (columnBound : column < Spartan.spartanColumnCount) :
      RunningTransitionDirectPlan.packageEnv application
        (PerApplicationSourceAssignment.ofCompleted application target suffix) column = target column := by
    apply PerApplicationSourceAssignment.packageEnv_ofCompleted
    simpa only [Spartan.spartanColumnCount_eq, PerApplicationPackage.basePackage_totalColumnCount_eq] using columnBound
  have same := PermutationOutput.Readout.env_congr_at PiCCSTranscriptReadout.phaseStart
    PiCCSOrdinarySourceSupport.transcriptInvocationCount
    (RunningTransitionDirectPlan.packageEnv application
      (PerApplicationSourceAssignment.ofCompleted application target suffix)) target
    (Spartan.sourceToSpartan source) (copied _ (Spartan.sourceToSpartan_lt source bounded))
    (fun index lane => copied _ (PiCCSTranscriptReadout.sboxColumn_lt_spartanColumnCount index lane))
  change PiCCSTranscriptReadout.env _ (Spartan.sourceToSpartan source) = _ at same
  change PiCCSTranscriptReadout.env _ (Spartan.sourceToSpartan source) = _
  exact same.trans (congrFun (readout_eq_target relation target physical) (Spartan.sourceToSpartan source))

/-- Canonical retained C outputs equal the stored final-state lanes of the
same accepted permutation, including the final C state consumed by R. -/
theorem outputValue_of_completed
    (application : Lifecycle.Stage1.Application.Program)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (target : Env)
    (suffix : Fin (PerApplicationPackage.addedPrivateColumnCount application) → F)
    (physical : NightstreamFPrime.Layout.PiCCS.v1_1.PhysicalHolds relation
      (PiCCSInputs.interface logicalWidth publicFits) PiCCSInputs.phaseOffset (Spartan.pullback target))
    (index : InvocationIndex) :
    let raw := canonicalRawValues application (PerApplicationSourceAssignment.ofCompleted application target suffix)
    outputValue (PerApplicationCanonicalEncodes.poseidonGeometry application) raw.assignment index =
      fun lane : Fin 8 => target ((physicalInvocation index).witnessStart + 584 + lane.val) := by
  intro raw
  let geometry := PerApplicationCanonicalEncodes.poseidonGeometry application
  have sboxes := PiCCSPoseidonPlan.retainedBlock_encodesAt geometry raw.assignment raw.retainedSource
    (PiRLCRetainedGeometry.laterPoseidonFits (PiCCSPoseidonPlan.prefixGeometry geometry))
    (PerApplicationCanonicalEncodes.retainedEncodes raw).laterPoseidon
  change SparseLayer.evalState raw.assignment (PiCCSPoseidonPlan.outputState geometry index) = _
  rw [outputState_baseEnv geometry raw.assignment raw.base raw.groupValue raw.products sboxes index]
  have slots : (fun lane : Fin 8 => PerApplicationPackage.baseEnv application (SourceCompiler.sourceEnv raw.base)
      ((physicalInvocation index).witnessStart + (PoseidonRetainedSlots.localOutput (PoseidonRetainedSlots.finalRow lane)).val)) =
      fun lane : Fin 8 => target
        ((physicalInvocation index).witnessStart + (PoseidonRetainedSlots.localOutput (PoseidonRetainedSlots.finalRow lane)).val) := by
    funext lane
    apply PerApplicationSourceAssignment.packageEnv_ofCompleted
    have before := (PiCCSInvocations.invocations_scheduleWithin logicalWidth publicFits relation).2
      (physicalInvocation index) (physicalInvocation_mem index)
    have localBound := (PoseidonRetainedSlots.localOutput (PoseidonRetainedSlots.finalRow lane)).isLt
    change (PoseidonRetainedSlots.localOutput (PoseidonRetainedSlots.finalRow lane)).val < 592 at localBound
    rw [PiCCSInvocations.invocationCeiling_eq] at before
    change (physicalInvocation index).witnessStart +
      (PoseidonRetainedSlots.localOutput (PoseidonRetainedSlots.finalRow lane)).val <
        PerApplicationPackage.basePackage.layout.totalColumnCount
    rw [PerApplicationPackage.basePackage_totalColumnCount_eq]
    omega
  rw [slots]
  exact (PermutationOutput.invocation_finalLayer (physicalInvocation index) target
    (invocation_holds relation target physical index)).symm

end NightstreamFPrime.Export.Stage1.PiCCSCompletedReadout
