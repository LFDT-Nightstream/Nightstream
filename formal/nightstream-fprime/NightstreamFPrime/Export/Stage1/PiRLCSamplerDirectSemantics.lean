import NightstreamFPrime.Export.Stage1.DirectPiRLCSamplerCompletePrefixPlan
import NightstreamFPrime.Export.Stage1.PiCCSTranscriptEndpointPlan
import NightstreamFPrime.Export.Stage1.PiRLCSamplerRetainedCustody
import NightstreamFPrime.Export.Stage1.PiCCSTranscriptDirectSemantics
import NightstreamFPrime.Layout.Stage1.SpartanValues

/-!
Owns the semantic composition from the direct retained PiRLC sampler plans to
the existing lifecycle sampler relation.

This module does not add rows, select an application, or close a phase status.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCSamplerDirectSemantics

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

/-- Below the sampler interval, the complete retained sampler view and the
PiCCS package view evaluate every source expression identically. -/
theorem semanticEnv_eq_packageEnv_belowSampler
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 1 → F)
    (column : Nat) (below : column < PiRLCStarts.samplerLogicalStart) :
    Spartan.pullback
        (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base)
        column =
      PiCCSActionPayloadBlock.packageEnv program
        (PiRLCRetainedPreservation.sourceAssignment program base groupValue) column := by
  have sourceBound : column < Spartan.SourceColumnCount := by
    apply lt_trans below
    rw [Spartan.sourceColumnCount_eq]
    norm_num [PiRLCStarts.samplerLogicalStart,
      PiRLCStarts.phaseLogicalStart, PiRLCInputs.phaseOffset,
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerOffset]
  have mappedBound := Spartan.sourceToSpartan_lt column sourceBound
  unfold Spartan.pullback
  rw [PiRLCSamplerRetainedCustody.semanticEnv_source_eq_transitionEnv_of_beforeSampler
    geometry assignment base below]
  change RunningTransitionDirectPlan.transitionEnv program base
      (Spartan.sourceToSpartan column) =
    PiCCSTranscriptEndpointPlan.transcriptEnv program base groupValue
      (Spartan.sourceToSpartan column)
  exact (PiCCSTranscriptEndpointPlan.transcriptEnv_eq_transitionEnv_of_lt
    program base groupValue (Spartan.sourceToSpartan column)
      mappedBound).symm

/-- The complete retained sampler view and the PiCCS package view give the
same value to each lane of the production PiCCS output-binding state. -/
theorem piCcsOutputFinalState_eval_eq
    {relationLogicalWidth : Nat}
    {relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth}
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 1 → F)
    (lane : Fin Spec.Poseidon2.width) :
    (NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.finalState
        (PiCCSInvocations.outputInterface relationLogicalWidth relationPublicFits)
        PiCCSInvocations.outputWitnessStart lane).eval
        (Spartan.pullback
          (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base)) =
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.finalState
        (PiCCSInvocations.outputInterface relationLogicalWidth relationPublicFits)
        PiCCSInvocations.outputWitnessStart lane).eval
        (PiCCSActionPayloadBlock.packageEnv program
          (PiRLCRetainedPreservation.sourceAssignment program base groupValue)) := by
  apply PiCCSInvocations.outputFinalState_eval_eq_of_agree_below_samplerStart
    relationLogicalWidth relationPublicFits relation
  intro column below
  exact semanticEnv_eq_packageEnv_belowSampler geometry assignment base
    groupValue column below

/-- The retained PiCCS output used by sampler invocation zero is the last
value state of the canonical PiCCS Poseidon2 schedule. -/
theorem piCcsFinalValue_eq_outputLast
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) :
    List.ofFn (PiRLCSamplerPoseidonPreservation.piCcsFinalValue geometry
      assignment) =
      PiCCSPoseidonPreservation.valueState geometry assignment
        PiCCSTranscriptDirectSemantics.outputLast := by
  unfold PiRLCSamplerPoseidonPreservation.piCcsFinalValue
    PiRLCSamplerPoseidonPlan.piCcsFinalOutput
    PiCCSPoseidonPreservation.valueState
    PiCCSPoseidonPreservation.outputValue
  rfl

/-- The 64 exact PiCCS endpoint rows connect the retained final Poseidon2
state to the lifecycle output-binding state under the complete sampler
environment. -/
theorem endpointRows_imply_piCcsFinalState
    {relationLogicalWidth : Nat}
    {relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth}
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (poseidonGeometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (ordinaryGeometry : PiCCSOrdinaryRetainedGeometry.Geometry program
      logicalWidth)
    (samplerGeometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 1 → F)
    (encoding : PiCCSOrdinaryRetainedGeometry.Encodes ordinaryGeometry
      assignment (PiRLCRetainedPreservation.sourceAssignment program base
        groupValue))
    (rowsZero : (PiCCSTranscriptEndpointPlan.plan poseidonGeometry
      ordinaryGeometry).RowsZero assignment) :
    List.ofFn (PiRLCSamplerPoseidonPreservation.piCcsFinalValue
        poseidonGeometry assignment) =
      List.ofFn
        (NightstreamFPrime.Gadgets.Poseidon2.Layer.evalState
          (Spartan.pullback
            (PiRLCSamplerRetainedCustody.semanticEnv samplerGeometry assignment
              base))
          (NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.finalState
            (PiCCSInvocations.outputInterface relationLogicalWidth
              relationPublicFits)
            PiCCSInvocations.outputWitnessStart)) := by
  calc
    List.ofFn (PiRLCSamplerPoseidonPreservation.piCcsFinalValue
        poseidonGeometry assignment) =
        PiCCSPoseidonPreservation.valueState poseidonGeometry assignment
          PiCCSTranscriptDirectSemantics.outputLast :=
      piCcsFinalValue_eq_outputLast poseidonGeometry assignment
    _ = List.ofFn
          (NightstreamFPrime.Gadgets.Poseidon2.Layer.evalState
            (PiCCSActionPayloadBlock.packageEnv program
              (PiRLCRetainedPreservation.sourceAssignment program base
                groupValue))
            (NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.finalState
              (PiCCSInvocations.outputInterface relationLogicalWidth
                relationPublicFits)
              PiCCSInvocations.outputWitnessStart)) :=
      PiCCSTranscriptEndpointPlan.outputEndpoint_eq_finalEval poseidonGeometry
        ordinaryGeometry assignment base groupValue encoding
        rowsZero
    _ = List.ofFn
          (NightstreamFPrime.Gadgets.Poseidon2.Layer.evalState
            (Spartan.pullback
              (PiRLCSamplerRetainedCustody.semanticEnv samplerGeometry
                assignment base))
            (NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.finalState
              (PiCCSInvocations.outputInterface relationLogicalWidth
                relationPublicFits)
              PiCCSInvocations.outputWitnessStart)) := by
      apply congrArg List.ofFn
      funext lane
      exact (piCcsOutputFinalState_eval_eq relation samplerGeometry assignment
        base groupValue lane).symm

/-- Exact lifecycle expression for one retained sampler Poseidon2 output
state. -/
def retainedStateExpr
    (source : Fin PiRLCSamplerPoseidonPlan.sourceCount)
    (step : Fin PiRLCSamplerPoseidonPlan.invocationsPerSource) :
    NightstreamFPrime.Gadgets.Poseidon2.Layer.EState :=
  fun lane => Expr.var
    (PiRLCSamplerRetainedCustody.StateLocation.sourceColumn
      { source := source, step := step, lane := lane })

/-- Every retained sampler Poseidon2 output is the exact value of its
lifecycle state column under the complete semantic environment. -/
theorem outputValue_eq_retainedStateEval
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (source : Fin PiRLCSamplerPoseidonPlan.sourceCount)
    (step : Fin PiRLCSamplerPoseidonPlan.invocationsPerSource) :
    PiRLCSamplerPoseidonPreservation.outputValue
        (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry geometry) assignment
        (PiRLCSamplerPoseidonPlan.invocation source step) =
      NightstreamFPrime.Gadgets.Poseidon2.Layer.evalState
        (Spartan.pullback
          (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base))
        (retainedStateExpr source step) := by
  funext lane
  let location : PiRLCSamplerRetainedCustody.StateLocation :=
    { source := source, step := step, lane := lane }
  have custody := PiRLCSamplerRetainedCustody.semanticEnv_state geometry
    assignment base location
  change (location.form geometry).eval assignment =
    PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base
      (Spartan.sourceToSpartan location.sourceColumn)
  exact custody.symm

/-- Step zero is the scalar-entry output in the canonical circuit. -/
theorem retainedStateExpr_entry
    {relationLogicalWidth : Nat}
    {relationPublicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth relationLogicalWidth}
    (source : Fin PiRLCSamplerPoseidonPlan.sourceCount) :
    retainedStateExpr source ⟨0, by decide⟩ =
      Sampler.enteredState
        (PiRLCSamplerInvocations.sourceInterface (logicalWidth := relationLogicalWidth)
          (publicFits := relationPublicFits) source.val)
        source.val (PiRLCStarts.samplerSourceLogicalStart source.val) := by
  have projected := PiRLCSamplerProjection.fastProductionEntryOutput_eq
    (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits) source.val
  rw [PiRLCSamplerProjection.fastProductionEntryOutput_eq_scheduleOutput] at projected
  apply Eq.trans _ projected
  funext lane
  unfold retainedStateExpr PiRLCSamplerRetainedCustody.StateLocation.sourceColumn
    NightstreamFPrime.Gadgets.Poseidon2.Permutation.scheduleOutput
    NightstreamFPrime.Gadgets.Poseidon2.Permutation.freshState
  simp only [PiRLCSamplerRetainedCustody.stateOutputOffset, Fin.val_zero, Nat.zero_mul,
    Nat.add_zero, PiRLCStarts.samplerSourceLogicalStart, SamplerChain.sourceOffset]

/-- Step one is the single advance permutation and the scalar's final state. -/
theorem retainedStateExpr_sourceFinal
    {relationLogicalWidth : Nat}
    {relationPublicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth relationLogicalWidth}
    (source : Fin PiRLCSamplerPoseidonPlan.sourceCount) :
    retainedStateExpr source ⟨1, by decide⟩ =
      Sampler.outputState
        (PiRLCSamplerInvocations.sourceInterface (logicalWidth := relationLogicalWidth)
          (publicFits := relationPublicFits) source.val)
        source.val (PiRLCStarts.samplerSourceLogicalStart source.val) := by
  funext lane
  change Expr.var (PiRLCStarts.samplerLogicalStart + source.val * Sampler.logicalPrivateCount +
      1080 + 1 * 3117 + lane.val) = Expr.var
    (Sampler.advanceOffset (PiRLCStarts.samplerSourceLogicalStart source.val) + 1080 + lane.val)
  apply congrArg Expr.var
  simp only [Sampler.advanceOffset, Sampler.rangeOffset,
    NightstreamFPrime.Gadgets.Sampling.WideReduction.Program.privateCount_eq,
    PiRLCStarts.samplerSourceLogicalStart, SamplerChain.sourceOffset]

/-- Entry invocation `previous + 1` reads the final retained invocation of
source `previous`. -/
theorem previousValue_entrySucc
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (previous : Nat) (currentLt : previous + 1 <
      PiRLCSamplerPoseidonPlan.sourceCount) :
    PiRLCSamplerPoseidonPreservation.previousValue geometry assignment
        (PiRLCSamplerPoseidonPlan.invocation
          ⟨previous + 1, currentLt⟩ ⟨0, by decide⟩) =
      PiRLCSamplerPoseidonPreservation.outputValue geometry assignment
        (PiRLCSamplerPoseidonPlan.invocation
          ⟨previous, by omega⟩ ⟨1, by decide⟩) := by
  unfold PiRLCSamplerPoseidonPreservation.previousValue
  rw [dif_neg]
  · apply congrArg
      (PiRLCSamplerPoseidonPreservation.outputValue geometry assignment)
    apply Fin.ext
    simp [PiRLCSamplerPoseidonPlan.invocation, Fin.encodeProd]
    omega
  · simp [PiRLCSamplerPoseidonPlan.invocation, Fin.encodeProd]

/-- Pointwise form of the complete 17-source predecessor theorem. -/
theorem previousValue_entry_eq_chainStateFn
    {relationLogicalWidth : Nat}
    {relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth}
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (ordinaryGeometry : PiCCSOrdinaryRetainedGeometry.Geometry program
      logicalWidth)
    (samplerGeometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 1 → F)
    (encoding : PiCCSOrdinaryRetainedGeometry.Encodes ordinaryGeometry
      assignment (PiRLCRetainedPreservation.sourceAssignment program base
        groupValue))
    (endpointRows : (PiCCSTranscriptEndpointPlan.plan
      (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry samplerGeometry)
      ordinaryGeometry).RowsZero assignment)
    (source : Fin PiRLCSamplerPoseidonPlan.sourceCount) :
    PiRLCSamplerPoseidonPreservation.previousValue
        (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry samplerGeometry)
        assignment
        (PiRLCSamplerPoseidonPlan.invocation source ⟨0, by decide⟩) =
      NightstreamFPrime.Gadgets.Poseidon2.Layer.evalState
        (Spartan.pullback
          (PiRLCSamplerRetainedCustody.semanticEnv samplerGeometry assignment
            base))
        (SamplerChain.stateAtExpr
          (PiRLCSamplerRows.samplerInterface
            (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits))
          PiRLCStarts.samplerLogicalStart source.val) := by
  by_cases zero : source.val = 0
  · have sourceEq : source = ⟨0, by decide⟩ := by apply Fin.ext; exact zero
    rw [sourceEq]
    apply List.ofFn_injective
    have endpoint := endpointRows_imply_piCcsFinalState relation
      (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry samplerGeometry)
      ordinaryGeometry samplerGeometry assignment base groupValue encoding endpointRows
    simpa [PiRLCSamplerPoseidonPreservation.previousValue,
      PiRLCSamplerPoseidonPlan.invocation, Fin.encodeProd, SamplerChain.stateAtExpr,
      PiRLCSamplerRows.samplerInterface, PiRLCSamplerRows.sharedInterface,
      Formal.samplerInterface, Formal.atOffset, PiRLCInputs.interface,
      PiRLCInputs.piCcsOutputState, PiRLCInputs.piCcsOutputInterface,
      PiRLCInputs.piCcsSharedInterface] using! endpoint
  · obtain ⟨previous, sourceValue⟩ := Nat.exists_eq_succ_of_ne_zero zero
    have currentLt : previous + 1 < PiRLCSamplerPoseidonPlan.sourceCount := by
      simpa [sourceValue] using source.isLt
    have sourceEq : source = ⟨previous + 1, currentLt⟩ := by apply Fin.ext; exact sourceValue
    rw [sourceEq, previousValue_entrySucc,
      outputValue_eq_retainedStateEval samplerGeometry assignment base,
      retainedStateExpr_sourceFinal (relationLogicalWidth := relationLogicalWidth)
        (relationPublicFits := relationPublicFits)]
    rfl

/-- The direct entry invocation input is the exact chain state plus the
verifier-owned scalar-domain lane vector. -/
theorem canonicalInput_entry
    {relationLogicalWidth : Nat}
    {relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth}
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (ordinaryGeometry : PiCCSOrdinaryRetainedGeometry.Geometry program
      logicalWidth)
    (samplerGeometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 1 → F)
    (encoding : PiCCSOrdinaryRetainedGeometry.Encodes ordinaryGeometry
      assignment (PiRLCRetainedPreservation.sourceAssignment program base
        groupValue))
    (endpointRows : (PiCCSTranscriptEndpointPlan.plan
      (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry samplerGeometry)
      ordinaryGeometry).RowsZero assignment)
    (source : Fin PiRLCSamplerPoseidonPlan.sourceCount) :
    PiRLCSamplerPoseidonPreservation.canonicalInput
        (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry samplerGeometry)
        assignment
        (PiRLCSamplerPoseidonPlan.invocation source ⟨0, by decide⟩) =
      fun lane =>
        NightstreamFPrime.Gadgets.Poseidon2.Layer.evalState
            (Spartan.pullback
              (PiRLCSamplerRetainedCustody.semanticEnv samplerGeometry
                assignment base))
            (SamplerChain.stateAtExpr
              (PiRLCSamplerRows.samplerInterface
                (logicalWidth := relationLogicalWidth)
                (publicFits := relationPublicFits))
              PiRLCStarts.samplerLogicalStart source.val) lane +
          PiRLCSamplerPoseidonPlan.entryWord source lane := by
  have previous := previousValue_entry_eq_chainStateFn relation
    ordinaryGeometry samplerGeometry assignment base groupValue
    encoding endpointRows source
  unfold PiRLCSamplerPoseidonPreservation.canonicalInput
  rw [PiRLCSamplerPoseidonPlan.descriptor_invocation]
  simp only [if_pos]
  rw [previous]

/-- The lifecycle scalar-entry absorb is one Poseidon2 permutation of the
incoming state plus the exact `[4, source, 0, ..., 0]` lane vector. -/
theorem enterScalar_ofFn
    (state : NightstreamFPrime.Gadgets.Poseidon2.Layer.FState)
    (source : Fin PiRLCSamplerPoseidonPlan.sourceCount) :
    NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.enterScalar
        (List.ofFn state) source.val =
      Spec.Poseidon2.permute
        (List.ofFn fun lane => state lane +
          PiRLCSamplerPoseidonPlan.entryWord source lane) := by
  unfold NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.enterScalar
    NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Transcript.enter
    Spec.Poseidon2.absorbBlock
  simp [Spec.Poseidon2.rate, Spec.Poseidon2.width,
    PiRLCSamplerPoseidonPlan.entryWord, List.ofFn_succ]
  apply congrArg Spec.Poseidon2.permute
  norm_num [List.range, List.range.loop, List.getD, Spec.Poseidon2.ofNat, natWord]

/-- Retained Poseidon2 semantics and the endpoint/state-chain custody proofs
supply the exact verifier-owned scalar-entry child for every source. -/
theorem canonicalSemantics_imply_entry
    {relationLogicalWidth : Nat}
    {relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth}
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (ordinaryGeometry : PiCCSOrdinaryRetainedGeometry.Geometry program
      logicalWidth)
    (samplerGeometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 1 → F)
    (encoding : PiCCSOrdinaryRetainedGeometry.Encodes ordinaryGeometry
      assignment (PiRLCRetainedPreservation.sourceAssignment program base
        groupValue))
    (endpointRows : (PiCCSTranscriptEndpointPlan.plan
      (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry samplerGeometry)
      ordinaryGeometry).RowsZero assignment)
    (semantics : PiRLCSamplerPoseidonPreservation.CanonicalSemantics
      (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry samplerGeometry)
      assignment)
    (source : Fin PiRLCSamplerPoseidonPlan.sourceCount) :
    TranscriptAbsorption.SpecHolds
      (PiRLCSamplerInvocations.sourceInterface
          (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits)
          source.val)
      source.val (PiRLCStarts.samplerSourceLogicalStart source.val)
      (Spartan.pullback
        (PiRLCSamplerRetainedCustody.semanticEnv samplerGeometry assignment
          base)) := by
  let env := Spartan.pullback
    (PiRLCSamplerRetainedCustody.semanticEnv samplerGeometry assignment base)
  let chainState :=
    NightstreamFPrime.Gadgets.Poseidon2.Layer.evalState env
      (SamplerChain.stateAtExpr
        (PiRLCSamplerRows.samplerInterface
          (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits))
        PiRLCStarts.samplerLogicalStart source.val)
  unfold TranscriptAbsorption.SpecHolds
  change NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.enterScalar
      (List.ofFn chainState) source.val =
    List.ofFn
      (NightstreamFPrime.Gadgets.Poseidon2.Layer.evalState env
        (TranscriptAbsorption.output
          (PiRLCSamplerInvocations.sourceInterface
              (logicalWidth := relationLogicalWidth)
              (publicFits := relationPublicFits) source.val)
          source.val (PiRLCStarts.samplerSourceLogicalStart source.val)))
  calc
    NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.enterScalar
        (List.ofFn chainState) source.val =
        Spec.Poseidon2.permute
          (List.ofFn fun lane => chainState lane +
            PiRLCSamplerPoseidonPlan.entryWord source lane) :=
      enterScalar_ofFn chainState source
    _ = Spec.Poseidon2.permute
          (List.ofFn
            (PiRLCSamplerPoseidonPreservation.canonicalInput
              (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry samplerGeometry)
              assignment
              (PiRLCSamplerPoseidonPlan.invocation source ⟨0, by decide⟩))) := by
      rw [canonicalInput_entry relation ordinaryGeometry samplerGeometry
        assignment base groupValue encoding endpointRows source]
    _ = List.ofFn
          (PiRLCSamplerPoseidonPreservation.outputValue
            (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry samplerGeometry)
            assignment
            (PiRLCSamplerPoseidonPlan.invocation source ⟨0, by decide⟩)) :=
      (semantics.invocation
        (PiRLCSamplerPoseidonPlan.invocation source ⟨0, by decide⟩)).symm
    _ = List.ofFn
          (NightstreamFPrime.Gadgets.Poseidon2.Layer.evalState env
            (retainedStateExpr source ⟨0, by decide⟩)) :=
      congrArg List.ofFn
        (outputValue_eq_retainedStateEval samplerGeometry assignment base
          source ⟨0, by decide⟩)
    _ = List.ofFn
          (NightstreamFPrime.Gadgets.Poseidon2.Layer.evalState env
            (TranscriptAbsorption.output
              (PiRLCSamplerInvocations.sourceInterface
                  (logicalWidth := relationLogicalWidth)
                  (publicFits := relationPublicFits) source.val)
              source.val
              (PiRLCStarts.samplerSourceLogicalStart source.val))) := by
      rw [retainedStateExpr_entry source]
      rfl


/-- The advance consumes the entry output of the same scalar. -/
theorem previousValue_advance
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (source : Fin PiRLCSamplerPoseidonPlan.sourceCount) :
    PiRLCSamplerPoseidonPreservation.previousValue geometry assignment
        (PiRLCSamplerPoseidonPlan.invocation source ⟨1, by decide⟩) =
      PiRLCSamplerPoseidonPreservation.outputValue geometry assignment
        (PiRLCSamplerPoseidonPlan.invocation source ⟨0, by decide⟩) := by
  unfold PiRLCSamplerPoseidonPreservation.previousValue
  rw [dif_neg]
  · apply congrArg (PiRLCSamplerPoseidonPreservation.outputValue geometry assignment)
    apply Fin.ext
    simp [PiRLCSamplerPoseidonPlan.invocation, Fin.encodeProd]
  · simp [PiRLCSamplerPoseidonPlan.invocation, Fin.encodeProd]

theorem canonicalSemantics_imply_advance
    {relationLogicalWidth : Nat}
    {relationPublicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth relationLogicalWidth}
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (semantics : PiRLCSamplerPoseidonPreservation.CanonicalSemantics
      (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry geometry) assignment)
    (source : Fin PiRLCSamplerPoseidonPlan.sourceCount) :
    NightstreamFPrime.Gadgets.Poseidon2.Permutation.Owned.SpecHolds
      (Sampler.advanceInterface
        (PiRLCSamplerInvocations.sourceInterface (logicalWidth := relationLogicalWidth)
          (publicFits := relationPublicFits) source.val)
        source.val (PiRLCStarts.samplerSourceLogicalStart source.val))
      (PiRLCStarts.advanceLogicalStart source.val)
      (Spartan.pullback (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base)) := by
  have checked := semantics.invocation (PiRLCSamplerPoseidonPlan.invocation source ⟨1, by decide⟩)
  unfold PiRLCSamplerPoseidonPreservation.canonicalInput at checked
  rw [PiRLCSamplerPoseidonPlan.descriptor_invocation] at checked
  simp only [Nat.one_ne_zero, if_false] at checked
  rw [previousValue_advance,
    outputValue_eq_retainedStateEval geometry assignment base source ⟨1, by decide⟩,
    outputValue_eq_retainedStateEval geometry assignment base source ⟨0, by decide⟩,
    retainedStateExpr_entry (relationLogicalWidth := relationLogicalWidth)
      (relationPublicFits := relationPublicFits),
    retainedStateExpr_sourceFinal (relationLogicalWidth := relationLogicalWidth)
      (relationPublicFits := relationPublicFits)] at checked
  exact checked

end NightstreamFPrime.Export.Stage1.PiRLCSamplerDirectSemantics
