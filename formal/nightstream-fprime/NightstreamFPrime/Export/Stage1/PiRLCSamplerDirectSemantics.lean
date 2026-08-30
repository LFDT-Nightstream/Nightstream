import NightstreamFPrime.Export.Stage1.DirectPiRLCSamplerCompletePrefixPlan
import NightstreamFPrime.Export.Stage1.PiCCSTranscriptEndpointPlan
import NightstreamFPrime.Export.Stage1.PiRLCSamplerSelectorCustody

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
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (column : Nat) (below : column < PiRLCStarts.samplerLogicalStart) :
    Spartan.pullback
        (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base)
        column =
      PiCCSActionPayloadBlock.packageEnv program
        (PiRLCRetainedPreservation.sourceAssignment program base groupValue
          products) column := by
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
    PiCCSTranscriptEndpointPlan.transcriptEnv program base groupValue products
      (Spartan.sourceToSpartan column)
  exact (PiCCSTranscriptEndpointPlan.transcriptEnv_eq_transitionEnv_of_lt
    program base groupValue products (Spartan.sourceToSpartan column)
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
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
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
          (PiRLCRetainedPreservation.sourceAssignment program base groupValue
            products)) := by
  apply PiCCSInvocations.outputFinalState_eval_eq_of_agree_below_samplerStart
    relationLogicalWidth relationPublicFits relation
  intro column below
  exact semanticEnv_eq_packageEnv_belowSampler geometry assignment base
    groupValue products column below

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

/-- The 32 exact PiCCS endpoint rows connect the retained final Poseidon2
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
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (one : assignment
      (PiCCSOrdinaryRetainedGeometry.oneColumn ordinaryGeometry) = 1)
    (encoding : PiCCSOrdinaryRetainedGeometry.Encodes ordinaryGeometry
      assignment (PiRLCRetainedPreservation.sourceAssignment program base
        groupValue products))
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
                groupValue products))
            (NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.finalState
              (PiCCSInvocations.outputInterface relationLogicalWidth
                relationPublicFits)
              PiCCSInvocations.outputWitnessStart)) :=
      PiCCSTranscriptEndpointPlan.outputEndpoint_eq_finalEval poseidonGeometry
        ordinaryGeometry assignment base groupValue products one encoding
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
        base groupValue products lane).symm

/-- The endpoint-connected retained state is exactly source zero of the
authoritative production sampler chain. -/
theorem endpointRows_imply_samplerInitialState
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
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (one : assignment
      (PiCCSOrdinaryRetainedGeometry.oneColumn ordinaryGeometry) = 1)
    (encoding : PiCCSOrdinaryRetainedGeometry.Encodes ordinaryGeometry
      assignment (PiRLCRetainedPreservation.sourceAssignment program base
        groupValue products))
    (rowsZero : (PiCCSTranscriptEndpointPlan.plan poseidonGeometry
      ordinaryGeometry).RowsZero assignment) :
    List.ofFn (PiRLCSamplerPoseidonPreservation.piCcsFinalValue
        poseidonGeometry assignment) =
      SamplerChain.evalInitialState
        (PiRLCSamplerRows.samplerInterface
          (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits))
        PiRLCStarts.samplerLogicalStart
        (Spartan.pullback
          (PiRLCSamplerRetainedCustody.semanticEnv samplerGeometry assignment
            base)) := by
  simpa [SamplerChain.evalInitialState, SamplerChain.evalStateAt,
    SamplerChain.stateAtExpr, Sampler.evalState,
    PiRLCSamplerRows.samplerInterface, PiRLCSamplerRows.sharedInterface,
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerInterface,
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.atOffset,
    PiRLCInputs.interface, PiRLCInputs.piCcsOutputState,
    PiRLCInputs.piCcsOutputInterface, PiRLCInputs.piCcsSharedInterface] using
      endpointRows_imply_piCcsFinalState relation poseidonGeometry
        ordinaryGeometry samplerGeometry assignment base groupValue products
        one encoding rowsZero

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

/-- Step zero of each retained source is exactly the lifecycle scalar-entry
output state. -/
theorem retainedStateExpr_entry
    {relationLogicalWidth : Nat}
    {relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth}
    (source : Fin PiRLCSamplerPoseidonPlan.sourceCount) :
    retainedStateExpr source ⟨0, by decide⟩ =
      TranscriptAbsorption.output
        (Sampler.entryInterface
          (PiRLCSamplerOrdinaryRows.sourceInterface
            (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits)
            source.val))
        source.val (PiRLCStarts.samplerSourceLogicalStart source.val) := by
  calc
    retainedStateExpr source ⟨0, by decide⟩ =
        PiRLCSamplerOrdinaryRows.fastEntryOutput
          (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits)
          source.val := by
      funext lane
      unfold retainedStateExpr
        PiRLCSamplerRetainedCustody.StateLocation.sourceColumn
        PiRLCSamplerOrdinaryRows.fastEntryOutput
      rw [PiRLCSamplerProjection.fastProductionEntryOutput_eq_scheduleOutput]
      unfold NightstreamFPrime.Gadgets.Poseidon2.Permutation.scheduleOutput
        NightstreamFPrime.Gadgets.Poseidon2.Permutation.freshState
      change Expr.var
          (PiRLCStarts.samplerLogicalStart +
            source.val * Sampler.logicalPrivateCount + 584 +
              0 * DigestWindow.logicalPrivateCount + lane.val) =
        Expr.var
          (PiRLCStarts.samplerSourceLogicalStart source.val + 584 + lane.val)
      rw [show
        PiRLCStarts.samplerLogicalStart +
              source.val * Sampler.logicalPrivateCount + 584 +
                0 * DigestWindow.logicalPrivateCount + lane.val =
            PiRLCStarts.samplerSourceLogicalStart source.val + 584 + lane.val by
        simp [PiRLCStarts.samplerSourceLogicalStart,
          Sampler.logicalPrivateCount]]
    _ = _ := PiRLCSamplerOrdinaryRows.fastEntryOutput_eq source.val

/-- Retained step `round + 1` is exactly the lifecycle output state of that
digest window. -/
theorem retainedStateExpr_window
    {relationLogicalWidth : Nat}
    {relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth}
    (source : Fin PiRLCSamplerPoseidonPlan.sourceCount)
    (round : Fin PiRLCSamplerOrdinaryRetainedBlocks.roundCount) :
    retainedStateExpr source
        ⟨round.val + 1, by
          have roundLt := round.isLt
          change round.val < 8 at roundLt
          change round.val + 1 < 9
          omega⟩ =
      DigestWindow.output
        (PiRLCSamplerOrdinaryRows.windowInterface
          (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits)
          source.val round.val)
        (PiRLCStarts.windowLogicalStart source.val round.val) := by
  funext lane
  unfold retainedStateExpr
    PiRLCSamplerRetainedCustody.StateLocation.sourceColumn
    DigestWindow.output
    NightstreamFPrime.Gadgets.Poseidon2.Permutation.Owned.output
    NightstreamFPrime.Gadgets.Poseidon2.Permutation.scheduleOutput
    NightstreamFPrime.Gadgets.Poseidon2.Permutation.freshState
    DigestWindow.permutationOffset
  change Expr.var
      (PiRLCStarts.samplerLogicalStart +
        source.val * Sampler.logicalPrivateCount + 584 +
          (round.val + 1) * DigestWindow.logicalPrivateCount + lane.val) =
    Expr.var
      (PiRLCStarts.windowLogicalStart source.val round.val +
        4 * DigestLane.logicalPrivateCount + 584 + lane.val)
  rw [show
    PiRLCStarts.samplerLogicalStart +
          source.val * Sampler.logicalPrivateCount + 584 +
            (round.val + 1) * DigestWindow.logicalPrivateCount + lane.val =
        PiRLCStarts.windowLogicalStart source.val round.val +
          4 * DigestLane.logicalPrivateCount + 584 + lane.val by
    simp [PiRLCStarts.windowLogicalStart,
      PiRLCStarts.samplerSourceLogicalStart, Sampler.logicalPrivateCount,
      DigestWindow.logicalPrivateCount, DigestLane.logicalPrivateCount]
    omega]

/-- Retained step `round` is exactly the lifecycle initial state of that
digest window. -/
theorem retainedStateExpr_windowInitialState
    {relationLogicalWidth : Nat}
    {relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth}
    (source : Fin PiRLCSamplerPoseidonPlan.sourceCount)
    (round : Fin PiRLCSamplerOrdinaryRetainedBlocks.roundCount) :
    retainedStateExpr source
        ⟨round.val, lt_trans round.isLt (by decide)⟩ =
      Sampler.windowInitialState
        (PiRLCSamplerOrdinaryRows.sourceInterface
          (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits)
          source.val)
        source.val (PiRLCStarts.samplerSourceLogicalStart source.val)
        round.val := by
  by_cases zero : round.val = 0
  · have stepEq :
        (⟨round.val, lt_trans round.isLt (by decide)⟩ :
          Fin PiRLCSamplerPoseidonPlan.invocationsPerSource) =
            ⟨0, by decide⟩ := by
      apply Fin.ext
      exact zero
    rw [stepEq, zero]
    simpa [Sampler.windowInitialState] using retainedStateExpr_entry source
  · obtain ⟨previous, roundEq⟩ := Nat.exists_eq_succ_of_ne_zero zero
    have previousLt : previous <
        PiRLCSamplerOrdinaryRetainedBlocks.roundCount := by
      have roundLt := round.isLt
      change round.val < 8 at roundLt
      rw [roundEq] at roundLt
      change previous < 8
      omega
    simpa [roundEq, Sampler.windowInitialState] using
      retainedStateExpr_window source ⟨previous, previousLt⟩

def priorStep (round : Fin PiRLCSamplerOrdinaryRetainedBlocks.roundCount) :
    Fin PiRLCSamplerPoseidonPlan.invocationsPerSource :=
  ⟨round.val, lt_trans round.isLt (by decide)⟩

def windowStep (round : Fin PiRLCSamplerOrdinaryRetainedBlocks.roundCount) :
    Fin PiRLCSamplerPoseidonPlan.invocationsPerSource :=
  ⟨round.val + 1, by
    have roundLt := round.isLt
    change round.val < 8 at roundLt
    change round.val + 1 < 9
    omega⟩

/-- The predecessor of one window invocation is the preceding retained step
of the same scalar source. -/
theorem previousValue_window
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (source : Fin PiRLCSamplerPoseidonPlan.sourceCount)
    (round : Fin PiRLCSamplerOrdinaryRetainedBlocks.roundCount) :
    PiRLCSamplerPoseidonPreservation.previousValue geometry assignment
        (PiRLCSamplerPoseidonPlan.invocation source (windowStep round)) =
      PiRLCSamplerPoseidonPreservation.outputValue geometry assignment
        (PiRLCSamplerPoseidonPlan.invocation source (priorStep round)) := by
  unfold PiRLCSamplerPoseidonPreservation.previousValue
  rw [dif_neg]
  · apply congrArg
      (PiRLCSamplerPoseidonPreservation.outputValue geometry assignment)
    apply Fin.ext
    simp [PiRLCSamplerPoseidonPlan.invocation, Fin.encodeProd,
      priorStep, windowStep]
  · simp [PiRLCSamplerPoseidonPlan.invocation, Fin.encodeProd, windowStep]

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
          ⟨previous, by omega⟩ ⟨8, by decide⟩) := by
  unfold PiRLCSamplerPoseidonPreservation.previousValue
  rw [dif_neg]
  · apply congrArg
      (PiRLCSamplerPoseidonPreservation.outputValue geometry assignment)
    apply Fin.ext
    simp [PiRLCSamplerPoseidonPlan.invocation, Fin.encodeProd]
    omega
  · simp [PiRLCSamplerPoseidonPlan.invocation, Fin.encodeProd]

/-- Retained step eight is exactly the lifecycle final state of one scalar
sampler. -/
theorem retainedStateExpr_sourceFinal
    {relationLogicalWidth : Nat}
    {relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth}
    (source : Fin PiRLCSamplerPoseidonPlan.sourceCount) :
    retainedStateExpr source ⟨8, by decide⟩ =
      Sampler.outputState
        (PiRLCSamplerOrdinaryRows.sourceInterface
          (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits)
          source.val)
        source.val (PiRLCStarts.samplerSourceLogicalStart source.val) := by
  simpa [Sampler.outputState, Sampler.digestRoundCount,
    PiRLCSamplerOrdinaryRows.windowInterface,
    PiRLCStarts.windowLogicalStart, Sampler.windowOffset, Sampler.windowBase,
    Sampler.entryPrivateCount, DigestWindow.logicalPrivateCount] using
      retainedStateExpr_window source ⟨7, by decide⟩

/-- The evaluated final retained state of source `previous` is exactly chain
state `previous + 1`. -/
theorem sourceFinalEval_eq_chainStateSucc
    {relationLogicalWidth : Nat}
    {relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth}
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (previous : Nat) (currentLt : previous + 1 <
      PiRLCSamplerPoseidonPlan.sourceCount) :
    List.ofFn
        (NightstreamFPrime.Gadgets.Poseidon2.Layer.evalState
          (Spartan.pullback
            (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base))
          (retainedStateExpr ⟨previous, by omega⟩ ⟨8, by decide⟩)) =
      SamplerChain.evalStateAt
        (PiRLCSamplerRows.samplerInterface
          (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits))
        PiRLCStarts.samplerLogicalStart
        (Spartan.pullback
          (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base))
        (previous + 1) := by
  rw [retainedStateExpr_sourceFinal ⟨previous, by omega⟩]
  rfl

/-- For every source, the direct entry invocation's previous value is exactly
the authoritative sampler-chain state at that source. Source zero is linked
through the PiCCS endpoint rows; later sources use the preceding final state. -/
theorem previousValue_entry_eq_chainState
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
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (one : assignment
      (PiCCSOrdinaryRetainedGeometry.oneColumn ordinaryGeometry) = 1)
    (encoding : PiCCSOrdinaryRetainedGeometry.Encodes ordinaryGeometry
      assignment (PiRLCRetainedPreservation.sourceAssignment program base
        groupValue products))
    (endpointRows : (PiCCSTranscriptEndpointPlan.plan
      (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry samplerGeometry)
      ordinaryGeometry).RowsZero assignment)
    (source : Fin PiRLCSamplerPoseidonPlan.sourceCount) :
    List.ofFn
        (PiRLCSamplerPoseidonPreservation.previousValue
          (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry samplerGeometry)
          assignment
          (PiRLCSamplerPoseidonPlan.invocation source ⟨0, by decide⟩)) =
      SamplerChain.evalStateAt
        (PiRLCSamplerRows.samplerInterface
          (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits))
        PiRLCStarts.samplerLogicalStart
        (Spartan.pullback
          (PiRLCSamplerRetainedCustody.semanticEnv samplerGeometry assignment
            base))
        source.val := by
  by_cases zero : source.val = 0
  · have sourceEq : source = ⟨0, by decide⟩ := by
      apply Fin.ext
      exact zero
    rw [sourceEq]
    simpa [PiRLCSamplerPoseidonPreservation.previousValue,
      PiRLCSamplerPoseidonPlan.invocation, Fin.encodeProd,
      SamplerChain.evalInitialState] using
      endpointRows_imply_samplerInitialState relation
        (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry samplerGeometry)
        ordinaryGeometry samplerGeometry assignment base groupValue products
        one encoding endpointRows
  · obtain ⟨previous, sourceValue⟩ := Nat.exists_eq_succ_of_ne_zero zero
    have currentLt : previous + 1 <
        PiRLCSamplerPoseidonPlan.sourceCount := by
      simpa [sourceValue] using source.isLt
    have sourceEq : source = ⟨previous + 1, currentLt⟩ := by
      apply Fin.ext
      exact sourceValue
    rw [sourceEq]
    calc
      List.ofFn
          (PiRLCSamplerPoseidonPreservation.previousValue
            (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry samplerGeometry)
            assignment
            (PiRLCSamplerPoseidonPlan.invocation
              ⟨previous + 1, currentLt⟩ ⟨0, by decide⟩)) =
          List.ofFn
            (PiRLCSamplerPoseidonPreservation.outputValue
              (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry samplerGeometry)
              assignment
              (PiRLCSamplerPoseidonPlan.invocation
                ⟨previous, by omega⟩ ⟨8, by decide⟩)) :=
        congrArg List.ofFn
          (previousValue_entrySucc
            (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry samplerGeometry)
            assignment previous currentLt)
      _ = List.ofFn
            (NightstreamFPrime.Gadgets.Poseidon2.Layer.evalState
              (Spartan.pullback
                (PiRLCSamplerRetainedCustody.semanticEnv samplerGeometry
                  assignment base))
              (retainedStateExpr ⟨previous, by omega⟩ ⟨8, by decide⟩)) :=
        congrArg List.ofFn
          (outputValue_eq_retainedStateEval samplerGeometry assignment base
            ⟨previous, by omega⟩ ⟨8, by decide⟩)
      _ = SamplerChain.evalStateAt
            (PiRLCSamplerRows.samplerInterface
              (logicalWidth := relationLogicalWidth)
              (publicFits := relationPublicFits))
            PiRLCStarts.samplerLogicalStart
            (Spartan.pullback
              (PiRLCSamplerRetainedCustody.semanticEnv samplerGeometry
                assignment base))
            (previous + 1) :=
        sourceFinalEval_eq_chainStateSucc samplerGeometry assignment base
          previous currentLt

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
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (one : assignment
      (PiCCSOrdinaryRetainedGeometry.oneColumn ordinaryGeometry) = 1)
    (encoding : PiCCSOrdinaryRetainedGeometry.Encodes ordinaryGeometry
      assignment (PiRLCRetainedPreservation.sourceAssignment program base
        groupValue products))
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
  apply List.ofFn_injective
  simpa [SamplerChain.evalStateAt, Sampler.evalState] using
    previousValue_entry_eq_chainState relation ordinaryGeometry samplerGeometry
      assignment base groupValue products one encoding endpointRows source

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
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (one : assignment
      (PiCCSOrdinaryRetainedGeometry.oneColumn ordinaryGeometry) = 1)
    (encoding : PiCCSOrdinaryRetainedGeometry.Encodes ordinaryGeometry
      assignment (PiRLCRetainedPreservation.sourceAssignment program base
        groupValue products))
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
    ordinaryGeometry samplerGeometry assignment base groupValue products one
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
    NightstreamFPrime.Lifecycle.Transcript.absorb
    Spec.Poseidon2.absorbBlock
  simp [Spec.Poseidon2.rate, Spec.Poseidon2.width,
    PiRLCSamplerPoseidonPlan.entryWord, List.ofFn_succ]
  apply congrArg Spec.Poseidon2.permute
  norm_num [List.range, List.range.loop, List.getD]

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
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (one : assignment
      (PiCCSOrdinaryRetainedGeometry.oneColumn ordinaryGeometry) = 1)
    (encoding : PiCCSOrdinaryRetainedGeometry.Encodes ordinaryGeometry
      assignment (PiRLCRetainedPreservation.sourceAssignment program base
        groupValue products))
    (endpointRows : (PiCCSTranscriptEndpointPlan.plan
      (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry samplerGeometry)
      ordinaryGeometry).RowsZero assignment)
    (semantics : PiRLCSamplerPoseidonPreservation.CanonicalSemantics
      (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry samplerGeometry)
      assignment)
    (source : Fin PiRLCSamplerPoseidonPlan.sourceCount) :
    TranscriptAbsorption.SpecHolds
      (Sampler.entryInterface
        (PiRLCSamplerOrdinaryRows.sourceInterface
          (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits)
          source.val))
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
          (Sampler.entryInterface
            (PiRLCSamplerOrdinaryRows.sourceInterface
              (logicalWidth := relationLogicalWidth)
              (publicFits := relationPublicFits) source.val))
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
        assignment base groupValue products one encoding endpointRows source]
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
              (Sampler.entryInterface
                (PiRLCSamplerOrdinaryRows.sourceInterface
                  (logicalWidth := relationLogicalWidth)
                  (publicFits := relationPublicFits) source.val))
              source.val
              (PiRLCStarts.samplerSourceLogicalStart source.val))) := by
      rw [retainedStateExpr_entry source]

/-- Every direct window invocation receives exactly the lifecycle window
initial state under the complete semantic environment. -/
theorem canonicalInput_window
    {relationLogicalWidth : Nat}
    {relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth}
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (source : Fin PiRLCSamplerPoseidonPlan.sourceCount)
    (round : Fin PiRLCSamplerOrdinaryRetainedBlocks.roundCount) :
    PiRLCSamplerPoseidonPreservation.canonicalInput
        (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry geometry) assignment
        (PiRLCSamplerPoseidonPlan.invocation source (windowStep round)) =
      NightstreamFPrime.Gadgets.Poseidon2.Layer.evalState
        (Spartan.pullback
          (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base))
        (Sampler.windowInitialState
          (PiRLCSamplerOrdinaryRows.sourceInterface
            (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits)
            source.val)
          source.val (PiRLCStarts.samplerSourceLogicalStart source.val)
          round.val) := by
  calc
    PiRLCSamplerPoseidonPreservation.canonicalInput
        (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry geometry) assignment
        (PiRLCSamplerPoseidonPlan.invocation source (windowStep round)) =
        PiRLCSamplerPoseidonPreservation.previousValue
          (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry geometry) assignment
          (PiRLCSamplerPoseidonPlan.invocation source
            (windowStep round)) := by
      unfold PiRLCSamplerPoseidonPreservation.canonicalInput
      rw [PiRLCSamplerPoseidonPlan.descriptor_invocation]
      simp [windowStep]
    _ = PiRLCSamplerPoseidonPreservation.outputValue
          (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry geometry) assignment
          (PiRLCSamplerPoseidonPlan.invocation source (priorStep round)) :=
      previousValue_window
        (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry geometry) assignment
        source round
    _ = NightstreamFPrime.Gadgets.Poseidon2.Layer.evalState
          (Spartan.pullback
            (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base))
          (retainedStateExpr source (priorStep round)) :=
      outputValue_eq_retainedStateEval geometry assignment base source
        (priorStep round)
    _ = NightstreamFPrime.Gadgets.Poseidon2.Layer.evalState
          (Spartan.pullback
            (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base))
          (Sampler.windowInitialState
            (PiRLCSamplerOrdinaryRows.sourceInterface
              (logicalWidth := relationLogicalWidth)
              (publicFits := relationPublicFits) source.val)
            source.val (PiRLCStarts.samplerSourceLogicalStart source.val)
            round.val) := by
      rw [show priorStep round =
          ⟨round.val, lt_trans round.isLt (by decide)⟩ by rfl]
      rw [retainedStateExpr_windowInitialState source round]

/-- Retained Poseidon2 semantics supplies the exact verifier-owned
permutation child for every digest window. -/
theorem canonicalSemantics_imply_windowPermutation
    {relationLogicalWidth : Nat}
    {relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth}
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (semantics : PiRLCSamplerPoseidonPreservation.CanonicalSemantics
      (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry geometry) assignment)
    (source : Fin PiRLCSamplerPoseidonPlan.sourceCount)
    (round : Fin PiRLCSamplerOrdinaryRetainedBlocks.roundCount) :
    NightstreamFPrime.Gadgets.Poseidon2.Permutation.Owned.SpecHolds
      (DigestWindow.permutationInterface
        (PiRLCSamplerOrdinaryRows.windowInterface
          (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits)
          source.val round.val)
        (PiRLCStarts.windowLogicalStart source.val round.val))
      (PiRLCStarts.digestPermutationLogicalStart source.val round.val)
      (Spartan.pullback
        (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base)) := by
  let env := Spartan.pullback
    (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base)
  change List.ofFn
      (NightstreamFPrime.Gadgets.Poseidon2.Layer.evalState env
        (DigestWindow.output
          (PiRLCSamplerOrdinaryRows.windowInterface
            (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits)
            source.val round.val)
          (PiRLCStarts.windowLogicalStart source.val round.val))) =
    Spec.Poseidon2.permute
      (List.ofFn
        (NightstreamFPrime.Gadgets.Poseidon2.Layer.evalState env
          (Sampler.windowInitialState
            (PiRLCSamplerOrdinaryRows.sourceInterface
              (logicalWidth := relationLogicalWidth)
              (publicFits := relationPublicFits) source.val)
            source.val (PiRLCStarts.samplerSourceLogicalStart source.val)
            round.val)))
  calc
    List.ofFn
        (NightstreamFPrime.Gadgets.Poseidon2.Layer.evalState env
          (DigestWindow.output
            (PiRLCSamplerOrdinaryRows.windowInterface
              (logicalWidth := relationLogicalWidth)
              (publicFits := relationPublicFits) source.val round.val)
            (PiRLCStarts.windowLogicalStart source.val round.val))) =
        List.ofFn
          (PiRLCSamplerPoseidonPreservation.outputValue
            (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry geometry)
            assignment
            (PiRLCSamplerPoseidonPlan.invocation source
              (windowStep round))) := by
      rw [← retainedStateExpr_window source round]
      change List.ofFn
          (NightstreamFPrime.Gadgets.Poseidon2.Layer.evalState
            (Spartan.pullback
              (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment
                base))
            (retainedStateExpr source (windowStep round))) = _
      exact congrArg List.ofFn
        (outputValue_eq_retainedStateEval geometry assignment base source
          (windowStep round)).symm
    _ = Spec.Poseidon2.permute
          (List.ofFn
            (PiRLCSamplerPoseidonPreservation.canonicalInput
              (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry geometry)
              assignment
              (PiRLCSamplerPoseidonPlan.invocation source
                (windowStep round)))) :=
      semantics.invocation
        (PiRLCSamplerPoseidonPlan.invocation source (windowStep round))
    _ = Spec.Poseidon2.permute
          (List.ofFn
            (NightstreamFPrime.Gadgets.Poseidon2.Layer.evalState env
              (Sampler.windowInitialState
                (PiRLCSamplerOrdinaryRows.sourceInterface
                  (logicalWidth := relationLogicalWidth)
                  (publicFits := relationPublicFits) source.val)
                source.val (PiRLCStarts.samplerSourceLogicalStart source.val)
                round.val))) := by
      rw [canonicalInput_window geometry assignment base source round]

/-- Exact ordinary-row satisfaction and retained Poseidon2 semantics compose
the four lane children and one permutation child of every digest window. -/
theorem samplerOrdinary_imply_digestWindow
    {relationLogicalWidth : Nat}
    {relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth}
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (ordinaryRows : R1CS.RowsHold
      (PiRLCSamplerOrdinaryDirectPlan.resolvedEnv geometry assignment)
      (PiRLCSamplerOrdinaryDirectSource.sourceRows
        (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits)))
    (poseidonSemantics :
      PiRLCSamplerPoseidonPreservation.CanonicalSemantics
        (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry geometry) assignment)
    (source : Fin PiRLCSamplerPoseidonPlan.sourceCount)
    (round : Fin PiRLCSamplerOrdinaryRetainedBlocks.roundCount)
    (assumptions : DigestWindow.Assumptions
      (PiRLCSamplerOrdinaryRows.windowInterface
        (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits)
        source.val round.val)
      (PiRLCStarts.windowLogicalStart source.val round.val)
      (Spartan.pullback
        (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base))) :
    DigestWindow.SpecHolds
      (PiRLCSamplerOrdinaryRows.windowInterface
        (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits)
        source.val round.val)
      (PiRLCStarts.windowLogicalStart source.val round.val)
      (Spartan.pullback
        (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base)) := by
  have completeRows := PiRLCSamplerRetainedCustody.rowsHold_semanticEnv
    geometry assignment base ordinaryRows
  constructor
  · intro lane
    apply PiRLCSamplerOrdinaryRows.rows_imply_laneSpec source.val round.val lane
      (by simpa [PiRLCSamplerPoseidonPlan.sourceCount,
        PiRLCSamplerOrdinaryRows.sourceCount] using source.isLt)
      (by simpa [PiRLCSamplerOrdinaryRetainedBlocks.roundCount,
        PiRLCSamplerOrdinaryRows.digestRoundCount] using round.isLt)
      (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base)
    · exact DigestWindow.laneAssumptions
        (PiRLCSamplerOrdinaryRows.windowInterface
          (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits)
          source.val round.val)
        (PiRLCStarts.windowLogicalStart source.val round.val) lane assumptions
    · exact completeRows
  · exact canonicalSemantics_imply_windowPermutation geometry assignment base
      poseidonSemantics source round

/-- The exact entry child and all eight exact digest windows compose the
complete scalar-sampler prefix for one source. -/
theorem directSemantics_imply_samplerPrefix
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
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (one : assignment
      (PiCCSOrdinaryRetainedGeometry.oneColumn ordinaryGeometry) = 1)
    (encoding : PiCCSOrdinaryRetainedGeometry.Encodes ordinaryGeometry
      assignment (PiRLCRetainedPreservation.sourceAssignment program base
        groupValue products))
    (endpointRows : (PiCCSTranscriptEndpointPlan.plan
      (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry samplerGeometry)
      ordinaryGeometry).RowsZero assignment)
    (ordinaryRows : R1CS.RowsHold
      (PiRLCSamplerOrdinaryDirectPlan.resolvedEnv samplerGeometry assignment)
      (PiRLCSamplerOrdinaryDirectSource.sourceRows
        (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits)))
    (poseidonSemantics :
      PiRLCSamplerPoseidonPreservation.CanonicalSemantics
        (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry samplerGeometry)
        assignment)
    (source : Fin PiRLCSamplerPoseidonPlan.sourceCount)
    (assumptions : Sampler.Assumptions
      (PiRLCSamplerOrdinaryRows.sourceInterface
        (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits)
        source.val)
      (PiRLCStarts.samplerSourceLogicalStart source.val)
      (Spartan.pullback
        (PiRLCSamplerRetainedCustody.semanticEnv samplerGeometry assignment
          base))) :
    Sampler.PrefixHolds
      (PiRLCSamplerOrdinaryRows.sourceInterface
        (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits)
        source.val)
      source.val (PiRLCStarts.samplerSourceLogicalStart source.val)
      (Spartan.pullback
        (PiRLCSamplerRetainedCustody.semanticEnv samplerGeometry assignment
          base)) := by
  constructor
  · exact canonicalSemantics_imply_entry relation ordinaryGeometry
      samplerGeometry assignment base groupValue products one encoding
      endpointRows poseidonSemantics source
  · intro round
    exact samplerOrdinary_imply_digestWindow samplerGeometry assignment base
      ordinaryRows poseidonSemantics source round
      (Sampler.windowAssumptions
        (PiRLCSamplerOrdinaryRows.sourceInterface
          (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits)
          source.val)
        source.val (PiRLCStarts.samplerSourceLogicalStart source.val)
        assumptions round)

/-- Every digest-lane logical column used by First54 has the same value in
the complete sampler environment and the canonical direct-plan base view. -/
theorem semanticEnv_logical_eq_baseEnv
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (encodes : PiRLCSamplerOrdinaryRetainedGeometry.Encodes geometry assignment
      (PiRLCRetainedPreservation.sourceAssignment program base groupValue
        products))
    (descriptor : PiRLCSamplerOrdinaryRetainedBlocks.Lane)
    (position : Fin PiRLCSamplerOrdinaryRetainedBlocks.logicalCountPerLane) :
    Spartan.pullback
        (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base)
        (PiRLCSamplerOrdinaryRetainedBlocks.logicalSource descriptor position) =
      PiRLCFirst54DirectPlan.baseEnv program base
        (PiRLCSamplerOrdinaryRetainedBlocks.logicalSource descriptor
          position) := by
  let column :=
    PiRLCSamplerOrdinaryRetainedBlocks.logicalSource descriptor position
  have target : PiRLCSamplerOrdinaryDirectSource.Target
      (Spartan.sourceToSpartan column) :=
    ⟨column,
      PiRLCSamplerOrdinaryDirectSource.Source.logical
        descriptor.source.val descriptor.round.val descriptor.lane.val
        position.val descriptor.source.isLt descriptor.round.isLt
        descriptor.lane.isLt position.isLt,
      rfl⟩
  have privateBound : column < PiRLCProductPlan.basePackage.layout.constantColumn := by
    have constant : PiRLCProductPlan.basePackage.layout.constantColumn =
        27695710 :=
      NightstreamFPrime.Export.Stage1.Package.circuitPackage_layout_values.2.2.1
    rw [constant]
    rcases descriptor with ⟨source, round, lane⟩
    have sourceLt := source.isLt
    have roundLt := round.isLt
    have laneLt := lane.isLt
    have positionLt := position.isLt
    change source.val < 17 at sourceLt
    change round.val < 8 at roundLt
    change lane.val < 4 at laneLt
    change position.val < 100 at positionLt
    dsimp [column]
    norm_num [PiRLCSamplerOrdinaryRetainedBlocks.logicalSource,
      PiRLCStarts.digestLaneLogicalStart, PiRLCStarts.windowLogicalStart,
      PiRLCStarts.samplerSourceLogicalStart, PiRLCStarts.samplerLogicalStart,
      PiRLCStarts.phaseLogicalStart, PiRLCInputs.phaseOffset,
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerOffset,
      PiRLCSamplerOrdinaryRetainedBlocks.sourceCount,
      PiRLCSamplerOrdinaryRetainedBlocks.roundCount,
      PiRLCSamplerOrdinaryRetainedBlocks.laneCount,
      PiRLCSamplerOrdinaryRetainedBlocks.logicalCountPerLane]
    omega
  calc
    Spartan.pullback
        (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base)
        column =
        PiRLCSamplerOrdinaryDirectPlan.resolvedEnv geometry assignment
          (Spartan.sourceToSpartan column) :=
      PiRLCSamplerRetainedCustody.semanticEnv_eq_resolved_of_target geometry
        assignment base target
    _ = RunningTransitionDirectPlan.transitionEnv program base
          (Spartan.sourceToSpartan column) :=
      PiRLCSamplerRetainedCustody.resolvedEnv_logical geometry assignment base
        groupValue products encodes descriptor position
    _ = PiRLCFirst54DirectPlan.baseEnv program base column :=
      (PiRLCSamplerRetainedCustody.baseEnv_eq_transitionEnv program base column
        privateBound).symm

def candidateDigestRound
    (candidate : PiRLCFirst54DirectSchedule.Candidate) :
    Fin PiRLCSamplerOrdinaryRetainedBlocks.roundCount :=
  ⟨candidate.round.val / 8, by
    have bounded := candidate.round.isLt
    change candidate.round.val < 64 at bounded
    change candidate.round.val / 8 < 8
    omega⟩

def candidateLane (candidate : PiRLCFirst54DirectSchedule.Candidate) :
    Fin PiRLCSamplerOrdinaryRetainedBlocks.laneCount :=
  ⟨candidate.round.val % 8 / 2, by
    have reduced := Nat.mod_lt candidate.round.val (by decide : 0 < 8)
    change candidate.round.val % 8 / 2 < 4
    omega⟩

def candidateDescriptor
    (candidate : PiRLCFirst54DirectSchedule.Candidate) :
    PiRLCSamplerOrdinaryRetainedBlocks.Lane :=
  { source := candidate.source
    round := candidateDigestRound candidate
    lane := candidateLane candidate }

def candidateRejectPosition
    (candidate : PiRLCFirst54DirectSchedule.Candidate) :
    Fin PiRLCSamplerOrdinaryRetainedBlocks.logicalCountPerLane :=
  ⟨NightstreamFPrime.Gadgets.Range.CanonicalU64.auxiliaryCount +
      candidate.round.val % 2 *
        NightstreamFPrime.Gadgets.Sampling.Candidate16Five.auxiliaryCount + 16, by
    have part := Nat.mod_lt candidate.round.val (by decide : 0 < 2)
    change 66 + candidate.round.val % 2 * 17 + 16 < 100
    omega⟩

def candidateSymbolPosition
    (candidate : PiRLCFirst54DirectSchedule.Candidate) :
    Fin PiRLCSamplerOrdinaryRetainedBlocks.logicalCountPerLane :=
  ⟨NightstreamFPrime.Gadgets.Range.CanonicalU64.auxiliaryCount +
      candidate.round.val % 2 *
        NightstreamFPrime.Gadgets.Sampling.Candidate16Five.auxiliaryCount + 1, by
    have part := Nat.mod_lt candidate.round.val (by decide : 0 < 2)
    change 66 + candidate.round.val % 2 * 17 + 1 < 100
    omega⟩

theorem candidateRejectColumn_eq (candidate :
    PiRLCFirst54DirectSchedule.Candidate) :
    PiRLCSamplerOrdinaryRetainedBlocks.logicalSource
        (candidateDescriptor candidate) (candidateRejectPosition candidate) =
      candidate.rejectColumn := by
  simp [PiRLCSamplerOrdinaryRetainedBlocks.logicalSource,
    candidateDescriptor, candidateDigestRound, candidateLane,
    candidateRejectPosition, PiRLCFirst54DirectSchedule.Candidate.rejectColumn,
    PiRLCFirst54Invocations.rejectSourceColumn,
    PiRLCFirst54Invocations.decoderLogicalStart,
    PiRLCFirst54Invocations.candidateDigestRound,
    PiRLCFirst54Invocations.candidateLane,
    PiRLCFirst54Invocations.candidatePart, Nat.add_assoc]

theorem candidateSymbolColumn_eq (candidate :
    PiRLCFirst54DirectSchedule.Candidate) :
    PiRLCSamplerOrdinaryRetainedBlocks.logicalSource
        (candidateDescriptor candidate) (candidateSymbolPosition candidate) =
      candidate.symbolColumn := by
  simp [PiRLCSamplerOrdinaryRetainedBlocks.logicalSource,
    candidateDescriptor, candidateDigestRound, candidateLane,
    candidateSymbolPosition, PiRLCFirst54DirectSchedule.Candidate.symbolColumn,
    PiRLCFirst54Invocations.remainderSourceColumn,
    PiRLCFirst54Invocations.decoderLogicalStart,
    PiRLCFirst54Invocations.candidateDigestRound,
    PiRLCFirst54Invocations.candidateLane,
    PiRLCFirst54Invocations.candidatePart, Nat.add_assoc]

/-- First54 reject and symbol inputs use the same canonical values in the
complete sampler environment and direct-plan base view. -/
theorem semanticEnv_candidateReject_eq_baseEnv
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (encodes : PiRLCSamplerOrdinaryRetainedGeometry.Encodes geometry assignment
      (PiRLCRetainedPreservation.sourceAssignment program base groupValue
        products))
    (candidate : PiRLCFirst54DirectSchedule.Candidate) :
    Spartan.pullback
        (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base)
        candidate.rejectColumn =
      PiRLCFirst54DirectPlan.baseEnv program base candidate.rejectColumn := by
  rw [← candidateRejectColumn_eq candidate]
  exact semanticEnv_logical_eq_baseEnv geometry assignment base groupValue
    products encodes (candidateDescriptor candidate)
      (candidateRejectPosition candidate)

theorem semanticEnv_candidateSymbol_eq_baseEnv
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (encodes : PiRLCSamplerOrdinaryRetainedGeometry.Encodes geometry assignment
      (PiRLCRetainedPreservation.sourceAssignment program base groupValue
        products))
    (candidate : PiRLCFirst54DirectSchedule.Candidate) :
    Spartan.pullback
        (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base)
        candidate.symbolColumn =
      PiRLCFirst54DirectPlan.baseEnv program base candidate.symbolColumn := by
  rw [← candidateSymbolColumn_eq candidate]
  exact semanticEnv_logical_eq_baseEnv geometry assignment base groupValue
    products encodes (candidateDescriptor candidate)
      (candidateSymbolPosition candidate)

end NightstreamFPrime.Export.Stage1.PiRLCSamplerDirectSemantics
