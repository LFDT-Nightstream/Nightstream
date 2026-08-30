import NightstreamFPrime.Export.Stage1.PiRLCSamplerDirectSemantics

/-!
Owns composition of the retained PiRLC sampler prefix and First54 selector
into the complete sampler and sampler-chain lifecycle relations.

This module does not add rows or close PiRLC status.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCSamplerFullSemantics

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open NightstreamFPrime.Gadgets.Sampling
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

theorem accepted_eval_eq
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (encodes : PiRLCSamplerOrdinaryRetainedGeometry.Encodes geometry assignment
      (PiRLCRetainedPreservation.sourceAssignment program base groupValue
        products))
    (interface : Sampler.Interface) (coordinate : Nat)
    (candidate : PiRLCFirst54DirectSchedule.Candidate) :
    ((Sampler.selectorInterface interface coordinate
        (PiRLCFirst54DirectBridge.samplerStart candidate.source)).accepted
          (PiRLCFirst54DirectBridge.selectorStart candidate.source)
          candidate.round).eval
        (Spartan.pullback
          (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base)) =
      ((Sampler.selectorInterface interface coordinate
        (PiRLCFirst54DirectBridge.samplerStart candidate.source)).accepted
          (PiRLCFirst54DirectBridge.selectorStart candidate.source)
          candidate.round).eval
        (PiRLCFirst54DirectPlan.baseEnv program base) := by
  let semantic := Spartan.pullback
    (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base)
  let canonical := PiRLCFirst54DirectPlan.baseEnv program base
  have oneEq : (1 : Expr).eval semantic = (1 : Expr).eval canonical := rfl
  have rejectEq : semantic candidate.rejectColumn =
      canonical candidate.rejectColumn :=
    PiRLCSamplerDirectSemantics.semanticEnv_candidateReject_eq_baseEnv
      geometry assignment base groupValue products encodes candidate
  rw [PiRLCFirst54DirectBridge.acceptedExpr_eq]
  rw [Expr.eval_sub, Expr.eval_sub]
  change (1 : Expr).eval semantic - semantic candidate.rejectColumn =
    (1 : Expr).eval canonical - canonical candidate.rejectColumn
  rw [oneEq, rejectEq]

theorem symbol_eval_eq
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (encodes : PiRLCSamplerOrdinaryRetainedGeometry.Encodes geometry assignment
      (PiRLCRetainedPreservation.sourceAssignment program base groupValue
        products))
    (interface : Sampler.Interface) (coordinate : Nat)
    (candidate : PiRLCFirst54DirectSchedule.Candidate) :
    ((Sampler.selectorInterface interface coordinate
        (PiRLCFirst54DirectBridge.samplerStart candidate.source)).symbol
          (PiRLCFirst54DirectBridge.selectorStart candidate.source)
          candidate.round).eval
        (Spartan.pullback
          (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base)) =
      ((Sampler.selectorInterface interface coordinate
        (PiRLCFirst54DirectBridge.samplerStart candidate.source)).symbol
          (PiRLCFirst54DirectBridge.selectorStart candidate.source)
          candidate.round).eval
        (PiRLCFirst54DirectPlan.baseEnv program base) := by
  rw [PiRLCFirst54DirectBridge.symbolExpr_eq]
  exact PiRLCSamplerDirectSemantics.semanticEnv_candidateSymbol_eq_baseEnv
    geometry assignment base groupValue products encodes candidate

theorem positionOutput_eval_eq
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
    (source : Fin PiRLCFirst54DirectSchedule.sourceCount)
    (sourceHolds : PiRLCFirst54DirectPlan.SourceHolds program base source)
    (round : Fin PiRLCFirst54DirectSchedule.roundCount)
    (slot : Fin First54Step.slotCount) :
    (First54Step.output
        (First54.positionOffset
          (PiRLCFirst54DirectBridge.selectorStart source) round.val)
        slot).eval
        (Spartan.pullback
          (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base)) =
      (First54Step.output
        (First54.positionOffset
          (PiRLCFirst54DirectBridge.selectorStart source) round.val)
        slot).eval (PiRLCFirst54DirectPlan.baseEnv program base) := by
  let descriptor : PiRLCFirst54DirectSchedule.Position :=
    ⟨⟨source, round⟩, slot⟩
  by_cases final : round.val = 63 ∧ slot.val = 54
  · let finalRound : Fin PiRLCFirst54DirectSchedule.roundCount :=
      ⟨63, by decide⟩
    have roundEq : round = finalRound := by
      apply Fin.ext
      exact final.1
    have slotEq : slot = First54.fullSlot := by
      apply Fin.ext
      exact final.2
    have completeRows := PiRLCSamplerRetainedCustody.rowsHold_semanticEnv
      geometry assignment base ordinaryRows
    have sourceLt : source.val < PiRLCSamplerOrdinaryRows.sourceCount := by
      simpa [PiRLCFirst54DirectSchedule.sourceCount,
        PiRLCFirst54Invocations.sourceCount,
        PiRLCSamplerOrdinaryRows.sourceCount] using source.isLt
    have semanticFull := PiRLCSamplerOrdinaryRows.rows_imply_selectorFull
      (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits)
      source.val sourceLt
      (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base)
      completeRows
    have canonicalFull :
        (First54.finalFull
          (PiRLCFirst54DirectBridge.selectorStart source)).eval
            (PiRLCFirst54DirectPlan.baseEnv program base) = 1 := by
      rw [← PiRLCFirst54DirectBridge.finalValue_eq_eval program base source]
      exact sourceHolds.full
    rw [roundEq, slotEq]
    simpa [First54.finalFull, finalRound, First54.candidateCount] using
      semanticFull.trans canonicalFull.symm
  · have notFinal : descriptor.positionColumn ≠
        PiRLCSamplerOrdinaryDirectSource.selectorSource source.val := by
      intro same
      rw [PiRLCSamplerSelectorCustody.positionColumn_eq_selectorColumn,
        PiRLCSamplerSelectorCustody.selectorSource_eq] at same
      change PiRLCStarts.samplerLogicalStart + source.val * 15504 +
          (8528 + round.val * 109 + slot.val) =
        PiRLCStarts.samplerLogicalStart + source.val * 15504 + 15449 at same
      have roundLt := round.isLt
      have slotLt := slot.isLt
      change round.val < 64 at roundLt
      change slot.val < 55 at slotLt
      apply final
      omega
    have custody :=
      PiRLCSamplerSelectorCustody.semanticEnv_position_eq_baseEnv
        geometry assignment base descriptor notFinal
    change Spartan.pullback
        (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base)
          descriptor.positionColumn =
      PiRLCFirst54DirectPlan.baseEnv program base descriptor.positionColumn
    exact custody

/-- Direct First54 rows and exact retained custody imply the canonical
First54 child specification in the complete sampler environment. -/
theorem directSemantics_imply_selectorSpec
    {relationLogicalWidth : Nat}
    {relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth}
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (encodes : PiRLCSamplerOrdinaryRetainedGeometry.Encodes geometry assignment
      (PiRLCRetainedPreservation.sourceAssignment program base groupValue
        products))
    (ordinaryRows : R1CS.RowsHold
      (PiRLCSamplerOrdinaryDirectPlan.resolvedEnv geometry assignment)
      (PiRLCSamplerOrdinaryDirectSource.sourceRows
        (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits)))
    (interface : Sampler.Interface) (coordinate : Nat)
    (source : Fin PiRLCFirst54DirectSchedule.sourceCount)
    (sourceHolds : PiRLCFirst54DirectPlan.SourceHolds program base source) :
    First54.SpecHolds
      (Sampler.selectorInterface interface coordinate
        (PiRLCFirst54DirectBridge.samplerStart source))
      (PiRLCFirst54DirectBridge.selectorStart source)
      (Spartan.pullback
        (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base)) := by
  let semantic := Spartan.pullback
    (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base)
  let canonical := PiRLCFirst54DirectPlan.baseEnv program base
  have canonicalSpec := PiRLCFirst54DirectBridge.sourceHolds_implies_specHolds
    program base interface coordinate source sourceHolds
  refine ⟨?_, ?_, ?_⟩
  · intro round slot
    let candidate : PiRLCFirst54DirectSchedule.Candidate := ⟨source, round⟩
    have roundEq : First54.candidateIndex round.val = round := by
      apply Fin.ext
      simp [First54.candidateIndex, Nat.mod_eq_of_lt round.isLt]
    have canonicalPosition := canonicalSpec.position round slot
    simp only [First54.positionInterface] at canonicalPosition ⊢
    rw [roundEq] at canonicalPosition ⊢
    have outputEq := positionOutput_eval_eq geometry assignment base ordinaryRows
      source sourceHolds round slot
    have acceptedEq := accepted_eval_eq geometry assignment base groupValue
      products encodes interface coordinate candidate
    have priorEq :
        (fun current =>
          (First54.priorPosition
            (PiRLCFirst54DirectBridge.selectorStart source) round.val
              current).eval semantic) =
        (fun current =>
          (First54.priorPosition
            (PiRLCFirst54DirectBridge.selectorStart source) round.val
              current).eval canonical) := by
      funext current
      exact PiRLCSamplerSelectorCustody.priorPosition_eval_eq geometry
        assignment base candidate current
    calc
      (First54Step.output
          (First54.positionOffset
            (PiRLCFirst54DirectBridge.selectorStart source) round.val)
          slot).eval semantic =
          (First54Step.output
            (First54.positionOffset
              (PiRLCFirst54DirectBridge.selectorStart source) round.val)
            slot).eval canonical := outputEq
      _ = First54Step.update
          (((Sampler.selectorInterface interface coordinate
            (PiRLCFirst54DirectBridge.samplerStart source)).accepted
              (PiRLCFirst54DirectBridge.selectorStart source) round).eval
            canonical)
          (fun current =>
            (First54.priorPosition
              (PiRLCFirst54DirectBridge.selectorStart source) round.val
                current).eval canonical) slot := canonicalPosition
      _ = First54Step.update
          (((Sampler.selectorInterface interface coordinate
            (PiRLCFirst54DirectBridge.samplerStart source)).accepted
              (PiRLCFirst54DirectBridge.selectorStart source) round).eval
            semantic)
          (fun current =>
            (First54.priorPosition
              (PiRLCFirst54DirectBridge.selectorStart source) round.val
                current).eval semantic) slot := by
        rw [acceptedEq, priorEq]
  · intro round slot
    let candidate : PiRLCFirst54DirectSchedule.Candidate := ⟨source, round⟩
    let descriptor : PiRLCFirst54DirectSchedule.Value := ⟨candidate, slot⟩
    have roundEq : First54.candidateIndex round.val = round := by
      apply Fin.ext
      simp [First54.candidateIndex, Nat.mod_eq_of_lt round.isLt]
    have canonicalValue := canonicalSpec.value round slot
    simp only [First54.valueInterface] at canonicalValue ⊢
    rw [roundEq] at canonicalValue ⊢
    have outputEq := PiRLCSamplerSelectorCustody.semanticEnv_value_eq_baseEnv
      geometry assignment base descriptor
    have acceptedEq := accepted_eval_eq geometry assignment base groupValue
      products encodes interface coordinate candidate
    have symbolEq := symbol_eval_eq geometry assignment base groupValue
      products encodes interface coordinate candidate
    have priorPositionEq :
        (fun current =>
          (First54.priorPosition
            (PiRLCFirst54DirectBridge.selectorStart source) round.val
              current).eval semantic) =
        (fun current =>
          (First54.priorPosition
            (PiRLCFirst54DirectBridge.selectorStart source) round.val
              current).eval canonical) := by
      funext current
      exact PiRLCSamplerSelectorCustody.priorPosition_eval_eq geometry
        assignment base candidate current
    have priorOutputEq :
        (fun current =>
          (First54.priorOutput
            (PiRLCFirst54DirectBridge.selectorStart source) round.val
              current).eval semantic) =
        (fun current =>
          (First54.priorOutput
            (PiRLCFirst54DirectBridge.selectorStart source) round.val
              current).eval canonical) := by
      funext current
      exact PiRLCSamplerSelectorCustody.priorOutput_eval_eq geometry
        assignment base candidate current
    change semantic descriptor.valueColumn = _
    calc
      semantic descriptor.valueColumn = canonical descriptor.valueColumn :=
        outputEq
      _ = First54ValueStep.update
          (((Sampler.selectorInterface interface coordinate
            (PiRLCFirst54DirectBridge.samplerStart source)).accepted
              (PiRLCFirst54DirectBridge.selectorStart source) round).eval
            canonical)
          (((Sampler.selectorInterface interface coordinate
            (PiRLCFirst54DirectBridge.samplerStart source)).symbol
              (PiRLCFirst54DirectBridge.selectorStart source) round).eval
            canonical)
          (fun current =>
            (First54.priorPosition
              (PiRLCFirst54DirectBridge.selectorStart source) round.val
                current).eval canonical)
          (fun current =>
            (First54.priorOutput
              (PiRLCFirst54DirectBridge.selectorStart source) round.val
                current).eval canonical) slot := canonicalValue
      _ = First54ValueStep.update
          (((Sampler.selectorInterface interface coordinate
            (PiRLCFirst54DirectBridge.samplerStart source)).accepted
              (PiRLCFirst54DirectBridge.selectorStart source) round).eval
            semantic)
          (((Sampler.selectorInterface interface coordinate
            (PiRLCFirst54DirectBridge.samplerStart source)).symbol
              (PiRLCFirst54DirectBridge.selectorStart source) round).eval
            semantic)
          (fun current =>
            (First54.priorPosition
              (PiRLCFirst54DirectBridge.selectorStart source) round.val
                current).eval semantic)
          (fun current =>
            (First54.priorOutput
              (PiRLCFirst54DirectBridge.selectorStart source) round.val
                current).eval semantic) slot := by
        rw [acceptedEq, symbolEq, priorPositionEq, priorOutputEq]
  · have completeRows := PiRLCSamplerRetainedCustody.rowsHold_semanticEnv
      geometry assignment base ordinaryRows
    have sourceLt : source.val < PiRLCSamplerOrdinaryRows.sourceCount := by
      simpa [PiRLCFirst54DirectSchedule.sourceCount,
        PiRLCFirst54Invocations.sourceCount,
        PiRLCSamplerOrdinaryRows.sourceCount] using source.isLt
    exact PiRLCSamplerOrdinaryRows.rows_imply_selectorFull
      (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits)
      source.val sourceLt
      (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base)
      completeRows

/-- The direct entry, window, and First54 evidence compose one complete scalar
sampler specification at the production Stage 1 offsets. -/
theorem directSemantics_imply_samplerSpec
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
    (piCcsEncoding : PiCCSOrdinaryRetainedGeometry.Encodes ordinaryGeometry
      assignment (PiRLCRetainedPreservation.sourceAssignment program base
        groupValue products))
    (samplerEncoding :
      PiRLCSamplerOrdinaryRetainedGeometry.Encodes samplerGeometry assignment
        (PiRLCRetainedPreservation.sourceAssignment program base groupValue
          products))
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
    (sourceHolds : ∀ source,
      PiRLCFirst54DirectPlan.SourceHolds program base source)
    (source : Fin PiRLCSamplerPoseidonPlan.sourceCount)
    (assumptions : Sampler.Assumptions
      (PiRLCSamplerOrdinaryRows.sourceInterface
        (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits)
        source.val)
      (PiRLCStarts.samplerSourceLogicalStart source.val)
      (Spartan.pullback
        (PiRLCSamplerRetainedCustody.semanticEnv samplerGeometry assignment
          base))) :
    Sampler.SpecHolds
      (PiRLCSamplerOrdinaryRows.sourceInterface
        (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits)
        source.val)
      source.val (PiRLCStarts.samplerSourceLogicalStart source.val)
      (Spartan.pullback
        (PiRLCSamplerRetainedCustody.semanticEnv samplerGeometry assignment
          base)) := by
  let interface := PiRLCSamplerOrdinaryRows.sourceInterface
    (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits)
      source.val
  let semantic := Spartan.pullback
    (PiRLCSamplerRetainedCustody.semanticEnv samplerGeometry assignment base)
  have samplerPrefix :=
    PiRLCSamplerDirectSemantics.directSemantics_imply_samplerPrefix
    relation ordinaryGeometry samplerGeometry assignment base groupValue products
    one piCcsEncoding endpointRows ordinaryRows poseidonSemantics source
    assumptions
  let selectorSource : Fin PiRLCFirst54DirectSchedule.sourceCount :=
    ⟨source.val, by
      simpa [PiRLCSamplerPoseidonPlan.sourceCount,
        PiRLCSamplerOrdinaryRetainedBlocks.sourceCount,
        PiRLCFirst54DirectSchedule.sourceCount,
        PiRLCFirst54Invocations.sourceCount] using source.isLt⟩
  have selectorSpec := directSemantics_imply_selectorSpec samplerGeometry
    assignment base groupValue products samplerEncoding ordinaryRows interface
    source.val selectorSource (sourceHolds selectorSource)
  have selectorAssumptions := Sampler.selectorAssumptions interface source.val
    (PiRLCStarts.samplerSourceLogicalStart source.val) semantic
      samplerPrefix.window
  have selectorOffset := PiRLCFirst54DirectBridge.selectorOffset_eq selectorSource
  change Sampler.selectorOffset
      (PiRLCStarts.samplerSourceLogicalStart source.val) =
    PiRLCFirst54DirectBridge.selectorStart selectorSource at selectorOffset
  rw [selectorOffset] at selectorAssumptions
  have selectorRelation := First54.parentCoverage
    (Sampler.selectorInterface interface source.val
      (PiRLCStarts.samplerSourceLogicalStart source.val))
    (PiRLCFirst54DirectBridge.selectorStart selectorSource) semantic
    selectorAssumptions selectorSpec
  exact ⟨samplerPrefix, selectorRelation⟩

/-- All 17 complete scalar samplers compose the exact lifecycle sampler-chain
relation, including the verifier challenge vector and final transcript state. -/
theorem directSemantics_imply_samplerChain
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
    (piCcsEncoding : PiCCSOrdinaryRetainedGeometry.Encodes ordinaryGeometry
      assignment (PiRLCRetainedPreservation.sourceAssignment program base
        groupValue products))
    (samplerEncoding :
      PiRLCSamplerOrdinaryRetainedGeometry.Encodes samplerGeometry assignment
        (PiRLCRetainedPreservation.sourceAssignment program base groupValue
          products))
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
    (sourceHolds : ∀ source,
      PiRLCFirst54DirectPlan.SourceHolds program base source)
    (assumptions : SamplerChain.Assumptions
      (PiRLCSamplerOrdinaryRows.chainInterface
        (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits))
      PiRLCStarts.samplerLogicalStart
      (Spartan.pullback
        (PiRLCSamplerRetainedCustody.semanticEnv samplerGeometry assignment
          base))) :
    SamplerChain.RelationHolds
      (PiRLCSamplerOrdinaryRows.chainInterface
        (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits))
      PiRLCStarts.samplerLogicalStart
      (Spartan.pullback
        (PiRLCSamplerRetainedCustody.semanticEnv samplerGeometry assignment
          base)) := by
  let chainInterface := PiRLCSamplerOrdinaryRows.chainInterface
    (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits)
  let semantic := Spartan.pullback
    (PiRLCSamplerRetainedCustody.semanticEnv samplerGeometry assignment base)
  have children : SamplerChain.ChildHolds chainInterface
      PiRLCStarts.samplerLogicalStart semantic := by
    intro source
    let directSource : Fin PiRLCSamplerPoseidonPlan.sourceCount :=
      ⟨source.val, by
        simpa [SamplerChain.sourceCount_eq,
          PiRLCSamplerPoseidonPlan.sourceCount,
          PiRLCSamplerOrdinaryRetainedBlocks.sourceCount] using source.isLt⟩
    have childAssumptions := SamplerChain.childAssumptions chainInterface
      PiRLCStarts.samplerLogicalStart source.val source.isLt semantic assumptions
    have childSpec := directSemantics_imply_samplerSpec relation ordinaryGeometry
      samplerGeometry assignment base groupValue products one piCcsEncoding
      samplerEncoding endpointRows ordinaryRows poseidonSemantics sourceHolds
      directSource childAssumptions
    have childRelation := Sampler.parentCoverage
      (PiRLCSamplerOrdinaryRows.sourceInterface
        (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits)
        source.val)
      source.val (PiRLCStarts.samplerSourceLogicalStart source.val) semantic
      childSpec
    simpa [chainInterface, PiRLCSamplerOrdinaryRows.sourceInterface,
      PiRLCSamplerOrdinaryRows.chainInterface, SamplerChain.sourceOffset,
      PiRLCStarts.samplerSourceLogicalStart, Sampler.logicalPrivateCount] using
        childRelation
  exact SamplerChain.parentCoverage chainInterface
    PiRLCStarts.samplerLogicalStart semantic children

end NightstreamFPrime.Export.Stage1.PiRLCSamplerFullSemantics
