import NightstreamFPrime.Export.Stage1.PiRLCSamplerDirectSemantics

/-!
Owns composition of the retained Poseidon2 and checked wide-reduction rows
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
open NightstreamFPrime.Lifecycle.PiRLC.v1_2
open NightstreamFPrime.Gadgets.Sampling
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

/-- Every retained scalar checks the total reduction and its single advance. -/
theorem directSemantics_imply_samplerSpec
    {relationLogicalWidth : Nat}
    {relationPublicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth relationLogicalWidth}
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (ordinaryGeometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (samplerGeometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 1 → F)
    (piCcsEncoding : PiCCSOrdinaryRetainedGeometry.Encodes ordinaryGeometry assignment
      (PiRLCRetainedPreservation.sourceAssignment program base groupValue))
    (endpointRows : (PiCCSTranscriptEndpointPlan.plan
      (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry samplerGeometry)
      ordinaryGeometry).RowsZero assignment)
    (ordinaryRows : R1CS.RowsHold
      (PiRLCSamplerOrdinaryDirectPlan.resolvedEnv samplerGeometry assignment)
      (PiRLCSamplerOrdinaryDirectSource.sourceRows
        (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits)))
    (poseidonSemantics : PiRLCSamplerPoseidonPreservation.CanonicalSemantics
      (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry samplerGeometry) assignment)
    (source : Fin PiRLCSamplerPoseidonPlan.sourceCount)
    (inputs : Sampler.Assumptions
      (PiRLCSamplerInvocations.sourceInterface (logicalWidth := relationLogicalWidth)
        (publicFits := relationPublicFits) source.val)
      (PiRLCStarts.samplerSourceLogicalStart source.val)) :
    Sampler.SpecHolds
      (PiRLCSamplerInvocations.sourceInterface (logicalWidth := relationLogicalWidth)
        (publicFits := relationPublicFits) source.val)
      source.val (PiRLCStarts.samplerSourceLogicalStart source.val)
      (Spartan.pullback (PiRLCSamplerRetainedCustody.semanticEnv samplerGeometry assignment base)) := by
  let interface := PiRLCSamplerInvocations.sourceInterface
    (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits) source.val
  let target := PiRLCSamplerRetainedCustody.semanticEnv samplerGeometry assignment base
  have entered := PiRLCSamplerDirectSemantics.canonicalSemantics_imply_entry
    relation ordinaryGeometry samplerGeometry assignment base groupValue piCcsEncoding
    endpointRows poseidonSemantics source
  have advanced := PiRLCSamplerDirectSemantics.canonicalSemantics_imply_advance
    (relationLogicalWidth := relationLogicalWidth) (relationPublicFits := relationPublicFits)
    samplerGeometry assignment base poseidonSemantics source
  have held := PiRLCSamplerRetainedCustody.rowsHold_semanticEnv
    samplerGeometry assignment base ordinaryRows
  change R1CS.RowsHold target ((PiRLCSamplerOrdinaryRows.rows
    (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits)).map Rows.CompiledRow.toR1CS) at held
  have packet := PiRLCSamplerOrdinaryRows.rows_imply_sourceRows
    (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits) source target (@held)
  rw [PiRLCSamplerOrdinaryRows.sourceRows, List.map_append, R1CS.rowsHold_append] at packet
  have rangeInputs : WideReduction.Assumptions
      (PiRLCSamplerOrdinaryRows.rangeInterface (logicalWidth := relationLogicalWidth)
        (publicFits := relationPublicFits) source.val)
      (PiRLCStarts.rangeLogicalStart source.val) := by
    rw [PiRLCSamplerOrdinaryRows.rangeInterface_eq]
    exact Sampler.range_inputs interface source.val _ inputs
  have decoded := PiRLCSamplerOrdinaryRows.rangeRows_imply_spec
    (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits) source.val target rangeInputs packet.1
  have words := PiRLCSamplerOrdinaryRows.wordRows_imply_spec source.val target packet.2
  apply Sampler.spec_of_children interface source.val (Spartan.pullback target)
    (PiRLCStarts.samplerSourceLogicalStart source.val) inputs entered _ advanced words
  rw [PiRLCSamplerOrdinaryRows.rangeInterface_eq] at decoded
  exact decoded

/-- All 17 scalar results compose the verifier's exact transcript schedule. -/
theorem directSemantics_imply_samplerChain
    {relationLogicalWidth : Nat}
    {relationPublicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth relationLogicalWidth}
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (ordinaryGeometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (samplerGeometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 1 → F)
    (piCcsEncoding : PiCCSOrdinaryRetainedGeometry.Encodes ordinaryGeometry assignment
      (PiRLCRetainedPreservation.sourceAssignment program base groupValue))
    (endpointRows : (PiCCSTranscriptEndpointPlan.plan
      (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry samplerGeometry)
      ordinaryGeometry).RowsZero assignment)
    (ordinaryRows : R1CS.RowsHold
      (PiRLCSamplerOrdinaryDirectPlan.resolvedEnv samplerGeometry assignment)
      (PiRLCSamplerOrdinaryDirectSource.sourceRows
        (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits)))
    (poseidonSemantics : PiRLCSamplerPoseidonPreservation.CanonicalSemantics
      (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry samplerGeometry) assignment)
    (inputs : SamplerChain.Assumptions
      (PiRLCSamplerRows.samplerInterface (logicalWidth := relationLogicalWidth)
        (publicFits := relationPublicFits)) PiRLCStarts.samplerLogicalStart) :
    SamplerChain.SpecHolds
      (PiRLCSamplerRows.samplerInterface (logicalWidth := relationLogicalWidth)
        (publicFits := relationPublicFits)) PiRLCStarts.samplerLogicalStart
      (Spartan.pullback (PiRLCSamplerRetainedCustody.semanticEnv samplerGeometry assignment base)) := by
  apply SamplerChain.spec_of_children _ _ _ inputs
  intro source
  let directSource : Fin PiRLCSamplerPoseidonPlan.sourceCount :=
    ⟨source.val, by simpa [SamplerChain.sourceCount_eq, PiRLCSamplerPoseidonPlan.sourceCount] using source.isLt⟩
  exact directSemantics_imply_samplerSpec relation ordinaryGeometry samplerGeometry assignment
    base groupValue piCcsEncoding endpointRows ordinaryRows poseidonSemantics directSource
    (SamplerChain.child_inputs _ _ source.val inputs)

end NightstreamFPrime.Export.Stage1.PiRLCSamplerFullSemantics
