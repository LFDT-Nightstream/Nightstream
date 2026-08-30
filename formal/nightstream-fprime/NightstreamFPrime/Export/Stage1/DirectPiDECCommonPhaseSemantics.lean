import NightstreamFPrime.Export.Stage1.DirectPiRLCSamplerCompletePrefixPlan
import NightstreamFPrime.Export.Stage1.PiDECEnvironmentCustody

/-!
Owns the deterministic PiDEC phase theorem in the complete PiRLC sampler
environment. The proof transports only the exact PiDEC row support and adds no
row. This is phase-local evidence and does not close PiDEC status.
-/

namespace NightstreamFPrime.Export.Stage1.DirectPiDECCommonPhaseSemantics

open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

private theorem sourceRows_varsSatisfy
    {relationLogicalWidth : Nat}
    {relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits) :
    ∀ row ∈ PiDECOrdinaryDirectSource.sourceRows relationLogicalWidth
        relationPublicFits,
      row.VarsSatisfy PiDECSourceSupport.Target := by
  intro row member
  change row ∈
    ((PiDECOrdinaryDirectSource.publicRows relationLogicalWidth
        relationPublicFits ++
      PiDECOrdinaryDirectSource.commitmentRows relationLogicalWidth
        relationPublicFits) ++
      PiDECOrdinaryDirectSource.evalKRows relationLogicalWidth
        relationPublicFits) ++
      PiDECOrdinaryDirectSource.evalARows relationLogicalWidth
        relationPublicFits at member
  rcases List.mem_append.mp member with beforeEvalA | evalAMember
  · rcases List.mem_append.mp beforeEvalA with beforeEvalK | evalKMember
    · rcases List.mem_append.mp beforeEvalK with publicMember | commitmentMember
      · exact PiDECOrdinaryDirectSource.publicRows_varsSatisfy relation row
          publicMember
      · exact PiDECOrdinaryDirectSource.commitmentRows_varsSatisfy relation row
          commitmentMember
    · exact PiDECOrdinaryDirectSource.evalKRows_varsSatisfy relation row
        evalKMember
  · exact PiDECOrdinaryDirectSource.evalARows_varsSatisfy relation row
      evalAMember

/-- The sampler-complete direct-prefix semantics entail PiDEC in the same
environment as PiRLC. -/
theorem semantics_imply_piDecPhaseHolds
    {relationLogicalWidth : Nat}
    {relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth}
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (ajtai : AjtaiKey
      (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits))
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry application
      logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth application) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (semantics : DirectPiRLCSamplerCompletePrefixPlan.Semantics relation
      geometry assignment base groupValue products)
    (assumptions : Lifecycle.PiDEC.v1_1.Formal.Assumptions relation
      (PiDECArithmetic.phaseInterface relationLogicalWidth relationPublicFits)
      PiDECInputs.phaseOffset
      (Spartan.pullback
        (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base))) :
    Lifecycle.PiDEC.v1_1.Semantics.PhaseHolds relation ajtai
      (PiDECArithmetic.phaseInterface relationLogicalWidth relationPublicFits)
      PiDECInputs.phaseOffset
      (Spartan.pullback
        (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base)) := by
  let transitionEnv := RunningTransitionDirectPlan.transitionEnv application base
  let semanticEnv :=
    PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base
  have sourceRows : R1CS.RowsHold semanticEnv
      (PiDECOrdinaryDirectSource.sourceRows relationLogicalWidth
        relationPublicFits) := by
    apply R1CS.rowsHold_of_agree
      (PiDECOrdinaryDirectSource.sourceRows relationLogicalWidth
        relationPublicFits)
      PiDECSourceSupport.Target transitionEnv semanticEnv
      (sourceRows_varsSatisfy relation)
    · intro column support
      exact PiDECEnvironmentCustody.semanticEnv_eq_transitionEnv_of_target
        geometry assignment base support
    · exact semantics.prior.piDec
  have canonicalRows : R1CS.RowsHold semanticEnv
      ((PiDECArithmetic.canonicalPlan relationLogicalWidth
        relationPublicFits).rows.map Rows.CompiledRow.toR1CS) := by
    rw [← PiDECOrdinaryDirectSource.sourceRows_eq_canonical]
    exact sourceRows
  have exactRows := PiDECArithmetic.Plan.rows_to_layout
    (PiDECArithmetic.canonicalPlan relationLogicalWidth relationPublicFits)
    (PiDECArithmetic.canonicalLayoutPlan relation)
    (PiDECArithmetic.canonicalPlan_matches relation)
  have remappedRows : R1CS.RowsHold semanticEnv
      (Spartan.remapRows (PiDECArithmetic.canonicalLayoutPlan relation).rows) := by
    rw [exactRows] at canonicalRows
    exact canonicalRows
  have physicalRows : R1CS.RowsHold (Spartan.pullback semanticEnv)
      (PiDECArithmetic.canonicalLayoutPlan relation).rows :=
    (Spartan.remapRows_hold semanticEnv
      (PiDECArithmetic.canonicalLayoutPlan relation).rows).mp remappedRows
  exact Layout.PiDEC.v1_1.physical_implies_phaseHolds relation ajtai
    (PiDECArithmetic.phaseInterface relationLogicalWidth relationPublicFits)
    PiDECInputs.phaseOffset (Spartan.pullback semanticEnv) assumptions
    physicalRows

end NightstreamFPrime.Export.Stage1.DirectPiDECCommonPhaseSemantics
