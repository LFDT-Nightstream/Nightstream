import NightstreamFPrime.Export.Stage1.PiCCSDecodedEndpoints
import NightstreamFPrime.Export.Stage1.PerApplicationFixedPoint

/-!
Owns the emitted-row soundness connection to the existing PiCCS phase
assembler. All twelve opaque child contracts use one decoded environment.
No raw encoding or semantic representation is an input to this theorem.
-/

namespace NightstreamFPrime.Export.Stage1.PiCCSDecodedPhase

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

/-- Arbitrary assignments satisfying the ordinary, Poseidon, and endpoint
row families satisfy the sole logical phase assembler in their decoded
environment. The enforced constant coordinate is the only value premise. -/
theorem rowsZero_implies_specHolds
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    {relationLogicalWidth : Nat}
    {relationPublicFits : ringDegree * Lifecycle.PaperAlgebra.publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (ordinary : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (poseidon : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (PiCCSOrdinaryRetainedGeometry.oneColumn ordinary) = 1)
    (ordinaryRows : (PiCCSOrdinaryDirectPlan.plan relation ordinary).RowsZero assignment)
    (transcriptRows : (PiCCSPoseidonPlan.plan (PiCCSPayloadWiring.form ordinary)
      poseidon).RowsZero assignment)
    (endpointRows : (PiCCSTranscriptEndpointPlan.plan poseidon ordinary).RowsZero
      assignment) :
    Lifecycle.PiCCS.v1_1.Formal.SpecHolds relation
      (PiCCSInvocations.parentInterface relationLogicalWidth relationPublicFits)
      PiCCSInputs.phaseOffset
      (Spartan.pullback (PiCCSAssignmentSoundness.decodedEnv ordinary assignment)) := by
  have arithmetic := PiCCSAssignmentSoundness.rowsZero_implies_arithmeticSpecs
    relation ordinary assignment one ordinaryRows
  have transcripts := PiCCSDecodedEndpoints.rowsZero_implies_transcriptSpecs
    (relationLogicalWidth := relationLogicalWidth)
    (relationPublicFits := relationPublicFits)
    ordinary poseidon assignment one transcriptRows endpointRows
  exact {
    statementBinding := arithmetic.statementBinding_parent
    statementAbsorption := transcripts.statementAbsorption_parent
    challenge := transcripts.challengeDerivation_parent
    roundTranscript := transcripts.roundTranscript_parent
    initialClaim := arithmetic.initialClaim_parent
    sumcheck := arithmetic.sumcheck_parent
    eval_K := arithmetic.evalK_parent
    eval_A := arithmetic.evalA_parent
    ccs := arithmetic.ccs_parent
    norm := arithmetic.norm_parent
    finalIdentity := arithmetic.finalIdentity_parent
    outputBinding := transcripts.outputBinding_parent relation }

/-- Acceptance of the selected complete Stage 1 row plan implies the exact
PiCCS phase predicate in decoded values. The fixed-point theorem selects the
key-facing relation before the three PiCCS row families are projected. -/
theorem selectedRowsZero_implies_phaseHolds
    (application : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (ajtai : AjtaiKey
      (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (template : Proof (ProductionKey.degreeBound
      (PerApplicationFixedPoint.relation application fits)))
    (assignment : Assignment F (PerApplicationFixedPoint.logicalWidth application))
    (one : assignment (ApplicationRetainedGeometry.oneColumn
      (PerApplicationFixedPoint.geometry application)) = 1)
    (accepted : (PerApplicationFixedPoint.structuralPlan application fits).RowsZero
      assignment) :
    Lifecycle.PiCCS.v1_1.Formal.PhaseHolds
      (PerApplicationFixedPoint.relation application fits) ajtai
      (PiCCSInvocations.parentInterface
        (PerApplicationFixedPoint.logicalWidth application)
        (PerApplicationFixedPoint.publicFits application))
      PiCCSInputs.phaseOffset
      (Spartan.pullback (PiCCSAssignmentSoundness.decodedEnv
        (DirectApplicationPrefixPlan.piCcsOrdinaryGeometry
          (PerApplicationFixedPoint.geometry application)) assignment))
      template := by
  let relation := PerApplicationFixedPoint.relation application fits
  let geometry := PerApplicationFixedPoint.geometry application
  have selected : (DirectApplicationPrefixPlan.plan relation fits.package geometry
      ).RowsZero assignment := by
    rw [PerApplicationFixedPoint.plan_fixedPoint]
    exact accepted
  have applicationRows := (DirectApplicationPrefixPlan.rowsZero_iff relation
    fits.package geometry assignment).mp selected
  have prefixRows := (DirectPiRLCSamplerCompletePrefixPlan.rowsZero_iff relation
    (DirectApplicationPrefixPlan.prefixGeometry geometry) assignment).mp
      applicationRows.1.1.1
  have samplerPrefixRows := prefixRows.1
  simp only [DirectPiRLCSamplerCompletePrefixPlan.samplerPrefixPlan,
    DirectPiDECPrefixPlan.samplerPrefixPlan, DirectPiDECPrefixPlan.piCcsCompletePlan,
    DirectPiDECPrefixPlan.pilotBindingPrefixPlan,
    DirectPiDECPrefixPlan.pilotOrdinaryPrefixPlan,
    DirectPiDECPrefixPlan.piCcsCorePlan, DirectPiDECPrefixPlan.piCcsPoseidonPrefix,
    ProductionRelation.Plan.append_rowsZero_iff] at samplerPrefixRows
  rcases samplerPrefixRows with
    ⟨⟨⟨⟨⟨⟨_pilot, transcriptRows⟩, ordinaryRows⟩, _pilotOrdinary⟩,
      _pilotBinding⟩, endpointRows⟩, _sampler⟩
  apply Lifecycle.PiCCS.v1_1.Formal.spec_implies_phaseHolds
  exact rowsZero_implies_specHolds relation
    (DirectApplicationPrefixPlan.piCcsOrdinaryGeometry geometry)
    (DirectPiDECPrefixPlan.poseidonGeometry
      (DirectApplicationPrefixPlan.piDecGeometry geometry)) assignment one
    ordinaryRows transcriptRows endpointRows

end NightstreamFPrime.Export.Stage1.PiCCSDecodedPhase
