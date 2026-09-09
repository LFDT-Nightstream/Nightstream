import NightstreamFPrime.Export.Stage1.ActualPiRLCSampling
import NightstreamFPrime.Export.Stage1.PiCCSDecodedPhase
import NightstreamFPrime.Export.Stage1.ActualPiRLCValues

/-!
Owns the selected-row connection from the exact PiCCS outgoing state to all
PiRLC challenges and the final sampler state. All values come from the actual
assignment. Canonical witness encoding is not a premise.
-/

namespace NightstreamFPrime.Export.Stage1.ActualPiRLC

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open PiCCSTranscriptEndpointPlan (endpointInvocation endpointColumn outputFamily)

variable {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
  {relationLogicalWidth : Nat}
  {relationPublicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth relationLogicalWidth}

theorem initial_eq_decoded_state
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (ordinary : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (poseidon : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (PiCCSOrdinaryRetainedGeometry.oneColumn ordinary) = 1)
    (rows : (PiCCSTranscriptEndpointPlan.plan poseidon ordinary).RowsZero assignment) :
    ActualPiRLCStates.initialState poseidon assignment =
      PiCCS.v1_1.StatementAbsorption.evalState
        (Spartan.pullback (PiCCSAssignmentSoundness.decodedEnv ordinary assignment))
        (PiCCS.v1_1.Formal.outputBindingFinalState relation
          (PiCCSInvocations.parentInterface relationLogicalWidth relationPublicFits)
          PiCCSInputs.phaseOffset) := by
  have last : endpointInvocation outputFamily =
      (⟨PiCCSPoseidonPlan.invocationCount - 1, by
        rw [PiCCSPoseidonPlan.invocationCount_eq]; decide⟩ : Fin PiCCSPoseidonPlan.invocationCount) := by
    apply Fin.ext
    simp [endpointInvocation, outputFamily, PiCCSTranscriptDirectSemantics.outputLast,
      PiCCSPoseidonPlan.invocationCount_eq]
  have endpoint := PiCCSDecodedEndpoints.rowsZero_implies_endpointStates ordinary poseidon
    assignment one rows outputFamily
  rw [last] at endpoint
  change ActualPiRLCStates.initialState poseidon assignment = _ at endpoint
  rw [endpoint]
  apply congrArg List.ofFn
  funext lane
  unfold PiCCS.v1_1.Formal.outputBindingFinalState
  rw [← PiCCSInvocations.outputWitnessStart_matches relationLogicalWidth relationPublicFits relation]
  change _ = (PiCCS.v1_1.OutputBinding.finalState
    (PiCCSInvocations.outputInterface relationLogicalWidth relationPublicFits)
    PiCCSInvocations.outputWitnessStart lane).eval
      (Spartan.pullback (PiCCSAssignmentSoundness.decodedEnv ordinary assignment))
  rw [PiCCSTranscriptEndpointPlan.outputFinalState_endpoint_of_shape]
  rfl

/-- The complete physical prefix supplies every sampler and endpoint premise. -/
theorem prefixRowsZero_implies_batch
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (PiRLCSamplerOrdinaryRetainedGeometry.oneColumn geometry) = 1)
    (rows : (DirectPiRLCSamplerCompletePrefixPlan.plan relation geometry).RowsZero assignment) :
    let ordinary := DirectPiDECPrefixPlan.piCcsOrdinaryGeometry
      (DirectPiRLCSamplerCompletePrefixPlan.piDecGeometry geometry)
    Transcript.PiRlcSampler.piRlcChallengesWithState
        (PiCCS.v1_1.StatementAbsorption.evalState
          (Spartan.pullback (PiCCSAssignmentSoundness.decodedEnv ordinary assignment))
          (PiCCS.v1_1.Formal.outputBindingFinalState relation
            (PiCCSInvocations.parentInterface relationLogicalWidth relationPublicFits)
            PiCCSInputs.phaseOffset)) PiRLCFirst54DirectSchedule.sourceCount =
      some ⟨ActualPiRLCSampling.challenge geometry assignment,
        ActualPiRLCStates.state (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry geometry)
          assignment ⟨16, by decide⟩ ⟨8, by decide⟩⟩ := by
  have parts := (DirectPiRLCSamplerCompletePrefixPlan.rowsZero_iff relation geometry assignment).mp rows
  have samplerPrefix := parts.1
  change (DirectPiDECPrefixPlan.samplerPrefixPlan relation
    (DirectPiRLCSamplerCompletePrefixPlan.piDecGeometry geometry)).RowsZero assignment at samplerPrefix
  rw [DirectPiDECPrefixPlan.samplerPrefixPlan, Plan.append_rowsZero_iff] at samplerPrefix
  have piCcsRows := samplerPrefix.1
  rw [DirectPiDECPrefixPlan.piCcsCompletePlan, Plan.append_rowsZero_iff] at piCcsRows
  have piRlcRows := parts.2.2.1
  change (PiRLCRetainedPlan.plan _ _).RowsZero assignment at piRlcRows
  have selectorRows := (PiRLCRetainedPlan.rowsZero_iff _ _ assignment).mp piRlcRows
  have semantics := PiRLCSamplerPoseidonPreservation.rowsZero_implies_canonicalSemantics
    (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry geometry) assignment one samplerPrefix.2
  dsimp only
  rw [← initial_eq_decoded_state relation _ _ assignment one piCcsRows.2]
  exact ActualPiRLCSampling.rowsZero_implies_batch relation geometry assignment one parts.2.1
    selectorRows.2 semantics

theorem productChallenge_eq
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) (descriptor : PiRLCProductSchedule.Descriptor) :
    ActualPiRLCValues.challenge (DirectPiRLCSamplerCompletePrefixPlan.piDecGeometry geometry)
        assignment descriptor = ActualPiRLCSampling.challenge geometry assignment descriptor.source := by
  funext lane
  simp only [ActualPiRLCValues.challenge, ActualPiRLCValues.inputs,
    PiRLCProductMatrixProgram.inputs, PiRLCRetainedInputs.productInputs,
    PiRLCProductSchedule.descriptor_invocation]
  rfl

/-- Actual selected rows and actual public input force the complete verifier
batch from the actual PiCCS execution. No sampler premise is supplied by the caller. -/
theorem selectedRowsAndPublic_imply_batch
    (application : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (ajtai : AjtaiKey (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (template : Proof (ProductionKey.degreeBound (PerApplicationFixedPoint.relation application fits)))
    (assignment : Assignment F (PerApplicationFixedPoint.logicalWidth application))
    (digest : Digest)
    (publicEqual : Phi81Relation.projectPublicInput
      (shape := FullShape (PerApplicationFixedPoint.logicalWidth application)
        (PerApplicationFixedPoint.publicFits application))
      (Phi81CarrierLayout.extendAssignment 0 assignment) =
        encHash (publicFits := PerApplicationFixedPoint.publicFits application) digest)
    (accepted : (PerApplicationFixedPoint.structuralPlan application fits).RowsZero assignment) :
    let geometry := PerApplicationFixedPoint.geometry application
    let relation := PerApplicationFixedPoint.relation application fits
    let interface := PiCCSInvocations.parentInterface (PerApplicationFixedPoint.logicalWidth application)
      (PerApplicationFixedPoint.publicFits application)
    let env := Spartan.pullback (PiCCSAssignmentSoundness.decodedEnv
      (DirectApplicationPrefixPlan.piCcsOrdinaryGeometry geometry) assignment)
    Transcript.PiRlcSampler.piRlcChallengesWithState
        ((ProductionKey.key relation ajtai).piCcsExecution
          (PiCCS.v1_1.Formal.evalRunning interface PiCCSInputs.phaseOffset env)
          (PiCCS.v1_1.Formal.evalFresh interface PiCCSInputs.phaseOffset env)
          (PiCCS.v1_1.Formal.evalProof relation interface PiCCSInputs.phaseOffset env template)).outgoingState
        PiRLCFirst54DirectSchedule.sourceCount =
      some ⟨ActualPiRLCSampling.challenge (DirectApplicationPrefixPlan.prefixGeometry geometry) assignment,
        ActualPiRLCStates.state (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry
          (DirectApplicationPrefixPlan.prefixGeometry geometry)) assignment ⟨16, by decide⟩ ⟨8, by decide⟩⟩ := by
  let geometry := PerApplicationFixedPoint.geometry application
  let relation := PerApplicationFixedPoint.relation application fits
  have publicBound : RecursivePublicOutputPlan.publicInput geometry assignment =
      encHash (publicFits := RecursivePublicOutputPlan.carrierPublicFits geometry) digest := by
    rw [RecursivePublicOutputPlan.publicInput_eq_projectPublicInput]
    exact publicEqual
  have one := RecursivePublicOutputPlan.publicEqual_implies_one geometry assignment digest publicBound
  have selected : (DirectApplicationPrefixPlan.plan relation fits.package geometry).RowsZero assignment := by
    rw [PerApplicationFixedPoint.plan_fixedPoint]
    exact accepted
  have parts := (DirectApplicationPrefixPlan.rowsZero_iff relation fits.package geometry assignment).mp selected
  have sampler := prefixRowsZero_implies_batch relation
    (DirectApplicationPrefixPlan.prefixGeometry geometry) assignment one parts.1.1.1
  have piCcs := PiCCSDecodedPhase.selectedRowsZero_implies_phaseHolds
    application fits ajtai template assignment one accepted
  dsimp only
  exact (congrArg (fun state => Transcript.PiRlcSampler.piRlcChallengesWithState state
    PiRLCFirst54DirectSchedule.sourceCount) piCcs.outgoingState).symm.trans sampler

/-- The actual product challenges are precisely the production NIFS key response. -/
theorem selectedRowsAndPublic_imply_keyChallenges
    (application : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (ajtai : AjtaiKey (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (template : Proof (ProductionKey.degreeBound (PerApplicationFixedPoint.relation application fits)))
    (assignment : Assignment F (PerApplicationFixedPoint.logicalWidth application))
    (digest : Digest)
    (publicEqual : Phi81Relation.projectPublicInput
      (shape := FullShape (PerApplicationFixedPoint.logicalWidth application)
        (PerApplicationFixedPoint.publicFits application))
      (Phi81CarrierLayout.extendAssignment 0 assignment) =
        encHash (publicFits := PerApplicationFixedPoint.publicFits application) digest)
    (accepted : (PerApplicationFixedPoint.structuralPlan application fits).RowsZero assignment) :
    let geometry := PerApplicationFixedPoint.geometry application
    let relation := PerApplicationFixedPoint.relation application fits
    let interface := PiCCSInvocations.parentInterface (PerApplicationFixedPoint.logicalWidth application)
      (PerApplicationFixedPoint.publicFits application)
    let env := Spartan.pullback (PiCCSAssignmentSoundness.decodedEnv
      (DirectApplicationPrefixPlan.piCcsOrdinaryGeometry geometry) assignment)
    (ProductionKey.key relation ajtai).piRlcChallenges
        (PiCCS.v1_1.Formal.evalRunning interface PiCCSInputs.phaseOffset env)
        (PiCCS.v1_1.Formal.evalFresh interface PiCCSInputs.phaseOffset env)
        (PiCCS.v1_1.Formal.evalProof relation interface PiCCSInputs.phaseOffset env template) =
      some (ActualPiRLCSampling.challenge (DirectApplicationPrefixPlan.prefixGeometry geometry) assignment) := by
  have batch := selectedRowsAndPublic_imply_batch application fits ajtai template assignment digest
    publicEqual accepted
  exact congrArg (Option.map Transcript.PiRlcSampler.Batch.challenges) batch

end NightstreamFPrime.Export.Stage1.ActualPiRLC
