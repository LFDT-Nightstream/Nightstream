import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint

/-!
Focused regression for the production delayed packed-`y_zcol` active boundary.

The completed selective PiRLC slice is intentionally absent. This regression
checks the full packed-witness production contract, exact fixed-point domain,
semantic pending erasure, adjacent one-fold closure, and terminal closure.
-/

namespace Nightstream.Tests.FPrimeSelectiveFixedPointPiCcsNcActiveBoundary

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection
open Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc

#check ProductionDomain.semanticShape
#check ProductionDomain.semanticShape_logicalWidth_exact
#check ProductionDomain.semanticShape_blockCount
#check ProductionDomain.blockLaneRoundCount
#check ProductionDomain.live_add_virtual_lanes
#check PackedWitnessSourceProjection.production_live_eq_witness
#check PackedWitnessSourceProjection.production_lane_padding_zero
#check PackedWitnessSourceProjection.production_block_padding_zero
#check PackedWitnessSourceProjection.production_lane_partition
#check PackedWitnessDecoder.unpack_at_generatedAddress
#check PackedWitnessDecoder.production_live_eq_generatedWitnessCell
#check PackedWitnessDecoder.production_padding_eq_zero
#check PackedWitnessDecoder.generated_full_decoder_and_lane_partition
#check Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority.DelayedRawChildren.CanonicalParentBinding
#check Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority.DelayedRawChildren.RawRunningCommitmentsBound
#check Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority.DelayedRawChildren.rawChildren_recompose_eq_canonicalParent_or_bindingCollision
#check Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane.DelayedChallengeDomain.producerBeta_ne_batchWeight
#check Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane.derivePreSumcheck_producerBeta
#check Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane.derivePreSumcheck_batchWeight
#check combinedAtPoint_eq_terminalFromMessage_of_bound
#check combinedAtPoint_block_quartic
#check combinedAtPoint_lane_quartic
#check combinedAtPoint_eq_ordinary_of_batchWeight_eq_zero
#check Acceptance.residualWeightIdentity_exact_iff
#check Acceptance.expectedRoundsRepresentable
#check ProductionSequence.ParentOpeningClosureBadEvent

example :
    (Function.Injective
        PackedWitnessDecoder.Artifact.childLogicalColumnAt /\
      Function.Surjective
        PackedWitnessDecoder.Artifact.childLogicalColumnAt) /\
      PackedWitnessDecoder.GeneratedLayout.matrixRows = 54 /\
      PackedWitnessDecoder.GeneratedLayout.booleanLaneCount -
          PackedWitnessDecoder.GeneratedLayout.matrixRows = 10 :=
  PackedWitnessDecoder.generated_full_decoder_and_lane_partition

#check PendingErasure.semanticFoldHolds_iff_withoutPending
#check ActiveBoundary.Accepted
#check ActiveBoundary.ClaimsAccepted
#check ActiveBoundary.claimsCheck
#check ActiveBoundary.claimsCheck_eq_true_iff
#check ActiveTrace.Step.accepted
#check ActiveTrace.Trace.baseCheck
#check ActiveTrace.Trace.baseCheck_eq_true_iff
#check ActiveTrace.Trace.terminalCheck
#check ActiveTrace.Trace.terminalCheck_eq_true_implies
#check ActiveTrace.Trace.RuntimeAccepted
#check MessageTerminal.transcriptAcceptedFromMessage_implies_rawAccepted_or_outputBindingFailure
#check ProductionPiCcs.messageAccepted_implies_accepted_or_outputBindingFailure
#check ProductionPiCcs.accepted_of_messageAccepted_and_packed
#check ProductionNifs.accepted_of_messageAccepted_and_packed
#check ProductionChecker.messageCheck_eq_true_iff_accepted
#check PackedWitnessProduction.messageCheck_implies_check_or_outputBindingFailure
#check ActiveBoundary.ClaimsAccepted.extracted_or_outputBindingFailure
#check ActiveBoundary.acceptedBase_implies_ordinaryNc
#check ActiveBoundary.acceptedPair_implies_previousActiveHolds_or_namedFailure
#check ActiveBoundary.acceptedPair_implies_previousConstruction2_or_namedFailure
#check ActiveBoundary.claimsAcceptedPair_implies_previousConstruction2_or_namedFailure
#check ActiveBoundary.claimsAcceptedPair_of_nextPacked_implies_previousPackedAndConstruction2_or_namedFailure
#check ActiveBoundary.claimsAcceptedPair_of_nextPacked_implies_previousConstruction2_or_namedFailure
#check ActiveOpenedBoundary.claimsAcceptedPair_of_nextPacked_of_openedPackedWitnesses_implies_previousPacked_or_namedFailure
#check ActiveBoundary.acceptedTerminal_implies_activeHolds_or_namedFailure
#check ActiveBoundary.acceptedTerminal_implies_construction2_or_namedFailure
#check ActiveBoundary.claimsAcceptedTerminal_implies_construction2_or_namedFailure
#check ActiveBoundary.claimsAcceptedTerminal_implies_packedAndConstruction2_or_namedFailure
#check ActiveBoundary.claimsAcceptedPairAndTerminal_implies_construction2_or_namedFailure
#check ActiveTrace.Trace.terminalChecked_implies_baseAndAllPaper_or_namedFailure
#check ProductionTerminal.check_eq_true_iff
#check ProductionTerminal.accepted_of_check
#check ProductionTerminal.projectionCheck_eq_true_iff
#check ProductionTerminal.accepted_of_component_checks
#check ProductionTerminal.TerminalCEBridge.claimHolds_iff_childAccepted
#check ProductionTerminal.TerminalCEBridge.holds_implies_childrenCheck
#check ProductionTerminal.TerminalCEBridge.accepted_of_terminalCE_and_projectionCheck
#check ProductionTerminal.TerminalCEBridge.rustVerifyPairs
#check ProductionTerminal.TerminalCEBridge.rustVerifyPairsSuccess_implies_childrenCheck
#check ProductionTerminal.TerminalCEBridge.accepted_of_rustVerifyPairs_and_projectionCheck
#check ProductionTerminal.accepted_implies_packedYZcolBound_or_badEvent
#check PackedWitnessProduction.terminalCheck_eq_true_iff_accepted
#check PackedWitnessProduction.terminalCheck_of_terminalCE_and_projection
#check PackedWitnessProduction.terminalCheck_of_rustVerifyPairs_and_projection
#check PackedWitnessProduction.messageCheckedTerminal_implies_semanticFold_or_badEvent
#check PackedWitnessProduction.messageCheckedTerminal_of_terminalCE_and_projection_implies_semanticFold_or_badEvent
#check PackedWitnessProduction.messageCheckedTerminal_of_rustVerifyPairs_and_projection_implies_semanticFold_or_badEvent
#check RefinementBoundary.messageCheckedTerminal_implies_semanticFold_or_namedFailure
#check ProductionSequence.acceptedNext_implies_previousPackedYZcolBound_or_badEvent
#check ProductionState.acceptedNext_of_stateBinding_implies_previousPackedYZcolBound_or_badEvent
#check ProductionBoundary.messageAcceptedPair_of_nextPacked_implies_previousSemanticFold_or_badEvent
#check PackedWitnessProduction.messageCheckedPair_of_nextPacked_of_stateChecks_implies_previousSemanticFold_or_badEvent
#check RefinementBoundary.messageCheckedPair_of_nextPacked_implies_previousSemanticFold_or_namedFailure
#check ActiveTrace.Trace.terminalChecked_implies_baseAllPackedAndAllPaper_or_parentOpeningFailure_or_paperFailure
#check ActiveTrace.Trace.runtimeAccepted_implies_baseAllPackedAndAllPaper_or_parentOpeningFailure_or_paperFailure
#check ActiveTrace.Trace.terminalChecked_implies_baseAllPackedAndAllPaper_or_namedFailure

/-! ## Exact headline contracts

These examples deliberately apply the recursive and terminal theorems using
only the executable/refinement inputs named by the active boundary, and demand
the narrow result partition. They fail if either theorem acquires an extra
semantic-authority premise or widens its conclusion with a generic unbound
branch.
-/

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.Protocol.FPrime.ConcretePhi81
open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics
open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator
open Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding.DelayedPending
open Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitness

universe uOuterKey uAppState uWitness uDigest uTranscriptState uEncoding

section

variable {OuterKey : Type uOuterKey}
variable {AppState : Type uAppState}
variable {Witness : Type uWitness}
variable {Digest : Type uDigest}
variable {TranscriptState : Type uTranscriptState}
variable {Encoding : Type uEncoding}
variable {shape : SemanticShape}
variable {publicRingColumns verifierRows : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= shape.carrierWidth}

example
    [DecidableEq Digest]
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (previousIncomingDigest sharedStateDigest nextOutgoingDigest : Digest)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (functionIndex : Fin 1)
    (previousInput :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (previousTemplate : Data shape)
    (previousWitnesses : Fin shape.runningCount -> Matrix shape)
    (previousCertificate : FixedActive.Certificate
      (ProductionContext.full setup previousInput))
    (previousAccepted : ActiveBoundary.ClaimsAccepted scheme previousIncomingDigest
      sharedStateDigest machine setup previousInput previousTemplate
      previousWitnesses previousCertificate)
    (nextInput :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (nextTemplate : Data shape)
    (nextWitnesses : Fin shape.runningCount -> Matrix shape)
    (nextCertificate : FixedActive.Certificate
      (ProductionContext.full setup nextInput))
    (nextAccepted : ActiveBoundary.ClaimsAccepted scheme sharedStateDigest
      nextOutgoingDigest machine setup nextInput nextTemplate nextWitnesses
      nextCertificate) :
    Nightstream.Protocol.FPrime.Paper.Construction2.RecursiveHolds
        (SelectedNifsSemantics.family
          (ActiveSemantics.Construction2.selectedNifsSetup setup))
        machine functionIndex (previousInput.fixedOne.toActive setup).toPaper
        (ActiveBoundary.outputOf machine setup previousInput
          previousCertificate).toPaper \/
      ProductionPiCcs.YRingUnbound
        (ProductionContext.full setup previousInput)
        (decodedData previousTemplate previousWitnesses)
        previousCertificate \/
      ProductionBoundary.RecursiveBadEvent scheme
        (ProductionContext.full setup previousInput)
        (decodedData previousTemplate previousWitnesses)
        previousCertificate (ProductionContext.full setup nextInput)
        (decodedData nextTemplate nextWitnesses) nextCertificate \/
      RefinementBoundary.RecursiveRefinementFailure
        (ProductionContext.canonical setup previousInput) previousTemplate
        previousWitnesses previousCertificate
        (ProductionContext.canonical setup nextInput) nextTemplate
        nextWitnesses \/
      ProductionPiCcs.OutputBindingFailure
        (ProductionContext.full setup previousInput)
        (decodedData previousTemplate previousWitnesses)
        previousCertificate \/
      ProductionPiCcs.OutputBindingFailure
        (ProductionContext.full setup nextInput)
        (decodedData nextTemplate nextWitnesses) nextCertificate := by
  exact ActiveBoundary.claimsAcceptedPair_implies_previousConstruction2_or_namedFailure
    noZeroDivisors scheme previousIncomingDigest sharedStateDigest
    nextOutgoingDigest machine setup functionIndex previousInput
    previousTemplate previousWitnesses previousCertificate previousAccepted
    nextInput nextTemplate nextWitnesses nextCertificate nextAccepted

example
    [DecidableEq Digest]
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (incomingStateDigest outgoingStateDigest : Digest)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (functionIndex : Fin 1)
    (input :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (template : Data shape)
    (witnesses : Fin shape.runningCount -> Matrix shape)
    (certificate : FixedActive.Certificate
      (ProductionContext.full setup input))
    (accepted : ActiveBoundary.ClaimsAccepted scheme incomingStateDigest
      outgoingStateDigest machine setup input template witnesses certificate)
    (terminalWitnesses : Fin productionGlobalParams.k -> Matrix shape)
    (terminal : PackedWitnessProduction.terminalCheck
      (ProductionContext.canonical setup input) certificate terminalWitnesses =
        true) :
    Nightstream.Protocol.FPrime.Paper.Construction2.RecursiveHolds
        (SelectedNifsSemantics.family
          (ActiveSemantics.Construction2.selectedNifsSetup setup))
        machine functionIndex (input.fixedOne.toActive setup).toPaper
        (ActiveBoundary.outputOf machine setup input certificate).toPaper \/
      ProductionPiCcs.YRingUnbound (ProductionContext.full setup input)
        (decodedData template witnesses) certificate \/
      ProductionBoundary.TerminalBadEvent (ProductionContext.full setup input)
        (decodedData template witnesses) certificate \/
      RefinementBoundary.TerminalRefinementFailure
        (ProductionContext.canonical setup input) template witnesses
        certificate := by
  exact ActiveBoundary.claimsAcceptedTerminal_implies_construction2_or_namedFailure
    noZeroDivisors scheme incomingStateDigest outgoingStateDigest machine setup
    functionIndex input template witnesses certificate accepted
    terminalWitnesses terminal

example
    [DecidableEq Digest]
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (previousIncomingDigest sharedStateDigest nextOutgoingDigest : Digest)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (functionIndex : Fin 1)
    (previousInput :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (previousTemplate : Data shape)
    (previousWitnesses : Fin shape.runningCount -> Matrix shape)
    (previousCertificate : FixedActive.Certificate
      (ProductionContext.full setup previousInput))
    (previousAccepted : ActiveBoundary.ClaimsAccepted scheme
      previousIncomingDigest sharedStateDigest machine setup previousInput
      previousTemplate previousWitnesses previousCertificate)
    (nextInput :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (nextTemplate : Data shape)
    (nextWitnesses : Fin shape.runningCount -> Matrix shape)
    (nextCertificate : FixedActive.Certificate
      (ProductionContext.full setup nextInput))
    (nextAccepted : ActiveBoundary.ClaimsAccepted scheme sharedStateDigest
      nextOutgoingDigest machine setup nextInput nextTemplate nextWitnesses
      nextCertificate)
    (terminalWitnesses : Fin productionGlobalParams.k -> Matrix shape)
    (terminal : PackedWitnessProduction.terminalCheck
      (ProductionContext.canonical setup nextInput) nextCertificate
      terminalWitnesses = true) :
    (Nightstream.Protocol.FPrime.Paper.Construction2.RecursiveHolds
        (SelectedNifsSemantics.family
          (ActiveSemantics.Construction2.selectedNifsSetup setup))
        machine functionIndex
        (previousInput.fixedOne.toActive setup).toPaper
        (ActiveBoundary.outputOf machine setup previousInput
          previousCertificate).toPaper ∧
      Nightstream.Protocol.FPrime.Paper.Construction2.RecursiveHolds
        (SelectedNifsSemantics.family
          (ActiveSemantics.Construction2.selectedNifsSetup setup))
        machine functionIndex (nextInput.fixedOne.toActive setup).toPaper
        (ActiveBoundary.outputOf machine setup nextInput
          nextCertificate).toPaper) ∨
      ProductionPiCcs.YRingUnbound
        (ProductionContext.full setup previousInput)
        (decodedData previousTemplate previousWitnesses)
        previousCertificate ∨
      ProductionPiCcs.YRingUnbound
        (ProductionContext.full setup nextInput)
        (decodedData nextTemplate nextWitnesses) nextCertificate ∨
      ProductionBoundary.RecursiveBadEvent scheme
        (ProductionContext.full setup previousInput)
        (decodedData previousTemplate previousWitnesses)
        previousCertificate (ProductionContext.full setup nextInput)
        (decodedData nextTemplate nextWitnesses) nextCertificate ∨
      ProductionBoundary.TerminalBadEvent
        (ProductionContext.full setup nextInput)
        (decodedData nextTemplate nextWitnesses) nextCertificate ∨
      RefinementBoundary.RecursiveRefinementFailure
        (ProductionContext.canonical setup previousInput) previousTemplate
        previousWitnesses previousCertificate
        (ProductionContext.canonical setup nextInput) nextTemplate
        nextWitnesses ∨
      RefinementBoundary.TerminalRefinementFailure
        (ProductionContext.canonical setup nextInput) nextTemplate
        nextWitnesses nextCertificate := by
  exact
    ActiveBoundary.claimsAcceptedPairAndTerminal_implies_construction2_or_namedFailure
      noZeroDivisors scheme previousIncomingDigest sharedStateDigest
      nextOutgoingDigest machine setup functionIndex previousInput
      previousTemplate previousWitnesses previousCertificate previousAccepted
      nextInput nextTemplate nextWitnesses nextCertificate nextAccepted
      terminalWitnesses terminal

example
    [DecidableEq Digest]
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (functionIndex : Fin 1)
    {incoming outgoing : Digest}
    (trace : ActiveTrace.Trace scheme machine setup incoming outgoing)
    (base : trace.BaseBoundary)
    (terminal : trace.TerminalChecked) :
    (trace.BaseNc ∧ trace.AllPaper functionIndex) ∨ trace.Failure := by
  exact
    ActiveTrace.Trace.terminalChecked_implies_baseAndAllPaper_or_namedFailure
      noZeroDivisors scheme machine setup functionIndex trace base terminal

example
    [DecidableEq Digest]
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (functionIndex : Fin 1)
    {incoming outgoing : Digest}
    (trace : ActiveTrace.Trace scheme machine setup incoming outgoing)
    (base : trace.BaseBoundary)
    (terminal : trace.TerminalChecked) :
    (trace.BaseNc ∧ trace.AllPacked ∧ trace.AllPaper functionIndex) ∨
      trace.ParentOpeningFailure ∨
      (trace.BaseNc ∧ trace.AllPacked ∧ trace.Failure) := by
  exact
    ActiveTrace.Trace.terminalChecked_implies_baseAllPackedAndAllPaper_or_parentOpeningFailure_or_paperFailure
      noZeroDivisors scheme machine setup functionIndex trace base terminal

example
    [DecidableEq Digest]
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (functionIndex : Fin 1)
    {incoming outgoing : Digest}
    (trace : ActiveTrace.Trace scheme machine setup incoming outgoing)
    (accepted : trace.RuntimeAccepted) :
    (trace.BaseNc ∧ trace.AllPacked ∧ trace.AllPaper functionIndex) ∨
      trace.ParentOpeningFailure ∨
      (trace.BaseNc ∧ trace.AllPacked ∧ trace.Failure) := by
  exact
    ActiveTrace.Trace.runtimeAccepted_implies_baseAllPackedAndAllPaper_or_parentOpeningFailure_or_paperFailure
      noZeroDivisors scheme machine setup functionIndex trace accepted

end

example : ProductionDomain.semanticShape.rowVariables = 24 :=
  ProductionDomain.semanticShape_rowVariables

example : ProductionDomain.semanticShape.logicalWidth = 11437038 :=
  ProductionDomain.semanticShape_logicalWidth_exact

example :
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81ColumnLayout.blockCount
        ProductionDomain.semanticShape.carrierWidth = 211797 :=
  ProductionDomain.semanticShape_blockCount

example : ProductionDomain.semanticShape.runningCount = 14 :=
  ProductionDomain.semanticShape_runningCount

example : ProductionDomain.liveLaneCount = 54 /\
    ProductionDomain.virtualLaneCount = 10 :=
  ⟨ProductionDomain.liveLaneCount_exact,
    ProductionDomain.virtualLaneCount_exact⟩

end Nightstream.Tests.FPrimeSelectiveFixedPointPiCcsNcActiveBoundary
