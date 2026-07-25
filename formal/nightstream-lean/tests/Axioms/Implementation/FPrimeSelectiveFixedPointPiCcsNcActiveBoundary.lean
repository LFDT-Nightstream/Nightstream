import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveTrace
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveOpenedBoundary
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessSourceProjection
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessDecoder
import tests.Axioms.Support

/-! Fail-closed dependencies for production delayed `y_zcol` active closure. -/

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection
open Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveBoundary.claimsCheck_eq_true_iff' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ActiveBoundary.claimsCheck_eq_true_iff

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveOpenedBoundary.claimsAcceptedPair_of_nextPacked_of_openedPackedWitnesses_implies_previousPacked_or_namedFailure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ActiveOpenedBoundary.claimsAcceptedPair_of_nextPacked_of_openedPackedWitnesses_implies_previousPacked_or_namedFailure

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PendingErasure.semanticFoldHolds_iff_withoutPending' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PendingErasure.semanticFoldHolds_iff_withoutPending

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveBoundary.acceptedBase_implies_ordinaryNc' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ActiveBoundary.acceptedBase_implies_ordinaryNc

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveBoundary.acceptedPair_implies_previousActiveHolds_or_namedFailure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ActiveBoundary.acceptedPair_implies_previousActiveHolds_or_namedFailure

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveBoundary.acceptedPair_implies_previousConstruction2_or_namedFailure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ActiveBoundary.acceptedPair_implies_previousConstruction2_or_namedFailure

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveBoundary.acceptedTerminal_implies_activeHolds_or_namedFailure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ActiveBoundary.acceptedTerminal_implies_activeHolds_or_namedFailure

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveBoundary.acceptedTerminal_implies_construction2_or_namedFailure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ActiveBoundary.acceptedTerminal_implies_construction2_or_namedFailure

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.MessageTerminal.transcriptAcceptedFromMessage_implies_rawAccepted_or_outputBindingFailure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms MessageTerminal.transcriptAcceptedFromMessage_implies_rawAccepted_or_outputBindingFailure

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionPiCcs.messageAccepted_implies_accepted_or_outputBindingFailure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ProductionPiCcs.messageAccepted_implies_accepted_or_outputBindingFailure

/-- info: 'Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionChecker.messageCheck_eq_true_iff_accepted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ProductionChecker.messageCheck_eq_true_iff_accepted

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessProduction.messageCheck_implies_check_or_outputBindingFailure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PackedWitnessProduction.messageCheck_implies_check_or_outputBindingFailure

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveBoundary.ClaimsAccepted.extracted_or_outputBindingFailure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ActiveBoundary.ClaimsAccepted.extracted_or_outputBindingFailure

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveBoundary.claimsAcceptedPair_implies_previousConstruction2_or_namedFailure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ActiveBoundary.claimsAcceptedPair_implies_previousConstruction2_or_namedFailure

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveBoundary.claimsAcceptedTerminal_implies_construction2_or_namedFailure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ActiveBoundary.claimsAcceptedTerminal_implies_construction2_or_namedFailure

/-- info: 'Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionTerminal.check_eq_true_iff' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ProductionTerminal.check_eq_true_iff

/-- info: 'Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionTerminal.accepted_of_check' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ProductionTerminal.accepted_of_check

/-- info: 'Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionTerminal.accepted_of_component_checks' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ProductionTerminal.accepted_of_component_checks

/-- info: 'Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionTerminal.TerminalCEBridge.accepted_of_terminalCE_and_projectionCheck' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ProductionTerminal.TerminalCEBridge.accepted_of_terminalCE_and_projectionCheck

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessProduction.terminalCheck_of_terminalCE_and_projection' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PackedWitnessProduction.terminalCheck_of_terminalCE_and_projection

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessProduction.messageCheckedTerminal_of_terminalCE_and_projection_implies_semanticFold_or_badEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PackedWitnessProduction.messageCheckedTerminal_of_terminalCE_and_projection_implies_semanticFold_or_badEvent

/-- info: 'Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionTerminal.TerminalCEBridge.accepted_of_rustVerifyPairs_and_projectionCheck' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ProductionTerminal.TerminalCEBridge.accepted_of_rustVerifyPairs_and_projectionCheck

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessProduction.messageCheckedTerminal_of_rustVerifyPairs_and_projection_implies_semanticFold_or_badEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PackedWitnessProduction.messageCheckedTerminal_of_rustVerifyPairs_and_projection_implies_semanticFold_or_badEvent

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionPiCcs.accepted_of_messageAccepted_and_packed' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ProductionPiCcs.accepted_of_messageAccepted_and_packed

/-- info: 'Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionNifs.accepted_of_messageAccepted_and_packed' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ProductionNifs.accepted_of_messageAccepted_and_packed

/-- info: 'Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionTerminal.accepted_implies_packedYZcolBound_or_badEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ProductionTerminal.accepted_implies_packedYZcolBound_or_badEvent

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessProduction.messageCheckedTerminal_implies_semanticFold_or_badEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PackedWitnessProduction.messageCheckedTerminal_implies_semanticFold_or_badEvent

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RefinementBoundary.messageCheckedTerminal_implies_semanticFold_or_namedFailure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms RefinementBoundary.messageCheckedTerminal_implies_semanticFold_or_namedFailure

/-- info: 'Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionState.acceptedNext_of_stateBinding_implies_previousPackedYZcolBound_or_badEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ProductionState.acceptedNext_of_stateBinding_implies_previousPackedYZcolBound_or_badEvent

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveBoundary.claimsAcceptedPair_of_nextPacked_implies_previousConstruction2_or_namedFailure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ActiveBoundary.claimsAcceptedPair_of_nextPacked_implies_previousConstruction2_or_namedFailure

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveBoundary.claimsAcceptedPairAndTerminal_implies_construction2_or_namedFailure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ActiveBoundary.claimsAcceptedPairAndTerminal_implies_construction2_or_namedFailure

/-- info: 'Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionBoundary.messageAcceptedPair_of_nextPacked_implies_previousPackedAndSemanticFold_or_badEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ProductionBoundary.messageAcceptedPair_of_nextPacked_implies_previousPackedAndSemanticFold_or_badEvent

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessProduction.messageCheckedPair_of_nextPacked_of_stateChecks_implies_previousPackedAndSemanticFold_or_badEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PackedWitnessProduction.messageCheckedPair_of_nextPacked_of_stateChecks_implies_previousPackedAndSemanticFold_or_badEvent

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RefinementBoundary.messageCheckedPair_of_nextPacked_implies_previousPackedAndSemanticFold_or_namedFailure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms RefinementBoundary.messageCheckedPair_of_nextPacked_implies_previousPackedAndSemanticFold_or_namedFailure

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveBoundary.claimsAcceptedBase_of_packed_implies_ordinaryNc' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ActiveBoundary.claimsAcceptedBase_of_packed_implies_ordinaryNc

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveBoundary.claimsAcceptedPair_of_nextPacked_implies_previousPackedAndConstruction2_or_namedFailure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ActiveBoundary.claimsAcceptedPair_of_nextPacked_implies_previousPackedAndConstruction2_or_namedFailure

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveBoundary.claimsAcceptedTerminal_implies_packedAndConstruction2_or_namedFailure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ActiveBoundary.claimsAcceptedTerminal_implies_packedAndConstruction2_or_namedFailure

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveTrace.Trace.terminalChecked_implies_baseAndAllPaper_or_namedFailure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ActiveTrace.Trace.terminalChecked_implies_baseAndAllPaper_or_namedFailure

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessSourceProjection.production_live_eq_witness' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PackedWitnessSourceProjection.production_live_eq_witness

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessSourceProjection.production_lane_padding_zero' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PackedWitnessSourceProjection.production_lane_padding_zero

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessSourceProjection.production_block_padding_zero' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PackedWitnessSourceProjection.production_block_padding_zero

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessSourceProjection.production_lane_partition' does not depend on any axioms -/
#guard_msgs in
#audit_axioms PackedWitnessSourceProjection.production_lane_partition

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessDecoder.unpack_at_generatedAddress' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PackedWitnessDecoder.unpack_at_generatedAddress

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessDecoder.production_live_eq_generatedWitnessCell' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PackedWitnessDecoder.production_live_eq_generatedWitnessCell

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessDecoder.production_padding_eq_zero' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PackedWitnessDecoder.production_padding_eq_zero

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessDecoder.generated_full_decoder_and_lane_partition' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PackedWitnessDecoder.generated_full_decoder_and_lane_partition

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority.DelayedRawChildren.rawRunningAssignments_recompose_eq_parent_or_bindingCollision_of_ncTruth' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority.DelayedRawChildren.rawRunningAssignments_recompose_eq_parent_or_bindingCollision_of_ncTruth

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority.DelayedRawChildren.rawChildren_recompose_eq_canonicalParent_or_bindingCollision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority.DelayedRawChildren.rawChildren_recompose_eq_canonicalParent_or_bindingCollision

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.DelayedPackedYZcol.CombinedNc.Step.accepted_next_of_parentOpening_implies_previous_packedYZcolBound_or_bindingEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ProductionStep.accepted_next_of_parentOpening_implies_previous_packedYZcolBound_or_bindingEvent

/-- info: 'Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionSequence.acceptedNext_of_parentOpening_implies_previousPackedYZcolBound_or_badEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ProductionSequence.acceptedNext_of_parentOpening_implies_previousPackedYZcolBound_or_badEvent

/-- info: 'Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionState.acceptedNext_of_stateBinding_of_parentOpening_implies_previousPackedYZcolBound_or_badEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ProductionState.acceptedNext_of_stateBinding_of_parentOpening_implies_previousPackedYZcolBound_or_badEvent

/-- info: 'Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionBoundary.messageAcceptedPair_of_nextPacked_of_parentOpening_implies_previousPacked_or_badEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ProductionBoundary.messageAcceptedPair_of_nextPacked_of_parentOpening_implies_previousPacked_or_badEvent

/-- info: 'Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionTerminal.accepted_of_parentOpening_implies_packedYZcolBound_or_badEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ProductionTerminal.accepted_of_parentOpening_implies_packedYZcolBound_or_badEvent

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessProduction.messageCheckedPair_of_nextPacked_of_stateChecks_implies_previousPacked_or_parentOpeningBadEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PackedWitnessProduction.messageCheckedPair_of_nextPacked_of_stateChecks_implies_previousPacked_or_parentOpeningBadEvent

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessProduction.terminalCheck_of_parentOpening_implies_packed_or_badEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PackedWitnessProduction.terminalCheck_of_parentOpening_implies_packed_or_badEvent

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveBoundary.claimsAcceptedPair_of_nextPacked_implies_previousPacked_or_parentOpeningBadEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ActiveBoundary.claimsAcceptedPair_of_nextPacked_implies_previousPacked_or_parentOpeningBadEvent

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveBoundary.claimsAcceptedTerminal_implies_packed_or_parentOpeningBadEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ActiveBoundary.claimsAcceptedTerminal_implies_packed_or_parentOpeningBadEvent

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveTrace.Trace.terminalChecked_implies_baseAllPackedAndAllPaper_or_namedFailure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ActiveTrace.Trace.terminalChecked_implies_baseAllPackedAndAllPaper_or_namedFailure

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveTrace.Trace.terminalChecked_implies_baseAllPackedAndAllPaper_or_parentOpeningFailure_or_paperFailure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ActiveTrace.Trace.terminalChecked_implies_baseAllPackedAndAllPaper_or_parentOpeningFailure_or_paperFailure

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveTrace.Trace.runtimeAccepted_implies_baseAllPackedAndAllPaper_or_parentOpeningFailure_or_paperFailure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ActiveTrace.Trace.runtimeAccepted_implies_baseAllPackedAndAllPaper_or_parentOpeningFailure_or_paperFailure

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.DelayedCombinedNc.combinedAtPoint_eq_terminalFromMessage_of_bound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.combinedAtPoint_eq_terminalFromMessage_of_bound

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.DelayedCombinedNc.combinedAtPoint_block_quartic' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.combinedAtPoint_block_quartic

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.DelayedCombinedNc.combinedAtPoint_lane_quartic' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.combinedAtPoint_lane_quartic

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.DelayedCombinedNc.combinedAtPoint_eq_ordinary_of_batchWeight_eq_zero' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.combinedAtPoint_eq_ordinary_of_batchWeight_eq_zero

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.DelayedCombinedNc.Acceptance.residualWeightIdentity_exact_iff' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.Acceptance.residualWeightIdentity_exact_iff

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.DelayedCombinedNc.Acceptance.expectedRoundsRepresentable' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.Acceptance.expectedRoundsRepresentable
