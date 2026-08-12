import Nightstream.Protocol.FPrime.Frozen
import tests.Axioms.Support

/-!
Fail-closed trusted-dependency gate for the frozen paper-authoritative facade.

This file guards proved headline equations, reductions, and formula
obstructions. Unproved target propositions remain definitions and are not
presented as established security theorems here.
-/

/-- info: 'Nightstream.Protocol.FPrime.Frozen.Obligations.superNeoCompositionReductionOfKnowledge' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.superNeoCompositionReductionOfKnowledge

/-- info: 'Nightstream.Protocol.FPrime.Frozen.Obligations.superNeoPaperObligations_of_components' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.superNeoPaperObligations_of_components

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.FinitePaperStrong.finitePaperStrong' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.finitePaperStrong

/-- info: 'Nightstream.Protocol.FPrime.Frozen.PiCcsFirstSuccessBridge.piCcsStrong_of_successGatedRetry' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.piCcsStrong_of_successGatedRetry

/-- info: 'Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteUniform.paperWeak' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.paperWeak

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.PiRlcBatch.CompatibleContext.batchPhi_eq_piCcsOutputPhi' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.batchPhi_eq_piCcsOutputPhi

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.PiRlcBatch.CompatibleContext.repeatedBatch_samePhi' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.repeatedBatch_samePhi

/-- info: 'Nightstream.Protocol.FPrime.Frozen.NonInteractiveOracleObstruction.distinct_contexts_same_derived' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.distinct_contexts_same_derived

/-- info: 'Nightstream.Protocol.FPrime.Frozen.NonInteractiveOracleObstruction.distinct_labels_same_squeeze' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.distinct_labels_same_squeeze

/-- info: 'Nightstream.Protocol.FPrime.Frozen.NonInteractiveOracleObstruction.distinct_public_inputs_same_bound_state' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.distinct_public_inputs_same_bound_state

/-- info: 'Nightstream.Protocol.FPrime.Frozen.NonInteractiveContinuationObstruction.replaceForkOracle_acceptedOutcome_iff' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.replaceForkOracle_acceptedOutcome_iff

/-- info: 'Nightstream.Protocol.FPrime.Frozen.NonInteractiveContinuationObstruction.replaceForkOracle_transitionOutcome_iff' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.replaceForkOracle_transitionOutcome_iff

/-- info: 'Nightstream.Protocol.FPrime.Frozen.NonInteractiveContinuationObstruction.distinct_replacement_oracles_same_nifs_execution' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.distinct_replacement_oracles_same_nifs_execution

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.piCcsExecution_coins_eq_replayInput' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.piCcsExecution_coins_eq_replayInput

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.piCcsExecution_outgoingState_eq_postOutput' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.piCcsExecution_outgoingState_eq_postOutput

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.piRlcChallenge_eq_response_after_piCcsOutput' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.piRlcChallenge_eq_response_after_piCcsOutput

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.not_piRlcSamplingSetFailure' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.not_piRlcSamplingSetFailure

/-- info: 'Nightstream.SuperNeo.InteractiveReduction.FiatShamirContract.anyFailure_iff_exists_event' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.anyFailure_iff_exists_event

/-- info: 'Nightstream.SuperNeo.InteractiveReduction.FiatShamirContract.anyFailure_probability_le_total' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.anyFailure_probability_le_total

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.TranscriptSecurityEvent.securityClass' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.TranscriptSecurityEvent.securityClass

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.transcriptSecurityEvent_implies_eventPredicate' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.transcriptSecurityEvent_implies_eventPredicate

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.acceptedFork_implies_ambientTargetOpenings' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.acceptedFork_implies_ambientTargetOpenings

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.piRlcExtractionFailure_implies_forkSampling_or_programmingFailure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.piRlcExtractionFailure_implies_forkSampling_or_programmingFailure

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.verify_sound_or_residual_or_multiFork' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.verify_sound_or_residual_or_multiFork

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.residualBadEvent_iff_residualFailure' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.residualBadEvent_iff_residualFailure

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.residualFailure_probability_le_total' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.residualFailure_probability_le_total

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.acceptedOutcome_implies_transition_or_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.acceptedOutcome_implies_transition_or_failure

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.nonInteractiveFailure_probability_le_total' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.nonInteractiveFailure_probability_le_total

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.accepted_probability_sub_total_le_transition' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.accepted_probability_sub_total_le_transition

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.RewindableProver.piRlcChallenges_baseProof' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.RewindableProver.piRlcChallenges_baseProof

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.RewindableForkOutcome.toAlignedForkOutcome_oracle' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.RewindableForkOutcome.toAlignedForkOutcome_oracle

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.RewindableForkOutcome.toAlignedForkOutcome_batch' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.RewindableForkOutcome.toAlignedForkOutcome_batch

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.RewindableForkOutcome.continuationSuccessAt_implies_parentOpening' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.RewindableForkOutcome.continuationSuccessAt_implies_parentOpening

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.RewindableForkOutcome.continuationSuccessAt_implies_piRlcVerifies' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.RewindableForkOutcome.continuationSuccessAt_implies_piRlcVerifies

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.RewindableForkOutcome.continuationSuccesses_imply_acceptedFork' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.RewindableForkOutcome.continuationSuccesses_imply_acceptedFork

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.piRlcForkSamplingFailure_implies_piDecContinuationFailure' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.piRlcForkSamplingFailure_implies_piDecContinuationFailure

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.rewindable_accepted_probability_sub_total_le_transition' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.rewindable_accepted_probability_sub_total_le_transition

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperCorrections.carriedTargetExponent_eq_absolute' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperCorrections.carriedTargetExponent_eq_absolute

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperCorrections.literalTargetExponent_ne_frozen' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperCorrections.literalTargetExponent_ne_frozen

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperCorrections.literalSection73NormIndices_ne_strictCentered_at_two' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperCorrections.literalSection73NormIndices_ne_strictCentered_at_two

/-- info: 'Nightstream.HyperNova.NonInteractiveMultiFold.accepts_iff_verify' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.HyperNova.NonInteractiveMultiFold.accepts_iff_verify

/-- info: 'Nightstream.HyperNova.Construction2.Paper.transition_iff_honestOuterDispatch_and_fixedAugmented' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.HyperNova.Construction2.Paper.transition_iff_honestOuterDispatch_and_fixedAugmented

/-- info: 'Nightstream.HyperNova.Construction2.Paper.holds_iff_transition' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.HyperNova.Construction2.Paper.holds_iff_transition

/-- info: 'Nightstream.HyperNova.Construction2.Paper.holds_iff_honestOuterDispatch_and_fixedAugmented' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.HyperNova.Construction2.Paper.holds_iff_honestOuterDispatch_and_fixedAugmented

/-- info: 'Nightstream.HyperNova.Construction2.Paper.terminalHolds_iff_transition' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.HyperNova.Construction2.Paper.terminalHolds_iff_transition

/-- info: 'Nightstream.HyperNova.Construction2.Paper.outerTerminalHolds_iff_transition' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.HyperNova.Construction2.Paper.outerTerminalHolds_iff_transition

/-- info: 'Nightstream.Protocol.FPrime.Frozen.HyperNova.nifsV_accepts_iff' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.HyperNova.nifsV_accepts_iff

/-- info: 'Nightstream.Protocol.FPrime.Frozen.HyperNova.fprime_accepts_iff_transition' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.HyperNova.fprime_accepts_iff_transition

/-- info: 'Nightstream.Protocol.FPrime.Frozen.HyperNova.canonicalFprime_accepts_iff_transition' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.HyperNova.canonicalFprime_accepts_iff_transition

/-- info: 'Nightstream.Protocol.FPrime.Frozen.HyperNova.canonicalFprime_accepts_implies_semanticTransition_or_selectedNifsBadEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.HyperNova.canonicalFprime_accepts_implies_semanticTransition_or_selectedNifsBadEvent

/-- info: 'Nightstream.Protocol.FPrime.Frozen.HyperNova.canonicalFprime_semanticTransition_implies_exists_nifsProof_accepts' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.HyperNova.canonicalFprime_semanticTransition_implies_exists_nifsProof_accepts

/-- info: 'Nightstream.Protocol.FPrime.Frozen.HyperNova.terminal_accepts_iff_transition' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.HyperNova.terminal_accepts_iff_transition

/-- info: 'Nightstream.Protocol.FPrime.Frozen.HyperNova.canonicalTerminal_accepts_iff_transition' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.HyperNova.canonicalTerminal_accepts_iff_transition

/-- info: 'Nightstream.Protocol.FPrime.Frozen.HyperNova.canonicalTerminal_exact_without_nifs' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.HyperNova.canonicalTerminal_exact_without_nifs

/-- info: 'Nightstream.Protocol.FPrime.Frozen.SuperNeo.piDec_reductionOfKnowledge' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.piDec_reductionOfKnowledge

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.PiRlcComposition.PiDec.finiteReductionOfKnowledge' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.finiteReductionOfKnowledge

/-- info: 'Nightstream.Protocol.FPrime.Frozen.SuperNeo.piRlc_literalAmbientBound_obstruction' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.piRlc_literalAmbientBound_obstruction

/-- info: 'Nightstream.Protocol.FPrime.Frozen.SuperNeo.piRlc_correctedAmbientBound_covers' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.piRlc_correctedAmbientBound_covers

/-- info: 'Nightstream.Protocol.FPrime.CanonicalVerifier.PaperNonInteractiveNifs.nifsSoundAndCompleteModulo' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.nifsSoundAndCompleteModulo

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.sourceValid_exists_verifiedTransition' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.sourceValid_exists_verifiedTransition

/-- info: 'Nightstream.Protocol.FPrime.CanonicalVerifier.PaperNonInteractiveNifs.canonicalFprime_accepts_implies_paperTransition_or_nifsBadEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.HyperNova.canonicalFprime_accepts_implies_paperTransition_or_nifsBadEvent

/-- info: 'Nightstream.Protocol.FPrime.CanonicalVerifier.PaperNonInteractiveNifs.canonicalFprime_paperTransition_implies_exists_nifsProof_accepts' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.HyperNova.canonicalFprime_paperTransition_implies_exists_nifsProof_accepts

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Key.statement_sumcheckDegreeBound_le' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Key.statement_sumcheckDegreeBound_le

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.piCcsCheck_eq_true_iff' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.piCcsCheck_eq_true_iff

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.piCcsRoundChain_of_check' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.piCcsRoundChain_of_check

/-- info: 'Nightstream.Protocol.FPrime.Frozen.NonInteractiveFixedKeyObstruction.programmingReceipts_force_same_base' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.programmingReceipts_force_same_base

/-- info: 'Nightstream.Protocol.FPrime.Frozen.NonInteractiveFixedKeyObstruction.distinct_bases_force_programming_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.distinct_bases_force_programming_failure

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.postPrefixForkExperiment_expectedQueriesAtMost' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.postPrefixForkExperiment_expectedQueriesAtMost

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.postPrefixOutcome_worldAccepted_iff' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.postPrefixOutcome_worldAccepted_iff

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.postPrefixChallengeSamplingFailure_probability_eq_zero' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.postPrefixChallengeSamplingFailure_probability_eq_zero

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.piRlcWorldProgrammingFailure_probability_le_paper' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.piRlcWorldProgrammingFailure_probability_le_paper

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.postPrefixExplicitRandomOracleContract' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.postPrefixExplicitRandomOracleContract

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.piRlcWorldAccepted_probability_sub_total_le_transition' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.piRlcWorldAccepted_probability_sub_total_le_transition

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.postPrefixAccepted_probability_sub_total_le_transition' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.postPrefixAccepted_probability_sub_total_le_transition

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.fullOracleAccepted_implies_transition_or_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.fullOracleAccepted_implies_transition_or_failure

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.fullOracleChallengeSamplingFailure_probability_eq_zero' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.fullOracleChallengeSamplingFailure_probability_eq_zero

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.fullOracleProgrammingFailure_probability_le_paper' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.fullOracleProgrammingFailure_probability_le_paper

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.fullOracleMixtureExplicitRandomOracleContract' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.fullOracleMixtureExplicitRandomOracleContract

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.fullOracleAccepted_probability_sub_total_le_transition' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.fullOracleAccepted_probability_sub_total_le_transition

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.fullOracleMixtureAccepted_probability_sub_total_le_transition' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.fullOracleMixtureAccepted_probability_sub_total_le_transition

/-- info: 'Nightstream.Protocol.FPrime.Frozen.NifsNonInteractiveBridge.fullOracleMixtureNifsNonInteractiveSound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.fullOracleMixtureNifsNonInteractiveSound

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Key.compatibleContext_piRlc' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.Key.compatibleContext_piRlc

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Key.compatiblePiDecContext_paper' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.Key.compatiblePiDecContext_paper

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.batchOfPrefix_eq_nifsPiRlcBatch' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.batchOfPrefix_eq_nifsPiRlcBatch

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.combinedParent_eq_nifsParent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.combinedParent_eq_nifsParent

/-- info: 'Nightstream.Protocol.FPrime.Frozen.NonInteractiveAdaptiveWitnessObstruction.fixed_witness_bound_does_not_bound_adaptive_existential' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.fixed_witness_bound_does_not_bound_adaptive_existential

/-- info: 'Nightstream.Protocol.FPrime.Frozen.PiDecTargetWitnessObstruction.accepted_without_piDec_target_witness' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.accepted_without_piDec_target_witness

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.RewindableProver.toInteractivePiDecReply_childAssignments' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.RewindableProver.toInteractivePiDecReply_childAssignments

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.CausalPrefixCouplingContract.support_eq_product' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.CausalPrefixCouplingContract.support_eq_product

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.CausalPrefixCouplingContract.mem_support_iff' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.CausalPrefixCouplingContract.mem_support_iff

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.CausalPrefixCouplingContract.support_cardinality' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.CausalPrefixCouplingContract.support_cardinality

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.CausalPrefixCouplingContract.toPrefixExperiment_prefixAligned' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.CausalPrefixCouplingContract.toPrefixExperiment_prefixAligned

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.CausalPrefixCouplingContract.toPrefixExperiment_piCcsCheck_extracts_sourceValid_or_badEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.CausalPrefixCouplingContract.toPrefixExperiment_piCcsCheck_extracts_sourceValid_or_badEvent

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.CausalPrefixCouplingContract.toPrefixExperiment_batch_eq' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.CausalPrefixCouplingContract.toPrefixExperiment_batch_eq

/-! Exact causal composition and D.6 target-success bridges. -/

/-- info: 'Nightstream.Protocol.FPrime.Frozen.SumCheckEncodingObstruction.fixed_width_acceptance_is_not_canonical_raw_acceptance' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.fixed_width_acceptance_is_not_canonical_raw_acceptance

/-! Fixed-width paper gate and exact NIFS alignment. -/

/-- info: 'Nightstream.SuperNeo.SumCheck.Finite.FixedPhase.RawCertificate.check_encode' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.SumCheck.Finite.FixedPhase.RawCertificate.check_encode

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolPolynomial.FixedWidth.accepted_implies_tableTruth_or_badEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.accepted_implies_tableTruth_or_badEvent

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction.fixedWidthAcceptedProbe_extracts_source_or_badEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.fixedWidthAcceptedProbe_extracts_source_or_badEvent

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.acceptedCheck_eq_piCcsCheck' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.acceptedCheck_eq_piCcsCheck

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.mixingFailure_iff_piCcsMixingRoot' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.mixingFailure_iff_piCcsMixingRoot

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.sumCheckFailure_iff_piCcsSumCheckCollision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.sumCheckFailure_iff_piCcsSumCheckCollision

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.piCcsCheck_extracts_sourceValid_or_badEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.piCcsCheck_extracts_sourceValid_or_badEvent

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.RewindableProver.interactivePiDecExecution_eq_continuation' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.RewindableProver.interactivePiDecExecution_eq_continuation

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.RewindableProver.continuationPiDecExecution_baseChallenges_attempt' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.RewindableProver.continuationPiDecExecution_baseChallenges_attempt

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.RewindableForkOutcome.piDecExecutionAt_eq_continuation' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.RewindableForkOutcome.piDecExecutionAt_eq_continuation

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.RewindableForkOutcome.continuationSuccessAt_baseChallenges_iff' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.RewindableForkOutcome.continuationSuccessAt_baseChallenges_iff

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.CausalPrefixCouplingContract.interactivePiDecExecution_eq_continuation' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.CausalPrefixCouplingContract.interactivePiDecExecution_eq_continuation

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.CausalPrefixCouplingContract.interactivePiDecExecution_eq_postPrefix' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.CausalPrefixCouplingContract.interactivePiDecExecution_eq_postPrefix

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.RewindablePiRlcWorldOutcome.piDecExecutionAt_world_attempt_eq_nifs' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.RewindablePiRlcWorldOutcome.piDecExecutionAt_world_attempt_eq_nifs

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.RewindablePiRlcWorldOutcome.continuationSuccessAt_world_iff_nifs_target' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.RewindablePiRlcWorldOutcome.continuationSuccessAt_world_iff_nifs_target

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.CausalPrefixCouplingContract.interactivePiDecSuccess_iff_postPrefixNifsTarget' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.CausalPrefixCouplingContract.interactivePiDecSuccess_iff_postPrefixNifsTarget
