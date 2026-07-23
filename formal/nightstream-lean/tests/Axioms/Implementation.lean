import Nightstream.Implementation
import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.SourceRoleCensus
import tests.Axioms.Implementation.PiCcsTranscript
import tests.Axioms.Implementation.PiRlcPackedMod5
import tests.Axioms.Implementation.PiRlcSelectionAggregateExactness
import tests.Axioms.Implementation.PiRlcChunkAggregateAcceptance
import tests.Axioms.Implementation.PiRlcChunkAggregateArtifact
import tests.Axioms.Implementation.PiRlcAggregateAcceptanceOuterImage
import tests.Axioms.Implementation.SeededPhi81Sampler
import tests.Axioms.Implementation.PiCcsMatrix
import tests.Axioms.Implementation.PiCcsNc
import tests.Axioms.Implementation.FPrimeFullHistoryNifsPaper
import tests.Axioms.Implementation.FPrimeRecursiveYZcolProjection
import tests.Axioms.Implementation.FPrimeRecursiveYZcolProjectionRefinement
import tests.Axioms.Implementation.FPrimeSelectiveFixedPointPiRlcProjectionYZcol
import tests.Axioms.Implementation.FPrimeSelectiveFixedPointCarrier270PublicPadding
import tests.Axioms.Implementation.FPrimeSelectiveFixedPointCarrier270ProductionPublicWriteTrace
import tests.Axioms.Implementation.FPrimeSelectiveFixedPointCarrier270PrivatePadding
import tests.Axioms.Implementation.FPrimeSelectiveFixedPointCarrier270Selectors
import tests.Axioms.Implementation.FPrimeSelectiveFixedPointPiCcsNcRawRunningDecoder
import tests.Axioms.Implementation.FPrimeSelectiveFixedPointPiCcsNcFreshSourceDecoder
import tests.Axioms.Implementation.FPrimeSelectiveFixedPointPiCcsNcPackedWitness
import tests.Axioms.Implementation.FPrimeSelectiveFixedPointPiCcsNcProductionDomain
import tests.Axioms.Implementation.FPrimeSelectiveFixedPointPiCcsNcActiveBoundary
import tests.Axioms.Implementation.FPrimeSelectiveFixedPointPiCcsNcRawOldBlockExecution
import tests.Axioms.Implementation.FPrimeSelectiveFixedPointPiDecActiveResultBridge
import tests.Axioms.Implementation.FPrimeSelectiveFixedPointPiCcsNcSourceRefinement
import tests.Axioms.Implementation.FPrimeSelectiveFixedPointAccumulatorPendingFamilyCodec
import tests.Axioms.Implementation.FPrimeRecursivePiRlcProjectionBetaLadder
import tests.Axioms.Implementation.FPrimeRecursivePiRlcProjectionBetaLadderRefinement
import tests.Axioms.Implementation.FPrimeRecursivePiRlcProjectionRhoEvaluations
import tests.Axioms.Implementation.FPrimeRecursivePiRlcProjectionRhoEvaluationsRefinement
import tests.Axioms.Implementation.FPrimeRecursivePiRlcProjectionYZcolIdentities
import tests.Axioms.Implementation.FPrimeRecursivePiRlcProjectionYZcolIdentitiesRefinement
import tests.Axioms.Implementation.FPrimeRecursivePiRlcProjectionYZcolNormalForm
import tests.Axioms.Implementation.FPrimeRecursivePiRlcChallengeWiring
import tests.Axioms.Implementation.FPrimeRecursivePiRlcChallengeSamplerLayout
import tests.Axioms.Implementation.FPrimeRecursivePiRlcChallengeTranscriptLayout
import tests.Axioms.Implementation.FPrimeRecursivePiRlcChallengeSamplerFirstAccepted
import tests.Axioms.Implementation.FPrimeRecursivePiRlcChallengeProjectionConsumer
import tests.Axioms.Implementation.FPrimeRecursivePiRlcChallengeTranscriptHandoff
import tests.Axioms.Implementation.FPrimeRecursiveProfileScope
import tests.Axioms.Implementation.FPrimeFullHistorySelectiveCarrier270
import tests.Axioms.Implementation.FPrimeFullHistorySelectiveCcsPolynomial
import tests.Axioms.Implementation.FPrimeFullHistorySelectiveCcsSelectorComposition
import tests.Axioms.Implementation.FPrimeFullHistorySelectiveCcsRowAction
import tests.Axioms.Implementation.FPrimeFullHistorySelectiveCcsRowArtifact
import tests.Axioms.Implementation.FPrimeFullHistorySelectiveCcsPayloadRefinement
import tests.Axioms.Implementation.FPrimeFullHistorySelectiveRelationProfile
import tests.Axioms.Implementation.FPrimeFullHistorySelectiveFixedPointShape
import tests.Axioms.Implementation.FPrimeFullHistoryPiRlcClaimEvaluationCarrier
import tests.Axioms.Implementation.FPrimeFullHistoryPiRlcClaimShapeAlignment
import tests.Axioms.Implementation.PiCcsOutputDigestPoseidon
import tests.Axioms.Implementation.PiCcsOutputActiveEnvelopeSemantics
import tests.Axioms.Implementation.PiCcsOutputDigestProfile
import tests.Axioms.Implementation.PiCcsOutputActiveSemantics
import tests.Axioms.Implementation.PiCcsOutputActiveSourceLayout
import tests.Axioms.Implementation.PiCcsOutputYZcolConsumer
import tests.Axioms.Implementation.PiCcsOutputActiveSisBoundary
import tests.Axioms.Implementation.PiCcsOutputActiveProjectionIdentity
import tests.Axioms.Implementation.ProjectionArtifactProgram
import tests.Axioms.Implementation.PiCcsOutputSemanticHandoff
import tests.Axioms.Implementation.PiCcsOutputProjection
import tests.Axioms.Implementation.PiDecStrict
import tests.Axioms.Implementation.PiDecStrictReducedY
import tests.Axioms.Implementation.PiDecStrictCanonicalX
import tests.Axioms.Implementation.PiDecTypedCarrier
import tests.Axioms.Implementation.PiDecStrictProductionCompiler
import tests.Axioms.Implementation.FPrimeFixedOneTypedLowering
import tests.Axioms.Support

/-!
Fail-closed implementation correspondence axioms gate. Every expectation is checked when this
module is built; the aggregate entrypoint imports all ownership groups.
-/

/-- info: 'Nightstream.Implementation.FPrime.Envelope.check_sound' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.FPrime.Envelope.check_sound

/-- info: 'Nightstream.Implementation.FPrime.CounterRefinement.counter_refinement' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.FPrime.CounterRefinement.counter_refinement

/-- info: 'Nightstream.Implementation.R1CS.bitRow_le_one' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.bitRow_le_one

/-- info: 'Nightstream.Implementation.R1CS.Program.run_agrees_of_satisfies' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.Program.run_agrees_of_satisfies

/-- info: 'Nightstream.Implementation.R1CS.Program.run_agrees_of_builder_satisfies' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.Program.run_agrees_of_builder_satisfies

/-- info: 'Nightstream.Implementation.R1CS.Program.run_satisfies_builder_rows' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.Program.run_satisfies_builder_rows

/-- info: 'Nightstream.Implementation.R1CS.CheckedProgram.sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.CheckedProgram.sound

/-- info: 'Nightstream.Implementation.R1CS.CheckedProgram.complete' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.CheckedProgram.complete

/-- info: 'Nightstream.Implementation.R1CS.SeededPhi81.sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SeededPhi81.sound

/-- info: 'Nightstream.Implementation.R1CS.SeededPhi81.complete' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SeededPhi81.complete

/-- info: 'Nightstream.Implementation.R1CS.ShiftedTernarySound.canonicalOpening_of_satisfies' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.ShiftedTernarySound.canonicalOpening_of_satisfies

/-- info: 'Nightstream.Implementation.R1CS.ShiftedTernarySound.commitmentHolds_of_satisfies' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.ShiftedTernarySound.commitmentHolds_of_satisfies

/-- info: 'Nightstream.Implementation.R1CS.ShiftedTernarySound.oneField_sound' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.ShiftedTernarySound.oneField_sound

/-- info: 'Nightstream.Implementation.R1CS.ShiftedTernarySound.canonicalOpening_of_canonicalRows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.ShiftedTernarySound.canonicalOpening_of_canonicalRows

/-- info: 'Nightstream.Implementation.R1CS.ShiftedTernaryComplete.canonicalRows_complete' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.ShiftedTernaryComplete.canonicalRows_complete

/-- info: 'Nightstream.Implementation.R1CS.LinearSubstitution.lcEval_terms' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.LinearSubstitution.lcEval_terms

/-- info: 'Nightstream.Implementation.R1CS.BooleanRowDedup.substituted_bitRow_iff_slot_bitRow' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.BooleanRowDedup.substituted_bitRow_iff_slot_bitRow

/-- info: 'Nightstream.Implementation.R1CS.BooleanRowDedup.substituted_swappedBitRow_iff_slot_bitRow' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.BooleanRowDedup.substituted_swappedBitRow_iff_slot_bitRow

/-- info: 'Nightstream.Implementation.R1CS.ShiftedTernarySharedSlots.productionAccepts_iff_canonicalRows' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.ShiftedTernarySharedSlots.productionAccepts_iff_canonicalRows

/-- info: 'Nightstream.Implementation.R1CS.ShiftedTernarySharedSlots.production_decoded_sharedAlias' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.ShiftedTernarySharedSlots.production_decoded_sharedAlias

/-- info: 'Nightstream.Implementation.R1CS.ShiftedTernarySharedSlots.artifactGateAccepts_iff_productionAccepts' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.ShiftedTernarySharedSlots.artifactGateAccepts_iff_productionAccepts

/-- info: 'Nightstream.Implementation.R1CS.ShiftedTernarySharedSlots.artifactGateAccepts_iff_canonicalRows' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.ShiftedTernarySharedSlots.artifactGateAccepts_iff_canonicalRows

/-- info: 'Nightstream.Implementation.R1CS.ShiftedTernaryCenteredZero.centered_zero_unique' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.ShiftedTernaryCenteredZero.centered_zero_unique

/-- info: 'Nightstream.Implementation.R1CS.ShiftedTernaryReducedCore.reduced_iff_canonicalRows' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.ShiftedTernaryReducedCore.reduced_iff_canonicalRows

/-- info: 'Nightstream.Implementation.R1CS.ShiftedTernaryReducedCore.canonicalOpening_of_reduced' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.ShiftedTernaryReducedCore.canonicalOpening_of_reduced

/-- info: 'Nightstream.Implementation.R1CS.ShiftedTernaryReducedCore.Accepts.borrow_bitness_follows' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.ShiftedTernaryReducedCore.Accepts.borrow_bitness_follows

/-- info: 'Nightstream.Implementation.R1CS.ShiftedTernaryReducedCore.CanonicalWitness.reducedCore_complete' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.ShiftedTernaryReducedCore.CanonicalWitness.reducedCore_complete

/-- info: 'Nightstream.Implementation.R1CS.Poseidon2PermutationSound.poseidon2Permutation_sound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.Poseidon2PermutationSound.poseidon2Permutation_sound

/-- info: 'Nightstream.Implementation.R1CS.Poseidon2PermutationSound.poseidon2Permutation_complete' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.Poseidon2PermutationSound.poseidon2Permutation_complete

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine.digestChunks_lane_part' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine.digestChunks_lane_part

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine.digestBlock_absorbed_zero' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine.digestBlock_absorbed_zero

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine.successfulExecution_successorState' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine.successfulExecution_successorState

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.DigestRounds.callAccepted_permute' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.DigestRounds.callAccepted_permute

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.DigestRounds.scheduledCallsAccepted' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.DigestRounds.scheduledCallsAccepted

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.PinSchedule.facts' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.PinSchedule.facts

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.ScheduleRefinement.accepted_refines_stateSchedule' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.ScheduleRefinement.accepted_refines_stateSchedule

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.ChunkOrder.accepted_refines_oneScalarSchedule' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.ChunkOrder.accepted_refines_oneScalarSchedule

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.ChunkRows.rows_eq_generated' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.ChunkRows.rows_eq_generated

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.SelectionRows.rows_eq_generated' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.SelectionRows.rows_eq_generated

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.acceptanceRows_refine_verifier' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.acceptanceRows_refine_verifier

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.decompositionRow_sound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.decompositionRow_sound

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.symbolRow_refines_verifier' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.symbolRow_refines_verifier

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.cumulativeRow_refines_verifier' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.cumulativeRow_refines_verifier

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Lane.refines' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Lane.refines

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.ChunkOrder.accepted_refines_lane' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.ChunkOrder.accepted_refines_lane

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.ChunkOrder.satisfyingLane_refines' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.ChunkOrder.satisfyingLane_refines

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.LaneRows.refines' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.LaneRows.refines

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.LaneRows.accepted_refines_machineLane' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.LaneRows.accepted_refines_machineLane

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.OneScalarRows.accepted_initialCount_zero' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.OneScalarRows.accepted_initialCount_zero

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.OneScalarRows.accepted_laneRows' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.OneScalarRows.accepted_laneRows

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.OneScalarRows.accepted_tailRows' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.OneScalarRows.accepted_tailRows

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.OneScalar.accepted_refines' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.OneScalar.accepted_refines

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.ScalarRows.canonicalPiece_eq' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.ScalarRows.canonicalPiece_eq

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.ScalarRows.lanePiece_eq' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.ScalarRows.lanePiece_eq

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.ScalarRows.tailPiece_eq' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.ScalarRows.tailPiece_eq

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.ScalarRows.accepted_canonicalRows' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.ScalarRows.accepted_canonicalRows

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.ScalarRows.accepted_laneRows' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.ScalarRows.accepted_laneRows

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.ScalarRows.accepted_tailRows' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.ScalarRows.accepted_tailRows

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.ScalarRows.accepted_initialCount_zero' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.ScalarRows.accepted_initialCount_zero

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.ScalarRows.accepted_canonicalLane_refines' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.ScalarRows.accepted_canonicalLane_refines

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.ScalarLanes.satisfyingRows_refine' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.ScalarLanes.satisfyingRows_refine

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.ScalarSemantics.counterChain' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.ScalarSemantics.counterChain

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.ScalarSemantics.accepted_refines_lanes' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.ScalarSemantics.accepted_refines_lanes

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.TailRows.accepted_satisfies' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.TailRows.accepted_satisfies

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.CandidateOrder.address_recomposes' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.CandidateOrder.address_recomposes

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.TailCandidateSemantics.laneIndex_block' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.TailCandidateSemantics.laneIndex_block

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.TailCandidateSemantics.laneIndex_lane' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.TailCandidateSemantics.laneIndex_lane

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.TailCandidateSemantics.candidate_refines' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.TailCandidateSemantics.candidate_refines

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.TailCandidateSemantics.cumulative_step' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.TailCandidateSemantics.cumulative_step

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.TailSources.acceptColumnMap' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.TailSources.acceptColumnMap

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.TailSources.symbolColumnMap' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.TailSources.symbolColumnMap

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.TailSources.cumulativeColumnMap' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.TailSources.cumulativeColumnMap

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.TailSources.nonzeroPriorColumnMap' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.TailSources.nonzeroPriorColumnMap

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.TailSources.accepted_sourceBindings' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.TailSources.accepted_sourceBindings

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.TailSources.accepted_candidate_refines' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.TailSources.accepted_candidate_refines

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.TailSources.accepted_cumulative_step' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.TailSources.accepted_cumulative_step

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.TailPrefixCounts.prefixCount_succ' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.TailPrefixCounts.prefixCount_succ

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.TailPrefixCounts.prefixWire_refines' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.TailPrefixCounts.prefixWire_refines

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.TailPrefixCounts.cumulativeWire_refines' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.TailPrefixCounts.cumulativeWire_refines

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.TailPrefixCounts.enoughAccepted' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.TailPrefixCounts.enoughAccepted

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.TailFirstAccepted.semanticOutput_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.TailFirstAccepted.semanticOutput_length

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.TailFirstAccepted.outputAt_refines' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.TailFirstAccepted.outputAt_refines

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.TailFirstAccepted.productionOutput_eq_semanticFieldOutput' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.TailFirstAccepted.productionOutput_eq_semanticFieldOutput

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.FirstAccepted.accepted_genericTailSatisfies' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.FirstAccepted.accepted_genericTailSatisfies

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.FirstAccepted.semanticOutput_length' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.FirstAccepted.semanticOutput_length

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.FirstAccepted.outputAt_refines' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.FirstAccepted.outputAt_refines

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.FirstAccepted.accepted_refines' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.FirstAccepted.accepted_refines

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.Schedule.completeCallTree_eq' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.Schedule.completeCallTree_eq

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.PinSchedule.Artifact.pinTree_eq' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.PinSchedule.Artifact.pinTree_eq

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.PinSchedule.facts' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.PinSchedule.facts

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.PinSchedule.Facts.entryCoordinate' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.PinSchedule.Facts.entryCoordinate

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.PinSchedule.Facts.block0Counter' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.PinSchedule.Facts.block0Counter

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.PinSchedule.Facts.laterCounter' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.PinSchedule.Facts.laterCounter

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.DigestRounds.entryBoundaryCallAccepted' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.DigestRounds.entryBoundaryCallAccepted

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.DigestRounds.scalar0Block0FullCursorCallAccepted' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.DigestRounds.scalar0Block0FullCursorCallAccepted

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.DigestRounds.scheduledCallsAccepted' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.DigestRounds.scheduledCallsAccepted

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.DigestRounds.laterDigestCallAccepted' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.DigestRounds.laterDigestCallAccepted

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.ScheduleRefinement.laterDigestBlock_refines' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.ScheduleRefinement.laterDigestBlock_refines

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.ScheduleRefinement.stateBeforeScalar_next' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.ScheduleRefinement.stateBeforeScalar_next

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.ScheduleRefinement.stateScheduleRefined' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.ScheduleRefinement.stateScheduleRefined

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.OutputDigestSemantics.inputClaimsDigestLabelNats_eq' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.OutputDigestSemantics.inputClaimsDigestLabelNats_eq

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.Semantics.serializeTerminalOutputs_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.Semantics.serializeTerminalOutputs_length

/-- info: 'Nightstream.Implementation.R1CS.ShiftedTernaryCanonicalWord.opening_trit_eq_native' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.ShiftedTernaryCanonicalWord.opening_trit_eq_native

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.EncodingSchedule.accepted_mainPiece_word' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.EncodingSchedule.accepted_mainPiece_word

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.EncodingSchedule.accepted_primaryCommitment' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.EncodingSchedule.accepted_primaryCommitment

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.SourceLayout.roleColumns_eq_artifact' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.SourceLayout.roleColumns_eq_artifact

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.SourceLayout.accepted_initialPins' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.SourceLayout.accepted_initialPins

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.SourceLayout.accepted_decodedSerialization' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.SourceLayout.accepted_decodedSerialization

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.Sis.Refinement.valueTerms_eq_semanticTerms' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.Sis.Refinement.valueTerms_eq_semanticTerms

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.Sis.Refinement.outputCoordinate_eq' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.Sis.Refinement.outputCoordinate_eq

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.Sis.Refinement.outputs_eq_apply' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.Sis.Refinement.outputs_eq_apply

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.Sis.ProductionBinding.accepted_primaryWordAgreement' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.Sis.ProductionBinding.accepted_primaryWordAgreement

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.Sis.ProductionBinding.accepted_primaryOutputs' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.Sis.ProductionBinding.accepted_primaryOutputs

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.Sis.ProductionBinding.accepted_composedOutputs' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.Sis.ProductionBinding.accepted_composedOutputs

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.Sis.SeedBinding.primaryGeometry' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.Sis.SeedBinding.primaryGeometry

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.Sis.SeedBinding.compressionGeometry' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.Sis.SeedBinding.compressionGeometry

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.Sis.SeedBinding.primarySeeds_derived' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.Sis.SeedBinding.primarySeeds_derived

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.Sis.SeedBinding.compressionSeeds_derived' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.Sis.SeedBinding.compressionSeeds_derived

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.Sis.SeedBinding.rejectionFuel_eq' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.Sis.SeedBinding.rejectionFuel_eq

/-- info: 'Nightstream.Implementation.R1CS.ChaCha8Refinement.quarterRound_refines' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.ChaCha8Refinement.quarterRound_refines

/-- info: 'Nightstream.Implementation.R1CS.ChaCha8Refinement.blockWords_eq' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.ChaCha8Refinement.blockWords_eq

/-- info: 'Nightstream.Implementation.R1CS.ChaCha8Refinement.words_eq' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.ChaCha8Refinement.words_eq

/-- info: 'Nightstream.Implementation.R1CS.ChaCha8Refinement.u64s_eq' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.ChaCha8Refinement.u64s_eq

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.OutputDigestSchedule.scheduleTree_eq' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.OutputDigestSchedule.scheduleTree_eq

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.OutputDigestPins.labelValues_match_semantics' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.OutputDigestPins.labelValues_match_semantics

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.OutputDigestBinding.catchupState_computed' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.OutputDigestBinding.catchupState_computed

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.OutputDigestBinding.completeBinding_eq_initialState' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.OutputDigestBinding.completeBinding_eq_initialState

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.OutputDigestBinding.accepted_refines_outputDigestBinding' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.OutputDigestBinding.accepted_refines_outputDigestBinding

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.CandidateRefinement.accepted_refines_candidateStream' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.CandidateRefinement.accepted_refines_candidateStream

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.CandidateRefinement.accepted_refines_machineCandidates' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.CandidateRefinement.accepted_refines_machineCandidates

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.MachineOutput.fieldCandidates_eq_machineCandidates' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.MachineOutput.fieldCandidates_eq_machineCandidates

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.MachineOutput.semanticOutput_length' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.MachineOutput.semanticOutput_length

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.MachineOutput.productionOutput_refines' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.MachineOutput.productionOutput_refines

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Selection.Initialization.zeroPrefix_eq_zero' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Selection.Initialization.zeroPrefix_eq_zero

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Selection.Acceptance.enoughAccepted' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Selection.Acceptance.enoughAccepted

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Selection.OneHot.exists_selectedOffset' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Selection.OneHot.exists_selectedOffset

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Selection.Position.refines' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Selection.Position.refines

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.TailRows.accepted_satisfies' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.TailRows.accepted_satisfies

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.TailRows.satisfyingRows_refine' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.TailRows.satisfyingRows_refine

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.TailInputs.candidate_refines' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.TailInputs.candidate_refines

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.TailInputs.cumulative_step' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.TailInputs.cumulative_step

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.PrefixCounts.prefixWire_refines' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.PrefixCounts.prefixWire_refines

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.PrefixCounts.enoughAccepted' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.PrefixCounts.enoughAccepted

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Selection.FirstAcceptedRefinement.outputAt_refines' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Selection.FirstAcceptedRefinement.outputAt_refines

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Selection.FirstAcceptedRefinement.accepted_refines' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Selection.FirstAcceptedRefinement.accepted_refines

/-- info: 'Nightstream.Implementation.R1CS.OutputAuthoritySboxManifest.candidateColumns_nodup' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.OutputAuthoritySboxManifest.candidateColumns_nodup

/-- info: 'Nightstream.Implementation.R1CS.OutputAuthoritySboxManifest.allCalls_transport' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.OutputAuthoritySboxManifest.allCalls_transport

/-- info: 'Nightstream.Implementation.R1CS.OutputAuthoritySboxManifest.noEscape_use_roles' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.OutputAuthoritySboxManifest.noEscape_use_roles

/-- info: 'Nightstream.Implementation.R1CS.canonicalU64_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.canonicalU64_sound

/-- info: 'Nightstream.Implementation.R1CS.u64Increment_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.u64Increment_sound

/-- info: 'Nightstream.Implementation.R1CS.u64Add_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.u64Add_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeCounterSound.fPrimeCounter_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeCounterSound.fPrimeCounter_sound

/-- info: 'Nightstream.Implementation.Encoding.FPrime.encInst_injective' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Encoding.FPrime.encInst_injective

/-- info: 'Nightstream.Implementation.Encoding.FPrime.encInst_bits_injective' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Encoding.FPrime.encInst_bits_injective

/-- info: 'Nightstream.Implementation.R1CS.FPrimeEncodingSound.fPrimeEncoding_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeEncodingSound.fPrimeEncoding_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeEncodingSound.accepted_public_bits_injective' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeEncodingSound.accepted_public_bits_injective

/-- info: 'Nightstream.Implementation.R1CS.FPrimeTerminalLinkSound.fPrimeTerminalLink_sound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeTerminalLinkSound.fPrimeTerminalLink_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeStateLinkSound.fPrimeStateLink_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeStateLinkSound.fPrimeStateLink_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeBaseStateSound.fPrimeBaseState_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeBaseStateSound.fPrimeBaseState_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeBaseProgramSound.fPrimeBaseProgram_sound' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeBaseProgramSound.fPrimeBaseProgram_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeBaseProgramSound.fPrimeBaseProgram_complete' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeBaseProgramSound.fPrimeBaseProgram_complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseStepSound.fPrimeFullHistoryBase_step_local_sound' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseStepSound.fPrimeFullHistoryBase_step_local_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseOutgoingSound.outgoing_sound' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseOutgoingSound.outgoing_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseOutgoingSound.base_step_holds' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseOutgoingSound.base_step_holds

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseGenericSound.local_sound' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseGenericSound.local_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseGenericSound.outgoing_sound' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseGenericSound.outgoing_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseGenericSound.step_holds' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseGenericSound.step_holds

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryProjection.recursive_exact_or_badRoot' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryProjection.recursive_exact_or_badRoot

/-- info: 'Nightstream.Implementation.R1CS.FPrimeChunkDigestSound.fPrimeChunkDigest_binding_sound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeChunkDigestSound.fPrimeChunkDigest_binding_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeChunkDigestSound.fPrimeChunkDigest_sound' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeChunkDigestSound.fPrimeChunkDigest_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeChunkDigestSound.fPrimeChunkDigest_complete' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeChunkDigestSound.fPrimeChunkDigest_complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeCeContinuitySound.fPrimeCeContinuity_sound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeCeContinuitySound.fPrimeCeContinuity_sound

/-- info: 'Nightstream.Implementation.R1CS.ProjectionProgram.ProjectionTrace.evaluation_sound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.ProjectionProgram.ProjectionTrace.evaluation_sound

/-- info: 'Nightstream.Implementation.R1CS.ProjectionProgram.ProjectionTrace.census_batchAccepted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.ProjectionProgram.ProjectionTrace.census_batchAccepted

/-- info: 'Nightstream.Implementation.R1CS.PiRLCProjection.exactRows_imply_batchAccepted' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRLCProjection.exactRows_imply_batchAccepted

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursiveManifest.topLevel_covers_program' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeRecursiveManifest.topLevel_covers_program

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursiveManifest.nifs_covers_block' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeRecursiveManifest.nifs_covers_block

/-- info: 'Nightstream.Implementation.Rust.FPrime.verify_eq_ok_iff_checkLocal' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.FPrime.verify_eq_ok_iff_checkLocal

/-- info: 'Nightstream.Implementation.Rust.FPrime.success_with_outgoing_refines_step' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.FPrime.success_with_outgoing_refines_step

/-- info: 'Nightstream.Implementation.Rust.FPrime.invalid_has_named_rejection' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.FPrime.invalid_has_named_rejection

/-- info: 'Nightstream.Implementation.Rust.Terminal.success_refines_terminalCE' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.Terminal.success_refines_terminalCE

/-- info: 'Nightstream.Implementation.Rust.Terminal.invalid_has_named_rejection' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.Terminal.invalid_has_named_rejection

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryProjection.exact_or_badRoot' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryProjection.exact_or_badRoot

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalContinuity.sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalContinuity.sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalContinuitySound.sound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalContinuitySound.sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalContinuitySound.complete' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalContinuitySound.complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalContinuitySound.satisfies_iff_holds' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalContinuitySound.satisfies_iff_holds

/-- info: 'Nightstream.Implementation.R1CS.CheckedProgram.satisfies_iff_assignmentHolds' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.CheckedProgram.satisfies_iff_assignmentHolds

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalLinkSound.sound' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalLinkSound.sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalLinkSound.complete' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalLinkSound.complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalLinkSound.satisfies_iff_holds' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalLinkSound.satisfies_iff_holds

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryOutputEncodingSound.sound' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryOutputEncodingSound.sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryOutputEncodingSound.terminalFreshDigest_eq_outputDigest' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryOutputEncodingSound.terminalFreshDigest_eq_outputDigest

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveOutputSound.sound' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveOutputSound.sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveOutputSound.xOutValues_sound' depends on axioms: [propext,
 Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveOutputSound.xOutValues_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveOutputSound.terminalFreshDigest_eq_xOut' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveOutputSound.terminalFreshDigest_eq_xOut

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveOutputSound.complete' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveOutputSound.complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryCounterLocalSound.local_sound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryCounterLocalSound.local_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryCounterSound.sound' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryCounterSound.sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryCounterLocalSound.Compiler.complete' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryCounterLocalSound.Compiler.complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryCounterSound.Compiler.complete' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryCounterSound.Compiler.complete

/-- info: 'Nightstream.Implementation.R1CS.PiDecStrictSound.Exact.recursive_sound' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiDecStrictSound.Exact.recursive_sound

/-- info: 'Nightstream.Implementation.R1CS.PiDecStrictSound.Exact.complete' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiDecStrictSound.Exact.complete

/-- info: 'Nightstream.Implementation.R1CS.PiDecAjtaiOpeningCollision.parentOpeningBindingCollision_to_ajtaiOpeningCollision' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiDecAjtaiOpeningCollision.parentOpeningBindingCollision_to_ajtaiOpeningCollision

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryPointBindingSound.recursive_sound' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryPointBindingSound.recursive_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryPointBindingSound.recursive_complete' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryPointBindingSound.recursive_complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryProjection.terminal_exact_or_badRoot' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryProjection.terminal_exact_or_badRoot

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCeSound.canonical_sound' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCeSound.canonical_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCeSound.canonical_complete' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCeSound.canonical_complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCeSound.all_claims_sound' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCeSound.all_claims_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryAffineSound.Recursive.sound' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryAffineSound.Recursive.sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryAffineSound.Recursive.complete' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryAffineSound.Recursive.complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryAffineSound.Terminal.sound' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryAffineSound.Terminal.sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryAffineSound.Terminal.complete' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryAffineSound.Terminal.complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles.recursive_roles_native_order' depends on axioms: [Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles.recursive_roles_native_order

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles.terminal_roles_native_order' depends on axioms: [Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles.terminal_roles_native_order

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles.full_history_profile_arities' depends on axioms: [Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles.full_history_profile_arities

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles.recursive_y_zcol_identity_census' depends on axioms: [Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles.recursive_y_zcol_identity_census

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles.terminal_y_zcol_identity_census' depends on axioms: [Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles.terminal_y_zcol_identity_census

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles.y_zcol_padding_census' depends on axioms: [Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles.y_zcol_padding_census

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles.y_zcol_active_coefficient_width' depends on axioms: [Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles.y_zcol_active_coefficient_width

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles.y_zcol_output_evaluation_width' depends on axioms: [Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles.y_zcol_output_evaluation_width

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles.y_zcol_padding_width' depends on axioms: [Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles.y_zcol_padding_width

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles.y_zcol_padded_output_width' depends on axioms: [Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles.y_zcol_padded_output_width

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles.y_zcol_padded_output_columns_disjoint' depends on axioms: [Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles.y_zcol_padded_output_columns_disjoint

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles.y_zcol_output_padding_is_glue' depends on axioms: [Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles.y_zcol_output_padding_is_glue

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles.y_zcol_output_padding_rows_match_glue_owner' depends on axioms: [Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles.y_zcol_output_padding_rows_match_glue_owner

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles.y_zcol_identity_ranges_nonempty' depends on axioms: [Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles.y_zcol_identity_ranges_nonempty

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles.recursive_glue_sound' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles.recursive_glue_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles.terminal_glue_sound' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles.terminal_glue_sound

/-- info: 'Nightstream.Implementation.R1CS.PiDecStrictCompiler.check_eq_true_iff' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiDecStrictCompiler.check_eq_true_iff

/-- info: 'Nightstream.Implementation.R1CS.ProjectionProgram.ProjectionTrace.native_complete' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.ProjectionProgram.ProjectionTrace.native_complete

/-- info: 'Nightstream.Implementation.R1CS.SumcheckRoundSound.native_complete' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SumcheckRoundSound.native_complete

/-- info: 'Nightstream.Implementation.R1CS.SumcheckChainSound.complete' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SumcheckChainSound.complete

/-- info: 'Nightstream.Implementation.R1CS.PiDecStrictSound.Exact.native_complete' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiDecStrictSound.Exact.native_complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryPublicPinsSound.Artifact.sound' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryPublicPinsSound.Artifact.sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryPublicPinsSound.Artifact.complete' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryPublicPinsSound.Artifact.complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStateLinkSound.complete' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryStateLinkSound.complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalParentLinkSound.sound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalParentLinkSound.sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalParentLinkSound.complete' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalParentLinkSound.complete

/-- info: 'Nightstream.Implementation.R1CS.TranscriptCertificate.ordered_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.TranscriptCertificate.ordered_sound

/-- info: 'Nightstream.Implementation.R1CS.TranscriptCertificate.ordered_complete' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.TranscriptCertificate.ordered_complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalRunningLinkSound.sound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalRunningLinkSound.sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalRunningLinkSound.complete' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalRunningLinkSound.complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalTranscriptSound.sound' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalTranscriptSound.sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalTranscriptSound.complete' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalTranscriptSound.complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCeSound.all_claims_complete' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCeSound.all_claims_complete

/-- info: 'Nightstream.Implementation.R1CS.CanonicalU64Complete.complete' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.CanonicalU64Complete.complete

/-- info: 'Nightstream.Implementation.R1CS.CanonicalU64Complete.mapped_complete' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.CanonicalU64Complete.mapped_complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursiveSourceRoleCensus.base_data_check' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeRecursiveSourceRoleCensus.base_data_check

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursiveSourceRoleCensus.recursive_data_check' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeRecursiveSourceRoleCensus.recursive_data_check

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursiveSourceRoleCensus.base_ordinaryRunSubtotal_count' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeRecursiveSourceRoleCensus.base_ordinaryRunSubtotal_count

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursiveSourceRoleCensus.recursive_ordinaryRunSubtotal_count' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeRecursiveSourceRoleCensus.recursive_ordinaryRunSubtotal_count

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursiveSourceRoleCensus.base_perField41_width_floor' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeRecursiveSourceRoleCensus.base_perField41_width_floor

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursiveSourceRoleCensus.recursive_perField41_width_floor' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeRecursiveSourceRoleCensus.recursive_perField41_width_floor

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursiveSourceRoleCensus.combined_perField41_width_floor' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeRecursiveSourceRoleCensus.combined_perField41_width_floor

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursiveSourceRoleCensus.recursive_one_million_perField41_budget_is_no_go' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeRecursiveSourceRoleCensus.recursive_one_million_perField41_budget_is_no_go
