import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript

/-! Kernel checks for the independent production-shaped PiCCS transcript schedule. -/

namespace tests.PiCcsTranscriptSchedule

open Nightstream.Implementation.R1CS.PiCcsTranscript
open Nightstream.Implementation.R1CS.PiCcsTranscript.Primitives
open Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.RawAbsorption
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine

#check Primitives.squeezeBlocks_fields_length
#check Primitives.squeezeN_two_fields_length
#check Primitives.squeezeN_two_absorbed_zero
#check Primitives.squeezeN_two_exact
#check Primitives.pairFields_extensionFields
#check Refinement.RawAbsorption.absorbElem_normalizeFull
#check Refinement.RawAbsorption.normalizedEq_refl
#check Refinement.RawAbsorption.normalizedEq_trans
#check Refinement.RawAbsorption.normalizeFull_idempotent
#check Refinement.RawAbsorption.appendRaw_eq_of_normalizedEq
#check Refinement.RawAbsorption.constant_normalizes_to_native
#check Refinement.RawAbsorption.variable_eq_native
#check Refinement.RawAbsorption.constantAppend_normalizedEq
#check Refinement.RawAbsorption.variableAppend_eq_of_normalizedEq
#check Refinement.RawAbsorption.constant_then_append_eq_native
#check Refinement.RawAbsorption.constant_then_digest_eq_native
#check Refinement.RawAbsorption.digest_eq_of_normalizedEq
#check Binding.run_eq_appendParentHandle
#check Challenges.run_gamma
#check SumCheck.runFe_shape
#check SumCheck.runNc_shape
#check SumCheck.runRound_absorbed_zero
#check SumCheck.runRounds_cons_absorbed_zero
#check Schedule.run_catchup_joint
#check Schedule.headerDigest_unique
#check Schedule.feChallengeCount
#check Schedule.ncChallengeCount
#check Schedule.replay_deterministic
#check Refinement.Terminal.Rows.ownerPieces_length
#check Refinement.Terminal.Schedule.familyLengths
#check Refinement.Terminal.Schedule.phaseTree_eq
#check Refinement.Terminal.Schedule.mainChallengeCall_formula
#check Refinement.Terminal.Schedule.betaMCall_formula
#check Refinement.Terminal.Schedule.instancePiece_mem
#check Refinement.Terminal.Schedule.bindingPiece_mem
#check Refinement.Terminal.Schedule.mainChallengePiece_mem
#check Refinement.Terminal.Schedule.betaMPiece_mem
#check Refinement.Terminal.DigestRounds.instanceDigestCallAccepted
#check Refinement.Terminal.DigestRounds.bindingCallAccepted
#check Refinement.Terminal.DigestRounds.mainChallengeCallAccepted
#check Refinement.Terminal.DigestRounds.betaMCallAccepted
#check Refinement.Terminal.DigestRounds.scheduledCallsAccepted
#check Refinement.Terminal.PinSchedule.Artifact.pinTree_eq
#check Refinement.Terminal.PinSchedule.facts
#check Refinement.Terminal.PinSchedule.Facts.runningCount
#check Refinement.Terminal.PinSchedule.Facts.mainLaterSqueeze
#check Refinement.Terminal.PinSchedule.Facts.betaMLaterSqueeze
#check Refinement.Terminal.ScheduleRefinement.headerBoundaryCallInput
#check Refinement.Terminal.ScheduleRefinement.headerPayloadCallInput
#check Refinement.Terminal.ScheduleRefinement.afterHeader_refines
#check Refinement.Terminal.ScheduleRefinement.afterInstance_refines
#check Refinement.Terminal.ScheduleRefinement.afterRunningDomain_refines
#check Refinement.Terminal.ScheduleRefinement.afterRunningCount_refines
#check Refinement.Terminal.ScheduleRefinement.bindingRun_refines

/-- A lazy constant append composes directly into the next eager variable
append without exposing the pending full-rate permutation. -/
example (state : State) (fields next : List Field) :
    gadgetVariableAppend (gadgetConstantAppend state fields) next =
      appendRaw (appendRaw state fields) next := by
  exact variableAppend_eq_of_normalizedEq
    (constantAppend_normalizedEq (normalizedEq_refl state) fields) next

/-- The same compositional relation is sufficient at a digest boundary,
including when no intervening round message exists. -/
example (state : State) (fields : List Field) :
    digest (gadgetConstantAppend state fields) =
      digest (appendRaw state fields) := by
  exact digest_eq_of_normalizedEq
    (constantAppend_normalizedEq (normalizedEq_refl state) fields)

end tests.PiCcsTranscriptSchedule
