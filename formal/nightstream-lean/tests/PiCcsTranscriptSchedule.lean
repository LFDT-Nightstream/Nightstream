import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Schedule
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.RawAbsorption
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.DigestRounds
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.PinSchedule
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.ScheduleRefinement

/-! Kernel checks for the independent production-shaped PiCCS transcript schedule. -/

namespace tests.PiCcsTranscriptSchedule

open Nightstream.Implementation.R1CS.PiCcsTranscript

#check Primitives.squeezeBlocks_fields_length
#check Primitives.pairFields_extensionFields
#check Refinement.RawAbsorption.absorbElem_normalizeFull
#check Refinement.RawAbsorption.constant_normalizes_to_native
#check Refinement.RawAbsorption.variable_eq_native
#check Refinement.RawAbsorption.constant_then_append_eq_native
#check Refinement.RawAbsorption.constant_then_digest_eq_native
#check Binding.run_eq_appendParentHandle
#check Challenges.run_gamma
#check SumCheck.runFe_shape
#check SumCheck.runNc_shape
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

end tests.PiCcsTranscriptSchedule
