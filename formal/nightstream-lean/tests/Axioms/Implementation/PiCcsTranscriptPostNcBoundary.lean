import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.PostNcBoundary
import tests.Axioms.Support

/-! Fail-closed kernel dependency gate for the minimal post-NC boundary. -/

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.PostNcBoundary.Bound.ofExactSchedule' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.PostNcBoundary.Bound.ofExactSchedule

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.PostNcBoundary.laneZero_irrelevant' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.PostNcBoundary.laneZero_irrelevant

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.PostNcBoundary.refines_catchupInput' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.PostNcBoundary.refines_catchupInput
