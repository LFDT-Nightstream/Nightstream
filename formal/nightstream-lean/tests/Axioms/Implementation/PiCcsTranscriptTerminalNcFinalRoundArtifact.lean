import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.FinalRound.Artifact
import tests.Axioms.Support

/-! Fail-closed dependency gate for exact final terminal-NC artifact owners. -/

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.FinalRound.Artifact.firstMessagePiece_eq' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.FinalRound.Artifact.firstMessagePiece_eq

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.FinalRound.Artifact.finalSqueezeOutputBase_eq' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.FinalRound.Artifact.finalSqueezeOutputBase_eq

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.FinalRound.Artifact.messageLengthPins_included' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.FinalRound.Artifact.messageLengthPins_included

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.FinalRound.Artifact.finalSqueezeCallAccepted' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.FinalRound.Artifact.finalSqueezeCallAccepted

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.FinalRound.Artifact.facts' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.FinalRound.Artifact.facts
