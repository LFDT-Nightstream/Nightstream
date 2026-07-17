import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Fe.Rows
import tests.Axioms.Support

/-! Fail-closed dependency gate for the legacy terminal FE owner tree. -/

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Fe.Rows.ownerPieces_length' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Fe.Rows.ownerPieces_length

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Fe.Rows.pieceAt_mem' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Fe.Rows.pieceAt_mem
