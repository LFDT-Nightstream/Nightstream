import Nightstream.Implementation.NebulaV2.NIFS.PiCCS.TranscriptCursorFor
import tests.Axioms.Support

set_option autoImplicit false

namespace tests.Axioms.NebulaV2ProductPiCcsTranscriptCursorFor

/-! Dependency gate for the exponent-indexed PiCCS transcript cursor. -/

/-- info: 'Nightstream.Implementation.NebulaV2.ProductPiCcsTranscriptCursorFor.afterFullOutput_absorbed' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductPiCcsTranscriptCursorFor.afterFullOutput_absorbed

end tests.Axioms.NebulaV2ProductPiCcsTranscriptCursorFor
