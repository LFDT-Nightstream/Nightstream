import Nightstream.Implementation.Nebula.NIFS.PiCCS.TranscriptCursorFor
import tests.Axioms.Support

set_option autoImplicit false

namespace tests.Axioms.NebulaProductPiCcsTranscriptCursorFor

/-! Dependency gate for the exponent-indexed PiCCS transcript cursor. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductPiCcsTranscriptCursorFor.afterFullOutput_absorbed' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductPiCcsTranscriptCursorFor.afterFullOutput_absorbed

end tests.Axioms.NebulaProductPiCcsTranscriptCursorFor
