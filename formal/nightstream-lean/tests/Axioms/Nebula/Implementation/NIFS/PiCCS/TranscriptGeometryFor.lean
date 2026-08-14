import Nightstream.Implementation.Nebula.NIFS.PiCCS.TranscriptGeometryFor
import tests.Axioms.Support

set_option autoImplicit false

namespace tests.Axioms.NebulaProductPiCcsTranscriptGeometryFor

/-! Dependency gate for the exponent-indexed PiCCS row census. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductPiCcsTranscriptGeometryFor.rows_length_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductPiCcsTranscriptGeometryFor.rows_length_exact

end tests.Axioms.NebulaProductPiCcsTranscriptGeometryFor
