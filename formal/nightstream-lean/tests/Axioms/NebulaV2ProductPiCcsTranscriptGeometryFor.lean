import Nightstream.Implementation.NebulaV2.ProductPiCcsTranscriptGeometryFor
import tests.Axioms.Support

set_option autoImplicit false

namespace tests.Axioms.NebulaV2ProductPiCcsTranscriptGeometryFor

/-! Dependency gate for the exponent-indexed PiCCS row census. -/

/-- info: 'Nightstream.Implementation.NebulaV2.ProductPiCcsTranscriptGeometryFor.rows_length_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductPiCcsTranscriptGeometryFor.rows_length_exact

end tests.Axioms.NebulaV2ProductPiCcsTranscriptGeometryFor
