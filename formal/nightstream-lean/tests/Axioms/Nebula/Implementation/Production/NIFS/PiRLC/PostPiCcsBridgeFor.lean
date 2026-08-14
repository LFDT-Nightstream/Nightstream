import Nightstream.Implementation.Nebula.Production.NIFS.PiRLC.PostPiCcsBridgeFor
import tests.Axioms.Support

set_option autoImplicit false

namespace tests.Axioms.NebulaProductionProductPiRlcPostPiCcsBridgeFor

/-! Dependency gate for the exponent-indexed PiCCS-to-PiRLC bridge. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductionProductPiRlcPostPiCcsBridgeFor.rows_imply_candidate_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionProductPiRlcPostPiCcsBridgeFor.rows_imply_candidate_exact

end tests.Axioms.NebulaProductionProductPiRlcPostPiCcsBridgeFor
