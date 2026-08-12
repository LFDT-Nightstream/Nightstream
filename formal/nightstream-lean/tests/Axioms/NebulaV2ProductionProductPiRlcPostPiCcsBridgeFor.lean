import Nightstream.Implementation.NebulaV2.ProductionProductPiRlcPostPiCcsBridgeFor
import tests.Axioms.Support

set_option autoImplicit false

namespace tests.Axioms.NebulaV2ProductionProductPiRlcPostPiCcsBridgeFor

/-! Dependency gate for the exponent-indexed PiCCS-to-PiRLC bridge. -/

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionProductPiRlcPostPiCcsBridgeFor.rows_imply_candidate_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionProductPiRlcPostPiCcsBridgeFor.rows_imply_candidate_exact

end tests.Axioms.NebulaV2ProductionProductPiRlcPostPiCcsBridgeFor
