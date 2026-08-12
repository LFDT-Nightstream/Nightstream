import Nightstream.Implementation.NebulaV2.ProductionProductPiCcsPublicBridge
import tests.Axioms.Support

/-! Dependency audit for the production-profile PiCCS public-field bridge. -/

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionProductPiCcsPublicBridge.rows_imply_successor_public_state' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionProductPiCcsPublicBridge.rows_imply_successor_public_state

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionProductPiCcsPublicBridge.rows_imply_value_public_state' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionProductPiCcsPublicBridge.rows_imply_value_public_state

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionProductPiCcsPublicBridge.no_cross_candidate_dual_placement' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionProductPiCcsPublicBridge.no_cross_candidate_dual_placement
