import Nightstream.Implementation.Nebula.Production.NIFS.PiCCS.PublicBridge
import tests.Axioms.Support

/-! Dependency audit for the production-profile PiCCS public-field bridge. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductionProductPiCcsPublicBridge.rows_imply_successor_public_state' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionProductPiCcsPublicBridge.rows_imply_successor_public_state

/-- info: 'Nightstream.Implementation.Nebula.ProductionProductPiCcsPublicBridge.rows_imply_value_public_state' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionProductPiCcsPublicBridge.rows_imply_value_public_state

/-- info: 'Nightstream.Implementation.Nebula.ProductionProductPiCcsPublicBridge.no_cross_candidate_dual_placement' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionProductPiCcsPublicBridge.no_cross_candidate_dual_placement
