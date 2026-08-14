import Nightstream.Implementation.Nebula.Production.NIFS.PiCCS.TypedBridgeFor
import tests.Axioms.Support

/-! Dependency audit for the exponent-indexed production PiCCS bridge. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductionProductPiCcsTypedBridgeFor.rows_imply_piCcsCheck_true' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionProductPiCcsTypedBridgeFor.rows_imply_piCcsCheck_true

/-- info: 'Nightstream.Implementation.Nebula.ProductionProductPiCcsTypedBridgeFor.rows_imply_outgoingState' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionProductPiCcsTypedBridgeFor.rows_imply_outgoingState
