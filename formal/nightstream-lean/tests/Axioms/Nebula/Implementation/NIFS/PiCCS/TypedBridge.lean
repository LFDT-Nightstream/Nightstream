import Nightstream.Implementation.Nebula.NIFS.PiCCS.TypedBridge
import tests.Axioms.Support

/-! Dependency audit for the exact V2 product-PiCCS row bridge. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductPiCcsTranscriptSemantics.rows_replay_semantics' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductPiCcsTranscriptSemantics.rows_replay_semantics

/-- info: 'Nightstream.Implementation.Nebula.ProductPiCcsTranscriptSemantics.absorbPublicInput_rows_semantics' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductPiCcsTranscriptSemantics.absorbPublicInput_rows_semantics

/-- info: 'Nightstream.Implementation.Nebula.ProductPiCcsTypedBridge.rows_imply_piCcsCheck_true' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductPiCcsTypedBridge.rows_imply_piCcsCheck_true

/-- info: 'Nightstream.Implementation.Nebula.ProductPiCcsTypedBridge.rows_imply_outgoingState' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductPiCcsTypedBridge.rows_imply_outgoingState
