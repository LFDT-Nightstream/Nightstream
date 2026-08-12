import Nightstream.Implementation.NebulaV2.NIFS.PiCCS.TypedBridge
import tests.Axioms.Support

/-! Dependency audit for the exact V2 product-PiCCS row bridge. -/

/-- info: 'Nightstream.Implementation.NebulaV2.ProductPiCcsTranscriptSemantics.rows_replay_semantics' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductPiCcsTranscriptSemantics.rows_replay_semantics

/-- info: 'Nightstream.Implementation.NebulaV2.ProductPiCcsTranscriptSemantics.absorbPublicInput_rows_semantics' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductPiCcsTranscriptSemantics.absorbPublicInput_rows_semantics

/-- info: 'Nightstream.Implementation.NebulaV2.ProductPiCcsTypedBridge.rows_imply_piCcsCheck_true' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductPiCcsTypedBridge.rows_imply_piCcsCheck_true

/-- info: 'Nightstream.Implementation.NebulaV2.ProductPiCcsTypedBridge.rows_imply_outgoingState' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductPiCcsTypedBridge.rows_imply_outgoingState
