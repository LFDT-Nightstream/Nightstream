import Nightstream.Implementation.NebulaV2.NIFS.PiDEC.TypedBridge
import tests.Axioms.Support

/-! Dependency audit for the exact V2 product-PiDEC row bridge. -/

/-- info: 'Nightstream.Implementation.NebulaV2.ProductPiDecRows.rows_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductPiDecRows.rows_sound

/-- info: 'Nightstream.Implementation.NebulaV2.ProductPiDecTypedBridge.typedEquations_of_rows' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductPiDecTypedBridge.typedEquations_of_rows

/-- info: 'Nightstream.Implementation.NebulaV2.ProductPiDecTypedBridge.paperAccepted_of_rows_for_attempt' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductPiDecTypedBridge.paperAccepted_of_rows_for_attempt

/-- info: 'Nightstream.Implementation.NebulaV2.ProductPiDecTypedBridge.paperAccepted_of_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductPiDecTypedBridge.paperAccepted_of_rows

/-- info: 'Nightstream.Implementation.NebulaV2.ProductPiDecTypedBridge.piDecCheck_true_of_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductPiDecTypedBridge.piDecCheck_true_of_rows
