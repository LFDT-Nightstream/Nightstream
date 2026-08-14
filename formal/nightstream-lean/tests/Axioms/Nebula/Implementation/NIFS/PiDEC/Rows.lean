import Nightstream.Implementation.Nebula.NIFS.PiDEC.TypedBridge
import tests.Axioms.Support

/-! Dependency audit for the exact V2 product-PiDEC row bridge. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductPiDecRows.rows_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductPiDecRows.rows_sound

/-- info: 'Nightstream.Implementation.Nebula.ProductPiDecTypedBridge.typedEquations_of_rows' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductPiDecTypedBridge.typedEquations_of_rows

/-- info: 'Nightstream.Implementation.Nebula.ProductPiDecTypedBridge.paperAccepted_of_rows_for_attempt' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductPiDecTypedBridge.paperAccepted_of_rows_for_attempt

/-- info: 'Nightstream.Implementation.Nebula.ProductPiDecTypedBridge.paperAccepted_of_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductPiDecTypedBridge.paperAccepted_of_rows

/-- info: 'Nightstream.Implementation.Nebula.ProductPiDecTypedBridge.piDecCheck_true_of_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductPiDecTypedBridge.piDecCheck_true_of_rows
