import Nightstream.Implementation.NebulaV2.ProductPiRlcParentBridge
import tests.Axioms.Support

/-! Dependency audit for the exact row-derived V2 PiRLC parent. -/

/-- info: 'Nightstream.Implementation.NebulaV2.ProductPiRlcParentBridge.combineEvaluations_singletons' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductPiRlcParentBridge.combineEvaluations_singletons

/-- info: 'Nightstream.Implementation.NebulaV2.ProductPiRlcParentBridge.challenges_eq_selected' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductPiRlcParentBridge.challenges_eq_selected

/-- info: 'Nightstream.Implementation.NebulaV2.ProductPiRlcParentBridge.parentFields_of_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductPiRlcParentBridge.parentFields_of_rows
