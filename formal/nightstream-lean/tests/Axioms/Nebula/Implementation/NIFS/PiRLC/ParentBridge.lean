import Nightstream.Implementation.Nebula.NIFS.PiRLC.ParentBridge
import tests.Axioms.Support

/-! Dependency audit for the exact row-derived V2 PiRLC parent. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductPiRlcParentBridge.combineEvaluations_singletons' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductPiRlcParentBridge.combineEvaluations_singletons

/-- info: 'Nightstream.Implementation.Nebula.ProductPiRlcParentBridge.challenges_eq_selected' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductPiRlcParentBridge.challenges_eq_selected

/-- info: 'Nightstream.Implementation.Nebula.ProductPiRlcParentBridge.parentFields_of_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductPiRlcParentBridge.parentFields_of_rows
