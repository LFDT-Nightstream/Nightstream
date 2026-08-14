import Nightstream.Implementation.Nebula.Production.Memory.CheckedStepRows
import tests.Axioms.Support

/-! Dependency audit for one production field-native checked memory step. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductionMemoryCheckedStepRows.derive' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionMemoryCheckedStepRows.derive
