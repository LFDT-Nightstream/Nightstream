import Nightstream.Implementation.NebulaV2.Production.Memory.CheckedStepRows
import tests.Axioms.Support

/-! Dependency audit for one production field-native checked memory step. -/

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionMemoryCheckedStepRows.derive' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionMemoryCheckedStepRows.derive
