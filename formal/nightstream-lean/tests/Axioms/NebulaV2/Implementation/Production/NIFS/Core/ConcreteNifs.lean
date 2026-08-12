import Nightstream.Implementation.NebulaV2.Production.NIFS.Core.ConcreteNifs
import tests.Axioms.Support

/-! Dependency audit for the production-profile executable paper-NIFS key. -/

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionProductConcreteNifsKey.SelectedKey.publicInputState_eq' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionProductConcreteNifsKey.SelectedKey.publicInputState_eq

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionProductConcreteNifs.selectedKey_publicInputState' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionProductConcreteNifs.selectedKey_publicInputState

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionProductConcreteNifs.rows_imply_selectedKey_publicInputState' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionProductConcreteNifs.rows_imply_selectedKey_publicInputState
