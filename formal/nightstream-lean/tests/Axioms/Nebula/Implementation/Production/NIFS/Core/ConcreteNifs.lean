import Nightstream.Implementation.Nebula.Production.NIFS.Core.ConcreteNifs
import tests.Axioms.Support

/-! Dependency audit for the production-profile executable paper-NIFS key. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductionProductConcreteNifsKey.SelectedKey.publicInputState_eq' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionProductConcreteNifsKey.SelectedKey.publicInputState_eq

/-- info: 'Nightstream.Implementation.Nebula.ProductionProductConcreteNifs.selectedKey_publicInputState' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionProductConcreteNifs.selectedKey_publicInputState

/-- info: 'Nightstream.Implementation.Nebula.ProductionProductConcreteNifs.rows_imply_selectedKey_publicInputState' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionProductConcreteNifs.rows_imply_selectedKey_publicInputState
