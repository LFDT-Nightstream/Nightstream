import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcPointAudit
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcParentSemantics
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcPointSemantics
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsSelectedFePoint
import tests.Axioms.Support

set_option autoImplicit false

namespace NightstreamTests.Axioms.ConcreteNifsPiRlcPoint

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

/-! The selected FE point, physical point rows, semantic refinement, honest
completeness, receipts, and conservation may use only Lean's standard
propositional/quotient principles.  Compiler trust is forbidden. -/

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsSelectedFePoint.selectedFePoint_eq' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsSelectedFePoint.selectedFePoint_eq

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcPointRows.rows_length' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsPiRlcPointRows.rows_length

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcPointRows.cost_rows_eq' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsPiRlcPointRows.cost_rows_eq

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcPointSemantics.coordinate_eq_derived' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsPiRlcPointSemantics.coordinate_eq_derived

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcPointSemantics.point_eq_derived' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsPiRlcPointSemantics.point_eq_derived

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcPointSemantics.rows_honest' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsPiRlcPointSemantics.rows_honest

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcPointAudit.ownership_is_positional' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsPiRlcPointAudit.ownership_is_positional

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcPointAudit.rows_conservation' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsPiRlcPointAudit.rows_conservation

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcParentSemantics.parent_eq_derived' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsPiRlcParentSemantics.parent_eq_derived

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcParentSemantics.materialized_parent_eq_derived' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsPiRlcParentSemantics.materialized_parent_eq_derived

end NightstreamTests.Axioms.ConcreteNifsPiRlcPoint
