import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOutputAudit
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOutputSemantics
import tests.Axioms.Support

set_option autoImplicit false

namespace NightstreamTests.Axioms.ConcreteNifsOutput

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

/-! The selected NIFS output rows, exact result refinement, honest
completeness, positional receipts, and conservation may use only Lean's
standard propositional/choice/quotient principles. Compiler trust is
forbidden. -/

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOutputRows.rows_length' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsOutputRows.rows_length

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOutputRows.cost_rows_eq' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsOutputRows.cost_rows_eq

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOutputRows.cost_columns_eq' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsOutputRows.cost_columns_eq

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOutputSemantics.child_equations_of_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsOutputSemantics.child_equations_of_rows

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOutputSemantics.rows_honest' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsOutputSemantics.rows_honest

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOutputSemantics.output_eq_selectedResult_of_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsOutputSemantics.output_eq_selectedResult_of_rows

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOutputAudit.ownership_is_positional' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsOutputAudit.ownership_is_positional

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOutputAudit.rows_conservation' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsOutputAudit.rows_conservation

end NightstreamTests.Axioms.ConcreteNifsOutput
