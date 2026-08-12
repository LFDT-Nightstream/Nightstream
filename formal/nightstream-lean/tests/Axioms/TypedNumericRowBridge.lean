import tests.TypedNumericRowBridge
import tests.Axioms.Support

/-! Fail-closed axiom guards for typed-to-numeric row lowering. -/

set_option autoImplicit false

namespace NightstreamTests.Axioms.TypedNumericRowBridge

open Nightstream.Implementation.Lowering.Goldilocks.TypedNumericRowBridge
open NightstreamTests.TypedNumericRowBridge

/-- info: 'Nightstream.Implementation.Lowering.Goldilocks.TypedNumericRowBridge.residue_lcEval' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms residue_lcEval

/-- info: 'Nightstream.Implementation.Lowering.Goldilocks.TypedNumericRowBridge.numericRow_holds_iff' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms numericRow_holds_iff

/-- info: 'Nightstream.Implementation.Lowering.Goldilocks.TypedNumericRowBridge.rows_satisfied_iff' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms rows_satisfied_iff

/-- info: 'Nightstream.Implementation.Lowering.Goldilocks.TypedNumericRowBridge.typedAssignment_lift' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms typedAssignment_lift

/-- info: 'Nightstream.Implementation.Lowering.Goldilocks.TypedNumericRowBridge.exists_numeric_assignment_of_satisfies' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms exists_numeric_assignment_of_satisfies

/-- info: 'Nightstream.Implementation.Lowering.Goldilocks.TypedNumericRowBridge.SplitEmbedding.satisfies' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SplitEmbedding.satisfies

/-- info: 'Nightstream.Implementation.Lowering.Goldilocks.TypedNumericRowBridge.SplitEmbedding.complete' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SplitEmbedding.complete

/-- info: 'NightstreamTests.TypedNumericRowBridge.noninjective_allocation_has_no_exact_lift' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms noninjective_allocation_has_no_exact_lift

/-- info: 'NightstreamTests.TypedNumericRowBridge.collidingIndex_not_injective' does not depend on any axioms -/
#guard_msgs in
#audit_axioms collidingIndex_not_injective

end NightstreamTests.Axioms.TypedNumericRowBridge
