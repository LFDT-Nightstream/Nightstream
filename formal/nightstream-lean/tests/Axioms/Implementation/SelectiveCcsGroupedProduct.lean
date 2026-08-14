import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Rewrite.GroupedProduct
import tests.Axioms.Support

/-! Fail-closed dependency gate for grouped-product rewrite algebra. -/

/-- info: 'Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.GroupedProduct.chainHolds_sound' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.GroupedProduct.chainHolds_sound

/-- info: 'Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.GroupedProduct.compile_chainHolds' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.GroupedProduct.compile_chainHolds

/-- info: 'Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.GroupedProduct.evaluationRow_residual' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.GroupedProduct.evaluationRow_residual

/-- info: 'Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.GroupedProduct.evaluationRow_zero_iff_stepHolds' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.GroupedProduct.evaluationRow_zero_iff_stepHolds

/-- info: 'Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.GroupedProduct.evaluationPoint_zero_iff_fiveProduct' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.GroupedProduct.evaluationPoint_zero_iff_fiveProduct
