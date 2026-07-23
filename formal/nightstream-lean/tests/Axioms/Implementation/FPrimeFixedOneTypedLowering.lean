import Nightstream.Implementation.Lowering.FPrimeFixedOne
import tests.Axioms.Support

/-!
Fail-closed kernel-dependency guard for obligation 9: the artifact-independent
typed lowering semantics, complete structural receipt, definitional cost, and
fixed-one step/terminal refinements.
-/

/-- info: 'Nightstream.Implementation.Lowering.Typed.Primitive.exec_eq_some_iff_holds' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Typed.Primitive.exec_eq_some_iff_holds

/-- info: 'Nightstream.Implementation.Lowering.Typed.Block.exec_eq_some_iff_holds' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Typed.Block.exec_eq_some_iff_holds

/-- info: 'Nightstream.Implementation.Lowering.Typed.Program.exec_eq_some_iff_holds' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Typed.Program.exec_eq_some_iff_holds

/-- info: 'Nightstream.Implementation.Lowering.Typed.Program.flattened_conservation' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Typed.Program.flattened_conservation

/-- info: 'Nightstream.Implementation.Lowering.Typed.Program.cost_eq_receipt_event_cost' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Typed.Program.cost_eq_receipt_event_cost

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.accepts_iff_transition' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.accepts_iff_transition

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.accepts_iff_transition' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.accepts_iff_transition
