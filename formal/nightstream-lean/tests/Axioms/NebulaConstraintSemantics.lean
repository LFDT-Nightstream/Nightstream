import Nightstream.Implementation.Lowering.Nebula.ConstraintSemantics
import tests.Axioms.Support

/-! Fail-closed dependency guards for exact Nebula row correspondence. -/

/-- info: 'Nightstream.Implementation.Lowering.Nebula.ConstraintSemantics.accepted_of_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Nebula.ConstraintSemantics.accepted_of_rows

/-- info: 'Nightstream.Implementation.Lowering.Nebula.ConstraintSemantics.rows_honest_of_accepted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Nebula.ConstraintSemantics.rows_honest_of_accepted

/-- info: 'Nightstream.Implementation.Lowering.Nebula.ConstraintSemantics.satisfies_iff_accepted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Nebula.ConstraintSemantics.satisfies_iff_accepted
