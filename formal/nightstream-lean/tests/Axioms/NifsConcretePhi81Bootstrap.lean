import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedBootstrap
import tests.Axioms.Support

/-! Fail-closed dependency gate for the independent zero-running bootstrap. -/

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedBootstrap.arity_total' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedBootstrap.arity_total

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedBootstrap.runningAuthority_iff_parentAbsent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedBootstrap.runningAuthority_iff_parentAbsent

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedBootstrap.ResultTransition.parentAbsent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedBootstrap.ResultTransition.parentAbsent

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedBootstrap.ResultTransition.children_transition' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedBootstrap.ResultTransition.children_transition

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedBootstrap.ResultTransition.childStructure_eq_fresh' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedBootstrap.ResultTransition.childStructure_eq_fresh
