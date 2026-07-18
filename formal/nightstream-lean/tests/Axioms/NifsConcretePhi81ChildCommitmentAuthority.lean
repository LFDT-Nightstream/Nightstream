import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.ChildCommitmentAuthority
import tests.Axioms.Support

/-! Fail-closed dependency gate for ordered child-commitment authority. -/

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.ChildCommitmentAuthority.children_eq_or_freshBindingCollision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.ChildCommitmentAuthority.children_eq_or_freshBindingCollision

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.ChildCommitmentAuthority.parent_children_eq_or_freshBindingCollision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.ChildCommitmentAuthority.parent_children_eq_or_freshBindingCollision
