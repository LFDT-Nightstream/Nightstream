import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.CanonicalParentAuthority
import tests.Axioms.Support

/-! Fail-closed dependency gate for canonical-parent authority. -/

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.CanonicalParentAuthority.parent_opening_eq_or_bindingCollision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.CanonicalParentAuthority.parent_opening_eq_or_bindingCollision

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.CanonicalParentAuthority.parent_children_eq_or_bindingCollision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.CanonicalParentAuthority.parent_children_eq_or_bindingCollision
