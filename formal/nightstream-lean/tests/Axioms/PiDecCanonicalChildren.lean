import Nightstream.SuperNeo.Folding.PiDEC.CanonicalChildren
import tests.Axioms.Support

/-! Fail-closed dependency gate for canonical PiDEC child authority. -/

/-- info: 'Nightstream.SuperNeo.Folding.PiDEC.CanonicalChildren.ForOpening.complete' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiDEC.CanonicalChildren.ForOpening.complete

/-- info: 'Nightstream.SuperNeo.Folding.PiDEC.CanonicalChildren.children_eq_or_bindingCollision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiDEC.CanonicalChildren.children_eq_or_bindingCollision

/-- info: 'Nightstream.SuperNeo.Folding.PiDEC.CanonicalChildren.children_eq_of_no_bindingCollision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiDEC.CanonicalChildren.children_eq_of_no_bindingCollision
