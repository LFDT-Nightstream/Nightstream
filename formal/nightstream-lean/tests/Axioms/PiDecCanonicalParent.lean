import Nightstream.SuperNeo.Folding.PiDEC.CanonicalParent
import tests.Axioms.Support

/-! Fail-closed dependency gate for computed PiDEC parent authority. -/

/-- info: 'Nightstream.SuperNeo.Folding.PiDEC.CanonicalParent.accepted_iff_compatible' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiDEC.CanonicalParent.accepted_iff_compatible

/-- info: 'Nightstream.SuperNeo.Folding.PiDEC.CanonicalParent.eq_canonical_of_accepted' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiDEC.CanonicalParent.eq_canonical_of_accepted

/-- info: 'Nightstream.SuperNeo.Folding.PiDEC.CanonicalParent.holds_of_children' depends on axioms: [Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiDEC.CanonicalParent.holds_of_children
