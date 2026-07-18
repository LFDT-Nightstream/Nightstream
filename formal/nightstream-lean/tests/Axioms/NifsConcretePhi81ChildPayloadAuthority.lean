import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.ChildPayloadAuthority
import tests.Axioms.Support

/-! Fail-closed dependency gate for minimal canonical child-payload authority. -/

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.ChildPayloadAuthority.canonicalFamily_childrenOf' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.ChildPayloadAuthority.canonicalFamily_childrenOf

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.ChildPayloadAuthority.canonicalFamily_of_forOpening' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.ChildPayloadAuthority.canonicalFamily_of_forOpening

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.ChildPayloadAuthority.canonicalFamily_of_accepted' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.ChildPayloadAuthority.canonicalFamily_of_accepted

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.ChildPayloadAuthority.family_eq_of_payloadList_eq' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.ChildPayloadAuthority.family_eq_of_payloadList_eq

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.ChildPayloadAuthority.parent_children_eq_of_familyPayload_eq' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.ChildPayloadAuthority.parent_children_eq_of_familyPayload_eq
