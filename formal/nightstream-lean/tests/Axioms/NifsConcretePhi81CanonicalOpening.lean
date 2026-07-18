import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.CanonicalOpening
import tests.Axioms.Support

/-! Fail-closed dependency gate for the opening-derived NIFS carrier. -/

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.CanonicalOpening.Context.full_running' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.CanonicalOpening.Context.full_running

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.CanonicalOpening.OpeningPayload.children_ofCanonical' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.CanonicalOpening.OpeningPayload.children_ofCanonical

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.CanonicalOpening.resultCarrier_complete' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.CanonicalOpening.resultCarrier_complete

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.CanonicalOpening.runningAuthority' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.CanonicalOpening.runningAuthority
