import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Fe.HonestProver
import tests.Axioms.Support

/-! Fail-closed dependency gate for sequential honest FE construction. -/

/--
info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Fe.HonestProver.exists_honest_certificate' depends on axioms: [propext,
 Classical.choice,
 Quot.sound]
-/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Fe.HonestProver.exists_honest_certificate

/--
info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Fe.HonestProver.complete_of_truth' depends on axioms: [propext,
 Classical.choice,
 Quot.sound]
-/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Fe.HonestProver.complete_of_truth
