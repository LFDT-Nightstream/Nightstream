import Nightstream.SuperNeo.Folding.PiRLC.PaperCompleteness
import tests.Axioms.Support

/-! Fail-closed trusted-dependency gate for deterministic paper `Pi_RLC`. -/

/-- info: 'Nightstream.SuperNeo.Folding.PiRLC.PaperCompleteness.perfectComplete' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiRLC.PaperCompleteness.perfectComplete

/-- info: 'Nightstream.SuperNeo.Folding.PiRLC.PaperCompleteness.canonicalPerfectComplete' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiRLC.PaperCompleteness.canonicalPerfectComplete

/-- info: 'Nightstream.SuperNeo.Folding.PiRLC.PaperCompleteness.publicCoin' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiRLC.PaperCompleteness.publicCoin
