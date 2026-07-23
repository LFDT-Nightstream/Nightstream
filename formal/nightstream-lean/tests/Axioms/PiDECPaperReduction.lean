import Nightstream.SuperNeo.Folding.PiDEC.PaperReduction
import tests.Axioms.Support

/-! Fail-closed trusted-dependency gate for the paper-exact `Pi_DEC` reduction. -/

/-- info: 'Nightstream.SuperNeo.Folding.PiDEC.PaperReduction.success_implies_extractedSource' depends on axioms: [Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiDEC.PaperReduction.success_implies_extractedSource

/-- info: 'Nightstream.SuperNeo.Folding.PiDEC.PaperReduction.perfectComplete' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiDEC.PaperReduction.perfectComplete

/-- info: 'Nightstream.SuperNeo.Folding.PiDEC.PaperReduction.reductionOfKnowledge' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiDEC.PaperReduction.reductionOfKnowledge
