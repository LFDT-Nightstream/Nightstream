import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Fe
import tests.Axioms.Support

/-! Fail-closed dependency gate for the canonical Split-NC FE phase. -/

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Fe.check_eq_true_iff_accepted' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Fe.check_eq_true_iff_accepted

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Fe.accepted_of_truth_and_honestAt' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Fe.accepted_of_truth_and_honestAt

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Fe.accepted_implies_truth_or_mismatch_or_badEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Fe.accepted_implies_truth_or_mismatch_or_badEvent
