import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Nc
import tests.Axioms.Support

/-! Fail-closed dependency gate for the canonical Split-NC phase. -/

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Nc.check_eq_true_iff_accepted' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Nc.check_eq_true_iff_accepted

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Nc.accepted_implies_truth_or_unbound_or_badEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Nc.accepted_implies_truth_or_unbound_or_badEvent
