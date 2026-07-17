import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Nc.BlockLane
import tests.Axioms.Support

/-! Fail-closed dependency gate for the canonical block×lane NC phase. -/

open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Nc

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Nc.BlockLane.check_eq_true_iff_accepted' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms BlockLane.check_eq_true_iff_accepted

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Nc.BlockLane.accepted_implies_truth_or_unbound_or_badEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms BlockLane.accepted_implies_truth_or_unbound_or_badEvent
