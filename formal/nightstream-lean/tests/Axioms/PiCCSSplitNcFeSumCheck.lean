import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe
import tests.Axioms.Support

/-! Fail-closed dependency gate for mixed-width Split-NC FE SumCheck replay. -/

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.lane_evaluate_uniform' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.lane_evaluate_uniform

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.Certificate.laneRawRounds_width' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.Certificate.laneRawRounds_width

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.check_eq_true_iff_accepted' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.check_eq_true_iff_accepted

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.honestAt_implies_fixedPhaseHonest' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.honestAt_implies_fixedPhaseHonest

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.exists_honestAt' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.exists_honestAt

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.expectedRoundsRepresentable' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.expectedRoundsRepresentable

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.complete_of_truth_and_honestAt' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.complete_of_truth_and_honestAt

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.accepted_implies_truth_or_badEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.accepted_implies_truth_or_badEvent
