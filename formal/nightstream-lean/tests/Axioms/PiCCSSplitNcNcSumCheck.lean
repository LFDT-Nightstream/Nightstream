import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc
import tests.Axioms.Support

/-! Fail-closed dependency gate for exact-width Split-NC SumCheck replay. -/

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.RoundMessage.toRaw_coefficients_length' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.RoundMessage.toRaw_coefficients_length

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.check_eq_true_iff_accepted' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.check_eq_true_iff_accepted

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.semanticAccepted_of_terminal_binding' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.semanticAccepted_of_terminal_binding

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.expectedRoundsRepresentable' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.expectedRoundsRepresentable

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.false_acceptance_implies_bad_challenge' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.false_acceptance_implies_bad_challenge

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.complete_of_truth' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.complete_of_truth

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.accepted_implies_truth_or_badEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.accepted_implies_truth_or_badEvent
