import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.BlockLane
import tests.Axioms.Support

/-! Fail-closed dependency gate for canonical block×lane NC SumCheck. -/

open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.BlockLane.semanticAccepted_of_terminal_binding' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms BlockLane.semanticAccepted_of_terminal_binding

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.BlockLane.expectedRoundsRepresentable' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms BlockLane.expectedRoundsRepresentable

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.BlockLane.accepted_rounds_length' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms BlockLane.accepted_rounds_length

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.BlockLane.complete_of_truth' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms BlockLane.complete_of_truth

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.BlockLane.false_acceptance_implies_bad_challenge' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms BlockLane.false_acceptance_implies_bad_challenge

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.BlockLane.accepted_implies_truth_or_badEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms BlockLane.accepted_implies_truth_or_badEvent
