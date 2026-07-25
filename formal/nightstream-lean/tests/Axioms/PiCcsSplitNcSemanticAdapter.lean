import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.SemanticAdapter
import tests.Axioms.Support

open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.SemanticAdapter

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.SemanticAdapter.feAccepted_implies_genericAccepted' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms feAccepted_implies_genericAccepted

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.SemanticAdapter.feAccepted_implies_truthPath' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms feAccepted_implies_truthPath

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.SemanticAdapter.feClaimTrue_of_truth' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms feClaimTrue_of_truth

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.SemanticAdapter.ncAccepted_implies_genericAccepted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ncAccepted_implies_genericAccepted

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.SemanticAdapter.ncAccepted_implies_truthPath' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ncAccepted_implies_truthPath

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.SemanticAdapter.ncClaimTrue_of_truth' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ncClaimTrue_of_truth
