import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.OutputRefinement
import tests.Axioms.Support

/-! Fail-closed dependency gate for the canonical block×lane Π_CCS-to-CE bridge. -/

/--
info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.OutputRefinement.accepted_and_outputBound_implies_outputsHold_or_badEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound]
-/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.OutputRefinement.accepted_and_outputBound_implies_outputsHold_or_badEvent

/--
info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.OutputRefinement.complete_of_paperObligations' depends on axioms: [propext,
 Classical.choice,
 Quot.sound]
-/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.OutputRefinement.complete_of_paperObligations
