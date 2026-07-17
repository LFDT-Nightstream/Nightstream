import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.HonestProver
import tests.Axioms.Support

/-! Fail-closed dependency gate for canonical FE→block×lane-NC construction. -/

/--
info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.HonestProver.canonicalOutput_bound' depends on axioms: [propext,
 Quot.sound]
-/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.HonestProver.canonicalOutput_bound

/--
info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.HonestProver.complete_of_paperObligations' depends on axioms: [propext,
 Classical.choice,
 Quot.sound]
-/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.HonestProver.complete_of_paperObligations
