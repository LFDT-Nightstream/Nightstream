import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.HonestProver
import tests.Axioms.Support

/-! Fail-closed dependency gate for honest Split-NC protocol composition. -/

/--
info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.HonestProver.complete_of_paperObligations' depends on axioms: [propext,
 Classical.choice,
 Quot.sound]
-/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.HonestProver.complete_of_paperObligations
