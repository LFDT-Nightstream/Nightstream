import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.OutputAuthority.Nc
import tests.Axioms.Support

/-! Fail-closed dependency gate for Split-NC output authority. -/

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.OutputAuthority.Nc.acceptedFromMessage_implies_truth_or_unbound_or_badEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.OutputAuthority.Nc.acceptedFromMessage_implies_truth_or_unbound_or_badEvent
