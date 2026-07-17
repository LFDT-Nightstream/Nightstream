import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol
import tests.Axioms.Support

/-! Fail-closed dependency gate for protocol-level Split-NC composition. -/

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.check_eq_true_iff_accepted' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.check_eq_true_iff_accepted

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.yRingBoundToSources_iff_yRing_eq' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.yRingBoundToSources_iff_yRing_eq

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.accepted_implies_paperObligations_or_unbound_or_badEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.accepted_implies_paperObligations_or_unbound_or_badEvent
