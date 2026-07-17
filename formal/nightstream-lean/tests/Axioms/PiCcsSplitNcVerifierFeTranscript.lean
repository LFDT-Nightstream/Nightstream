import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Fe
import tests.Axioms.Support

/-! Fail-closed dependency gate for sequential mixed-width FE replay. -/

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Fe.replay_eq_row_then_lane' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Fe.replay_eq_row_then_lane

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Fe.derive_point_coordinates' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Fe.derive_point_coordinates

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Fe.derive_coordinates_finalState' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Fe.derive_coordinates_finalState

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Fe.check_eq_true_iff_accepted' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Fe.check_eq_true_iff_accepted
