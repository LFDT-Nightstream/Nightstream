import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc
import tests.Axioms.Support

/-! Fail-closed dependency gate for sequential exact-width NC replay. -/

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc.runRoundsFrom_append' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc.runRoundsFrom_append

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc.check_eq_true_iff_accepted' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc.check_eq_true_iff_accepted
