import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.InputAuthority
import tests.Axioms.Support

/-! Fail-closed dependency gate for the public-input authority bridge. -/

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.InputAuthority.relationEvaluations_eq_priorEvaluations_of_carriedTruth' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.InputAuthority.relationEvaluations_eq_priorEvaluations_of_carriedTruth

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.InputAuthority.allSourcesHold' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.InputAuthority.allSourcesHold
