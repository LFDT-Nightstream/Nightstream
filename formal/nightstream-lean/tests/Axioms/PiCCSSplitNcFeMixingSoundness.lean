import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.MixingSoundness
import tests.Axioms.Support

/-! Fail-closed dependency gate for deterministic Split-NC FE compression soundness. -/

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.MixingSoundness.mixedResidual_eq_zero_iff_truth_or_mixingRoot' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.MixingSoundness.mixedResidual_eq_zero_iff_truth_or_mixingRoot
