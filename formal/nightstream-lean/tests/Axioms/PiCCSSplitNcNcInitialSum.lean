import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.InitialSum
import tests.Axioms.Support

/-! Fail-closed dependency gate for the independent Split-NC NC initial sum. -/

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.InitialSum.sumcheckHypercubeSum_eq_hypercubeSum' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.InitialSum.sumcheckHypercubeSum_eq_hypercubeSum

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.InitialSum.hypercubeSum_eq_mixedResidualAtBeta' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.InitialSum.hypercubeSum_eq_mixedResidualAtBeta

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.InitialSum.mixedResidualAtBeta_eq_zero_of_truth' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.InitialSum.mixedResidualAtBeta_eq_zero_of_truth

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.InitialSum.claimedInitial_eq_sumcheckHypercubeSum_of_truth' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.InitialSum.claimedInitial_eq_sumcheckHypercubeSum_of_truth
