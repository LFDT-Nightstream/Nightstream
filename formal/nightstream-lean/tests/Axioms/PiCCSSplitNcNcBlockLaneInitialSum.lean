import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.InitialSum
import tests.Axioms.Support

/-! Fail-closed dependency gate for the canonical block×lane initial sum. -/

open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.InitialSum.hypercubeSum_eq_mixedResidualAtBeta' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms InitialSum.hypercubeSum_eq_mixedResidualAtBeta

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.InitialSum.sumcheckHypercubeSum_eq_hypercubeSum' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms InitialSum.sumcheckHypercubeSum_eq_hypercubeSum

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.InitialSum.sourceResidualAtBeta_eq_zero_of_truth' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms InitialSum.sourceResidualAtBeta_eq_zero_of_truth

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.InitialSum.claimedInitial_eq_hypercubeSum_of_truth' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms InitialSum.claimedInitial_eq_hypercubeSum_of_truth

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.InitialSum.claimedInitial_eq_sumcheckHypercubeSum_of_truth' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms InitialSum.claimedInitial_eq_sumcheckHypercubeSum_of_truth
