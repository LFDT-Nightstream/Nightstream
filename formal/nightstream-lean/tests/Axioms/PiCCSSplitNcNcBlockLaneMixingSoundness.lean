import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.MixingSoundness
import tests.Axioms.Support

/-! Fail-closed dependency gate for canonical block×lane mixing soundness. -/

open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.MixingSoundness.gammaPolynomial_evaluate_eq_mixedResidualAtBeta' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms MixingSoundness.gammaPolynomial_evaluate_eq_mixedResidualAtBeta

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.MixingSoundness.sourceResidualAtBeta_eq_zero_of_all_lane_specializations_zero' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms MixingSoundness.sourceResidualAtBeta_eq_zero_of_all_lane_specializations_zero

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.MixingSoundness.mixedResidualAtBeta_eq_zero_iff_truth_or_laneSelectorRoot_or_blockSelectorRoot_or_gammaPolynomialRoot' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms MixingSoundness.mixedResidualAtBeta_eq_zero_iff_truth_or_laneSelectorRoot_or_blockSelectorRoot_or_gammaPolynomialRoot
