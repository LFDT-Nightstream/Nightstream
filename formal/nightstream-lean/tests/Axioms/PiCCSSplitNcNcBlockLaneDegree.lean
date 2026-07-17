import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.Degree
import tests.Axioms.Support

/-! Fail-closed dependency gate for canonical block×lane NC degree proofs. -/

open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.Degree.Source.sourceValueAt_block_affine' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Degree.Source.sourceValueAt_block_affine

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.Degree.Source.rangeValueAt_lane_cubic' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Degree.Source.rangeValueAt_lane_cubic

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.Degree.Polynomial.qAtPoint_block_quartic' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Degree.Polynomial.qAtPoint_block_quartic

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.Degree.Polynomial.qAtPoint_lane_has_five_coefficients' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Degree.Polynomial.qAtPoint_lane_has_five_coefficients

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.Degree.SumCheck.sumcheckPolynomial_slice_quartic' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Degree.SumCheck.sumcheckPolynomial_slice_quartic

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.Degree.SumCheck.expectedRound_has_five_coefficients' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Degree.SumCheck.expectedRound_has_five_coefficients
