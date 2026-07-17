import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.Mixing
import tests.Axioms.Support

/-! Fail-closed dependency gate for canonical block×lane NC mixing. -/

open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.Mixing.polynomial_coordinates_eq_qAtPoint' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Mixing.polynomial_coordinates_eq_qAtPoint
