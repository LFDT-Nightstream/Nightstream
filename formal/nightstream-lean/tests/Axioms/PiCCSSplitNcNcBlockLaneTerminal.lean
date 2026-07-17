import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.Terminal
import tests.Axioms.Support

/-! Fail-closed dependency gate for the canonical block×lane packed terminal. -/

open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.Terminal.paddedYZcol_eq_blockValueAt_of_bound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Terminal.paddedYZcol_eq_blockValueAt_of_bound

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.Terminal.valueAt_eq_sourceValueAt_of_bound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Terminal.valueAt_eq_sourceValueAt_of_bound

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.Terminal.terminal_eq_qAtPoint_of_bound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Terminal.terminal_eq_qAtPoint_of_bound
