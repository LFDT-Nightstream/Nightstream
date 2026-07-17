import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc.BlockLane
import tests.Axioms.Support

/-! Fail-closed dependency gate for canonical block×lane NC transcript replay. -/

open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc.BlockLane.replay_eq_block_then_lane' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms BlockLane.replay_eq_block_then_lane

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc.BlockLane.derive_coordinates_finalState' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms BlockLane.derive_coordinates_finalState

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc.BlockLane.check_eq_true_iff_accepted' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms BlockLane.check_eq_true_iff_accepted
