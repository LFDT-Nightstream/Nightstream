import Nightstream.Implementation.NebulaV2.Commitment.Terminal.ProductCommitmentBridge
import tests.Axioms.Support

open Nightstream.Implementation.NebulaV2

/-- info: 'Nightstream.Implementation.NebulaV2.AlignedLaneAction.Slice.project_act' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms AlignedLaneAction.Slice.project_act

/-- info: 'Nightstream.Implementation.NebulaV2.AlignedLaneAction.Slice.project_combineAssignments' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms AlignedLaneAction.Slice.project_combineAssignments

/-- info: 'Nightstream.Implementation.NebulaV2.AlignedLaneAction.Slice.project_recomposeAssignment' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms AlignedLaneAction.Slice.project_recomposeAssignment

/-- info: 'Nightstream.Implementation.NebulaV2.ProductCommitmentAlgebra.commit_combine' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ProductCommitmentAlgebra.commit_combine

/-- info: 'Nightstream.Implementation.NebulaV2.ProductCommitmentAlgebra.commit_recompose' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms ProductCommitmentAlgebra.commit_recompose

/-- info: 'Nightstream.Implementation.NebulaV2.ProductCommitmentAlgebra.paperProfile' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ProductCommitmentAlgebra.paperProfile

/-- info: 'Nightstream.Implementation.NebulaV2.TerminalProductCommitmentBridge.commit_eq_exactBundle' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TerminalProductCommitmentBridge.commit_eq_exactBundle

/-- info: 'Nightstream.Implementation.NebulaV2.TerminalProductCommitmentBridge.product_opening_of_terminal_rows' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TerminalProductCommitmentBridge.product_opening_of_terminal_rows
