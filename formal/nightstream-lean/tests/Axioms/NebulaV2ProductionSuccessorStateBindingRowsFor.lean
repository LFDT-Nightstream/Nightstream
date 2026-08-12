import Nightstream.Implementation.NebulaV2.ProductionSuccessorStateBindingRowsFor
import tests.Axioms.Support

/-! Dependency audit for the exponent-indexed F-prime successor rows. -/

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionSuccessorStateBindingRowsFor.successorPermutationCount_25' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionSuccessorStateBindingRowsFor.successorPermutationCount_25

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionSuccessorStateBindingRowsFor.rows_length_exact' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionSuccessorStateBindingRowsFor.rows_length_exact

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionSuccessorStateBindingRowsFor.rows_imply_outputState' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionSuccessorStateBindingRowsFor.rows_imply_outputState

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionSuccessorStateBindingRowsFor.rows_imply_outputDigest_lane' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionSuccessorStateBindingRowsFor.rows_imply_outputDigest_lane
