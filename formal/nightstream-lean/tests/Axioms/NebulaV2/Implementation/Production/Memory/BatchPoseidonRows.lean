import Nightstream.Implementation.NebulaV2.Production.Memory.BatchPoseidonRows
import tests.Axioms.Support

/-! Dependency audit for the exact production memory-batch Poseidon2 rows. -/

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionMemoryBatchPoseidonRows.output_columns_eq_digest' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionMemoryBatchPoseidonRows.output_columns_eq_digest

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionMemoryBatchPoseidonRows.candidate_row_count_table' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionMemoryBatchPoseidonRows.candidate_row_count_table
