import Nightstream.Implementation.NebulaV2.ProductionMemoryBatchCcsLinkRows
import tests.Axioms.Support

/-! Dependency audit for complete production memory-to-CCS authority rows. -/

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionMemoryBatchCcsLinkRows.rows_imply_fullMatches_of_ccsPublicPlaced' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionMemoryBatchCcsLinkRows.rows_imply_fullMatches_of_ccsPublicPlaced

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionMemoryBatchCcsLinkRows.candidate_row_count_table' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionMemoryBatchCcsLinkRows.candidate_row_count_table
