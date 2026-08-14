import Nightstream.Implementation.Nebula.Production.Memory.CheckedBatchRows
import tests.Axioms.Support

/-! Dependency audit for production field-native checked memory batches. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductionMemoryCheckedBatchRows.consumesList_of_indexed' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionMemoryCheckedBatchRows.consumesList_of_indexed

/-- info: 'Nightstream.Implementation.Nebula.ProductionMemoryCheckedBatchRows.Result.consumes_suffixBatch' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionMemoryCheckedBatchRows.Result.consumes_suffixBatch

/-- info: 'Nightstream.Implementation.Nebula.ProductionMemoryCheckedBatchRows.derive' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionMemoryCheckedBatchRows.derive

/-- info: 'Nightstream.Implementation.Nebula.ProductionMemoryCheckedBatchRows.rows_imply_exact_ordered_batch' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionMemoryCheckedBatchRows.rows_imply_exact_ordered_batch
