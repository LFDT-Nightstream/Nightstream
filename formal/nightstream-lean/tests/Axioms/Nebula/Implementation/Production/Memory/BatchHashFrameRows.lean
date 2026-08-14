import Nightstream.Implementation.Nebula.Production.Memory.BatchHashFrameRows
import tests.Axioms.Support

/-! Dependency audit for the exact production memory-batch hash frame. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductionMemoryBatchHashFrameRows.input_column_values' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionMemoryBatchHashFrameRows.input_column_values

/-- info: 'Nightstream.Implementation.Nebula.ProductionMemoryBatchHashFrameRows.inputColumns_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionMemoryBatchHashFrameRows.inputColumns_length
