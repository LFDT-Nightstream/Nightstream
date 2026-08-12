import Nightstream.Implementation.NebulaV2.ProductionMemoryCarryRows
import tests.Axioms.Support

/-! Dependency audit for the production field-native memory-carry decoder. -/

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionMemoryCarryRows.rows_imply_exact_carry' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionMemoryCarryRows.rows_imply_exact_carry

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionMemoryCarryRows.parsed_unique' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionMemoryCarryRows.parsed_unique

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionMemoryCarryRows.derive_unique' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionMemoryCarryRows.derive_unique
