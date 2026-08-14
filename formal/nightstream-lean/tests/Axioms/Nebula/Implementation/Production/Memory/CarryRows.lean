import Nightstream.Implementation.Nebula.Production.Memory.CarryRows
import tests.Axioms.Support

/-! Dependency audit for the production field-native memory-carry decoder. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductionMemoryCarryRows.rows_imply_exact_carry' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionMemoryCarryRows.rows_imply_exact_carry

/-- info: 'Nightstream.Implementation.Nebula.ProductionMemoryCarryRows.parsed_unique' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionMemoryCarryRows.parsed_unique

/-- info: 'Nightstream.Implementation.Nebula.ProductionMemoryCarryRows.derive_unique' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionMemoryCarryRows.derive_unique
