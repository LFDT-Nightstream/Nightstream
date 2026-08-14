import Nightstream.Implementation.Nebula.Production.Memory.ClaimRows
import tests.Axioms.Support

/-! Dependency audit for the production field-native memory-claim decoder. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductionMemoryClaimRows.rows_imply_exact_claim' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionMemoryClaimRows.rows_imply_exact_claim

/-- info: 'Nightstream.Implementation.Nebula.ProductionMemoryClaimRows.parsed_unique' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionMemoryClaimRows.parsed_unique

/-- info: 'Nightstream.Implementation.Nebula.ProductionMemoryClaimRows.derive_unique' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionMemoryClaimRows.derive_unique
