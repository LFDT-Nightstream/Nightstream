import tests.FPrimeTerminalLinkPhysicalRowsRefinement
import tests.Axioms.Support

/-!
Fail-closed guards for the bounded production terminal-link row refinement.
-/

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.PhysicalRowsRefinement.generated_batchSize_eq' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.PhysicalRowsRefinement.generated_batchSize_eq

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.PhysicalRowsRefinement.generated_rowCount_eq' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.PhysicalRowsRefinement.generated_rowCount_eq

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.PhysicalRowsRefinement.generated_columnCount_eq' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.PhysicalRowsRefinement.generated_columnCount_eq

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.PhysicalRowsRefinement.generated_rows_eq_selected' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.PhysicalRowsRefinement.generated_rows_eq_selected

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.PhysicalRowsRefinement.generated_rows_eq_compiler_output' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.PhysicalRowsRefinement.generated_rows_eq_compiler_output

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.PhysicalRowsRefinement.generated_rows_exact_receipt_ownership' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.PhysicalRowsRefinement.generated_rows_exact_receipt_ownership
