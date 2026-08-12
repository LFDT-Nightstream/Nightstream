import tests.FPrimeTerminalLinkProgramRefinement
import tests.Axioms.Support

/-!
Fail-closed guards for the Rust-emitted terminal-link source program.
-/

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.Program.plain_cost' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.Program.plain_cost

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.Program.plain_expansion' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.Program.plain_expansion

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.Program.compile_plain' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.Program.compile_plain

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.Program.accepts_plain_iff' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.Program.accepts_plain_iff

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.ProgramRefinement.generated_plain_eq_canonical' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.ProgramRefinement.generated_plain_eq_canonical

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.ProgramRefinement.generated_plain_cost' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.ProgramRefinement.generated_plain_cost

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.ProgramRefinement.generated_plain_expansion' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.ProgramRefinement.generated_plain_expansion

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.ProgramRefinement.generated_plain_compile' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.ProgramRefinement.generated_plain_compile

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.ProgramRefinement.generated_plain_accepts_iff_selectedRows' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.ProgramRefinement.generated_plain_accepts_iff_selectedRows

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.ProgramRefinement.generated_batchCost_eq_rowCount' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.ProgramRefinement.generated_batchCost_eq_rowCount

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.ProgramRefinement.generated_program_exact_row_ownership' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.ProgramRefinement.generated_program_exact_row_ownership
