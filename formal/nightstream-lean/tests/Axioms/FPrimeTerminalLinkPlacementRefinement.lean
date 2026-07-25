import tests.FPrimeTerminalLinkPlacementRefinement
import tests.Axioms.Support

/-!
Fail-closed guards for the checked terminal-link program's exact current
full-history placement refinement.
-/

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.PlacementRefinement.generatedPlain_compile_eq_currentPlacement' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.PlacementRefinement.generatedPlain_compile_eq_currentPlacement

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.PlacementRefinement.generatedPlain_accepts_pulled_iff_generatedRows' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.PlacementRefinement.generatedPlain_accepts_pulled_iff_generatedRows

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.PlacementRefinement.generatedPlain_accepts_pulled_iff_loweringPriorLinkAccepted' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.PlacementRefinement.generatedPlain_accepts_pulled_iff_loweringPriorLinkAccepted
