import tests.FPrimeTerminalLinkLoweringRefinement
import tests.Axioms.Support

/-!
Fail-closed guard for the generated fused terminal-link program's exact typed
Terminal prior-link refinement.
-/

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.LoweringRefinement.generatedPlain_accepts_iff_priorLinkAccepted' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.LoweringRefinement.generatedPlain_accepts_iff_priorLinkAccepted
