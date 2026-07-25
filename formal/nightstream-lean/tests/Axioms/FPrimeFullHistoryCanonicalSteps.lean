import tests.FPrimeFullHistoryCanonicalSteps
import tests.Axioms.Support

open Nightstream.Assurance.FPrimeFullHistoryCanonicalSteps

/-- info: 'Nightstream.Assurance.FPrimeFullHistoryCanonicalSteps.fullRows_imply_frozenSteps_or_bad' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms fullRows_imply_frozenSteps_or_bad
