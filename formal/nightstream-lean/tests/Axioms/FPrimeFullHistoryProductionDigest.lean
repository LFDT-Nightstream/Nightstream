import tests.FPrimeFullHistoryProductionDigest
import tests.Axioms.Support

/-!
Fail-closed guards for the exact full-history final-state production digest
refinement.
-/

/-- info: 'Nightstream.Assurance.FPrimeFullHistoryProductionDigest.finalState_latestPublicXOuts' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.FPrimeFullHistoryProductionDigest.finalState_latestPublicXOuts

/-- info: 'Nightstream.Assurance.FPrimeFullHistoryProductionDigest.fullRows_finalState_latest_digest' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.FPrimeFullHistoryProductionDigest.fullRows_finalState_latest_digest

/-- info: 'Nightstream.Assurance.FPrimeFullHistoryProductionDigest.fullRows_finalState_latest_digest_and_logical_public' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.FPrimeFullHistoryProductionDigest.fullRows_finalState_latest_digest_and_logical_public

/-- info: 'Nightstream.Assurance.FPrimeFullHistoryProductionDigest.fullRows_construct_currentPlainOwner' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.FPrimeFullHistoryProductionDigest.fullRows_construct_currentPlainOwner

/-- info: 'Nightstream.Assurance.FPrimeFullHistoryProductionDigest.fullRows_and_currentTerminalPlacement_construct_plainOwner' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.FPrimeFullHistoryProductionDigest.fullRows_and_currentTerminalPlacement_construct_plainOwner
