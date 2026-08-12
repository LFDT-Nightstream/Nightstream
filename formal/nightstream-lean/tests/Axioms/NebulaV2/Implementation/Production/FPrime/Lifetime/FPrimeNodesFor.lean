import Nightstream.Implementation.NebulaV2.Production.FPrime.Lifetime.FPrimeNodesFor
import tests.Axioms.Support

/-! Dependency audit for exponent-indexed local F-prime nodes. -/

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactFPrimeLifetimeFor.BaseNode.baseArtifactProfileExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactFPrimeLifetimeFor.BaseNode.baseArtifactProfileExact

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactFPrimeLifetimeFor.BaseNode.memoryResult' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactFPrimeLifetimeFor.BaseNode.memoryResult

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactFPrimeLifetimeFor.BaseNode.seedManifestExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactFPrimeLifetimeFor.BaseNode.seedManifestExact

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactFPrimeLifetimeFor.BaseNode.invocationIndex_is_one' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactFPrimeLifetimeFor.BaseNode.invocationIndex_is_one

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactFPrimeLifetimeFor.BaseNode.challengeAuthorityExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactFPrimeLifetimeFor.BaseNode.challengeAuthorityExact

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactFPrimeLifetimeFor.BaseNode.freshSelectsFixedBase' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactFPrimeLifetimeFor.BaseNode.freshSelectsFixedBase

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactFPrimeLifetimeFor.RecursiveNode.ofAcceptedRows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactFPrimeLifetimeFor.RecursiveNode.ofAcceptedRows

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactFPrimeLifetimeFor.RecursiveNode.seedManifestExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactFPrimeLifetimeFor.RecursiveNode.seedManifestExact

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactFPrimeLifetimeFor.RecursiveNode.consumes_previous' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactFPrimeLifetimeFor.RecursiveNode.consumes_previous

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactFPrimeLifetimeFor.RecursiveNode.proof_is_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactFPrimeLifetimeFor.RecursiveNode.proof_is_exact

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactFPrimeLifetimeFor.RecursiveNode.accepted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactFPrimeLifetimeFor.RecursiveNode.accepted

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactFPrimeLifetimeFor.RecursiveNode.inputIterationExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactFPrimeLifetimeFor.RecursiveNode.inputIterationExact

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactFPrimeLifetimeFor.RecursiveNode.freshSelectsFixedRecursive' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactFPrimeLifetimeFor.RecursiveNode.freshSelectsFixedRecursive

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactFPrimeLifetimeFor.TerminalNode.consumes_trailing' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactFPrimeLifetimeFor.TerminalNode.consumes_trailing

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactFPrimeLifetimeFor.TerminalNode.proof_is_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactFPrimeLifetimeFor.TerminalNode.proof_is_exact

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactFPrimeLifetimeFor.TerminalNode.accepted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactFPrimeLifetimeFor.TerminalNode.accepted

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactFPrimeLifetimeFor.TerminalNode.fixedProgramSatisfied' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactFPrimeLifetimeFor.TerminalNode.fixedProgramSatisfied

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactFPrimeLifetimeFor.TerminalNode.commonFoldExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactFPrimeLifetimeFor.TerminalNode.commonFoldExact

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactFPrimeLifetimeFor.TerminalNode.compactManifestExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactFPrimeLifetimeFor.TerminalNode.compactManifestExact
