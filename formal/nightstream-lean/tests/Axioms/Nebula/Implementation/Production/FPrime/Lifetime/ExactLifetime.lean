import Nightstream.Implementation.Nebula.Production.FPrime.Lifetime.ExactLifetime
import tests.Axioms.Support

/-! Dependency audit for the exact base-to-terminal paper F-prime lifetime. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.legacyDecoder_conflicts_with_candidate' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.legacyDecoder_conflicts_with_candidate

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.GeneratedContext.statementVerifierKeySelected' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.GeneratedContext.statementVerifierKeySelected

#print axioms Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.GeneratedContext.statementIdentitySelected

#print axioms Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.GeneratedContext.baseChallengeStatementIdentitySelected
#print axioms Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.GeneratedContext.baseChallengeStatementIdSelected

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.concreteBalanced_implies_balanced' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.concreteBalanced_implies_balanced

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.ProducedBatch.after_eq_consumer' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.ProducedBatch.after_eq_consumer

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.RecursiveNode.ofAcceptedRows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.RecursiveNode.ofAcceptedRows

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.RecursiveNode.seedManifestExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.RecursiveNode.seedManifestExact

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.TerminalNode.fixedProgramSatisfied' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.TerminalNode.fixedProgramSatisfied

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.TerminalNode.commonFoldExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.TerminalNode.commonFoldExact

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.TerminalNode.compactManifestExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.TerminalNode.compactManifestExact

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.Tail.fixedRecursiveBranches_or_collision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.Tail.fixedRecursiveBranches_or_collision

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.Tail.consumerInvocationIndices_or_collision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.Tail.consumerInvocationIndices_or_collision

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.Tail.fullStateContinuity_or_collision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.Tail.fullStateContinuity_or_collision

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.BaseNode.invocationIndexSchedule_or_collision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.BaseNode.invocationIndexSchedule_or_collision

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.terminal_extract_or_collision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.terminal_extract_or_collision

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.Tail.extract_or_collision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.Tail.extract_or_collision

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.BaseNode.extract_or_collision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.BaseNode.extract_or_collision

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.BaseNode.seedManifestExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.BaseNode.seedManifestExact

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.BaseNode.challengeAuthorityExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.BaseNode.challengeAuthorityExact

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.LifetimeExtraction.rowSegmentChain' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.LifetimeExtraction.rowSegmentChain

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.LifetimeExtraction.completedExecution' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.LifetimeExtraction.completedExecution

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.LifetimeExtraction.exactCompletedRun' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.LifetimeExtraction.exactCompletedRun

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.LifetimeExtraction.exactClaimSchedule' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.LifetimeExtraction.exactClaimSchedule

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.LifetimeExtraction.fixedBranchSchedule' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.LifetimeExtraction.fixedBranchSchedule

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.LifetimeExtraction.receipts_eq_consumedReceipts' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.LifetimeExtraction.receipts_eq_consumedReceipts

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.Tail.claimSchedule_consumerInvocationIndices' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.Tail.claimSchedule_consumerInvocationIndices

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.LifetimeExtraction.consumerInvocationIndices_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.LifetimeExtraction.consumerInvocationIndices_exact

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.LifetimeExtraction.fullStateContinuityExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.LifetimeExtraction.fullStateContinuityExact

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.GeneratedContext.context_terminalProgram' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperExactLifetime.GeneratedContext.context_terminalProgram
