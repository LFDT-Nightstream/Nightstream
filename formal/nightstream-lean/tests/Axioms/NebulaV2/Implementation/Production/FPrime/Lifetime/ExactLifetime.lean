import Nightstream.Implementation.NebulaV2.Production.FPrime.Lifetime.ExactLifetime
import tests.Axioms.Support

/-! Dependency audit for the exact base-to-terminal paper F-prime lifetime. -/

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.legacyDecoder_conflicts_with_candidate' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.legacyDecoder_conflicts_with_candidate

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.GeneratedContext.statementVerifierKeySelected' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.GeneratedContext.statementVerifierKeySelected

#print axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.GeneratedContext.statementIdentitySelected

#print axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.GeneratedContext.baseChallengeStatementIdentitySelected
#print axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.GeneratedContext.baseChallengeStatementIdSelected

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.concreteBalanced_implies_balanced' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.concreteBalanced_implies_balanced

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.ProducedBatch.after_eq_consumer' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.ProducedBatch.after_eq_consumer

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.RecursiveNode.ofAcceptedRows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.RecursiveNode.ofAcceptedRows

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.RecursiveNode.seedManifestExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.RecursiveNode.seedManifestExact

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.TerminalNode.fixedProgramSatisfied' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.TerminalNode.fixedProgramSatisfied

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.TerminalNode.commonFoldExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.TerminalNode.commonFoldExact

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.TerminalNode.compactManifestExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.TerminalNode.compactManifestExact

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.Tail.fixedRecursiveBranches_or_collision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.Tail.fixedRecursiveBranches_or_collision

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.Tail.consumerInvocationIndices_or_collision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.Tail.consumerInvocationIndices_or_collision

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.Tail.fullStateContinuity_or_collision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.Tail.fullStateContinuity_or_collision

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.BaseNode.invocationIndexSchedule_or_collision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.BaseNode.invocationIndexSchedule_or_collision

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.terminal_extract_or_collision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.terminal_extract_or_collision

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.Tail.extract_or_collision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.Tail.extract_or_collision

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.BaseNode.extract_or_collision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.BaseNode.extract_or_collision

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.BaseNode.seedManifestExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.BaseNode.seedManifestExact

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.BaseNode.challengeAuthorityExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.BaseNode.challengeAuthorityExact

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.LifetimeExtraction.rowSegmentChain' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.LifetimeExtraction.rowSegmentChain

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.LifetimeExtraction.completedExecution' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.LifetimeExtraction.completedExecution

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.LifetimeExtraction.exactCompletedRun' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.LifetimeExtraction.exactCompletedRun

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.LifetimeExtraction.exactClaimSchedule' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.LifetimeExtraction.exactClaimSchedule

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.LifetimeExtraction.fixedBranchSchedule' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.LifetimeExtraction.fixedBranchSchedule

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.LifetimeExtraction.receipts_eq_consumedReceipts' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.LifetimeExtraction.receipts_eq_consumedReceipts

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.Tail.claimSchedule_consumerInvocationIndices' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.Tail.claimSchedule_consumerInvocationIndices

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.LifetimeExtraction.consumerInvocationIndices_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.LifetimeExtraction.consumerInvocationIndices_exact

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.LifetimeExtraction.fullStateContinuityExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.LifetimeExtraction.fullStateContinuityExact

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.GeneratedContext.context_terminalProgram' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime.GeneratedContext.context_terminalProgram
