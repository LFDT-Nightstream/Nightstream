import Nightstream.Implementation.Nebula.Production.FPrime.Lifetime.ExactLifetime

/-! Regression surface for the exact base-to-terminal paper F-prime lifetime. -/

set_option autoImplicit false

namespace tests.NebulaProductionPaperExactLifetime

open Nightstream.Implementation.Nebula.ProductionPaperExactLifetime

#check Context
#check Context.machine
#check Context.snapshotRoot
#check Context.terminalProgram_fold
#check GeneratedContext
#check GeneratedContext.machine
#check GeneratedContext.snapshotRoot
#check GeneratedContext.context
#check GeneratedContext.context_relationAuthority
#check GeneratedContext.context_fPrimeProgram
#check GeneratedContext.context_terminalProgram
#check GeneratedContext.context_config
#check GeneratedContext.context_relationArtifact
#check legacyDecoder_conflicts_with_candidate
#check GeneratedContext.statementVerifierKeySelected
#check GeneratedContext.statementIdentitySelected
#check GeneratedContext.baseChallengeStatementIdentitySelected
#check GeneratedContext.baseChallengeStatementIdSelected
#check ProducerAuthority
#check ProducedBatch
#check ProducedBatch.after_eq_consumer
#check RecursiveNode
#check RecursiveNode.ofAcceptedRows
#check RecursiveNode.seedManifestExact
#check TerminalNode
#check TerminalNode.fixedProgramSatisfied
#check TerminalNode.commonFoldExact
#check TerminalNode.compactManifestExact
#check Tail
#check Tail.recursive
#check Extraction
#check CompletionRows
#check concreteBalanced_implies_balanced
#check producerCarry_eq_consumerStart
#check TerminalNode.state_equal_or_collision
#check Tail.consumerInvocationIndices
#check Tail.consumerInvocationIndices_length
#check Tail.claimSchedule_consumerInvocationIndices
#check Tail.consumerInvocationIndices_or_collision
#check Tail.fullStateContinuity_or_collision
#check Tail.fixedRecursiveBranches_or_collision
#check terminal_extract_or_collision
#check Tail.extract_or_collision
#check BaseNode
#check BaseNode.memoryAuthority
#check BaseNode.memoryResult
#check BaseNode.seedManifestExact
#check BaseNode.authority
#check BaseNode.producerCarry_eq_active
#check BaseNode.RowInvocationIndexSchedule
#check BaseNode.invocationIndexSchedule_or_collision
#check LifetimeExtraction
#check LifetimeExtraction.claimLifetime
#check LifetimeExtraction.exactClaimSchedule
#check LifetimeExtraction.fixedBranchSchedule
#check LifetimeExtraction.receipts_eq_consumedReceipts
#check LifetimeExtraction.consumerInvocationIndices_exact
#check LifetimeExtraction.fullStateContinuityExact
#check LifetimeExtraction.rowSegmentChain
#check BaseNode.extract_or_collision
#check LifetimeExtraction.completedExecution
#check LifetimeExtraction.exactCompletedRun
#check LifetimeExtraction.completedExecutionDerived

end tests.NebulaProductionPaperExactLifetime
