import Nightstream.Implementation.Nebula.Production.FPrime.Lifetime.FPrimeNodesFor

/-! Regression surface for exponent-indexed local F-prime nodes. -/

set_option autoImplicit false

namespace tests.NebulaProductionPaperFPrimeNodesFor

open Nightstream.Implementation.Nebula.ProductionPaperExactFPrimeLifetimeFor

#check BaseNode.baseArtifactProfileExact

#check Context
#check Context.terminalProgram_fold
#check BaseNode
#check BaseNode.memoryAuthority
#check BaseNode.memoryResult
#check BaseNode.seedManifestExact
#check BaseNode.invocationIndex_is_one
#check BaseNode.challengeAuthorityExact
#check BaseNode.freshSelectsFixedBase
#check RecursiveNode
#check RecursiveNode.ofAcceptedRows
#check RecursiveNode.seedManifestExact
#check RecursiveNode.consumes_previous
#check RecursiveNode.proof_is_exact
#check RecursiveNode.accepted
#check RecursiveNode.inputIterationExact
#check RecursiveNode.freshSelectsFixedRecursive
#check TerminalNode
#check TerminalNode.consumes_trailing
#check TerminalNode.proof_is_exact
#check TerminalNode.accepted
#check TerminalNode.fixedProgramSatisfied
#check TerminalNode.commonFoldExact
#check TerminalNode.compactManifestExact

end tests.NebulaProductionPaperFPrimeNodesFor
