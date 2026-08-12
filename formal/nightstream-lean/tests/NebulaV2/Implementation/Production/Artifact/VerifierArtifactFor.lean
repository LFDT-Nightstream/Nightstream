import Nightstream.Implementation.NebulaV2.Production.Artifact.VerifierArtifactFor

/-! Surface checks for the single verifier-owned V2 artifact. -/

set_option autoImplicit false

namespace tests.NebulaV2ProductionVerifierArtifactFor

open Nightstream.Implementation.NebulaV2.ProductionVerifierArtifactFor

#check SourceCompilerEvidence.program
#check SourceCompilerEvidence.program_rows
#check Artifact.config
#check Artifact.config_lanes
#check Artifact.config_fullKey
#check Artifact.config_operationsKey
#check Artifact.config_snapshotKey
#check Artifact.baseChallengeProgram
#check Artifact.baseChallengeRowsMatched
#check Artifact.baseChallengeStatementIdExact
#check Artifact.baseChallengeStatementIdentityExact
#check Artifact.fPrimeProgram
#check Artifact.sourceProgram
#check Artifact.relationArtifact
#check Artifact.terminalTypedNumericRows
#check Artifact.terminalTypedAssignment
#check Artifact.terminalTypedProgramIncluded
#check Artifact.terminalTypedProgramColumnsScoped
#check Artifact.relationAuthority
#check Artifact.exactAugmentedRowDomain
#check Artifact.augmentedRowsFit
#check Artifact.baseProfileSelected
#check Artifact.recursiveSeedProfileSelected
#check Artifact.selectedVerifierKeyDigest
#check Artifact.selectedRelationManifestDigest
#check Artifact.selectedTerminalManifestDigest
#check Artifact.exactDecodedBranch_iff_generated
#check Artifact.selectedGeneratedBranchOfCcsPublic
#check Artifact.generatedBranch_implies_coreBranch
#check Artifact.terminalProgramSatisfied
#check sameRowCountOnly_accepts_different_relations

end tests.NebulaV2ProductionVerifierArtifactFor
