import Nightstream.Implementation.NebulaV2.Production.Artifact.VerifierArtifactFor
import tests.Axioms.Support

/-! Axiom gate for the single verifier-owned V2 artifact. -/

set_option autoImplicit false

namespace tests.Axioms.NebulaV2ProductionVerifierArtifactFor

open Nightstream.Implementation.NebulaV2.ProductionVerifierArtifactFor

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionVerifierArtifactFor.Artifact.terminalTypedProgramIncluded' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Artifact.terminalTypedProgramIncluded

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionVerifierArtifactFor.Artifact.terminalTypedProgramColumnsScoped' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Artifact.terminalTypedProgramColumnsScoped

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionVerifierArtifactFor.Artifact.selectedVerifierKeyDigest' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Artifact.selectedVerifierKeyDigest

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionVerifierArtifactFor.Artifact.selectedRelationManifestDigest' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Artifact.selectedRelationManifestDigest

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionVerifierArtifactFor.Artifact.selectedTerminalManifestDigest' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Artifact.selectedTerminalManifestDigest

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionVerifierArtifactFor.Artifact.exactAugmentedRowDomain' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Artifact.exactAugmentedRowDomain

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionVerifierArtifactFor.Artifact.selectedGeneratedBranchOfCcsPublic' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Artifact.selectedGeneratedBranchOfCcsPublic

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionVerifierArtifactFor.Artifact.generatedBranch_implies_coreBranch' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Artifact.generatedBranch_implies_coreBranch

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionVerifierArtifactFor.Artifact.terminalProgramSatisfied' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Artifact.terminalProgramSatisfied

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionVerifierArtifactFor.sameRowCountOnly_accepts_different_relations' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms sameRowCountOnly_accepts_different_relations

end tests.Axioms.NebulaV2ProductionVerifierArtifactFor
