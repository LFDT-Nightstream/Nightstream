import Nightstream.Implementation.NebulaV2.RecursiveManifestStateAuthority
import tests.Axioms.Support

open Nightstream.Implementation.NebulaV2

/-- info: 'Nightstream.Implementation.NebulaV2.RecursiveManifestNifsCall.Call.CarryBlocks.priorAccepted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms RecursiveManifestNifsCall.Call.CarryBlocks.priorAccepted

/-- info: 'Nightstream.Implementation.NebulaV2.RecursiveManifestNifsCall.Call.CarryBlocks.intermediateAccepted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms RecursiveManifestNifsCall.Call.CarryBlocks.intermediateAccepted

/-- info: 'Nightstream.Implementation.NebulaV2.RecursiveManifestNifsCall.Call.CarryBlocks.outgoingAccepted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms RecursiveManifestNifsCall.Call.CarryBlocks.outgoingAccepted

/-- info: 'Nightstream.Implementation.NebulaV2.RecursiveManifestSchema.Artifact.stateOutput_satisfied' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms RecursiveManifestSchema.Artifact.stateOutput_satisfied

/-- info: 'Nightstream.Implementation.NebulaV2.RecursiveManifestNifsCall.Call.outgoingStateCarryPlaced' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms RecursiveManifestNifsCall.Call.outgoingStateCarryPlaced

/-- info: 'Nightstream.Implementation.NebulaV2.RecursiveManifestStateAuthority.priorAuthority_digest_eq_columns' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms RecursiveManifestStateAuthority.priorAuthority_digest_eq_columns

/-- info: 'Nightstream.Implementation.NebulaV2.RecursiveManifestStateAuthority.boundaryFromPrevious' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms RecursiveManifestStateAuthority.boundaryFromPrevious

/-- info: 'Nightstream.Implementation.NebulaV2.RecursiveManifestStateAuthority.outgoingAuthority_digest_eq_columns' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms RecursiveManifestStateAuthority.outgoingAuthority_digest_eq_columns
