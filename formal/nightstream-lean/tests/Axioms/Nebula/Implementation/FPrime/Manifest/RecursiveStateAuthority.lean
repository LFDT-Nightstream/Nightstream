import Nightstream.Implementation.Nebula.FPrime.Manifest.RecursiveStateAuthority
import tests.Axioms.Support

open Nightstream.Implementation.Nebula

/-- info: 'Nightstream.Implementation.Nebula.RecursiveManifestNifsCall.Call.CarryBlocks.priorAccepted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms RecursiveManifestNifsCall.Call.CarryBlocks.priorAccepted

/-- info: 'Nightstream.Implementation.Nebula.RecursiveManifestNifsCall.Call.CarryBlocks.intermediateAccepted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms RecursiveManifestNifsCall.Call.CarryBlocks.intermediateAccepted

/-- info: 'Nightstream.Implementation.Nebula.RecursiveManifestNifsCall.Call.CarryBlocks.outgoingAccepted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms RecursiveManifestNifsCall.Call.CarryBlocks.outgoingAccepted

/-- info: 'Nightstream.Implementation.Nebula.RecursiveManifestSchema.Artifact.stateOutput_satisfied' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms RecursiveManifestSchema.Artifact.stateOutput_satisfied

/-- info: 'Nightstream.Implementation.Nebula.RecursiveManifestNifsCall.Call.outgoingStateCarryPlaced' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms RecursiveManifestNifsCall.Call.outgoingStateCarryPlaced

/-- info: 'Nightstream.Implementation.Nebula.RecursiveManifestStateAuthority.priorAuthority_digest_eq_columns' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms RecursiveManifestStateAuthority.priorAuthority_digest_eq_columns

/-- info: 'Nightstream.Implementation.Nebula.RecursiveManifestStateAuthority.boundaryFromPrevious' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms RecursiveManifestStateAuthority.boundaryFromPrevious

/-- info: 'Nightstream.Implementation.Nebula.RecursiveManifestStateAuthority.outgoingAuthority_digest_eq_columns' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms RecursiveManifestStateAuthority.outgoingAuthority_digest_eq_columns
