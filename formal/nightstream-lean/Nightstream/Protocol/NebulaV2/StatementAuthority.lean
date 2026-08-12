import Nightstream.Protocol.NebulaV2.WasmStatement
import Nightstream.Protocol.NebulaV2.ProductionProfileCandidates

/-!
Contract: verifier-owned authority relation for a production V2 statement.

Assurance tier: model-level and cryptographic-reduction boundary.

Owns independent recomputation of every named statement digest, aggregate
verifier-key binding to seven manifest digests, and exact ownership of the
program, initial application state, and initial memory image.

Does not prove collision resistance, Poseidon2 conformance, byte parsing,
execution, generated-row correctness, or proof-system soundness.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.NebulaV2.StatementAuthority

open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.Profile
open Nightstream.Protocol.NebulaV2.Soundness
open Nightstream.Protocol.NebulaV2.WasmState
open Nightstream.Protocol.NebulaV2.WasmStatement

/-- Seven independently tagged manifest digests that define one verifier
key. The aggregate key digest is not one of these fields. -/
@[ext] structure ManifestDigests (Digest : Type) where
  relation : Digest
  laneLayout : Digest
  setup : Digest
  transcript : Digest
  codec : Digest
  terminal : Digest
  applicationStateSchema : Digest
deriving DecidableEq, Repr

/-- Concrete verifier-owned objects. None is supplied by a proof. -/
structure Inputs
    (RelationManifest LaneLayout SetupManifest TranscriptManifest
      CodecManifest TerminalManifest StateSchema ApplicationRelation
      Program MemoryPlan : Type) where
  relationManifest : RelationManifest
  laneLayout : LaneLayout
  setupManifest : SetupManifest
  transcriptManifest : TranscriptManifest
  codecManifest : CodecManifest
  terminalManifest : TerminalManifest
  stateSchema : StateSchema
  applicationRelation : ApplicationRelation
  program : Program
  memoryPlan : MemoryPlan
  initialApplicationState : AppStateVector

/-- Domain-specific digest functions selected by V2. Each function must be
refined to the exact Poseidon2 frame in the implementation layer. -/
structure DigestFunctions
    (RelationManifest LaneLayout SetupManifest TranscriptManifest
      CodecManifest TerminalManifest StateSchema ApplicationRelation
      Program MemoryPlan Digest : Type) where
  relationManifest : RelationManifest → Digest
  laneLayout : LaneLayout → Digest
  setupManifest : SetupManifest → Digest
  transcriptManifest : TranscriptManifest → Digest
  codecManifest : CodecManifest → Digest
  terminalManifest : TerminalManifest → Digest
  stateSchema : StateSchema → Digest
  applicationRelation : ApplicationRelation → Digest
  program : Program → Digest
  memoryPlan : MemoryPlan → Digest
  initialMemoryImage : MemoryPlan → Fin scannedCells → Nat
  verifierKey : Profile.Identity → ManifestDigests Digest → Digest

def DigestFunctions.manifestDigests
    {RelationManifest LaneLayout SetupManifest TranscriptManifest
      CodecManifest TerminalManifest StateSchema ApplicationRelation
      Program MemoryPlan Digest : Type}
    (functions :
      DigestFunctions RelationManifest LaneLayout SetupManifest
        TranscriptManifest CodecManifest TerminalManifest StateSchema
        ApplicationRelation Program MemoryPlan Digest)
    (inputs :
      Inputs RelationManifest LaneLayout SetupManifest TranscriptManifest
        CodecManifest TerminalManifest StateSchema ApplicationRelation
        Program MemoryPlan) : ManifestDigests Digest :=
  { relation := functions.relationManifest inputs.relationManifest
    laneLayout := functions.laneLayout inputs.laneLayout
    setup := functions.setupManifest inputs.setupManifest
    transcript := functions.transcriptManifest inputs.transcriptManifest
    codec := functions.codecManifest inputs.codecManifest
    terminal := functions.terminalManifest inputs.terminalManifest
    applicationStateSchema := functions.stateSchema inputs.stateSchema }

/-- Recompute one verifier-key identity for the exact selected profile. This
parameterized form is mandatory for the field-native production candidates,
whose relation and verifier key differ from the bit-serial V2 reference. -/
def DigestFunctions.expectedVerifierKeyFor
    {RelationManifest LaneLayout SetupManifest TranscriptManifest
      CodecManifest TerminalManifest StateSchema ApplicationRelation
      Program MemoryPlan Digest : Type}
    (functions :
      DigestFunctions RelationManifest LaneLayout SetupManifest
        TranscriptManifest CodecManifest TerminalManifest StateSchema
        ApplicationRelation Program MemoryPlan Digest)
    (inputs :
      Inputs RelationManifest LaneLayout SetupManifest TranscriptManifest
        CodecManifest TerminalManifest StateSchema ApplicationRelation
        Program MemoryPlan)
    (profile : Profile.Identity) : VerifierKeyIdentity Digest :=
  { digest := functions.verifierKey profile
      (functions.manifestDigests inputs)
    relationManifestDigest := functions.relationManifest inputs.relationManifest
    laneLayoutDigest := functions.laneLayout inputs.laneLayout
    setupManifestDigest := functions.setupManifest inputs.setupManifest
    transcriptManifestDigest :=
      functions.transcriptManifest inputs.transcriptManifest
    codecManifestDigest := functions.codecManifest inputs.codecManifest
    terminalManifestDigest := functions.terminalManifest inputs.terminalManifest
    applicationStateSchemaDigest := functions.stateSchema inputs.stateSchema }

/-- Recompute the complete statement identity for the selected profile. -/
def DigestFunctions.expectedIdentityFor
    {RelationManifest LaneLayout SetupManifest TranscriptManifest
      CodecManifest TerminalManifest StateSchema ApplicationRelation
      Program MemoryPlan Digest : Type}
    (functions :
      DigestFunctions RelationManifest LaneLayout SetupManifest
        TranscriptManifest CodecManifest TerminalManifest StateSchema
        ApplicationRelation Program MemoryPlan Digest)
    (inputs :
      Inputs RelationManifest LaneLayout SetupManifest TranscriptManifest
        CodecManifest TerminalManifest StateSchema ApplicationRelation
        Program MemoryPlan)
    (profile : Profile.Identity) : StatementIdentity Digest :=
  { profile := profile
    verifierKey := functions.expectedVerifierKeyFor inputs profile
    applicationRelationDigest :=
      functions.applicationRelation inputs.applicationRelation
    programDigest := functions.program inputs.program
    memoryPlanDigest := functions.memoryPlan inputs.memoryPlan }

/-- Bit-serial reference identity retained as an explicit specialization. -/
def DigestFunctions.expectedVerifierKey
    {RelationManifest LaneLayout SetupManifest TranscriptManifest
      CodecManifest TerminalManifest StateSchema ApplicationRelation
      Program MemoryPlan Digest : Type}
    (functions :
      DigestFunctions RelationManifest LaneLayout SetupManifest
        TranscriptManifest CodecManifest TerminalManifest StateSchema
        ApplicationRelation Program MemoryPlan Digest)
    (inputs :
      Inputs RelationManifest LaneLayout SetupManifest TranscriptManifest
        CodecManifest TerminalManifest StateSchema ApplicationRelation
        Program MemoryPlan) : VerifierKeyIdentity Digest :=
  functions.expectedVerifierKeyFor inputs Profile.v2

/-- Bit-serial reference statement identity. Production code must use
`expectedIdentityFor` with its candidate identity. -/
def DigestFunctions.expectedIdentity
    {RelationManifest LaneLayout SetupManifest TranscriptManifest
      CodecManifest TerminalManifest StateSchema ApplicationRelation
      Program MemoryPlan Digest : Type}
    (functions :
      DigestFunctions RelationManifest LaneLayout SetupManifest
        TranscriptManifest CodecManifest TerminalManifest StateSchema
        ApplicationRelation Program MemoryPlan Digest)
    (inputs :
      Inputs RelationManifest LaneLayout SetupManifest TranscriptManifest
        CodecManifest TerminalManifest StateSchema ApplicationRelation
        Program MemoryPlan) : StatementIdentity Digest :=
  functions.expectedIdentityFor inputs Profile.v2

/-- Exact authority opening of one typed statement. These fields contain no
accepted-proof or execution proposition. -/
structure Opens
    {RelationManifest LaneLayout SetupManifest TranscriptManifest
      CodecManifest TerminalManifest StateSchema ApplicationRelation
      Program MemoryPlan Digest : Type}
    (functions :
      DigestFunctions RelationManifest LaneLayout SetupManifest
        TranscriptManifest CodecManifest TerminalManifest StateSchema
        ApplicationRelation Program MemoryPlan Digest)
    (inputs :
      Inputs RelationManifest LaneLayout SetupManifest TranscriptManifest
        CodecManifest TerminalManifest StateSchema ApplicationRelation
        Program MemoryPlan)
    (statement : Statement Program Digest) : Prop where
  identity : statement.base.identity = functions.expectedIdentity inputs
  program : statement.base.program = inputs.program
  initialApplicationState :
    statement.base.initialApplicationState = inputs.initialApplicationState
  initialMemoryImage :
    statement.base.initialImage =
      functions.initialMemoryImage inputs.memoryPlan

/-- Exact authority opening for an explicitly selected profile. Unlike
`Opens`, this relation is suitable for every field-native production
candidate and cannot silently retain the bit-serial V2 identity. -/
structure OpensFor
    {RelationManifest LaneLayout SetupManifest TranscriptManifest
      CodecManifest TerminalManifest StateSchema ApplicationRelation
      Program MemoryPlan Digest : Type}
    (profile : Profile.Identity)
    (functions :
      DigestFunctions RelationManifest LaneLayout SetupManifest
        TranscriptManifest CodecManifest TerminalManifest StateSchema
        ApplicationRelation Program MemoryPlan Digest)
    (inputs :
      Inputs RelationManifest LaneLayout SetupManifest TranscriptManifest
        CodecManifest TerminalManifest StateSchema ApplicationRelation
        Program MemoryPlan)
    (statement : Statement Program Digest) : Prop where
  identity :
    statement.base.identity = functions.expectedIdentityFor inputs profile
  program : statement.base.program = inputs.program
  initialApplicationState :
    statement.base.initialApplicationState = inputs.initialApplicationState
  initialMemoryImage :
    statement.base.initialImage =
      functions.initialMemoryImage inputs.memoryPlan

/-- The legacy reference opening is exactly the profile-parameterized
opening at `Profile.v2`. -/
def Opens.toOpensFor
    {RelationManifest LaneLayout SetupManifest TranscriptManifest
      CodecManifest TerminalManifest StateSchema ApplicationRelation
      Program MemoryPlan Digest : Type}
    {functions :
      DigestFunctions RelationManifest LaneLayout SetupManifest
        TranscriptManifest CodecManifest TerminalManifest StateSchema
        ApplicationRelation Program MemoryPlan Digest}
    {inputs :
      Inputs RelationManifest LaneLayout SetupManifest TranscriptManifest
        CodecManifest TerminalManifest StateSchema ApplicationRelation
        Program MemoryPlan}
    {statement : Statement Program Digest}
    (opening : Opens functions inputs statement) :
    OpensFor Profile.v2 functions inputs statement where
  identity := opening.identity
  program := opening.program
  initialApplicationState := opening.initialApplicationState
  initialMemoryImage := opening.initialMemoryImage

namespace OpensFor

theorem profile_exact
    {RelationManifest LaneLayout SetupManifest TranscriptManifest
      CodecManifest TerminalManifest StateSchema ApplicationRelation
      Program MemoryPlan Digest : Type}
    {profile : Profile.Identity}
    {functions :
      DigestFunctions RelationManifest LaneLayout SetupManifest
        TranscriptManifest CodecManifest TerminalManifest StateSchema
        ApplicationRelation Program MemoryPlan Digest}
    {inputs :
      Inputs RelationManifest LaneLayout SetupManifest TranscriptManifest
        CodecManifest TerminalManifest StateSchema ApplicationRelation
        Program MemoryPlan}
    {statement : Statement Program Digest}
    (opening : OpensFor profile functions inputs statement) :
    statement.base.identity.profile = profile := by
  rw [opening.identity]
  rfl

theorem aggregate_key_is_recomputed
    {RelationManifest LaneLayout SetupManifest TranscriptManifest
      CodecManifest TerminalManifest StateSchema ApplicationRelation
      Program MemoryPlan Digest : Type}
    {profile : Profile.Identity}
    {functions :
      DigestFunctions RelationManifest LaneLayout SetupManifest
        TranscriptManifest CodecManifest TerminalManifest StateSchema
        ApplicationRelation Program MemoryPlan Digest}
    {inputs :
      Inputs RelationManifest LaneLayout SetupManifest TranscriptManifest
        CodecManifest TerminalManifest StateSchema ApplicationRelation
        Program MemoryPlan}
    {statement : Statement Program Digest}
    (opening : OpensFor profile functions inputs statement) :
    statement.base.identity.verifierKey.digest =
      functions.verifierKey profile (functions.manifestDigests inputs) := by
  rw [opening.identity]
  rfl

theorem initial_snapshot_is_verifier_owned
    {RelationManifest LaneLayout SetupManifest TranscriptManifest
      CodecManifest TerminalManifest StateSchema ApplicationRelation
      Program MemoryPlan Digest : Type}
    {profile : Profile.Identity}
    {functions :
      DigestFunctions RelationManifest LaneLayout SetupManifest
        TranscriptManifest CodecManifest TerminalManifest StateSchema
        ApplicationRelation Program MemoryPlan Digest}
    {inputs :
      Inputs RelationManifest LaneLayout SetupManifest TranscriptManifest
        CodecManifest TerminalManifest StateSchema ApplicationRelation
        Program MemoryPlan}
    {statement : Statement Program Digest}
    (opening : OpensFor profile functions inputs statement) :
    Snapshot.ofImage statement.base.initialImage =
      Snapshot.ofImage (functions.initialMemoryImage inputs.memoryPlan) := by
  rw [opening.initialMemoryImage]

theorem production_profile_exact
    {RelationManifest LaneLayout SetupManifest TranscriptManifest
      CodecManifest TerminalManifest StateSchema ApplicationRelation
      Program MemoryPlan Digest : Type}
    {candidate : ProductionProfileCandidates.Id}
    {functions :
      DigestFunctions RelationManifest LaneLayout SetupManifest
        TranscriptManifest CodecManifest TerminalManifest StateSchema
        ApplicationRelation Program MemoryPlan Digest}
    {inputs :
      Inputs RelationManifest LaneLayout SetupManifest TranscriptManifest
        CodecManifest TerminalManifest StateSchema ApplicationRelation
        Program MemoryPlan}
    {statement : Statement Program Digest}
    (opening : OpensFor (ProductionProfileCandidates.identity candidate)
      functions inputs statement) :
    statement.base.identity.profile =
      ProductionProfileCandidates.identity candidate :=
  opening.profile_exact

end OpensFor

namespace Opens

theorem profile_exact
    {RelationManifest LaneLayout SetupManifest TranscriptManifest
      CodecManifest TerminalManifest StateSchema ApplicationRelation
      Program MemoryPlan Digest : Type}
    {functions :
      DigestFunctions RelationManifest LaneLayout SetupManifest
        TranscriptManifest CodecManifest TerminalManifest StateSchema
        ApplicationRelation Program MemoryPlan Digest}
    {inputs :
      Inputs RelationManifest LaneLayout SetupManifest TranscriptManifest
        CodecManifest TerminalManifest StateSchema ApplicationRelation
        Program MemoryPlan}
    {statement : Statement Program Digest}
    (opening : Opens functions inputs statement) :
    statement.base.identity.profile = Profile.v2 := by
  rw [opening.identity]
  rfl

theorem aggregate_key_is_recomputed
    {RelationManifest LaneLayout SetupManifest TranscriptManifest
      CodecManifest TerminalManifest StateSchema ApplicationRelation
      Program MemoryPlan Digest : Type}
    {functions :
      DigestFunctions RelationManifest LaneLayout SetupManifest
        TranscriptManifest CodecManifest TerminalManifest StateSchema
        ApplicationRelation Program MemoryPlan Digest}
    {inputs :
      Inputs RelationManifest LaneLayout SetupManifest TranscriptManifest
        CodecManifest TerminalManifest StateSchema ApplicationRelation
        Program MemoryPlan}
    {statement : Statement Program Digest}
    (opening : Opens functions inputs statement) :
    statement.base.identity.verifierKey.digest =
      functions.verifierKey Profile.v2
        (functions.manifestDigests inputs) := by
  rw [opening.identity]
  rfl

theorem initial_snapshot_is_verifier_owned
    {RelationManifest LaneLayout SetupManifest TranscriptManifest
      CodecManifest TerminalManifest StateSchema ApplicationRelation
      Program MemoryPlan Digest : Type}
    {functions :
      DigestFunctions RelationManifest LaneLayout SetupManifest
        TranscriptManifest CodecManifest TerminalManifest StateSchema
        ApplicationRelation Program MemoryPlan Digest}
    {inputs :
      Inputs RelationManifest LaneLayout SetupManifest TranscriptManifest
        CodecManifest TerminalManifest StateSchema ApplicationRelation
        Program MemoryPlan}
    {statement : Statement Program Digest}
    (opening : Opens functions inputs statement) :
    Snapshot.ofImage statement.base.initialImage =
      Snapshot.ofImage (functions.initialMemoryImage inputs.memoryPlan) := by
  rw [opening.initialMemoryImage]

end Opens

end Nightstream.Protocol.NebulaV2.StatementAuthority
