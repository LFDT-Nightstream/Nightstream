import Nightstream.Protocol.Nebula.ApplicationTrace
import Nightstream.Protocol.Nebula.Profile

/-!
Contract: top-level theorem shape for PaddedRowIdentityMemoryV2.

Assurance tier: model-level and security-reduction boundary.

Owns the independent operational memory plus completed-application execution
conclusion and the list of named bad events used by staged assurance layers.

Does not prove any cryptographic bad-event bound or any Rust, generated-row,
codec, terminal-backend, or deployed-verifier refinement.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.Nebula.Soundness

open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.ApplicationTrace
open Nightstream.Protocol.Nebula.Completion
open Nightstream.Protocol.Nebula.Memory

/-- These constructors are theorem obligations, not assumptions hidden in a
single generic "soundness" event. -/
inductive BadEvent where
  | decode
  | circuitRefinement
  | applicationPortCoverage
  | fPrimeLifecycle
  | recursiveSizeClosure
  | bundlePropagation
  | commitmentBinding
  | compactTokenBinding
  | seededSetup
  | poseidonOrTranscript
  | memoryFingerprint
  | piRlcSampler
  | foldExtraction
  | terminalBackend
deriving DecidableEq, Repr

def AnyBad (occurs : BadEvent → Prop) : Prop :=
  ∃ event, occurs event

/-- Typed identity of every verifier-key-owned artifact that can change the
accepted relation. The aggregate digest is not used as a substitute for the
independent manifest identities. -/
structure VerifierKeyIdentity (Digest : Type) where
  digest : Digest
  relationManifestDigest : Digest
  laneLayoutDigest : Digest
  setupManifestDigest : Digest
  transcriptManifestDigest : Digest
  codecManifestDigest : Digest
  terminalManifestDigest : Digest
  applicationStateSchemaDigest : Digest
deriving DecidableEq, Repr

/-- Exact verifier-owned identity of the external V2 statement. A deployed
codec must refine each digest to the corresponding authoritative object. -/
structure StatementIdentity (Digest : Type) where
  profile : Profile.Identity
  verifierKey : VerifierKeyIdentity Digest
  applicationRelationDigest : Digest
  programDigest : Digest
  memoryPlanDigest : Digest
deriving DecidableEq, Repr

inductive IdentityDigestTag where
  | verifierKey
  | relationManifest
  | laneLayout
  | setupManifest
  | transcriptManifest
  | codecManifest
  | terminalManifest
  | applicationStateSchema
  | applicationRelation
  | program
  | memoryPlan
deriving DecidableEq, Repr

inductive IdentityAtom (Digest : Type) where
  | profile (identity : Profile.Identity)
  | digest (tag : IdentityDigestTag) (value : Digest)
deriving DecidableEq, Repr

/-- Canonical identity prefix of the external statement. Field tags and order
are authority-bearing. -/
def StatementIdentity.encode
    {Digest : Type} (identity : StatementIdentity Digest) :
    List (IdentityAtom Digest) :=
  [ .profile identity.profile
  , .digest .verifierKey identity.verifierKey.digest
  , .digest .relationManifest identity.verifierKey.relationManifestDigest
  , .digest .laneLayout identity.verifierKey.laneLayoutDigest
  , .digest .setupManifest identity.verifierKey.setupManifestDigest
  , .digest .transcriptManifest identity.verifierKey.transcriptManifestDigest
  , .digest .codecManifest identity.verifierKey.codecManifestDigest
  , .digest .terminalManifest identity.verifierKey.terminalManifestDigest
  , .digest .applicationStateSchema
      identity.verifierKey.applicationStateSchemaDigest
  , .digest .applicationRelation identity.applicationRelationDigest
  , .digest .program identity.programDigest
  , .digest .memoryPlan identity.memoryPlanDigest
  ]

theorem StatementIdentity.encode_length
    {Digest : Type} (identity : StatementIdentity Digest) :
    identity.encode.length = 12 :=
  rfl

theorem StatementIdentity.encode_injective
    {Digest : Type} [DecidableEq Digest] :
    Function.Injective
      (StatementIdentity.encode :
        StatementIdentity Digest → List (IdentityAtom Digest)) := by
  intro left right equal
  rcases left with
    ⟨leftProfile,
      ⟨leftKey, leftRelation, leftLayout, leftSetup, leftTranscript,
        leftCodec, leftTerminal, leftStateSchema⟩,
      leftApplication, leftProgram, leftPlan⟩
  rcases right with
    ⟨rightProfile,
      ⟨rightKey, rightRelation, rightLayout, rightSetup, rightTranscript,
        rightCodec, rightTerminal, rightStateSchema⟩,
      rightApplication, rightProgram, rightPlan⟩
  simp_all [StatementIdentity.encode]

structure PublicStatement (Program ApplicationState Digest : Type) where
  identity : StatementIdentity Digest
  program : Program
  initialApplicationState : ApplicationState
  initialImage : Fin scannedCells → Nat
  initialImageInRange : Snapshot.ImageInRange initialImage
  segmentCount : Nat
  finalGlobalTimestamp : Nat
  expectedResult : ExecutionResult ApplicationState Digest

/-- Independent semantic conclusion after deterministic refinement and after
all named computational bad events are excluded. -/
def HasSoundExecution
    {Program ApplicationState Digest : Type}
    (applicationSemantics :
      ApplicationTrace.Semantics Program ApplicationState)
    (statement : PublicStatement Program ApplicationState Digest)
    (snapshotRoot : Snapshot → Digest) : Prop :=
  ∃ (finalSnapshot : Snapshot)
      (segmentAccesses : List (List Access))
      (applicationExecution :
        ApplicationTrace.CompletedExecution applicationSemantics
          statement.program statement.initialApplicationState
          statement.expectedResult statement.segmentCount),
    Executes (Snapshot.ofImage statement.initialImage).tuples 0
        segmentAccesses.flatten finalSnapshot.tuples
        statement.finalGlobalTimestamp ∧
      segmentAccesses.length = statement.segmentCount ∧
      applicationExecution.CoversMemory segmentAccesses ∧
      statement.expectedResult.finalMemoryRoot = snapshotRoot finalSnapshot

end Nightstream.Protocol.Nebula.Soundness
