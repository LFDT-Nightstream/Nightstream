import Nightstream.Assurance.Nebula.CompactSequenceSecurity

/-!
Contract: staged deployed-acceptance pipeline for exact Nebula V2.

Assurance tier: implementation and cryptographic refinement boundary.

Owns a deterministic parser-based acceptance predicate, a narrow extraction
interface from terminal acceptance to raw `IdealAcceptV2`, and composition
with the independent ideal and compact-chain soundness theorems.

The extraction interface does not contain `ExecutionWitness`,
`HasSoundExecution`, `ValidSegment`, `ValidChain`, or a balance conclusion.
It must instead construct the raw ideal checks from an extracted artifact.

Does not implement the parser, generated rows, NIFS/fold extractor, terminal
backend, Rust verifier, or cryptographic probability games.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Assurance.Nebula.ReleasePipeline

open Nightstream.Assurance.Nebula.CompactSequenceSecurity
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.CompactChain
open Nightstream.Protocol.Nebula.CompactCommit
open Nightstream.Protocol.Nebula.IdealAcceptance
open Nightstream.Protocol.Nebula.Soundness

/-- External acceptance is not an arbitrary predicate. The exact decoder must
produce one parsed value, and the selected terminal verifier must accept that
same value. -/
def Accepts
    {Bytes Parsed : Type}
    (decode : Bytes → Option Parsed)
    (terminalAccepts : Parsed → Prop)
    (proof : Bytes) : Prop :=
  ∃ parsed, decode proof = some parsed ∧ terminalAccepts parsed

/-- Narrow bridge owned by the generated relation, recursive proof system,
and terminal backend. It extracts an implementation artifact first. Only the
second field refines that artifact to raw ideal checks. Neither field may
assume the final semantic execution conclusion. -/
structure ExtractionBoundary
    {Bytes Parsed Artifact ChallengeField Plan Digest : Type}
    [Field ChallengeField]
    {schema : FullClaim.Schema}
    (decode : Bytes → Option Parsed)
    (terminalAccepts : Parsed → Prop)
    (occurs : BadEvent → Prop)
    (config :
      Config ChallengeField Profile.Identity Plan CommitmentEncoding Digest)
    (bundleComponent :
      schema.CommitmentBundle → CommitmentBundle.Component →
        CommitmentEncoding)
    (verify : FullVerifier schema Digest ChallengeField)
    {Program ApplicationState : Type}
    (applicationSemantics :
      ApplicationTrace.Semantics Program ApplicationState)
    (statement : PublicStatement Program ApplicationState Digest) where
  extracted : Parsed → Artifact → Prop
  terminalExtracts : ∀ parsed,
    terminalAccepts parsed →
      AnyBad occurs ∨ ∃ artifact, extracted parsed artifact
  extractedRefinesIdeal : ∀ {parsed artifact},
    extracted parsed artifact →
      IdealAcceptV2 config schema bundleComponent verify applicationSemantics
        statement

/-- Final deterministic failure type. Implementation/extraction failures and
protocol-level failures remain distinct. -/
inductive Failure
    {ChallengeField Plan Seed Digest : Type}
    [Field ChallengeField]
    (occurs : BadEvent → Prop)
    (config :
      Config ChallengeField Profile.Identity Plan CommitmentEncoding Digest)
    (hash : HashInput Plan Digest → Digest)
    (key : Key Plan Seed) : Prop where
  | implementation (failure : AnyBad occurs) :
      Failure occurs config hash key
  | protocol (failure : ReleaseFailure config hash key) :
      Failure occurs config hash key

/-- Staged conditional soundness for deployed bytes. The theorem derives the
semantic conclusion. Its refinement premise ends at `IdealAcceptV2`; it does
not assume an execution witness or the desired conclusion. -/
theorem deployed_acceptance_implies_execution_or_named_failure
    {Bytes Parsed Artifact ChallengeField Plan Seed Digest : Type}
    [Field ChallengeField]
    {schema : FullClaim.Schema}
    {decode : Bytes → Option Parsed}
    {terminalAccepts : Parsed → Prop}
    {occurs : BadEvent → Prop}
    {config :
      Config ChallengeField Profile.Identity Plan CommitmentEncoding Digest}
    {bundleComponent :
      schema.CommitmentBundle → CommitmentBundle.Component →
        CommitmentEncoding}
    {verify : FullVerifier schema Digest ChallengeField}
    {Program ApplicationState : Type}
    {applicationSemantics :
      ApplicationTrace.Semantics Program ApplicationState}
    {statement : PublicStatement Program ApplicationState Digest}
    (hash : HashInput Plan Digest → Digest)
    (key : Key Plan Seed)
    (chainRootExact : config.chainRoot = chainRoot hash key)
    (boundary :
      ExtractionBoundary (Artifact := Artifact) decode terminalAccepts occurs
        config bundleComponent verify applicationSemantics statement)
    {proof : Bytes}
    (accepted : Accepts decode terminalAccepts proof) :
    Failure occurs config hash key ∨
      HasSoundExecution applicationSemantics statement config.snapshotRoot := by
  rcases accepted with ⟨parsed, _decoded, terminalAccepted⟩
  rcases boundary.terminalExtracts parsed terminalAccepted with
    implementationFailure | ⟨artifact, extracted⟩
  · exact Or.inl (.implementation implementationFailure)
  · let idealAcceptance := boundary.extractedRefinesIdeal extracted
    rcases compact_acceptance_implies_execution_or_release_failure
        hash key chainRootExact idealAcceptance with protocolFailure | execution
    · exact Or.inl (.protocol protocolFailure)
    · exact Or.inr execution.execution

end Nightstream.Assurance.Nebula.ReleasePipeline
