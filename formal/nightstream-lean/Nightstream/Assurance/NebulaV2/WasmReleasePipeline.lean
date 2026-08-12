import Nightstream.Assurance.NebulaV2.ReleasePipeline
import Nightstream.Protocol.NebulaV2.StatementAuthority
import Nightstream.Protocol.NebulaV2.WasmPublicStatementEncoding

/-!
Contract: production-shaped deployed soundness boundary for V2 WASM.

Assurance tier: implementation and cryptographic refinement boundary.

Owns composition of byte decoding, terminal extraction, fixed deterministic
WASM semantics, exact public-image decoding, and verifier-owned statement
authority. The success branch derives execution and terminal state.

Does not implement or assume away the parser, generated relation, recursive
extractor, terminal backend, digest security, or computational bounds. Those
remain the explicit `ExtractionBoundary` and named failure branches.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Assurance.NebulaV2.WasmReleasePipeline

open Nightstream.Assurance.NebulaV2.CompactSequenceSecurity
open Nightstream.Assurance.NebulaV2.ReleasePipeline
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.CompactChain
open Nightstream.Protocol.NebulaV2.CompactCommit
open Nightstream.Protocol.NebulaV2.IdealAcceptance
open Nightstream.Protocol.NebulaV2.Soundness
open Nightstream.Protocol.NebulaV2.StatementAuthority
open Nightstream.Protocol.NebulaV2.WasmPublicStatementEncoding
open Nightstream.Protocol.NebulaV2.WasmState
open Nightstream.Protocol.NebulaV2.WasmStatement

/-- The production theorem has no arbitrary application-semantics parameter
and no opaque initial-memory authority. Its refinement input still stops at
raw `IdealAcceptV2`; it cannot contain the conclusion below. -/
theorem deployed_acceptance_implies_fixed_wasm_execution_or_named_failure
    {Bytes Parsed Artifact ChallengeField Plan Seed : Type}
    [Field ChallengeField]
    {schema : FullClaim.Schema}
    {decode : Bytes → Option Parsed}
    {terminalAccepts : Parsed → Prop}
    {occurs : BadEvent → Prop}
    {config :
      Config ChallengeField Profile.Identity Plan CommitmentEncoding
        Digest.Value}
    {bundleComponent :
      schema.CommitmentBundle → CommitmentBundle.Component →
        CommitmentEncoding}
    {verify : FullVerifier schema Digest.Value ChallengeField}
    {Program : Type}
    {machine : Machine Program}
    {statement : ProductionStatement Program}
    {publicImage : PublicImage}
    (decoded : publicImage.Decodes statement)
    {RelationManifest LaneLayout SetupManifest TranscriptManifest
      CodecManifest TerminalManifest StateSchema ApplicationRelation
      MemoryPlan : Type}
    (digestFunctions :
      DigestFunctions RelationManifest LaneLayout SetupManifest
        TranscriptManifest CodecManifest TerminalManifest StateSchema
        ApplicationRelation Program MemoryPlan Digest.Value)
    (authorityInputs :
      Inputs RelationManifest LaneLayout SetupManifest TranscriptManifest
        CodecManifest TerminalManifest StateSchema ApplicationRelation
        Program MemoryPlan)
    (authority : Opens digestFunctions authorityInputs statement)
    (hash : HashInput Plan Digest.Value → Digest.Value)
    (key : Key Plan Seed)
    (chainRootExact : config.chainRoot = chainRoot hash key)
    (boundary :
      ExtractionBoundary (Artifact := Artifact) decode terminalAccepts occurs
        config bundleComponent verify machine.semantics statement.base)
    {proof : Bytes}
    (accepted : Accepts decode terminalAccepts proof) :
    ReleasePipeline.Failure occurs config hash key ∨
      (HasSoundExecution machine.semantics statement.base
          config.snapshotRoot ∧
        statement.base.expectedResult.finalApplicationState.Terminal
          statement.base.expectedResult.outcome ∧
        publicImage = PublicImage.ofStatement statement ∧
        statement.base.identity =
          digestFunctions.expectedIdentity authorityInputs) := by
  rcases deployed_acceptance_implies_execution_or_named_failure
      hash key chainRootExact boundary accepted with failure | execution
  · exact Or.inl failure
  · right
    have terminal :
        statement.base.expectedResult.finalApplicationState.Terminal
          statement.base.expectedResult.outcome := by
      rcases execution with
        ⟨_finalSnapshot, _segmentAccesses, applicationExecution,
          _memoryExecution, _segmentCount, _coverage, _finalRoot⟩
      exact machine.completedExecution_final_terminal applicationExecution
    exact
      ⟨execution, terminal, decoded.exactImage, authority.identity⟩

end Nightstream.Assurance.NebulaV2.WasmReleasePipeline
