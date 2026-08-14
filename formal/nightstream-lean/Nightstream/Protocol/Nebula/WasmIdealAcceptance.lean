import Nightstream.Protocol.Nebula.IdealAcceptance
import Nightstream.Protocol.Nebula.WasmPublicStatementEncoding

/-!
Contract: fixed-machine ideal soundness theorem for production V2 WASM.

Assurance tier: model-level and security-reduction boundary.

Owns the specialization of ideal Nebula acceptance to the verifier-key-owned
deterministic WASM machine and the exact public-statement decoder. It removes
the generic application-semantics parameter from the production theorem.

Does not prove that Rust, generated rows, recursive verification, a terminal
backend, or a byte parser refines ideal acceptance. It also does not bound a
named cryptographic failure.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.Nebula.WasmIdealAcceptance

open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.IdealAcceptance
open Nightstream.Protocol.Nebula.Soundness
open Nightstream.Protocol.Nebula.WasmPublicStatementEncoding
open Nightstream.Protocol.Nebula.WasmState
open Nightstream.Protocol.Nebula.WasmStatement

abbrev ProductionAccept
    {ChallengeField Profile Plan Commitment : Type}
    [Field ChallengeField]
    (config :
      Config ChallengeField Profile Plan Commitment Digest.Value)
    (schema : FullClaim.Schema)
    (bundleComponent :
      schema.CommitmentBundle → CommitmentBundle.Component → Commitment)
    (verify : FullVerifier schema Digest.Value ChallengeField)
    {Program : Type}
    (machine : Machine Program)
    (statement : ProductionStatement Program) :=
  IdealAcceptV2 config schema bundleComponent verify machine.semantics
    statement.base

/-- Exact success conclusion for the fixed WASM machine. Every field is
derived by `production_acceptance_implies_execution_or_failure`; callers do
not supply a semantic execution witness to this structure. -/
structure CertifiedWasmExecution
    {ChallengeField Profile Plan Commitment : Type}
    [Field ChallengeField]
    {config :
      Config ChallengeField Profile Plan Commitment Digest.Value}
    {schema : FullClaim.Schema}
    {bundleComponent :
      schema.CommitmentBundle → CommitmentBundle.Component → Commitment}
    {verify : FullVerifier schema Digest.Value ChallengeField}
    {Program : Type}
    {machine : Machine Program}
    {statement : ProductionStatement Program}
    {publicImage : PublicImage}
    (acceptance :
      ProductionAccept config schema bundleComponent verify machine statement) :
    Prop where
  core : CertifiedExecution acceptance
  finalStateFromMachine :
    statement.base.expectedResult.finalApplicationState.Terminal
      statement.base.expectedResult.outcome
  exactPublicImage : publicImage = PublicImage.ofStatement statement

/-- Production-shaped ideal soundness. The only success branch uses the
fixed deterministic machine and the exact decoded V2 statement. A constant
snapshot root remains an explicit `Failure.snapshotRoot` branch. -/
theorem production_acceptance_implies_execution_or_failure
    {ChallengeField Profile Plan Commitment : Type}
    [Field ChallengeField]
    {config :
      Config ChallengeField Profile Plan Commitment Digest.Value}
    {schema : FullClaim.Schema}
    {bundleComponent :
      schema.CommitmentBundle → CommitmentBundle.Component → Commitment}
    {verify : FullVerifier schema Digest.Value ChallengeField}
    {Program : Type}
    {machine : Machine Program}
    {statement : ProductionStatement Program}
    {publicImage : PublicImage}
    (decoded : publicImage.Decodes statement)
    (acceptance :
      ProductionAccept config schema bundleComponent verify machine statement) :
    Failure config ∨
      CertifiedWasmExecution (publicImage := publicImage) acceptance := by
  rcases ideal_acceptance_implies_execution_or_failure acceptance with
    failure | certified
  · exact Or.inl failure
  · have finalStateFromMachine :
        statement.base.expectedResult.finalApplicationState.Terminal
          statement.base.expectedResult.outcome := by
      rcases certified.execution with
        ⟨_finalSnapshot, _segmentAccesses, applicationExecution,
          _memoryExecution, _segmentCount, _coverage, _finalRoot⟩
      exact machine.completedExecution_final_terminal applicationExecution
    exact Or.inr
      { core := certified
        finalStateFromMachine := finalStateFromMachine
        exactPublicImage := decoded.exactImage }

end Nightstream.Protocol.Nebula.WasmIdealAcceptance
