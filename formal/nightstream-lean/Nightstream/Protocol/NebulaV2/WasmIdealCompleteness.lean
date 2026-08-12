import Nightstream.Protocol.NebulaV2.IdealCompleteness
import Nightstream.Protocol.NebulaV2.WasmStatement

/-!
Contract: constructive ideal completeness for the fixed V2 WASM machine.

Assurance tier: model-level.

Owns specialization of the general ideal-completeness construction to the
deterministic `WasmState.Machine` semantics and the exact production
statement. A caller cannot replace the application relation with a permissive
predicate.

Does not prove completeness of witness generation, commitments, NIFS,
generated rows, Rust, proof serialization, or the deployed terminal backend.
Those primitive artifacts remain explicit inputs to `CompletenessInput`.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.NebulaV2.WasmIdealCompleteness

open Nightstream.Protocol.NebulaV2.IdealAcceptance
open Nightstream.Protocol.NebulaV2.IdealCompleteness
open Nightstream.Protocol.NebulaV2.Soundness
open Nightstream.Protocol.NebulaV2.WasmState
open Nightstream.Protocol.NebulaV2.WasmStatement

/-- Exact completeness input for the verifier-owned WASM machine. -/
abbrev WasmCompletenessInput
    {ChallengeField Profile Plan Commitment : Type}
    [Field ChallengeField]
    (config : Config ChallengeField Profile Plan Commitment Digest.Value)
    (schema : FullClaim.Schema)
    (bundleComponent :
      schema.CommitmentBundle → CommitmentBundle.Component → Commitment)
    (verify : FullVerifier schema Digest.Value ChallengeField)
    {Program : Type}
    (machine : Machine Program)
    (statement : ProductionStatement Program) :=
  CompletenessInput config schema bundleComponent verify machine.semantics
    statement.base

/-- A valid execution of the fixed WASM machine, with honest primitive
artifacts, constructs the exact ideal V2 acceptance object. The conclusion is
derived by the general completeness construction; it is not a field of this
theorem's input. -/
def valid_fixed_wasm_execution_with_honest_artifacts_is_accepted
    {ChallengeField Profile Plan Commitment : Type}
    [Field ChallengeField]
    {schema : FullClaim.Schema}
    {config : Config ChallengeField Profile Plan Commitment Digest.Value}
    {bundleComponent :
      schema.CommitmentBundle → CommitmentBundle.Component → Commitment}
    {verify : FullVerifier schema Digest.Value ChallengeField}
    {Program : Type}
    {machine : Machine Program}
    {statement : ProductionStatement Program}
    (input : WasmCompletenessInput config schema bundleComponent verify
      machine statement) :
    IdealAcceptV2 config schema bundleComponent verify machine.semantics
      statement.base :=
  valid_execution_with_honest_artifacts_is_accepted input

/-- The same completeness input proves that its declared final WASM state is
terminal for the declared return or trap result. -/
theorem completeness_input_final_state_terminal
    {ChallengeField Profile Plan Commitment : Type}
    [Field ChallengeField]
    {schema : FullClaim.Schema}
    {config : Config ChallengeField Profile Plan Commitment Digest.Value}
    {bundleComponent :
      schema.CommitmentBundle → CommitmentBundle.Component → Commitment}
    {verify : FullVerifier schema Digest.Value ChallengeField}
    {Program : Type}
    {machine : Machine Program}
    {statement : ProductionStatement Program}
    (input : WasmCompletenessInput config schema bundleComponent verify
      machine statement) :
    statement.base.expectedResult.finalApplicationState.Terminal
      statement.base.expectedResult.outcome :=
  machine.completedExecution_final_terminal input.applicationExecution

end Nightstream.Protocol.NebulaV2.WasmIdealCompleteness
