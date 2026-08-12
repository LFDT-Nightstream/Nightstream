import Nightstream.Protocol.NebulaV2.IdealCompleteness

set_option autoImplicit false

namespace tests.NebulaV2IdealCompleteness

open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.Completion
open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.Protocol.NebulaV2.IdealAcceptance
open Nightstream.Protocol.NebulaV2.IdealCompleteness
open Nightstream.Protocol.NebulaV2.Soundness

variable {ChallengeField Profile Plan Commitment Digest : Type}
variable [Field ChallengeField]
variable {schema : FullClaim.Schema}
variable {config :
  Config ChallengeField Profile Plan Commitment Digest}
variable {bundleComponent :
  schema.CommitmentBundle → CommitmentBundle.Component → Commitment}
variable {Program ApplicationState : Type}
variable {applicationSemantics :
  ApplicationTrace.Semantics Program ApplicationState}
variable {statement : PublicStatement Program ApplicationState Digest}
variable {verify : FullVerifier schema Digest ChallengeField}

#check HonestSegment.toGlobalRun
#check HonestChain.toGlobalFPrime
#check CompletenessInput.globalFPrime

theorem valid_bounded_input_constructs_raw_acceptance
    (input : CompletenessInput config schema bundleComponent verify
      applicationSemantics statement) :
    Nonempty
      (IdealAcceptV2 config schema bundleComponent verify applicationSemantics
        statement) :=
  ⟨valid_execution_with_honest_artifacts_is_accepted input⟩

theorem constructed_acceptance_enters_the_soundness_reduction
    (input : CompletenessInput config schema bundleComponent verify
      applicationSemantics statement) :
    let acceptance := valid_execution_with_honest_artifacts_is_accepted input
    Failure config ∨ CertifiedExecution acceptance := by
  exact ideal_acceptance_implies_execution_or_failure
    (valid_execution_with_honest_artifacts_is_accepted input)

end tests.NebulaV2IdealCompleteness
