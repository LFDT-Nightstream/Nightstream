import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.FirstRound.Connectivity
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.LaterRound.Replay
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.FinalState

/-!
Complete typed replay of the terminal Split-NC transcript.

Assurance tier: conditional implementation/R1CS refinement.

Owns: the exact fifteen-message decomposition into the distinct first round
and fourteen uniform later rounds; composition of prologue, first-round, and
later-round execution; and the resulting exact final Poseidon2 call output.

Does not own: derivation of the FE successor; the typed-to-assignment carrier
boundary; SumCheck algebra; costs; necessity; or row removal.

Emits constraints: no.

Authority boundary: the raw transcript list is derived losslessly from one
typed carrier. Every state transition is computed by the independent
transcript machine and tied to accepted Poseidon2 rows. No intermediate
digest, state, or challenge is supplied as authority.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc_sumcheck.messages` | raw NC messages are round zero followed by rounds one through fourteen | derived | `rawRounds_eq_messages` |
| `nifs.pi_ccs.nc_sumcheck.replay` | prologue and all fifteen rounds reach the exact final call output | conditional composition | `runRounds_eq_finalCallOutput` |
| `nifs.pi_ccs.nc_sumcheck.final_permutation` | complete exact schedule satisfies the final permutation boundary | conditional composition | `finalPermutationBound_of_exactSchedule` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.Replay

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.DigestRounds
open Nightstream.Implementation.R1CS.PiCcsTranscript.SumCheck
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

set_option maxHeartbeats 1000000

private abbrev Input
    {shape : SemanticShape}
    (publicInput : PublicInput shape) :=
  PiCcsTranscript.Exact.Schedule.Input publicInput Carrier.domain

/-- Exact typed physical order: the distinct round zero followed by the
fourteen uniform later rounds. -/
def messages
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    (input : Input publicInput) :
    List RoundMessage :=
  FirstRound.Source.typedMessage input ::
    LaterRound.Replay.messages input

/-- The concrete raw NC list is exactly the phase-structured typed list. -/
theorem rawRounds_eq_messages
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    (input : Input publicInput) :
    (PiCcsTranscript.Exact.Schedule.rawMessages input).ncRounds =
      messages input := by
  rw [Carrier.rawRounds_eq_typed input,
    Carrier.typedRawRounds_eq_fixed input]
  unfold Carrier.fixedTypedRawRounds messages
  unfold FirstRound.Source.typedMessage
    LaterRound.Replay.messages LaterRound.Source.typedMessage
  apply List.ext_get
  · simpa only [List.length_ofFn, List.length_cons,
      LaterRound.Artifact.roundCount, Schedule.laterRoundCount,
      Carrier.roundCount, Nat.reduceAdd] using Carrier.domain_roundCount
  · intro index leftLt rightLt
    cases index with
    | zero =>
        simp only [List.get_eq_getElem, List.getElem_ofFn,
          List.getElem_cons_zero]
        apply congrArg PiCcsTranscript.ExactMessages.encodeFixed
        apply congrArg (Carrier.typedRound input)
        apply Fin.ext
        rfl
    | succ index =>
        simp only [List.get_eq_getElem, List.getElem_ofFn,
          List.getElem_cons_succ]
        apply congrArg PiCcsTranscript.ExactMessages.encodeFixed
        apply congrArg (Carrier.typedRound input)
        apply Fin.ext
        rfl

/-- The indexed final later-round call and the independently selected
final-round artifact are the same physical Poseidon2 call. -/
theorem laterFinalCall_eq_finalArtifact :
    LaterRound.Artifact.squeezeCall LaterRound.Replay.finalRound =
      FinalRound.Artifact.finalSqueezeCall := by
  decide

/-- Starting at one FE-successor state, the exact prologue and all fifteen
typed messages reach the exact accepted final squeeze-call output. -/
theorem runRounds_eq_finalCallOutput
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    (input : Input publicInput)
    (afterFe : State)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiCcsNcSumcheck.Accepted assignment)
    (carrierBound : Carrier.Bound input assignment canonical)
    (afterFeBound :
      Prologue.Execution.AfterFeBound afterFe assignment canonical) :
    (runRounds (ncPrologue afterFe) (messages input)).1 =
      callOutputState assignment canonical
        FinalRound.Artifact.finalSqueezeCall := by
  have firstIncoming :
      LaterRound.Execution.IncomingBound ⟨0, by decide⟩
        (runRound (ncPrologue afterFe)
          (FirstRound.Source.typedMessage input)).1
        assignment canonical := by
    simpa only [FirstRound.Connectivity.laterFirst] using
      FirstRound.Connectivity.nextIncoming_of_prologueAndCarrier
        input afterFe canonical one accepted carrierBound afterFeBound
  have laterReplay :=
    LaterRound.Replay.run_eq_finalCallOutput
      (runRound (ncPrologue afterFe)
        (FirstRound.Source.typedMessage input)).1
      input canonical one accepted carrierBound firstIncoming
  calc
    (runRounds (ncPrologue afterFe) (messages input)).1 =
        (runRounds
          (runRound (ncPrologue afterFe)
            (FirstRound.Source.typedMessage input)).1
          (LaterRound.Replay.messages input)).1 := by
      exact runRounds_cons_state
        (ncPrologue afterFe)
        (FirstRound.Source.typedMessage input)
        (LaterRound.Replay.messages input)
    _ =
        callOutputState assignment canonical
          (LaterRound.Artifact.squeezeCall
            LaterRound.Replay.finalRound) :=
      laterReplay
    _ =
        callOutputState assignment canonical
          FinalRound.Artifact.finalSqueezeCall := by
      exact congrArg
        (fun call => callOutputState assignment canonical call)
        laterFinalCall_eq_finalArtifact

/-- The complete exact NC schedule reaches the final accepted call output.
The only cross-phase premise is the explicit retained FE-successor surface. -/
theorem afterNc_eq_finalCallOutput
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    (input : Input publicInput)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiCcsNcSumcheck.Accepted assignment)
    (carrierBound : Carrier.Bound input assignment canonical)
    (afterFeBound :
      Prologue.Execution.AfterFeBound
        (PiCcsTranscript.Exact.Schedule.run input).afterFe
        assignment canonical) :
    (PiCcsTranscript.Exact.Schedule.run input).afterNc =
      callOutputState assignment canonical
        FinalRound.Artifact.finalSqueezeCall := by
  rw [PiCcsTranscript.Exact.Schedule.run_afterNc_uses_afterFe]
  unfold runNc
  rw [rawRounds_eq_messages input]
  exact runRounds_eq_finalCallOutput input
    (PiCcsTranscript.Exact.Schedule.run input).afterFe
    canonical one accepted carrierBound afterFeBound

/-- Complete typed replay discharges the old opaque final-prefix assumption.
The final semantic state is the pure permutation of the exact artifact call
input, with accepted rows used only to identify that permutation's output. -/
theorem finalPermutationBound_of_exactSchedule
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    (input : Input publicInput)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiCcsNcSumcheck.Accepted assignment)
    (carrierBound : Carrier.Bound input assignment canonical)
    (afterFeBound :
      Prologue.Execution.AfterFeBound
        (PiCcsTranscript.Exact.Schedule.run input).afterFe
        assignment canonical) :
    FinalState.FinalPermutationBound
      (PiCcsTranscript.Exact.Schedule.run input).afterNc
      assignment canonical := by
  refine
    ⟨callInputState assignment canonical
        FinalRound.Artifact.finalSqueezeCall ⟨rate, by decide⟩,
      rfl, ?_⟩
  rw [afterNc_eq_finalCallOutput input canonical one accepted
    carrierBound afterFeBound]
  exact
    (callAccepted_permute canonical one
      FinalRound.Artifact.finalSqueezeCall ⟨rate, by decide⟩
      (FinalRound.Artifact.finalSqueezeCallAccepted accepted)).symm

end Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.Replay
