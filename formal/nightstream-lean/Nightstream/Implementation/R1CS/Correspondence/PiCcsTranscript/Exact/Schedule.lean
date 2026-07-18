import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Exact.Carrier

/-!
Canonical exact FE-to-NC SumCheck sub-schedule for the minimal mixed-width
Split-NC `Pi_CCS` candidate verifier.

Assurance tier: executable candidate-verifier semantics.

Owns: one exact typed input surface, the raw boundary projection, one FE
transcript execution, direct threading of its successor into one NC
execution, the joint trace, exact challenge counts, and deterministic replay.

Does not own: derivation of the verifier-owned FE initial claim, pre-SumCheck
coin derivation, FE/NC polynomial truth, terminal identities, catch-up,
native/gadget lazy-prologue refinement, Poseidon2 row refinement, equality
with the current uniform-width Rust/R1CS encoding, costs, or row removal.

Emits constraints: no.

Authority boundary: `Input` contains an `Exact.Carrier`, not loose raw
messages or a `WellShaped` proof. `expectedFeInitial` is verifier-owned.
`rawMessages` is only the lossless boundary projection consumed by the
existing concrete transcript machine.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.exact.schedule.input` | carry only exact typed FE/NC rounds plus verifier state and FE initial | verifier boundary | `Input` |
| `nifs.pi_ccs.exact.schedule.encoding` | raw projection satisfies the complete exact language | derived | `rawMessages_exactLanguage` |
| `nifs.pi_ccs.exact.schedule.fe` | enter FE once and replay the exact row-then-lane list | direct dataflow | `feExecution` |
| `nifs.pi_ccs.exact.schedule.nc` | enter NC once from FE's exact successor | direct dataflow | `ncExecution`, `run_afterNc_uses_afterFe` |
| `nifs.pi_ccs.exact.schedule.counts.fe` | one challenge per exact FE row/lane round | derived | `feChallengeCount` |
| `nifs.pi_ccs.exact.schedule.counts.nc` | one challenge per exact NC round | derived | `ncChallengeCount` |
| `nifs.pi_ccs.exact.schedule.nc.cursor` | a positive exact NC phase ends with cursor zero | computed | `run_afterNc_absorbed_zero` |
| `nifs.pi_ccs.exact.schedule.replay` | one input determines one joint trace | computed | `replay_deterministic` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.Schedule

open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

private abbrev RawMessages :=
  Nightstream.Implementation.R1CS.PiCcsTranscript.SumCheck.Messages
private abbrev RawExtension :=
  Nightstream.Implementation.R1CS.PiCcsTranscript.Primitives.Extension
private abbrev NcRoundCount (domain : FlatNcDomain) : Nat :=
  Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc.roundCount
    domain

/-- Exact post-coin-derivation input to the two SumCheck transcript phases.

The FE initial claim is explicit because this module starts after its
verifier-side computation. It is not a field of `Carrier`, and no raw shape
predicate can be supplied in place of the exact typed rounds. -/
structure Input
    {shape : SemanticShape}
    (publicInput : PublicInput shape)
    (domain : FlatNcDomain) where
  initialState : State
  expectedFeInitial : K
  carrier : Carrier publicInput domain

/-- Lossless raw projection used only at the concrete transcript boundary. -/
def rawMessages
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input publicInput domain) : RawMessages :=
  encode input.expectedFeInitial input.carrier

/-- Execute the FE transcript phase exactly once from the incoming state. -/
def feExecution
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input publicInput domain) : State × List RawExtension :=
  Nightstream.Implementation.R1CS.PiCcsTranscript.SumCheck.runFe
    input.initialState (rawMessages input)

/-- Execute the NC transcript phase exactly once from FE's returned state. -/
def ncExecution
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input publicInput domain) : State × List RawExtension :=
  Nightstream.Implementation.R1CS.PiCcsTranscript.SumCheck.runNc
    (feExecution input).1 (rawMessages input)

/-- Joint verifier-visible trace of the exact FE-to-NC sub-schedule. -/
structure Trace where
  afterFe : State
  feChallenges : List RawExtension
  afterNc : State
  ncChallenges : List RawExtension

/-- Replay the sole exact FE-to-NC phase order. -/
def run
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input publicInput domain) : Trace :=
  let fe := feExecution input
  let nc :=
    Nightstream.Implementation.R1CS.PiCcsTranscript.SumCheck.runNc
      fe.1 (rawMessages input)
  {
    afterFe := fe.1
    feChallenges := fe.2
    afterNc := nc.1
    ncChallenges := nc.2
  }

/-- The raw boundary projection always belongs to the complete exact
physical language. -/
theorem rawMessages_exactLanguage
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input publicInput domain) :
    ExactLanguage publicInput domain input.expectedFeInitial
      (rawMessages input) :=
  exactLanguage_encode input.expectedFeInitial input.carrier

/-- Re-decoding the raw boundary projection recovers the exact typed carrier. -/
@[simp] theorem decode_rawMessages
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input publicInput domain) :
    decode publicInput domain input.expectedFeInitial (rawMessages input) =
      some input.carrier :=
  decode_encode input.expectedFeInitial input.carrier

/-- The FE fields of the joint trace are the two projections of one exact
concrete FE execution. -/
theorem run_feJoint
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input publicInput domain) :
    ((run input).afterFe, (run input).feChallenges) =
      feExecution input := by
  rfl

/-- NC starts from the exact state returned by FE; no caller can supply an
independent phase-boundary state. -/
theorem run_ncJoint
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input publicInput domain) :
    ((run input).afterNc, (run input).ncChallenges) =
      Nightstream.Implementation.R1CS.PiCcsTranscript.SumCheck.runNc
        (run input).afterFe (rawMessages input) := by
  rfl

/-- State-only form of the FE-to-NC successor binding. -/
@[simp] theorem run_afterNc_uses_afterFe
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input publicInput domain) :
    (run input).afterNc =
      (Nightstream.Implementation.R1CS.PiCcsTranscript.SumCheck.runNc
        (run input).afterFe (rawMessages input)).1 := by
  exact congrArg Prod.fst (run_ncJoint input)

/-- FE derives exactly one challenge per exact row or lane round. -/
theorem feChallengeCount
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input publicInput domain) :
    (run input).feChallenges.length =
      shape.rowVariables + domain.laneVariables := by
  change
    (Nightstream.Implementation.R1CS.PiCcsTranscript.SumCheck.runFe
      input.initialState (rawMessages input)).2.length =
        shape.rowVariables + domain.laneVariables
  rw [
    Nightstream.Implementation.R1CS.PiCcsTranscript.SumCheck.runFe_challengeCount]
  exact encode_feRounds_length input.expectedFeInitial input.carrier

/-- NC derives exactly one challenge per verifier-owned NC coordinate. -/
theorem ncChallengeCount
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input publicInput domain) :
    (run input).ncChallenges.length = NcRoundCount domain := by
  change
    (Nightstream.Implementation.R1CS.PiCcsTranscript.SumCheck.runNc
      (feExecution input).1 (rawMessages input)).2.length =
        NcRoundCount domain
  rw [
    Nightstream.Implementation.R1CS.PiCcsTranscript.SumCheck.runNc_challengeCount]
  exact encode_ncRounds_length input.expectedFeInitial input.carrier

/-- A positive verifier-owned NC dimension makes the exact NC message list
nonempty, so its final challenge permutation computes cursor zero. The
positivity premise is explicit because `FlatNcDomain` also models the valid
zero-dimensional edge case. -/
theorem run_afterNc_absorbed_zero
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input publicInput domain)
    (positive : 0 < NcRoundCount domain) :
    (run input).afterNc.absorbed.val = 0 := by
  rw [run_afterNc_uses_afterFe]
  unfold Nightstream.Implementation.R1CS.PiCcsTranscript.SumCheck.runNc
  have nonempty :
      (rawMessages input).ncRounds ≠ [] := by
    intro empty
    have count :=
      encode_ncRounds_length input.expectedFeInitial input.carrier
    change
      (rawMessages input).ncRounds.length = NcRoundCount domain
      at count
    rw [empty] at count
    simp only [List.length_nil] at count
    omega
  cases roundsEq : (rawMessages input).ncRounds with
  | nil => exact (nonempty roundsEq).elim
  | cons round rest =>
      exact
        Nightstream.Implementation.R1CS.PiCcsTranscript.SumCheck.runRounds_cons_absorbed_zero
          _ round rest

/-- Joint exact challenge-count statement for the complete sub-schedule. -/
theorem challengeCounts
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input publicInput domain) :
    (run input).feChallenges.length =
        shape.rowVariables + domain.laneVariables /\
      (run input).ncChallenges.length = NcRoundCount domain :=
  ⟨feChallengeCount input, ncChallengeCount input⟩

/-- Exact replay is functional: one typed input cannot determine two
different FE-to-NC traces. -/
theorem replay_deterministic
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input publicInput domain)
    (left right : Trace)
    (leftEq : left = run input)
    (rightEq : right = run input) :
    left = right := by
  rw [leftEq, rightEq]

end Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.Schedule
