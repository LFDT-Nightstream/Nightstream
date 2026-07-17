import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Parameters
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.Interface

/-!
Sequential verifier-owned transcript replay for the exact-width Split-NC
SumCheck phase.

Owns: an exact-round certificate, the abstract NC transcript state machine,
message-before-challenge replay, and the adapter from transcript-derived
challenges to the exact-width NC claimed-chain checker.

Does not own: NC polynomial semantics, honest message construction, terminal
binding, transcript encoding, Poseidon2, Fiat--Shamir security, Rust, R1CS, or
constraint counts.

Emits constraints: no.

Authority boundary: the certificate contains exactly one five-coefficient
message per NC variable and no challenges or transcript states. The verifier
enters the NC phase once, absorbs each message, squeezes the corresponding
challenge, and threads the returned state into the next round.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc.transcript.certificate` | exactly one five-slot message per NC variable | checked by type | `Certificate` |
| `nifs.pi_ccs.nc.transcript.round.absorb` | absorb the current message before sampling | verifier transcript | `runRound` |
| `nifs.pi_ccs.nc.transcript.round.challenge` | squeeze one `K` challenge and thread its successor state | verifier transcript | `runRoundsFrom` |
| `nifs.pi_ccs.nc.transcript.point` | derived challenges form the exact NC cube point | computed | `derive` |
| `nifs.pi_ccs.nc.transcript.chain` | claimed-chain acceptance uses only derived challenges | checked | `Accepted`, `check` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

universe uState

/-- Number of sequential NC SumCheck rounds for the flat column/lane domain. -/
abbrev roundCount (domain : FlatNcDomain) : Nat :=
  domain.columnVariables + domain.laneVariables

/-- One exact five-coefficient NC round message. -/
abbrev RoundMessage :=
  Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.RoundMessage

/-- The prover-visible NC transcript certificate.

Its function domain fixes the round count structurally. Challenges and
transcript states cannot be represented as certificate fields. -/
structure Certificate (domain : FlatNcDomain) where
  rounds : Fin (roundCount domain) -> RoundMessage

namespace Certificate

/-- Canonical finite-index projection into the exact-width claimed-chain
checker. -/
def toSumCheck
    {domain : FlatNcDomain}
    (certificate : Certificate domain) :
    Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.Certificate
    where
  rounds := List.ofFn certificate.rounds

/-- Projection preserves the domain's exact NC round count. -/
@[simp] theorem toSumCheck_rounds_length
    {domain : FlatNcDomain}
    (certificate : Certificate domain) :
    certificate.toSumCheck.rounds.length = roundCount domain := by
  simp [toSumCheck]

end Certificate

/-- Abstract deterministic NC transcript machine.

`enterNc` owns the phase boundary or prologue. A round has no index-dependent
operation because the concrete protocol binds ordering through canonical
certificate order and chained state. -/
structure Machine (State : Type uState) where
  enterNc : State -> State
  absorbRound : State -> RoundMessage -> State
  squeezeChallenge : State -> K × State

/-- Absorb one exact-width message before squeezing its challenge. -/
def runRound
    {State : Type uState}
    (machine : Machine State)
    (state : State)
    (message : RoundMessage) : K × State :=
  machine.squeezeChallenge (machine.absorbRound state message)

/-- One round is definitionally a message absorb followed by a challenge
squeeze. -/
theorem runRound_eq_squeeze_after_absorb
    {State : Type uState}
    (machine : Machine State)
    (state : State)
    (message : RoundMessage) :
    runRound machine state message =
      machine.squeezeChallenge (machine.absorbRound state message) :=
  rfl

/-- Replay a message list in order, threading every squeeze successor state
into the next absorb. -/
def runRoundsFrom
    {State : Type uState}
    (machine : Machine State) :
    State -> List RoundMessage -> List K × State
  | state, [] => ([], state)
  | state, message :: messages =>
      let sample := runRound machine state message
      let tail := runRoundsFrom machine sample.2 messages
      (sample.1 :: tail.1, tail.2)

/-- Replay produces exactly one challenge per absorbed message. -/
theorem runRoundsFrom_challenges_length
    {State : Type uState}
    (machine : Machine State)
    (state : State)
    (messages : List RoundMessage) :
    (runRoundsFrom machine state messages).1.length = messages.length := by
  induction messages generalizing state with
  | nil => rfl
  | cons message messages inductionHypothesis =>
      simp only [runRoundsFrom, List.length_cons]
      rw [inductionHypothesis]

/-- Replaying a concatenation starts the suffix from the exact state returned
by the prefix and concatenates the two challenge lists. -/
theorem runRoundsFrom_append
    {State : Type uState}
    (machine : Machine State)
    (state : State)
    (first rest : List RoundMessage) :
    runRoundsFrom machine state (first ++ rest) =
      let firstResult := runRoundsFrom machine state first
      let restResult := runRoundsFrom machine firstResult.2 rest
      (firstResult.1 ++ restResult.1, restResult.2) := by
  induction first generalizing state with
  | nil => simp [runRoundsFrom]
  | cons message first inductionHypothesis =>
      simp [runRoundsFrom, inductionHypothesis]

/-- Transcript-derived NC challenge point and outgoing state. -/
structure Derived (domain : FlatNcDomain) (State : Type uState) where
  challengePoint : CubePoint K (roundCount domain)
  finalState : State

/-- Enter the NC phase and replay every exact-width message in canonical finite
index order. -/
def derive
    {State : Type uState}
    {domain : FlatNcDomain}
    (machine : Machine State)
    (initialState : State)
    (certificate : Certificate domain) : Derived domain State :=
  let result := runRoundsFrom machine (machine.enterNc initialState)
    (List.ofFn certificate.rounds)
  {
    challengePoint := {
      coordinates := result.1
      dimension := by
        rw [runRoundsFrom_challenges_length]
        simp
    }
    finalState := result.2
  }

/-- The derived point contains exactly one challenge per NC variable. -/
@[simp] theorem derive_challenges_length
    {State : Type uState}
    {domain : FlatNcDomain}
    (machine : Machine State)
    (initialState : State)
    (certificate : Certificate domain) :
    (derive machine initialState certificate).challengePoint.coordinates.length =
      roundCount domain :=
  (derive machine initialState certificate).challengePoint.dimension

/-- Logical NC acceptance under the challenges derived from this exact
certificate replay. The initial and terminal claims remain explicit verifier
inputs. -/
def Accepted
    {State : Type uState}
    {domain : FlatNcDomain}
    (machine : Machine State)
    (transcriptState : State)
    (initial terminal : K)
    (certificate : Certificate domain) : Prop :=
  let derived := derive machine transcriptState certificate
  Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.Accepted
    initial derived.challengePoint.coordinates terminal certificate.toSumCheck

/-- Executable NC claimed-chain checker under transcript-derived challenges. -/
def check
    {State : Type uState}
    {domain : FlatNcDomain}
    (machine : Machine State)
    (transcriptState : State)
    (initial terminal : K)
    (certificate : Certificate domain) : Bool :=
  let derived := derive machine transcriptState certificate
  Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.check
    initial derived.challengePoint.coordinates terminal certificate.toSumCheck

/-- The transcript-bound executable checker is exactly its logical
claimed-chain relation. -/
theorem check_eq_true_iff_accepted
    {State : Type uState}
    {domain : FlatNcDomain}
    (machine : Machine State)
    (transcriptState : State)
    (initial terminal : K)
    (certificate : Certificate domain) :
    check machine transcriptState initial terminal certificate = true <->
      Accepted machine transcriptState initial terminal certificate := by
  simp only [check, Accepted]
  exact
    Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.check_eq_true_iff_accepted
      initial (derive machine transcriptState certificate).challengePoint.coordinates
      terminal certificate.toSumCheck

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc
