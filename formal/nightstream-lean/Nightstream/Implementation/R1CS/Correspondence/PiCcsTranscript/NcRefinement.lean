import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.SumCheck
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Transport
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc

/-!
Model-level refinement from the exact-width semantic NC transcript to the
production-shaped Poseidon2 transcript execution.

Owns: explicit transport between semantic `K` and the implementation
extension carrier, exact five-coefficient message serialization, the concrete
semantic-machine instantiation, and equality with concrete NC round replay.

Does not own: SumCheck algebra, terminal authority, `WellShaped`, equality
with Rust or an R1CS trace, Poseidon2 row soundness, Fiat--Shamir probability,
constraint totals, or permission to remove rows.

Emits constraints: no. This file identifies the concrete operations that later
row-refinement and cost accounting must own.

Authority boundary: semantic certificates still contain only fixed-width
messages. This adapter serializes those messages, executes the verifier-owned
NC prologue, absorbs each message, squeezes two base fields, and transports the
derived implementation response back to semantic `K`.

| Stage path | Concrete owner | Exact obligation | Cost leaf |
|---|---|---|---|
| `nifs.pi_ccs.nc.transcript.prologue` | `SumCheck.ncPrologue` | enter NC once before any round | prologue raw absorbs |
| `nifs.pi_ccs.nc.transcript.message.length_word` | `Primitives.appendRaw` | absorb payload length `10` | message length word |
| `nifs.pi_ccs.nc.transcript.message.coefficients` | `toConcreteRound` | five `K` coefficients become ten `(c0,c1)` fields | ten payload fields |
| `nifs.pi_ccs.nc.transcript.message.normalize_full` | `Primitives.appendRaw` | apply native eager full-rate normalization | message normalization |
| `nifs.pi_ccs.nc.transcript.challenge.squeeze_marker` | `Primitives.squeezeN` | absorb the digest marker through `digest` | squeeze marker |
| `nifs.pi_ccs.nc.transcript.challenge.permutation` | `Primitives.squeezeN` | execute the complete response permutation | Poseidon2 permutation |
| `nifs.pi_ccs.nc.transcript.challenge.output_pair` | `Primitives.firstExtension` | lanes zero and one form one `K` challenge | challenge output pair |
-/

namespace Nightstream.Implementation.R1CS.PiCcsTranscript.NcRefinement

open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.Implementation.R1CS.PiCcsTranscript.Primitives
open Nightstream.Implementation.R1CS.PiCcsTranscript.Transport
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc

/-- Serialize one exact-width semantic message into the concrete transcript
message carrier without trimming high zero coefficients. -/
def toConcreteRound (message : RoundMessage) : SumCheck.RoundMessage where
  coefficients := message.coefficients.map toExtension

/-- The implementation carrier receives exactly five extension coefficients. -/
@[simp] theorem toConcreteRound_coefficients_length
    (message : RoundMessage) :
    (toConcreteRound message).coefficients.length = 5 := by
  rw [show (toConcreteRound message).coefficients.length =
    message.coefficients.length by simp [toConcreteRound]]
  simpa [Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Degree.ncSumcheckDegreeBound]
    using message.coefficients_length

/-- Five extension coefficients serialize to exactly ten base-field payload
elements before `appendRaw` adds its authoritative length word. -/
@[simp] theorem toConcreteRound_fields_length
    (message : RoundMessage) :
    (SumCheck.roundFields (toConcreteRound message)).length = 10 := by
  change (extensionFields (toConcreteRound message).coefficients).length = 10
  rw [extensionFields_length, toConcreteRound_coefficients_length]

/-- Canonical concrete message list for one exact semantic certificate. -/
def concreteRounds
    {domain : FlatNcDomain}
    (certificate : Certificate domain) : List SumCheck.RoundMessage :=
  (List.ofFn certificate.rounds).map toConcreteRound

/-- Concrete projection preserves the exact NC round count. -/
@[simp] theorem concreteRounds_length
    {domain : FlatNcDomain}
    (certificate : Certificate domain) :
    (concreteRounds certificate).length = roundCount domain := by
  simp [concreteRounds]

/-- Production-shaped instantiation of the independent semantic NC machine. -/
def machine : Machine State where
  enterNc := SumCheck.ncPrologue
  absorbRound state message :=
    appendRaw state (SumCheck.roundFields (toConcreteRound message))
  squeezeChallenge state :=
    let response := squeezeN state 2
    (toK (firstExtension response.2), response.1)

/-- One semantic absorb-then-squeeze round is the concrete round execution,
with only the tuple order and extension carrier transported. -/
theorem runRound_refines (state : State) (message : RoundMessage) :
    runRound machine state message =
      let concrete := SumCheck.runRound state (toConcreteRound message)
      (toK concrete.2, concrete.1) := by
  rfl

/-- Ordered semantic replay equals concrete `runRounds` for every exact-width
message list. -/
theorem runRoundsFrom_refines
    (state : State)
    (messages : List RoundMessage) :
    runRoundsFrom machine state messages =
      let concrete := SumCheck.runRounds state (messages.map toConcreteRound)
      (concrete.2.map toK, concrete.1) := by
  induction messages generalizing state with
  | nil => rfl
  | cons message messages inductionHypothesis =>
      simp [runRoundsFrom, SumCheck.runRounds, runRound_refines,
        inductionHypothesis]

/-- Full semantic derivation equals concrete replay from the production NC
prologue, jointly for the challenge vector and successor state. -/
theorem derive_refines_runRounds
    {domain : FlatNcDomain}
    (initial : State)
    (certificate : Certificate domain) :
    ((derive machine initial certificate).challengePoint.coordinates,
        (derive machine initial certificate).finalState) =
      let concrete := SumCheck.runRounds (SumCheck.ncPrologue initial)
        (concreteRounds certificate)
      (concrete.2.map toK, concrete.1) := by
  have replay := runRoundsFrom_refines (SumCheck.ncPrologue initial)
    (List.ofFn certificate.rounds)
  apply Prod.ext
  · simpa [derive, machine, concreteRounds] using congrArg Prod.fst replay
  · simpa [derive, machine, concreteRounds] using congrArg Prod.snd replay

/-- The same joint equality through `SumCheck.runNc`. Only the NC message-list
binding is required; unrelated FE fields are not assigned semantic authority. -/
theorem derive_refines_runNc
    {domain : FlatNcDomain}
    (initial : State)
    (certificate : Certificate domain)
    (messages : SumCheck.Messages)
    (ncRounds : messages.ncRounds = concreteRounds certificate) :
    ((derive machine initial certificate).challengePoint.coordinates,
        (derive machine initial certificate).finalState) =
      let concrete := SumCheck.runNc initial messages
      (concrete.2.map toK, concrete.1) := by
  simpa [SumCheck.runNc, ncRounds] using
    derive_refines_runRounds initial certificate

end Nightstream.Implementation.R1CS.PiCcsTranscript.NcRefinement
