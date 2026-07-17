import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.SumCheck
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Transport
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.Interface
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Fe

/-!
Model-level refinement from the exact mixed-width semantic FE transcript to
the candidate Poseidon2 transcript execution.

Assurance tier: executable candidate-verifier semantics.

Owns: exact transport of physical FE row and lane messages, proof that lane
rounds contribute exactly three extension coefficients and six base fields,
the concrete semantic-machine instantiation, and equality with concrete FE
round replay.

Does not own: FE polynomial algebra, transcript-prefix authority, the loose
current-production `WellShaped` predicate, equality with the current
uniform-width Rust or R1CS traces, Poseidon2 row soundness, Fiat--Shamir
probability, costs, or row removal.

Emits constraints: no.

Authority boundary: only `Certificate.rawRounds` is transported. The
semantic-only `uniformRounds` projection does not occur in this module, so
its verifier-known high zeros cannot be serialized or absorbed. The FE
prologue is entered once, and concrete replay receives the row list followed
immediately by the lane list.

| Stage path | Concrete owner | Exact obligation | Cost leaf |
|---|---|---|---|
| `nifs.pi_ccs.fe.transcript.prologue` | `SumCheck.fePrologue` | enter FE once with the authoritative initial claim | prologue raw absorbs |
| `nifs.pi_ccs.fe.transcript.row.coefficients` | `concreteRowRounds` | transport `Drow + 1` physical coefficients | row coefficient payload |
| `nifs.pi_ccs.fe.transcript.lane.coefficients` | `concreteLaneRounds` | transport exactly three physical coefficients | six base-field payload fields |
| `nifs.pi_ccs.fe.transcript.message.length_word` | `Primitives.appendRaw` | absorb the physical payload length before its fields | message length word |
| `nifs.pi_ccs.fe.transcript.phase_cut` | `concreteRounds_eq_row_then_lane`, `concreteReplay_eq_row_then_lane` | lane replay starts at the exact row successor without marker or reset | zero-cost ordering node |
| `nifs.pi_ccs.fe.transcript.challenge` | `machine`, `runRound_refines` | absorb before squeezing exactly one extension challenge | squeeze/permutation/output pair |
| `nifs.pi_ccs.fe.transcript.phase` | `derive_refines_runFe` | typed semantic derive and concrete FE replay share one flat challenge/state result | phase composition |
-/

namespace Nightstream.Implementation.R1CS.PiCcsTranscript.FeRefinement

open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.Implementation.R1CS.PiCcsTranscript.Primitives
open Nightstream.Implementation.R1CS.PiCcsTranscript.Transport
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Fe

private abbrev RawMessage :=
  Nightstream.SuperNeo.SumCheck.Finite.Message K

private abbrev FeCertificate
    {shape : SemanticShape}
    (input : PublicInput shape)
    (domain : FlatNcDomain) :=
  Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.Certificate
    input domain

/-- Transport one physical semantic message without trimming or widening its
coefficient list. -/
def toConcreteRound (message : RawMessage) : SumCheck.RoundMessage where
  coefficients := message.coefficients.map toExtension

/-- Transport preserves the physical extension-coefficient count exactly. -/
@[simp] theorem toConcreteRound_coefficients_length (message : RawMessage) :
    (toConcreteRound message).coefficients.length =
      message.coefficients.length := by
  simp [toConcreteRound]

/-- Every physical semantic coefficient contributes exactly two concrete
base fields. -/
@[simp] theorem toConcreteRound_fields_length (message : RawMessage) :
    (SumCheck.roundFields (toConcreteRound message)).length =
      2 * message.coefficients.length := by
  simp [SumCheck.roundFields]

/-- Canonical concrete row-round list. -/
def concreteRowRounds
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (certificate : FeCertificate input domain) :
    List SumCheck.RoundMessage :=
  certificate.rowRawRounds.map toConcreteRound

/-- Canonical concrete lane-round list. -/
def concreteLaneRounds
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (certificate : FeCertificate input domain) :
    List SumCheck.RoundMessage :=
  certificate.laneRawRounds.map toConcreteRound

/-- Canonical concrete FE list, preserving the sole physical serialization
order. -/
def concreteRounds
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (certificate : FeCertificate input domain) :
    List SumCheck.RoundMessage :=
  certificate.rawRounds.map toConcreteRound

/-- Concrete transport is exactly the row prefix followed by the lane suffix. -/
theorem concreteRounds_eq_row_then_lane
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (certificate : FeCertificate input domain) :
    concreteRounds certificate =
      concreteRowRounds certificate ++ concreteLaneRounds certificate := by
  simp [concreteRounds, concreteRowRounds, concreteLaneRounds,
    Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.Certificate.rawRounds]

/-- Concrete projection preserves the exact FE round count. -/
@[simp] theorem concreteRounds_length
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (certificate : FeCertificate input domain) :
    (concreteRounds certificate).length =
      shape.rowVariables + domain.laneVariables := by
  simp [concreteRounds]

/-- Every transported row payload contains exactly twice the syntax-derived
extension width. -/
theorem concreteRowRound_coefficients_length
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (certificate : FeCertificate input domain)
    (message : RawMessage)
    (member : message ∈ certificate.rowRawRounds) :
    (toConcreteRound message).coefficients.length =
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.Drow
        input + 1 := by
  rw [toConcreteRound_coefficients_length]
  exact
    Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.Certificate.rowRawRounds_width
      certificate message member

/-- Every transported row payload contains exactly twice the syntax-derived
extension width in base-field coordinates. -/
theorem concreteRowRound_fields_length
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (certificate : FeCertificate input domain)
    (message : RawMessage)
    (member : message ∈ certificate.rowRawRounds) :
    (SumCheck.roundFields (toConcreteRound message)).length =
      2 * (Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.Drow
        input + 1) := by
  rw [toConcreteRound_fields_length]
  rw [Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.Certificate.rowRawRounds_width
    certificate message member]

/-- Every transported lane payload is exactly three extension coefficients,
hence exactly six base fields. -/
theorem concreteLaneRound_coefficients_length
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (certificate : FeCertificate input domain)
    (message : RawMessage)
    (member : message ∈ certificate.laneRawRounds) :
    (toConcreteRound message).coefficients.length = 3 := by
  rw [toConcreteRound_coefficients_length]
  exact
    Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.Certificate.laneRawRounds_width
      certificate message member

/-- Three physical lane coefficients become exactly six base-field payload
elements before `appendRaw` prepends the authoritative length word. -/
theorem concreteLaneRound_fields_length
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (certificate : FeCertificate input domain)
    (message : RawMessage)
    (member : message ∈ certificate.laneRawRounds) :
    (SumCheck.roundFields (toConcreteRound message)).length = 6 := by
  rw [toConcreteRound_fields_length]
  rw [Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.Certificate.laneRawRounds_width
    certificate message member]

/-- Concrete execution preserves the physical row/lane cut: lane replay
starts from the exact row successor state, with no second prologue or hidden
boundary absorb. -/
theorem concreteReplay_eq_row_then_lane
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (initial : State)
    (certificate : FeCertificate input domain) :
    SumCheck.runRounds initial (concreteRounds certificate) =
      let rowResult := SumCheck.runRounds initial
        (concreteRowRounds certificate)
      let laneResult := SumCheck.runRounds rowResult.1
        (concreteLaneRounds certificate)
      (laneResult.1, rowResult.2 ++ laneResult.2) := by
  rw [concreteRounds_eq_row_then_lane]
  exact SumCheck.runRounds_append initial
    (concreteRowRounds certificate) (concreteLaneRounds certificate)

/-- Production-shaped instantiation of the independent semantic FE machine.
The claimed initial value parameterizes only the one FE prologue. -/
def machine (claimed : K) : Machine State where
  enterFe state := SumCheck.fePrologue state (toExtension claimed)
  absorbRound state message :=
    appendRaw state (SumCheck.roundFields (toConcreteRound message))
  squeezeChallenge state :=
    let response := squeezeN state 2
    (toK (firstExtension response.2), response.1)

/-- One semantic absorb-then-squeeze round is concrete round execution, up to
tuple order and lossless challenge transport. -/
theorem runRound_refines
    (claimed : K)
    (state : State)
    (message : RawMessage) :
    runRound (machine claimed) state message =
      let concrete := SumCheck.runRound state (toConcreteRound message)
      (toK concrete.2, concrete.1) := by
  rfl

/-- Ordered semantic replay equals concrete replay for every physical message
list. -/
theorem runRoundsFrom_refines
    (claimed : K)
    (state : State)
    (messages : List RawMessage) :
    runRoundsFrom (machine claimed) state messages =
      let concrete := SumCheck.runRounds state
        (messages.map toConcreteRound)
      (concrete.2.map toK, concrete.1) := by
  induction messages generalizing state with
  | nil => rfl
  | cons message messages inductionHypothesis =>
      simp [runRoundsFrom, SumCheck.runRounds, runRound_refines,
        inductionHypothesis]

/-- Flat semantic replay equals concrete replay from the exact FE prologue,
jointly for challenge order and successor state. The independent semantic
module separately proves that this flat result is exactly the typed
`derive` point and final state. -/
theorem replay_refines_runRounds
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (initial : State)
    (claimed : K)
    (certificate : FeCertificate input domain) :
    runRoundsFrom (machine claimed)
        (SumCheck.fePrologue initial (toExtension claimed))
        certificate.rawRounds =
      let concrete := SumCheck.runRounds
        (SumCheck.fePrologue initial (toExtension claimed))
        (concreteRounds certificate)
      (concrete.2.map toK, concrete.1) := by
  simpa only [concreteRounds] using
    runRoundsFrom_refines claimed
      (SumCheck.fePrologue initial (toExtension claimed))
      certificate.rawRounds

/-- The same flat equality through `SumCheck.runFe`. Only the FE initial claim
and exact physical FE message list are bound; unrelated NC fields are not
assigned semantic authority. Compose with semantic
`derive_coordinates_finalState` to recover the typed row/lane point. -/
theorem replay_refines_runFe
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (initial : State)
    (claimed : K)
    (certificate : FeCertificate input domain)
    (messages : SumCheck.Messages)
    (initialClaim : messages.feInitial = toExtension claimed)
    (feRounds : messages.feRounds = concreteRounds certificate) :
    runRoundsFrom (machine claimed)
        ((machine claimed).enterFe initial) certificate.rawRounds =
      let concrete := SumCheck.runFe initial messages
      (concrete.2.map toK, concrete.1) := by
  simpa [machine, SumCheck.runFe, initialClaim, feRounds] using
    replay_refines_runRounds initial claimed certificate

/-- The typed FE point and successor state jointly refine the concrete
`runFe` result. Keeping this composition beside the FE replay lemmas avoids
dependent rewriting through the row/lane point in outer schedule proofs. -/
theorem derive_refines_runFe
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (initial : State)
    (claimed : K)
    (certificate : FeCertificate input domain)
    (messages : SumCheck.Messages)
    (initialClaim : messages.feInitial = toExtension claimed)
    (feRounds : messages.feRounds = concreteRounds certificate) :
    ((derive (machine claimed) initial certificate).challengePoint.coordinates,
        (derive (machine claimed) initial certificate).finalState) =
      let concrete := SumCheck.runFe initial messages
      (concrete.2.map toK, concrete.1) := by
  calc
    _ = runRoundsFrom (machine claimed)
          ((machine claimed).enterFe initial) certificate.rawRounds :=
      derive_coordinates_finalState (machine claimed) initial certificate
    _ = _ :=
      replay_refines_runFe initial claimed certificate messages
        initialClaim feRounds

end Nightstream.Implementation.R1CS.PiCcsTranscript.FeRefinement
