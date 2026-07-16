import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Challenges

/-!
Verifier-visible FE and NC SumCheck transcript phases for production-shaped
`Pi_CCS`.

Assurance tier: executable implementation semantics. This module serializes
finite coefficient messages and derives every round challenge from the
transcript. It does not restate the SumCheck acceptance equations.

Owns: coefficient-pair serialization; exact FE/NC domain and initial-claim
prologues; one raw coefficient message plus one two-field squeeze per round;
round-count and degree-bound shape predicates; and the phase successor state.

Does not own: polynomial truth, `g(0)+g(1)=claim`, terminal evaluation,
paper-joint-to-SplitNc refinement, authority of the initial claim or round
messages, native/gadget/R1CS correspondence, costs, or row removal.

Emits constraints: no.

Authority boundary: a prover supplies only finite coefficient messages.
Challenges and state transitions are verifier-derived. `WellShaped` checks
cardinality and degree metadata but is not a SumCheck soundness theorem.

| Protocol | Phase | Constraint family | Exact obligation |
|---|---|---|---|
| `Pi_CCS` | FE prologue | `fePrologue` | raw `[7]`, raw `[9]`, raw initial pair, raw `[10]` |
| `Pi_CCS` | NC prologue | `ncPrologue` | raw `[8]`, raw `[9]`, raw zero pair, raw `[10]` |
| `Pi_CCS` | round message | `roundFields` | flatten every coefficient in `(c0,c1)` order |
| `Pi_CCS` | round challenge | `runRound` | append one raw message and squeeze exactly two fields |
| `Pi_CCS` | shape | `WellShaped` | exact FE/NC round counts and at most `degreeBound+1` coefficients |
-/

namespace Nightstream.Implementation.R1CS.PiCcsTranscript.SumCheck

open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.Implementation.R1CS.PiCcsTranscript.Primitives
open Nightstream.Implementation.R1CS.PiCcsTranscript.Challenges

set_option maxHeartbeats 1000000

/-- One finite prover message. The coefficient list is the only transcript
payload; verifier challenges are deliberately absent. -/
structure RoundMessage where
  coefficients : List Extension
deriving DecidableEq

/-- Two concrete SumCheck channels carried by a `Pi_CCS` proof. -/
structure Messages where
  feInitial : Extension
  feRounds : List RoundMessage
  ncRounds : List RoundMessage
deriving DecidableEq

/-- Fixed-shape checks performed before transcript replay. -/
structure WellShaped (shape : Shape) (messages : Messages) : Prop where
  feRoundCount : messages.feRounds.length = shape.ellN + shape.ellD
  ncRoundCount : messages.ncRounds.length = shape.ellM + shape.ellD
  feDegree : forall round, round ∈ messages.feRounds ->
    round.coefficients.length <= shape.degreeBound + 1
  ncDegree : forall round, round ∈ messages.ncRounds ->
    round.coefficients.length <= shape.degreeBound + 1

/-- Exact base-field payload of one round polynomial. -/
def roundFields (round : RoundMessage) : List Field :=
  extensionFields round.coefficients

private def appendSingleton (state : State) (tag : Nat) : State :=
  appendRaw state [wordField tag]

/-- Four raw messages preceding FE round replay. -/
def fePrologue (initial : State) (claimed : Extension) : State :=
  let afterDomain := appendSingleton initial 7
  let afterInitialTag := appendSingleton afterDomain 9
  let afterInitial := appendRaw afterInitialTag [claimed.c0, claimed.c1]
  appendSingleton afterInitial 10

/-- Four raw messages preceding NC round replay. The initial claim is the
verifier-owned zero extension, not a proof field. -/
def ncPrologue (initial : State) : State :=
  let afterDomain := appendSingleton initial 8
  let afterInitialTag := appendSingleton afterDomain 9
  let afterInitial := appendRaw afterInitialTag
    [Extension.zero.c0, Extension.zero.c1]
  appendSingleton afterInitial 10

/-- One exact round transcript transition and its derived extension challenge. -/
def runRound (initial : State) (round : RoundMessage) : State × Extension :=
  let afterMessage := appendRaw initial (roundFields round)
  let response := squeezeN afterMessage 2
  (response.1, firstExtension response.2)

/-- Replay a complete ordered round list. -/
def runRounds : State -> List RoundMessage -> State × List Extension
  | initial, [] => (initial, [])
  | initial, round :: rest =>
      let current := runRound initial round
      let suffix := runRounds current.1 rest
      (suffix.1, current.2 :: suffix.2)

/-- Complete FE transcript phase. -/
def runFe (initial : State) (messages : Messages) :
    State × List Extension :=
  runRounds (fePrologue initial messages.feInitial) messages.feRounds

/-- Complete NC transcript phase. -/
def runNc (initial : State) (messages : Messages) :
    State × List Extension :=
  runRounds (ncPrologue initial) messages.ncRounds

@[simp] theorem runRounds_challengeCount (initial : State)
    (rounds : List RoundMessage) :
    (runRounds initial rounds).2.length = rounds.length := by
  induction rounds generalizing initial with
  | nil => rfl
  | cons round rest inductionHypothesis =>
      simp only [runRounds, List.length_cons]
      rw [inductionHypothesis]

@[simp] theorem runFe_challengeCount (initial : State)
    (messages : Messages) :
    (runFe initial messages).2.length = messages.feRounds.length := by
  exact runRounds_challengeCount _ _

@[simp] theorem runNc_challengeCount (initial : State)
    (messages : Messages) :
    (runNc initial messages).2.length = messages.ncRounds.length := by
  exact runRounds_challengeCount _ _

/-- Under the verifier-fixed shape, the derived FE point has exactly
`ellN+ellD` extension coordinates. -/
theorem runFe_shape (initial : State) {shape : Shape} {messages : Messages}
    (wellShaped : WellShaped shape messages) :
    (runFe initial messages).2.length = shape.ellN + shape.ellD := by
  rw [runFe_challengeCount, wellShaped.feRoundCount]

/-- Under the verifier-fixed shape, the derived NC point has exactly
`ellM+ellD` extension coordinates. -/
theorem runNc_shape (initial : State) {shape : Shape} {messages : Messages}
    (wellShaped : WellShaped shape messages) :
    (runNc initial messages).2.length = shape.ellM + shape.ellD := by
  rw [runNc_challengeCount, wellShaped.ncRoundCount]

end Nightstream.Implementation.R1CS.PiCcsTranscript.SumCheck
