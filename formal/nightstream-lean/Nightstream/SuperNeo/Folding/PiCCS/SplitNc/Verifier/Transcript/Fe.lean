import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.Interface

/-!
Sequential verifier-owned transcript replay for mixed-width Split-NC FE
SumCheck.

Owns: the abstract FE transcript state machine, one phase entry, physical
row-then-lane message replay, message-before-challenge order, direct state
threading across the row/lane cut, and construction of the typed FE challenge
point.

Does not own: FE polynomial semantics, degree proofs, claimed-chain algebra,
concrete transcript encoding, Poseidon2, Fiat--Shamir security, Rust, R1CS,
emitted rows, removals, or costs.

Emits constraints: no.

Authority boundary: only `SumCheck.Fe.Certificate.rawRounds` enters this
machine. `uniformRounds` is deliberately absent, so semantic high-zero lane
extensions cannot be absorbed. The machine enters FE once and threads the
row suffix state directly into the first lane absorb; there is no second tag,
prologue, marker, reset, or independently supplied boundary claim.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.fe.transcript.enter` | enter FE exactly once before any round | verifier transcript | `derive` |
| `nifs.pi_ccs.fe.transcript.round.absorb` | absorb each physical message before sampling | verifier transcript | `runRound` |
| `nifs.pi_ccs.fe.transcript.round.challenge` | squeeze one challenge and thread successor state | verifier transcript | `runRoundsFrom` |
| `nifs.pi_ccs.fe.transcript.phase_cut` | lane replay begins from the row replay's exact final state | direct dataflow | `replay_eq_row_then_lane` |
| `nifs.pi_ccs.fe.transcript.point.row` | row challenges are the authoritative prefix | computed | `derive` |
| `nifs.pi_ccs.fe.transcript.point.lane` | lane challenges are the authoritative suffix | computed | `derive` |
| `nifs.pi_ccs.fe.transcript.output` | typed point coordinates and final state are the two projections of the same flat replay | derived | `derive_coordinates_finalState` |
| `nifs.pi_ccs.fe.transcript.chain` | claimed-chain acceptance consumes only derived challenges | checked | `Accepted`, `check` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Fe

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe

universe uState

private abbrev RawMessage :=
  Nightstream.SuperNeo.SumCheck.Finite.Message K

/-- Abstract deterministic FE transcript machine. The concrete refinement
must instantiate `absorbRound` with the exact raw-message length word and
coefficient encoding. -/
structure Machine (State : Type uState) where
  enterFe : State -> State
  absorbRound : State -> RawMessage -> State
  squeezeChallenge : State -> K × State

/-- Absorb one physical message before squeezing its challenge. -/
def runRound
    {State : Type uState}
    (machine : Machine State)
    (state : State)
    (message : RawMessage) : K × State :=
  machine.squeezeChallenge (machine.absorbRound state message)

/-- The round order is definitionally absorb-then-squeeze. -/
theorem runRound_eq_squeeze_after_absorb
    {State : Type uState}
    (machine : Machine State)
    (state : State)
    (message : RawMessage) :
    runRound machine state message =
      machine.squeezeChallenge (machine.absorbRound state message) :=
  rfl

/-- Replay a physical message list in order, threading every squeeze
successor into the next absorb. -/
def runRoundsFrom
    {State : Type uState}
    (machine : Machine State) :
    State -> List RawMessage -> List K × State
  | state, [] => ([], state)
  | state, message :: messages =>
      let sample := runRound machine state message
      let tail := runRoundsFrom machine sample.2 messages
      (sample.1 :: tail.1, tail.2)

/-- Replay produces exactly one challenge per absorbed physical message. -/
theorem runRoundsFrom_challenges_length
    {State : Type uState}
    (machine : Machine State)
    (state : State)
    (messages : List RawMessage) :
    (runRoundsFrom machine state messages).1.length = messages.length := by
  induction messages generalizing state with
  | nil => rfl
  | cons message messages inductionHypothesis =>
      simp only [runRoundsFrom, List.length_cons]
      rw [inductionHypothesis]

/-- Replaying a concatenation starts its suffix from the exact state returned
by its prefix. -/
theorem runRoundsFrom_append
    {State : Type uState}
    (machine : Machine State)
    (state : State)
    (first rest : List RawMessage) :
    runRoundsFrom machine state (first ++ rest) =
      let firstResult := runRoundsFrom machine state first
      let restResult := runRoundsFrom machine firstResult.2 rest
      (firstResult.1 ++ restResult.1, restResult.2) := by
  induction first generalizing state with
  | nil => simp [runRoundsFrom]
  | cons message first inductionHypothesis =>
      simp [runRoundsFrom, inductionHypothesis]

/-- The concrete FE phase split is only a view of one flat replay: lane
messages start from the exact row final state, without re-entering FE. -/
theorem replay_eq_row_then_lane
    {State : Type uState}
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {input : PublicInput shape}
    (machine : Machine State)
    (initialState : State)
    (certificate :
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.Certificate
        input domain) :
    runRoundsFrom machine (machine.enterFe initialState)
        certificate.rawRounds =
      let rowResult := runRoundsFrom machine (machine.enterFe initialState)
        certificate.rowRawRounds
      let laneResult := runRoundsFrom machine rowResult.2
        certificate.laneRawRounds
      (rowResult.1 ++ laneResult.1, laneResult.2) := by
  exact runRoundsFrom_append machine (machine.enterFe initialState)
    certificate.rowRawRounds certificate.laneRawRounds

/-- Transcript-derived FE row/lane point and outgoing state. -/
structure Derived
    (shape : SemanticShape)
    (domain : FlatNcDomain)
    (State : Type uState) where
  challengePoint : Point shape domain
  finalState : State

/-- Enter FE once, replay only the physically serialized mixed-width rounds,
and split the resulting challenge vector at the verifier-owned row count. -/
def derive
    {State : Type uState}
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {input : PublicInput shape}
    (machine : Machine State)
    (initialState : State)
    (certificate :
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.Certificate
        input domain) : Derived shape domain State :=
  let result := runRoundsFrom machine (machine.enterFe initialState)
    certificate.rawRounds
  {
    challengePoint := {
      row := {
        coordinates := result.1.take shape.rowVariables
        dimension := by
          rw [List.length_take, runRoundsFrom_challenges_length,
            Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.Certificate.rawRounds_length]
          omega
      }
      lane := {
        coordinates := result.1.drop shape.rowVariables
        dimension := by
          rw [List.length_drop, runRoundsFrom_challenges_length,
            Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.Certificate.rawRounds_length]
          omega
      }
    }
    finalState := result.2
  }

/-- Derived point serialization recovers the flat transcript challenge order
exactly. -/
theorem derive_point_coordinates
    {State : Type uState}
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {input : PublicInput shape}
    (machine : Machine State)
    (initialState : State)
    (certificate :
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.Certificate
        input domain) :
    (derive machine initialState certificate).challengePoint.coordinates =
      (runRoundsFrom machine (machine.enterFe initialState)
        certificate.rawRounds).1 := by
  unfold derive Point.coordinates
  exact List.take_append_drop shape.rowVariables _

/-- The typed point and successor state are jointly the two projections of
one flat physical replay. This is the stable interface used by concrete
transcript refinements; they need not unfold the dependent row/lane point. -/
theorem derive_coordinates_finalState
    {State : Type uState}
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {input : PublicInput shape}
    (machine : Machine State)
    (initialState : State)
    (certificate :
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.Certificate
        input domain) :
    ((derive machine initialState certificate).challengePoint.coordinates,
        (derive machine initialState certificate).finalState) =
      runRoundsFrom machine (machine.enterFe initialState)
        certificate.rawRounds := by
  apply Prod.ext
  · exact derive_point_coordinates machine initialState certificate
  · rfl

/-- Logical FE acceptance under challenges derived from the physical
mixed-width certificate replay. -/
def Accepted
    {State : Type uState}
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {input : PublicInput shape}
    (machine : Machine State)
    (transcriptState : State)
    (initial terminal : K)
    (certificate :
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.Certificate
        input domain) : Prop :=
  let derived := derive machine transcriptState certificate
  Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.Accepted
    initial terminal derived.challengePoint certificate

/-- Executable FE checker under transcript-derived challenges. -/
def check
    {State : Type uState}
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {input : PublicInput shape}
    (machine : Machine State)
    (transcriptState : State)
    (initial terminal : K)
    (certificate :
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.Certificate
        input domain) : Bool :=
  let derived := derive machine transcriptState certificate
  Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.check
    initial terminal derived.challengePoint certificate

/-- Transcript-bound executable and logical FE acceptance coincide. -/
theorem check_eq_true_iff_accepted
    {State : Type uState}
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {input : PublicInput shape}
    (machine : Machine State)
    (transcriptState : State)
    (initial terminal : K)
    (certificate :
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.Certificate
        input domain) :
    check machine transcriptState initial terminal certificate = true <->
      Accepted machine transcriptState initial terminal certificate := by
  simp only [check, Accepted]
  exact
    Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.check_eq_true_iff_accepted
      initial terminal (derive machine transcriptState certificate).challengePoint
      certificate

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Fe
