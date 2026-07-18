import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Fe

/-!
Focused regressions for sequential mixed-width FE transcript replay.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.fe.transcript.enter` | FE is entered exactly once | duplicate prologue or phase reset |
| `nifs.pi_ccs.fe.transcript.round` | each physical message is absorbed before its squeeze | challenge-before-message drift |
| `nifs.pi_ccs.fe.transcript.phase_cut` | lane replay receives the row replay's exact final state | hidden boundary marker or state reset |
| `nifs.pi_ccs.fe.transcript.point` | the flat replay splits into one row and one lane challenge | row/lane decoder drift |
| `nifs.pi_ccs.fe.transcript.chain` | executable checking uses only the derived point | caller-supplied challenge authority |
-/

namespace NightstreamTests.PiCcsSplitNcVerifierFeTranscript

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Fe
open Nightstream.SuperNeo.SumCheck.Finite

private abbrev ops := ConcreteCarrier.extensionOps

private def oneRowShape : SemanticShape where
  rowVariables := 1
  logicalWidth := 1
  freshCount := 1
  runningCount := 0
  matrixCount := 1

private def oneLaneDomain : FlatNcDomain where
  columnVariables := 0
  laneVariables := 1

private def zeroRow
    (input : PublicInput oneRowShape) : RowMessage input :=
  FixedPolynomial.zero ops.toOps (Drow input)

private def zeroLane : LaneMessage :=
  FixedPolynomial.zero ops.toOps laneSumcheckDegreeBound

private def twoRoundCertificate
    (input : PublicInput oneRowShape) :
    Certificate input oneLaneDomain where
  rowRounds := fun _ => zeroRow input
  laneRounds := fun _ => zeroLane

private inductive Event where
  | enter
  | absorb
  | squeeze
deriving Repr, DecidableEq

private def traceMachine : Machine (List Event) where
  enterFe state := state ++ [.enter]
  absorbRound state _ := state ++ [.absorb]
  squeezeChallenge state := (K.zero, state ++ [.squeeze])

/-- One entry is followed by row absorb/squeeze and lane absorb/squeeze. -/
example (input : PublicInput oneRowShape) :
    (derive traceMachine [] (twoRoundCertificate input)).finalState =
      [.enter, .absorb, .squeeze, .absorb, .squeeze] := by
  rfl

/-- The typed point preserves the authoritative row/lane split. -/
example (input : PublicInput oneRowShape) :
    (derive traceMachine []
        (twoRoundCertificate input)).challengePoint.row.coordinates.length = 1 ∧
      (derive traceMachine []
        (twoRoundCertificate input)).challengePoint.lane.coordinates.length = 1 := by
  constructor <;> exact CubePoint.dimension _

/-- The flat derived coordinates are exactly the challenges emitted by the
single physical replay. -/
example (input : PublicInput oneRowShape) :
    (derive traceMachine []
        (twoRoundCertificate input)).challengePoint.coordinates =
      (runRoundsFrom traceMachine (traceMachine.enterFe [])
        (twoRoundCertificate input).rawRounds).1 :=
  derive_point_coordinates traceMachine [] (twoRoundCertificate input)

/-- The row/lane presentation is one concatenated replay, not two transcript
phases. -/
example (input : PublicInput oneRowShape) :
    runRoundsFrom traceMachine (traceMachine.enterFe [])
        (twoRoundCertificate input).rawRounds =
      let rowResult := runRoundsFrom traceMachine (traceMachine.enterFe [])
        (twoRoundCertificate input).rowRawRounds
      let laneResult := runRoundsFrom traceMachine rowResult.2
        (twoRoundCertificate input).laneRawRounds
      (rowResult.1 ++ laneResult.1, laneResult.2) :=
  replay_eq_row_then_lane traceMachine [] (twoRoundCertificate input)

/-- No caller-supplied point can override transcript-derived checking. -/
example
    (input : PublicInput oneRowShape)
    (initial terminal : K) :
    check traceMachine [] initial terminal (twoRoundCertificate input) = true ↔
      Accepted traceMachine [] initial terminal (twoRoundCertificate input) :=
  check_eq_true_iff_accepted traceMachine [] initial terminal
    (twoRoundCertificate input)

end NightstreamTests.PiCcsSplitNcVerifierFeTranscript
