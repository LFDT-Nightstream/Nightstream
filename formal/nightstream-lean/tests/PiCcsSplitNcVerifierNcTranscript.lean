import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc

/-!
Focused regressions for sequential exact-width NC transcript replay.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.nc.transcript.certificate` | two variables project to two five-slot messages | variable-width or list-shaped certificate drift |
| `nifs.pi_ccs.nc.transcript.round` | every absorb precedes its squeeze and successor state threads forward | pre-message challenge sampling or state reset |
| `nifs.pi_ccs.nc.transcript.point` | two messages derive a two-coordinate point | challenge/message count mismatch |
| `nifs.pi_ccs.nc.transcript.chain` | executable checking uses the derived point | transcript/checker disconnect |
-/

namespace NightstreamTests.PiCcsSplitNcVerifierNcTranscript

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc

private def zeroRound : RoundMessage where
  coefficients := [K.zero, K.zero, K.zero, K.zero, K.zero]
  coefficients_length := rfl

private def twoRoundDomain : FlatNcDomain where
  columnVariables := 1
  laneVariables := 1

private def twoRoundCertificate : Certificate twoRoundDomain where
  rounds := fun _ => zeroRound

private inductive Event where
  | enter
  | absorb
  | squeeze
deriving Repr, DecidableEq

private def traceMachine : Machine (List Event) where
  enterNc state := state ++ [.enter]
  absorbRound state _ := state ++ [.absorb]
  squeezeChallenge state := (K.zero, state ++ [.squeeze])

/-- Exact certificate shape survives projection to the algebra checker. -/
example : twoRoundCertificate.toSumCheck.rounds.length = 2 := by
  simp [twoRoundCertificate, twoRoundDomain, roundCount]

/-- Every statically carried NC round retains all five coefficient slots. -/
example (round : Fin 2) :
    (twoRoundCertificate.rounds round).toRaw.coefficients.length = 5 := by
  simp [twoRoundCertificate]

/-- The observable schedule is enter, then absorb/squeeze for each message;
the second absorb receives the first squeeze's successor state. -/
example :
    (derive traceMachine [] twoRoundCertificate).finalState =
      [.enter, .absorb, .squeeze, .absorb, .squeeze] := by
  decide

/-- Exactly one challenge is derived per fixed certificate message. -/
example :
    (derive traceMachine [] twoRoundCertificate).challengePoint.coordinates.length =
      2 := by
  exact derive_challenges_length traceMachine [] twoRoundCertificate

/-- Prefix replay determines the exact state from which suffix replay starts. -/
example :
    runRoundsFrom traceMachine [] ([zeroRound] ++ [zeroRound]) =
      let prefixResult := runRoundsFrom traceMachine [] [zeroRound]
      let suffixResult := runRoundsFrom traceMachine prefixResult.2 [zeroRound]
      (prefixResult.1 ++ suffixResult.1, suffixResult.2) :=
  runRoundsFrom_append traceMachine [] [zeroRound] [zeroRound]

/-- No caller-supplied challenge vector can override transcript-derived
checking. -/
example (initial terminal : K) :
    check traceMachine [] initial terminal twoRoundCertificate = true ↔
      Accepted traceMachine [] initial terminal twoRoundCertificate :=
  check_eq_true_iff_accepted traceMachine [] initial terminal
    twoRoundCertificate

end NightstreamTests.PiCcsSplitNcVerifierNcTranscript
