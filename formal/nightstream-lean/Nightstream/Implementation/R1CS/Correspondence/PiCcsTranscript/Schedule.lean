import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.SumCheck

/-!
Complete production-shaped `Pi_CCS` transcript schedule up to the `Pi_RLC`
handoff.

Assurance tier: executable implementation semantics. This module composes the
independently stated binding, challenge, FE, NC, and catch-up phases. It is the
semantic target for later native, gadget, and generated-row refinement.

Owns: one typed input surface; the only accepted phase order; every named
intermediate state; derived pre-SumCheck and round challenges; the derived
header digest; and uniqueness of any claimed header digest that matches the
replay.

Does not own: authority proofs for the outer initial state or binding fields,
SumCheck algebraic acceptance, paper-joint-to-SplitNc soundness, output-message
digest binding, equality with Rust/R1CS, cost totals, or row removal.

Emits constraints: no.

Authority boundary: `Input` contains no challenge and no header digest. Those
are outputs of deterministic verifier replay. A proof-carried header digest is
accepted only through `HeaderDigestMatches`, never by self-consistency.

| Protocol | Phase | Child owner | Mathematical guarantee |
|---|---|---|---|
| `Pi_CCS` | authority prefix | `Binding.run` | bind header, instance, count, and checked parent before sampling |
| `Pi_CCS` | split coins | `Challenges.run` | derive every production pre-SumCheck coin and successor state |
| `Pi_CCS` | FE SumCheck | `SumCheck.runFe` | absorb finite FE messages and derive one challenge per round |
| `Pi_CCS` | NC SumCheck | `SumCheck.runNc` | absorb finite NC messages and derive one challenge per round |
| `Pi_CCS` | catch-up | `Primitives.catchup` | derive the unique header digest and post-catch-up state |
-/

namespace Nightstream.Implementation.R1CS.PiCcsTranscript.Schedule

open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.Implementation.R1CS.PiCcsTranscript.Primitives

/-- Complete verifier-visible transcript input. The incoming state and binding
fields remain explicit authority obligations for upstream refinement. -/
structure Input where
  initialState : State
  shape : Challenges.Shape
  binding : Binding.Input
  sumcheck : SumCheck.Messages

/-- Exact fixed-shape checks required before interpreting the transcript as a
production `Pi_CCS` execution. -/
def WellShaped (input : Input) : Prop :=
  SumCheck.WellShaped input.shape input.sumcheck

/-- Every meaningful protocol boundary and every verifier-derived response. -/
structure Trace where
  binding : Binding.Trace
  challenges : Challenges.Output
  afterFe : State
  feChallenges : List Extension
  afterNc : State
  ncChallenges : List Extension
  afterCatchup : State
  headerDigest : Fin 4 -> Field

/-- Binding-prefix trace computed from the authoritative incoming state. -/
def bindingTrace (input : Input) : Binding.Trace :=
  Binding.trace input.initialState input.binding

/-- Pre-SumCheck challenges derived from the completed binding prefix. -/
def challengeTrace (input : Input) : Challenges.Output :=
  Challenges.run (Binding.run input.initialState input.binding) input.shape

/-- FE transcript execution from the verifier-derived challenge state. -/
def feTrace (input : Input) : State × List Extension :=
  SumCheck.runFe (challengeTrace input).state input.sumcheck

/-- NC transcript execution from the FE successor state. -/
def ncTrace (input : Input) : State × List Extension :=
  SumCheck.runNc (feTrace input).1 input.sumcheck

/-- Catch-up digest and successor state from the NC successor state. -/
def catchupTrace (input : Input) : State × (Fin 4 -> Field) :=
  Primitives.catchup (ncTrace input).1

/-- Replay the only accepted production-shaped phase order. -/
def run (input : Input) : Trace :=
  { binding := bindingTrace input
    challenges := challengeTrace input
    afterFe := (feTrace input).1
    feChallenges := (feTrace input).2
    afterNc := (ncTrace input).1
    ncChallenges := (ncTrace input).2
    afterCatchup := (catchupTrace input).1
    headerDigest := (catchupTrace input).2 }

/-- Acceptance predicate for the proof-carried four-lane header digest. -/
def HeaderDigestMatches (input : Input) (claimed : Fin 4 -> Field) : Prop :=
  forall lane, claimed lane = (run input).headerDigest lane

@[simp] theorem run_afterBinding (input : Input) :
    (run input).binding.afterParentHandle =
      Binding.run input.initialState input.binding := by
  simpa only [run, bindingTrace] using
    Binding.trace_afterParentHandle input.initialState input.binding

@[simp] theorem run_afterFe (input : Input) :
    (run input).afterFe =
      (SumCheck.runFe (run input).challenges.state input.sumcheck).1 := by
  simp only [run, feTrace]

@[simp] theorem run_afterNc (input : Input) :
    (run input).afterNc =
      (SumCheck.runNc (run input).afterFe input.sumcheck).1 := by
  simp only [run, ncTrace]

/-- The state and digest at the handoff are the two projections of the same
verifier-owned catch-up execution. -/
theorem run_catchup_joint (input : Input) :
    ((run input).afterCatchup, (run input).headerDigest) =
      Primitives.catchup (run input).afterNc := by
  simp only [run, catchupTrace]

/-- A proof cannot choose two different header digests that both match one
verifier replay. -/
theorem headerDigest_unique (input : Input)
    {left right : Fin 4 -> Field}
    (leftMatches : HeaderDigestMatches input left)
    (rightMatches : HeaderDigestMatches input right) :
    left = right := by
  funext lane
  rw [leftMatches lane, rightMatches lane]

/-- The verifier-derived FE response count is fixed by the semantic shape,
not by a carried challenge vector. -/
theorem feChallengeCount (input : Input) (wellShaped : WellShaped input) :
    (run input).feChallenges.length =
      input.shape.ellN + input.shape.ellD := by
  exact SumCheck.runFe_shape _ wellShaped

/-- The verifier-derived NC response count is fixed by the semantic shape. -/
theorem ncChallengeCount (input : Input) (wellShaped : WellShaped input) :
    (run input).ncChallenges.length =
      input.shape.ellM + input.shape.ellD := by
  exact SumCheck.runNc_shape _ wellShaped

/-- The complete schedule is a function. This theorem is intentionally simple:
later refinement must prove concrete execution equals this function rather
than postulating agreement between two independently supplied traces. -/
theorem replay_deterministic (input : Input) (left right : Trace)
    (leftEq : left = run input) (rightEq : right = run input) :
    left = right := by
  rw [leftEq, rightEq]

end Nightstream.Implementation.R1CS.PiCcsTranscript.Schedule
