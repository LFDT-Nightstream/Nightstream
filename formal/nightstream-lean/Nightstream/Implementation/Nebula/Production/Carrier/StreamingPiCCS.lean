import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolVerifier

/-!
Contract: bounded-round continuation semantics for the production PiCCS
verifier.

Assurance tier: model-level exact refinement.

Owns an interleaved message-before-challenge SumCheck checker, exact replay
equivalence to the existing monolithic finite checker, and the typed PiCCS
wrapper that preserves the paper verifier result and outgoing transcript
state.

Does not own generated rows, a continuation-state codec, Poseidon2
refinement, Rust refinement, recursive lifecycle integration, costs, or a
final relation size claim.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.ProductionStreamingPiCcs

open Nightstream.SuperNeo.SumCheck
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

universe uField uState

/-- Result of a bounded round suffix. The current claim is consumed inside
the checker, so only the verifier decision, final transcript state, and exact
challenge prefix remain. -/
structure RoundResult (Field : Type uField) (State : Type uState) where
  accepted : Bool
  transcriptState : State
  point : List Field

/-- Check one suffix of a finite SumCheck while deriving every challenge only
after the corresponding polynomial message has been absorbed. -/
def checkRoundsFrom
    {Field : Type uField}
    {State : Type uState}
    [DecidableEq Field]
    {shape : Shape}
    (oracle : FiatShamir.Oracle
      (ProtocolVerifier.Statement Field State shape) Field State shape)
    (ops : Finite.Ops Field)
    (maxDegree : Nat)
    (rounds : Fin shape.cubeVariables -> Finite.Message Field)
    (terminal : List Field -> Field) :
    State -> Field -> List Field -> List (Fin shape.cubeVariables) ->
      RoundResult Field State
  | state, current, point, [] =>
      {
        accepted := decide (current = terminal point)
        transcriptState := state
        point := point
      }
  | state, current, point, round :: remaining =>
      let message := rounds round
      let absorbed := oracle.absorbRound state round message
      let sample := oracle.squeeze absorbed (.sumcheck round)
      let tail := checkRoundsFrom oracle ops maxDegree rounds terminal
        sample.2 (message.evaluate ops sample.1)
        (point ++ [sample.1]) remaining
      {
        accepted :=
          message.canonicalCheck ops &&
          decide (message.degreeUpperBound <= maxDegree) &&
          decide (current = ops.add
            (message.evaluate ops ops.zero)
            (message.evaluate ops ops.one)) &&
          tail.accepted
        transcriptState := tail.transcriptState
        point := tail.point
      }

/-- The bounded checker is exactly the existing finite checker over the
challenge suffix produced by the same transcript replay. -/
theorem checkRoundsFrom_exact
    {Field : Type uField}
    {State : Type uState}
    [DecidableEq Field]
    {shape : Shape}
    (oracle : FiatShamir.Oracle
      (ProtocolVerifier.Statement Field State shape) Field State shape)
    (ops : Finite.Ops Field)
    (maxDegree : Nat)
    (rounds : Fin shape.cubeVariables -> Finite.Message Field)
    (terminal : List Field -> Field)
    (state : State)
    (current : Field)
    (challengesSoFar : List Field)
    (indices : List (Fin shape.cubeVariables)) :
    let streamed := checkRoundsFrom oracle ops maxDegree rounds terminal
      state current challengesSoFar indices
    let replay := FiatShamir.deriveRoundsFrom oracle rounds state indices
    streamed.transcriptState = replay.2 /\
      streamed.point = challengesSoFar ++ replay.1 /\
      streamed.accepted =
        Finite.checkChain ops maxDegree current
          (indices.map rounds) replay.1
          (terminal (challengesSoFar ++ replay.1)) := by
  induction indices generalizing state current challengesSoFar with
  | nil =>
      simp [checkRoundsFrom, FiatShamir.deriveRoundsFrom,
        Finite.checkChain]
  | cons round remaining inductionHypothesis =>
      simp [checkRoundsFrom, FiatShamir.deriveRoundsFrom,
        Finite.checkChain, inductionHypothesis, List.append_assoc]

/-- Total list-to-point conversion. A wrong length maps to the canonical zero
point. The production schedule proves that this fallback is unreachable. -/
def cubePointOrZero
    {Field : Type uField}
    {variables : Nat}
    (zero : Field)
    (coordinates : List Field) : CubePoint Field variables :=
  if dimension : coordinates.length = variables then
    { coordinates, dimension }
  else
    {
      coordinates := List.replicate variables zero
      dimension := by simp
    }

@[simp] theorem cubePointOrZero_coordinates_of_length
    {Field : Type uField}
    {variables : Nat}
    (zero : Field)
    (coordinates : List Field)
    (dimension : coordinates.length = variables) :
    (cubePointOrZero (variables := variables) zero coordinates).coordinates =
      coordinates := by
  simp [cubePointOrZero, dimension]

/-- Complete bounded-round execution for the typed PiCCS verifier. -/
structure Execution
    (Field : Type uField)
    (State : Type uState)
    (shape : Shape) where
  alpha : CubePoint Field shape.cubeVariables
  gamma : Field
  rounds : RoundResult Field State
  outgoingState : State

/-- Run alpha/gamma derivation once, check each SumCheck round in order, and
absorb the typed output only after the final round. -/
def derive
    {Field : Type uField}
    {State : Type uState}
    [DecidableEq Field]
    {shape : Shape}
    (oracle : ProtocolVerifier.Oracle Field State shape)
    (priorState : State)
    (ops : InterpolationOps Field)
    (input : ProtocolPolynomial.VerifierInput Field shape)
    (certificate : ProtocolVerifier.Certificate Field shape) :
    Execution Field State shape :=
  let statement : ProtocolVerifier.Statement Field State shape :=
    { priorState, input }
  let pre := FiatShamir.derivePreSumcheck oracle.transcript statement
  let terminal := fun coordinates =>
    ProtocolPolynomial.terminalFromMessage ops input pre.alpha pre.gamma
      (cubePointOrZero ops.toOps.zero coordinates) certificate.output
  let rounds := checkRoundsFrom oracle.transcript ops.toOps
    input.sumcheckDegreeBound certificate.rounds terminal pre.state
    (input.initial ops pre.gamma) []
    (canonicalFinIndices shape.cubeVariables)
  {
    alpha := pre.alpha
    gamma := pre.gamma
    rounds := rounds
    outgoingState :=
      oracle.absorbOutput rounds.transcriptState certificate.output
  }

/-- The bounded verifier accepts exactly when all bounded rounds and the
verifier-computed terminal accept. -/
def check
    {Field : Type uField}
    {State : Type uState}
    [DecidableEq Field]
    {shape : Shape}
    (oracle : ProtocolVerifier.Oracle Field State shape)
    (priorState : State)
    (ops : InterpolationOps Field)
    (input : ProtocolPolynomial.VerifierInput Field shape)
    (certificate : ProtocolVerifier.Certificate Field shape) : Bool :=
  (derive oracle priorState ops input certificate).rounds.accepted

/-- The bounded execution preserves the final round state, complete challenge
point, and post-output transcript state of the existing monolithic replay. -/
theorem derive_transcript_exact
    {Field : Type uField}
    {State : Type uState}
    [DecidableEq Field]
    {shape : Shape}
    (oracle : ProtocolVerifier.Oracle Field State shape)
    (priorState : State)
    (ops : InterpolationOps Field)
    (input : ProtocolPolynomial.VerifierInput Field shape)
    (certificate : ProtocolVerifier.Certificate Field shape) :
    let streamed := derive oracle priorState ops input certificate
    let monolithic := ProtocolVerifier.derive oracle priorState input certificate
    streamed.rounds.transcriptState = monolithic.coins.finalState /\
      streamed.rounds.point = monolithic.coins.roundPoint.coordinates /\
      streamed.outgoingState = monolithic.outgoingState := by
  let statement : ProtocolVerifier.Statement Field State shape :=
    { priorState, input }
  let pre := FiatShamir.derivePreSumcheck oracle.transcript statement
  let indices := canonicalFinIndices shape.cubeVariables
  let replay := FiatShamir.deriveRoundsFrom oracle.transcript
    certificate.rounds pre.state indices
  let terminal := fun coordinates =>
    ProtocolPolynomial.terminalFromMessage ops input pre.alpha pre.gamma
      (cubePointOrZero ops.toOps.zero coordinates) certificate.output
  let streamedRounds := checkRoundsFrom oracle.transcript ops.toOps
    input.sumcheckDegreeBound certificate.rounds terminal pre.state
    (input.initial ops pre.gamma) [] indices
  have exactReplay := checkRoundsFrom_exact oracle.transcript ops.toOps
    input.sumcheckDegreeBound certificate.rounds terminal pre.state
    (input.initial ops pre.gamma) [] indices
  change streamedRounds.transcriptState = replay.2 /\
    streamedRounds.point = replay.1 /\
    oracle.absorbOutput streamedRounds.transcriptState certificate.output =
      oracle.absorbOutput replay.2 certificate.output
  exact
    ⟨exactReplay.1,
      by simpa using exactReplay.2.1,
      congrArg
        (fun state => oracle.absorbOutput state certificate.output)
        exactReplay.1⟩

/-- The bounded checker has exactly the same acceptance decision as the
existing monolithic protocol verifier. -/
theorem check_eq_protocolVerifier_check
    {Field : Type uField}
    {State : Type uState}
    [DecidableEq Field]
    {shape : Shape}
    (oracle : ProtocolVerifier.Oracle Field State shape)
    (priorState : State)
    (ops : InterpolationOps Field)
    (input : ProtocolPolynomial.VerifierInput Field shape)
    (certificate : ProtocolVerifier.Certificate Field shape) :
    check oracle priorState ops input certificate =
      ProtocolVerifier.check oracle priorState ops input certificate := by
  let statement : ProtocolVerifier.Statement Field State shape :=
    { priorState, input }
  let pre := FiatShamir.derivePreSumcheck oracle.transcript statement
  let indices := canonicalFinIndices shape.cubeVariables
  let replay := FiatShamir.deriveRoundsFrom oracle.transcript
    certificate.rounds pre.state indices
  let terminal := fun coordinates =>
    ProtocolPolynomial.terminalFromMessage ops input pre.alpha pre.gamma
      (cubePointOrZero ops.toOps.zero coordinates) certificate.output
  let streamed := checkRoundsFrom oracle.transcript ops.toOps
    input.sumcheckDegreeBound certificate.rounds terminal pre.state
    (input.initial ops pre.gamma) [] indices
  have exactReplay := checkRoundsFrom_exact oracle.transcript ops.toOps
    input.sumcheckDegreeBound certificate.rounds terminal pre.state
    (input.initial ops pre.gamma) [] indices
  have replayLength : replay.1.length = shape.cubeVariables := by
    dsimp only [replay]
    rw [FiatShamir.deriveRoundsFrom_values_length]
    exact canonicalFinIndices_length shape.cubeVariables
  let replayPoint : CubePoint Field shape.cubeVariables :=
    { coordinates := replay.1, dimension := replayLength }
  have replayPointExact :
      cubePointOrZero ops.toOps.zero replay.1 = replayPoint := by
    simp [cubePointOrZero, replayLength, replayPoint]
  have messageOrder : indices.map certificate.rounds =
      certificate.toFinite.rounds := by
    simp [indices, canonicalFinIndices,
      ProtocolVerifier.Certificate.toFinite,
      ProtocolVerifier.Certificate.toTranscript,
      FiatShamir.Certificate.toFinite, Function.comp_def]
  change streamed.accepted = _
  rw [exactReplay.2.2]
  simp only [List.nil_append]
  rw [messageOrder]
  unfold ProtocolVerifier.check ProtocolPolynomial.check Finite.check
  dsimp only [ProtocolVerifier.derive, FiatShamir.derive]
  change
    Finite.checkChain ops.toOps input.sumcheckDegreeBound
        (input.initial ops pre.gamma) certificate.toFinite.rounds replay.1
        (terminal replay.1) =
      Finite.checkChain ops.toOps input.sumcheckDegreeBound
        (input.initial ops pre.gamma) certificate.toFinite.rounds replay.1
        (ProtocolPolynomial.terminalFromMessage ops input pre.alpha pre.gamma
          replayPoint certificate.output)
  unfold terminal
  rw [replayPointExact]

/-- The existing deterministic PiCCS soundness reduction transfers without
change because the bounded and monolithic acceptance decisions are equal. -/
theorem check_implies_tableTruth_or_badEvent
    {Field : Type uField}
    {State : Type uState}
    [DecidableEq Field]
    {shape : Shape}
    (oracle : ProtocolVerifier.Oracle Field State shape)
    (priorState : State)
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (zeroLaws : InterpolationZeroLaws ops)
    (data : ProtocolPolynomial.Data Field shape)
    (challengeSetSize : Nat)
    (certificate : ProtocolVerifier.Certificate Field shape)
    (checked : check oracle priorState ops data.toVerifierInput
      certificate = true) :
    let execution := ProtocolVerifier.derive oracle priorState
      data.toVerifierInput certificate
    (TableResidualData.toTableObligations ops
        (SignedCoefficientObject.toTableResidualData ops
          (data.toJointData ops))).AllHold \/
      SignedCoefficientObject.MixingRoot ops (data.toJointData ops)
        execution.coins.alpha execution.coins.gamma \/
      (exists round,
        Nightstream.SuperNeo.SumCheck.BadChallenge
          (SumCheckInitial.symbolicInstance ops (data.toJointData ops)
            execution.coins.alpha execution.coins.gamma
            data.toVerifierInput.sumcheckDegreeBound
            challengeSetSize execution.coins.roundPoint.coordinates
            (ProtocolPolynomial.terminalFromMessage ops data.toVerifierInput
              execution.coins.alpha execution.coins.gamma
              execution.coins.roundPoint certificate.output)
            certificate.toFinite
            (ProtocolPolynomial.canonicalExpected ops data
              execution.coins.alpha execution.coins.gamma
              execution.coins.roundPoint.coordinates))
          round) \/
      ProtocolPolynomial.OutputMismatch ops data execution.coins.alpha
        execution.coins.gamma execution.coins.roundPoint certificate.output := by
  have monolithicChecked :
      ProtocolVerifier.check oracle priorState ops data.toVerifierInput
        certificate = true := by
    rw [← check_eq_protocolVerifier_check]
    exact checked
  exact ProtocolVerifier.check_implies_tableTruth_or_badEvent oracle
    priorState ops laws zeroLaws data challengeSetSize certificate
    monolithicChecked

end Nightstream.Implementation.Nebula.ProductionStreamingPiCcs
