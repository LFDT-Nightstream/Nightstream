import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FiatShamir
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolPolynomial

/-!
Transcript-bound verifier for the actual nonlinear paper joint `Pi_CCS`
polynomial.

Protocol: SuperNeo `Pi_CCS` (Section 7.3 / Appendix D.4).
Phase: verifier-owned Fiat--Shamir replay, SumCheck terminal verification, and
outgoing transcript handoff.
Constraint family: semantic verifier composition only; this file emits no
rows.

Owns: one certificate containing only round polynomial messages and typed
output evaluations; a statement carrier containing the prior verifier state
and complete public polynomial input; derivation of every
alpha/gamma/SumCheck challenge from that statement; executable checking from
the minimal `ProtocolPolynomial.VerifierInput`; and an explicit abstract
output-absorption call after the final SumCheck challenge.

Does not own: hidden semantic assignments or image tables, construction of
production PiCCS inputs, projection of output values from a concrete CE
proof object, semantic degree bounds, a concrete Poseidon2 encoding,
random-oracle security, Pi_RLC handoff refinement, Rust, R1CS, or counts.

Emits constraints: no.

Authority boundary: the certificate carries no challenges, point, terminal,
degree, or outgoing state. The statement passed to transcript initialization
contains the prior state and the complete public polynomial input; there is no
arbitrary caller context beside it. The algebraic checker cannot read semantic
assignment/image tables. The abstract transcript and output-absorption
functions may still ignore or collide on their arguments. Whole-replay
challenge/state collisions and whole `(state, output)` absorption collisions
are therefore named below; Poseidon2 refinement and their security bounds
remain explicit obligations.

| Protocol | Phase | Family | Mathematical obligation |
|---|---|---|---|
| `Pi_CCS` | pre-SumCheck | alpha / gamma | derived by `FiatShamir.derive` from the complete typed statement |
| `Pi_CCS` | statement | prior state / complete public polynomial input | `Statement`; passed together to transcript initialization |
| `Pi_CCS` | verifier input | structure / prior point / public claims / degree | `ProtocolPolynomial.VerifierInput`; degree derives from explicit terms |
| `Pi_CCS` | SumCheck rounds | messages / challenges | each message is absorbed before its challenge is squeezed |
| `Pi_CCS` | terminal | nonlinear `F` / `NC` / `Eval` / `Q` | verifier computes `ProtocolPolynomial.terminalFromMessage` at the derived point |
| `Pi_CCS` | outgoing transcript | complete output message | verifier calls `absorbOutput`; certificate cannot supply final state |
| assurance | transcript binding boundary | statement/messages determine challenges and final round state | `TranscriptReplayCollision`, `TranscriptStateCollision` |
| assurance | output binding boundary | complete `(round state, output)` pair determines outgoing state | `OutputAbsorptionCollision` |
| assurance | executable correspondence | checker iff finite accepted relation | `check_eq_true_iff_accepted` |
| assurance | deterministic reduction | table truth or explicit mixing/round/output bad event | `check_implies_tableTruth_or_badEvent` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolVerifier

open Nightstream.SuperNeo.SumCheck
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

universe uField uState

/-- Complete statement handed to abstract transcript initialization. This
removes the former arbitrary `Context` surface: every execution supplies the
prior verifier state and the exact public polynomial input together. -/
structure Statement
    (Field : Type uField)
    (State : Type uState)
    (shape : Shape) where
  priorState : State
  input : ProtocolPolynomial.VerifierInput Field shape

/-- The verifier transcript schedule plus the mandatory post-SumCheck output
absorption used for the outgoing state. -/
structure Oracle
    (Field : Type uField)
    (State : Type uState)
    (shape : Shape) where
  transcript : FiatShamir.Oracle (Statement Field State shape) Field State shape
  absorbOutput : State -> ProtocolPolynomial.OutputMessage Field shape -> State

/-- Explicit failure event when the abstract post-SumCheck operation does not
distinguish two different complete `(incoming state, output message)` pairs.
This catches omission of either the prior transcript or the output payload. -/
def OutputAbsorptionCollision
    {Field : Type uField}
    {State : Type uState}
    {shape : Shape}
    (oracle : Oracle Field State shape)
    (leftState rightState : State)
    (leftMessage rightMessage : ProtocolPolynomial.OutputMessage Field shape) :
    Prop :=
  (leftState, leftMessage) ≠ (rightState, rightMessage) /\
    oracle.absorbOutput leftState leftMessage =
      oracle.absorbOutput rightState rightMessage

/-- Complete prover certificate for this semantic phase. It contains values
and polynomial messages only; all challenges and verifier state are derived. -/
structure Certificate (Field : Type uField) (shape : Shape) where
  rounds : Fin shape.cubeVariables -> SumCheck.Finite.Message Field
  output : ProtocolPolynomial.OutputMessage Field shape

namespace Certificate

/-- Project only the round messages into the transcript-schedule certificate. -/
def toTranscript
    {Field : Type uField}
    {shape : Shape}
    (certificate : Certificate Field shape) :
    FiatShamir.Certificate Field shape where
  rounds := certificate.rounds

/-- Project the same round messages into the finite SumCheck certificate. -/
def toFinite
    {Field : Type uField}
    {shape : Shape}
    (certificate : Certificate Field shape) :
    SumCheck.Finite.Certificate Field :=
  certificate.toTranscript.toFinite

/-- The finite checker sees exactly one message per semantic cube variable. -/
theorem toFinite_rounds_length
    {Field : Type uField}
    {shape : Shape}
    (certificate : Certificate Field shape) :
    certificate.toFinite.rounds.length = shape.cubeVariables :=
  certificate.toTranscript.toFinite_rounds_length

end Certificate

/-- Complete abstract transcript replay input: public statement plus every
round polynomial message. The post-SumCheck output is intentionally excluded;
it is covered by `OutputAbsorptionCollision` together with the final round
state. -/
structure ReplayInput
    (Field : Type uField)
    (State : Type uState)
    (shape : Shape) where
  statement : Statement Field State shape
  rounds : FiatShamir.Certificate Field shape

namespace ReplayInput

/-- Replay the generic transcript from this complete typed input. -/
def derive
    {Field : Type uField}
    {State : Type uState}
    {shape : Shape}
    (oracle : Oracle Field State shape)
    (input : ReplayInput Field State shape) :
    FiatShamir.DerivedCoins Field State shape :=
  FiatShamir.derive oracle.transcript input.statement input.rounds

end ReplayInput

/-- Two distinct replay inputs produce the same complete verifier challenge
view. This catches statement omission, round-message omission, and a squeeze
operation that ignores its incoming state. Concrete Poseidon2 refinement must
reduce this event to the corresponding transcript-security assumption. -/
def TranscriptReplayCollision
    {Field : Type uField}
    {State : Type uState}
    {shape : Shape}
    (oracle : Oracle Field State shape)
    (left right : ReplayInput Field State shape) : Prop :=
  left ≠ right /\
    (left.derive oracle).alpha = (right.derive oracle).alpha /\
    (left.derive oracle).gamma = (right.derive oracle).gamma /\
    (left.derive oracle).roundPoint = (right.derive oracle).roundPoint

/-- Two distinct replay inputs end at the same pre-output transcript state.
This separately names loss of chaining authority even when some sampled
challenge differs. -/
def TranscriptStateCollision
    {Field : Type uField}
    {State : Type uState}
    {shape : Shape}
    (oracle : Oracle Field State shape)
    (left right : ReplayInput Field State shape) : Prop :=
  left ≠ right /\
    (left.derive oracle).finalState = (right.derive oracle).finalState

/-- Verifier-derived coins and the outgoing state after the abstract output
absorption call. Non-omission is a separate refinement obligation. -/
structure Derived
    (Field : Type uField)
    (State : Type uState)
    (shape : Shape) where
  coins : FiatShamir.DerivedCoins Field State shape
  outgoingState : State

/-- Replay the challenge schedule, then pass the typed output values to the
abstract operation producing the state handed to the next protocol phase. -/
def derive
    {Field : Type uField}
    {State : Type uState}
    {shape : Shape}
    (oracle : Oracle Field State shape)
    (priorState : State)
    (input : ProtocolPolynomial.VerifierInput Field shape)
    (certificate : Certificate Field shape) : Derived Field State shape :=
  let statement : Statement Field State shape := { priorState, input }
  let coins := FiatShamir.derive oracle.transcript statement certificate.toTranscript
  {
    coins := coins
    outgoingState := oracle.absorbOutput coins.finalState certificate.output
  }

/-- The challenge vector is computed from the complete statement and round
messages. The outgoing state additionally calls the abstract output operation. -/
theorem derive_coins_eq_transcript
    {Field : Type uField}
    {State : Type uState}
    {shape : Shape}
    (oracle : Oracle Field State shape)
    (priorState : State)
    (input : ProtocolPolynomial.VerifierInput Field shape)
    (certificate : Certificate Field shape) :
    (derive oracle priorState input certificate).coins =
      FiatShamir.derive oracle.transcript
        ({ priorState, input } : Statement Field State shape)
        certificate.toTranscript := by
  rfl

/-- The prover cannot nominate the outgoing state: it is exactly the
configured output operation applied to the replayed round state. This theorem
does not claim that the abstract operation is injective or non-omitting. -/
theorem derive_outgoingState_eq_absorbOutput
    {Field : Type uField}
    {State : Type uState}
    {shape : Shape}
    (oracle : Oracle Field State shape)
    (priorState : State)
    (input : ProtocolPolynomial.VerifierInput Field shape)
    (certificate : Certificate Field shape) :
    (derive oracle priorState input certificate).outgoingState =
      oracle.absorbOutput
        (FiatShamir.derive oracle.transcript
          ({ priorState, input } : Statement Field State shape)
          certificate.toTranscript).finalState
        certificate.output := by
  rfl

/-- The actual protocol checker: transcript-derived coins, nonlinear paper
terminal, typed output message, and finite round messages. -/
def check
    {Field : Type uField}
    {State : Type uState}
    [DecidableEq Field]
    {shape : Shape}
    (oracle : Oracle Field State shape)
    (priorState : State)
    (ops : InterpolationOps Field)
    (input : ProtocolPolynomial.VerifierInput Field shape)
    (certificate : Certificate Field shape) : Bool :=
  let execution := derive oracle priorState input certificate
  ProtocolPolynomial.check ops input execution.coins.alpha
    execution.coins.gamma execution.coins.roundPoint
    certificate.output certificate.toFinite

/-- Exact executable/logical correspondence for the transcript-bound actual
paper-polynomial checker. -/
theorem check_eq_true_iff_accepted
    {Field : Type uField}
    {State : Type uState}
    [DecidableEq Field]
    {shape : Shape}
    (oracle : Oracle Field State shape)
    (priorState : State)
    (ops : InterpolationOps Field)
    (input : ProtocolPolynomial.VerifierInput Field shape)
    (certificate : Certificate Field shape) :
    check oracle priorState ops input certificate = true <->
      let execution := derive oracle priorState input certificate
      SumCheck.Finite.Accepted ops.toOps input.sumcheckDegreeBound
        (input.initial ops execution.coins.gamma)
        execution.coins.roundPoint.coordinates
        (ProtocolPolynomial.terminalFromMessage ops input
          execution.coins.alpha execution.coins.gamma
          execution.coins.roundPoint certificate.output)
        certificate.toFinite := by
  unfold check
  exact ProtocolPolynomial.check_eq_true_iff_accepted ops input
    (derive oracle priorState input certificate).coins.alpha
    (derive oracle priorState input certificate).coins.gamma
    (derive oracle priorState input certificate).coins.roundPoint certificate.output
    certificate.toFinite

/-- Completeness from the exact finite relation under the coins replayed by
the verifier. -/
theorem check_complete_of_accepted
    {Field : Type uField}
    {State : Type uState}
    [DecidableEq Field]
    {shape : Shape}
    (oracle : Oracle Field State shape)
    (priorState : State)
    (ops : InterpolationOps Field)
    (input : ProtocolPolynomial.VerifierInput Field shape)
    (certificate : Certificate Field shape)
    (accepted :
      let execution := derive oracle priorState input certificate
      SumCheck.Finite.Accepted ops.toOps input.sumcheckDegreeBound
        (input.initial ops execution.coins.gamma)
        execution.coins.roundPoint.coordinates
        (ProtocolPolynomial.terminalFromMessage ops input
          execution.coins.alpha execution.coins.gamma
          execution.coins.roundPoint certificate.output)
        certificate.toFinite) :
    check oracle priorState ops input certificate = true :=
  (check_eq_true_iff_accepted oracle priorState ops input certificate).2
    accepted

/-- Deterministic soundness boundary for the transcript-bound actual protocol
polynomial. Acceptance reaches independent residual-table truth or exposes a
specific algebraic, SumCheck, or output-message bad event. -/
theorem check_implies_tableTruth_or_badEvent
    {Field : Type uField}
    {State : Type uState}
    [DecidableEq Field]
    {shape : Shape}
    (oracle : Oracle Field State shape)
    (priorState : State)
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (zeroLaws : InterpolationZeroLaws ops)
    (data : ProtocolPolynomial.Data Field shape)
    (challengeSetSize : Nat)
    (certificate : Certificate Field shape)
    (checked : check oracle priorState ops data.toVerifierInput
      certificate = true) :
    let execution := derive oracle priorState data.toVerifierInput certificate
    (TableResidualData.toTableObligations ops
        (SignedCoefficientObject.toTableResidualData ops
          (data.toJointData ops))).AllHold \/
      SignedCoefficientObject.MixingRoot ops (data.toJointData ops)
        execution.coins.alpha execution.coins.gamma \/
      (exists round,
        SumCheck.BadChallenge
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
  exact ProtocolPolynomial.check_implies_tableTruth_or_badEvent
    ops laws zeroLaws data
    (derive oracle priorState data.toVerifierInput certificate).coins.alpha
    (derive oracle priorState data.toVerifierInput certificate).coins.gamma
    challengeSetSize
    (derive oracle priorState data.toVerifierInput certificate).coins.roundPoint
    certificate.output
    certificate.toFinite checked

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolVerifier
