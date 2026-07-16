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
output evaluations; derivation of every alpha/gamma/SumCheck challenge from
the verifier transcript; checking against `ProtocolPolynomial` rather than a
residual-table MLE; and absorption of the complete output message into the
outgoing transcript state after the final SumCheck challenge.

Does not own: construction of protocol image tables from production CCS data,
projection of output values from a concrete CE proof object, semantic degree
bounds, a concrete Poseidon2 encoding, random-oracle security, Pi_RLC handoff
refinement, Rust, R1CS, or constraint counts.

Emits constraints: no.

Authority boundary: the certificate carries no challenges, point, terminal,
degree, or outgoing state. The verifier derives the point, computes the
terminal from typed output values, and derives the outgoing state by absorbing
those same values. The abstract oracle can still collide; concrete Poseidon2
refinement and its security statement remain separate obligations.

| Protocol | Phase | Family | Mathematical obligation |
|---|---|---|---|
| `Pi_CCS` | pre-SumCheck | alpha / gamma | derived by `FiatShamir.derive` from verifier context |
| `Pi_CCS` | SumCheck rounds | messages / challenges | each message is absorbed before its challenge is squeezed |
| `Pi_CCS` | terminal | nonlinear `F` / `NC` / `Eval` / `Q` | verifier computes `ProtocolPolynomial.terminalFromMessage` at the derived point |
| `Pi_CCS` | outgoing transcript | complete output message | verifier calls `absorbOutput`; certificate cannot supply final state |
| assurance | executable correspondence | checker iff finite accepted relation | `check_eq_true_iff_accepted` |
| assurance | deterministic reduction | table truth or explicit mixing/round/output bad event | `check_implies_tableTruth_or_badEvent` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolVerifier

open Nightstream.SuperNeo.SumCheck
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

universe uContext uField uState

/-- The verifier transcript schedule plus the mandatory post-SumCheck output
absorption used for the outgoing state. -/
structure Oracle
    (Context : Type uContext)
    (Field : Type uField)
    (State : Type uState)
    (shape : Shape) where
  transcript : FiatShamir.Oracle Context Field State shape
  absorbOutput : State -> ProtocolPolynomial.OutputMessage Field shape -> State

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

/-- Verifier-derived coins and the outgoing state after binding the output
message. -/
structure Derived
    (Field : Type uField)
    (State : Type uState)
    (shape : Shape) where
  coins : FiatShamir.DerivedCoins Field State shape
  outgoingState : State

/-- Replay the challenge schedule, then absorb the typed output values into
the state handed to the next protocol phase. -/
def derive
    {Context : Type uContext}
    {Field : Type uField}
    {State : Type uState}
    {shape : Shape}
    (oracle : Oracle Context Field State shape)
    (context : Context)
    (certificate : Certificate Field shape) : Derived Field State shape :=
  let coins := FiatShamir.derive oracle.transcript context certificate.toTranscript
  {
    coins := coins
    outgoingState := oracle.absorbOutput coins.finalState certificate.output
  }

/-- The challenge vector depends only on the verifier context and round
messages, while the outgoing state additionally binds the output message. -/
theorem derive_coins_eq_transcript
    {Context : Type uContext}
    {Field : Type uField}
    {State : Type uState}
    {shape : Shape}
    (oracle : Oracle Context Field State shape)
    (context : Context)
    (certificate : Certificate Field shape) :
    (derive oracle context certificate).coins =
      FiatShamir.derive oracle.transcript context certificate.toTranscript := by
  rfl

/-- The prover cannot nominate the outgoing state: it is exactly the
verifier's output-message absorb applied to the replayed round state. -/
theorem derive_outgoingState_eq_absorbOutput
    {Context : Type uContext}
    {Field : Type uField}
    {State : Type uState}
    {shape : Shape}
    (oracle : Oracle Context Field State shape)
    (context : Context)
    (certificate : Certificate Field shape) :
    (derive oracle context certificate).outgoingState =
      oracle.absorbOutput
        (FiatShamir.derive oracle.transcript context
          certificate.toTranscript).finalState
        certificate.output := by
  rfl

/-- The actual protocol checker: transcript-derived coins, nonlinear paper
terminal, typed output message, and finite round messages. -/
def check
    {Context : Type uContext}
    {Field : Type uField}
    {State : Type uState}
    [DecidableEq Field]
    {shape : Shape}
    (oracle : Oracle Context Field State shape)
    (context : Context)
    (ops : InterpolationOps Field)
    (data : ProtocolPolynomial.Data Field shape)
    (maxDegree : Nat)
    (certificate : Certificate Field shape) : Bool :=
  let execution := derive oracle context certificate
  ProtocolPolynomial.check ops data execution.coins.alpha
    execution.coins.gamma maxDegree execution.coins.roundPoint
    certificate.output certificate.toFinite

/-- Exact executable/logical correspondence for the transcript-bound actual
paper-polynomial checker. -/
theorem check_eq_true_iff_accepted
    {Context : Type uContext}
    {Field : Type uField}
    {State : Type uState}
    [DecidableEq Field]
    {shape : Shape}
    (oracle : Oracle Context Field State shape)
    (context : Context)
    (ops : InterpolationOps Field)
    (data : ProtocolPolynomial.Data Field shape)
    (maxDegree : Nat)
    (certificate : Certificate Field shape) :
    check oracle context ops data maxDegree certificate = true <->
      let execution := derive oracle context certificate
      SumCheck.Finite.Accepted ops.toOps maxDegree
        (SumCheckInitial.verifierInitial ops (data.toJointData ops)
          execution.coins.gamma)
        execution.coins.roundPoint.coordinates
        (ProtocolPolynomial.terminalFromMessage ops data
          execution.coins.alpha execution.coins.gamma
          execution.coins.roundPoint certificate.output)
        certificate.toFinite := by
  unfold check
  exact ProtocolPolynomial.check_eq_true_iff_accepted ops data
    (derive oracle context certificate).coins.alpha
    (derive oracle context certificate).coins.gamma maxDegree
    (derive oracle context certificate).coins.roundPoint certificate.output
    certificate.toFinite

/-- Completeness from the exact finite relation under the coins replayed by
the verifier. -/
theorem check_complete_of_accepted
    {Context : Type uContext}
    {Field : Type uField}
    {State : Type uState}
    [DecidableEq Field]
    {shape : Shape}
    (oracle : Oracle Context Field State shape)
    (context : Context)
    (ops : InterpolationOps Field)
    (data : ProtocolPolynomial.Data Field shape)
    (maxDegree : Nat)
    (certificate : Certificate Field shape)
    (accepted :
      let execution := derive oracle context certificate
      SumCheck.Finite.Accepted ops.toOps maxDegree
        (SumCheckInitial.verifierInitial ops (data.toJointData ops)
          execution.coins.gamma)
        execution.coins.roundPoint.coordinates
        (ProtocolPolynomial.terminalFromMessage ops data
          execution.coins.alpha execution.coins.gamma
          execution.coins.roundPoint certificate.output)
        certificate.toFinite) :
    check oracle context ops data maxDegree certificate = true :=
  (check_eq_true_iff_accepted oracle context ops data maxDegree certificate).2
    accepted

/-- Deterministic soundness boundary for the transcript-bound actual protocol
polynomial. Acceptance reaches independent residual-table truth or exposes a
specific algebraic, SumCheck, or output-message bad event. -/
theorem check_implies_tableTruth_or_badEvent
    {Context : Type uContext}
    {Field : Type uField}
    {State : Type uState}
    [DecidableEq Field]
    {shape : Shape}
    (oracle : Oracle Context Field State shape)
    (context : Context)
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (zeroLaws : InterpolationZeroLaws ops)
    (data : ProtocolPolynomial.Data Field shape)
    (maxDegree challengeSetSize : Nat)
    (certificate : Certificate Field shape)
    (checked : check oracle context ops data maxDegree certificate = true) :
    let execution := derive oracle context certificate
    (TableResidualData.toTableObligations ops
        (SignedCoefficientObject.toTableResidualData ops
          (data.toJointData ops))).AllHold \/
      SignedCoefficientObject.MixingRoot ops (data.toJointData ops)
        execution.coins.alpha execution.coins.gamma \/
      (exists round,
        SumCheck.BadChallenge
          (SumCheckInitial.symbolicInstance ops (data.toJointData ops)
            execution.coins.alpha execution.coins.gamma maxDegree
            challengeSetSize execution.coins.roundPoint.coordinates
            (ProtocolPolynomial.terminalFromMessage ops data
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
    (derive oracle context certificate).coins.alpha
    (derive oracle context certificate).coins.gamma maxDegree challengeSetSize
    (derive oracle context certificate).coins.roundPoint certificate.output
    certificate.toFinite checked

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolVerifier
