import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.FiatShamir
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ProtocolPolynomial

/-! SuperNeo v1.2 Section 7.3, Appendix B.2. -/

namespace NightstreamFPrime.Spec.Folding.PiCCS.TranscriptReplay

open NightstreamFPrime.Spec.SumCheck
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

universe uField uState

structure Statement
    (Field : Type uField)
    (State : Type uState)
    (shape : Shape) where
  priorState : State
  input : ProtocolPolynomial.VerifierInput Field shape

structure Oracle
    (Field : Type uField)
    (State : Type uState)
    (shape : Shape) where
  transcript : FiatShamir.Oracle (Statement Field State shape) Field State shape
  initialState_is_prior : ∀ statement,
    transcript.initialState statement = statement.priorState

structure Certificate (Field : Type uField) (shape : Shape) where
  rounds : Fin shape.cubeVariables -> SumCheck.Finite.Message Field
  output : ProtocolPolynomial.OutputMessage Field shape

namespace Certificate

def toTranscript
    {Field : Type uField}
    {shape : Shape}
    (certificate : Certificate Field shape) :
    FiatShamir.Certificate Field shape where
  rounds := certificate.rounds

def toFinite
    {Field : Type uField}
    {shape : Shape}
    (certificate : Certificate Field shape) :
    SumCheck.Finite.Certificate Field :=
  certificate.toTranscript.toFinite

theorem toFinite_rounds_length
    {Field : Type uField}
    {shape : Shape}
    (certificate : Certificate Field shape) :
    certificate.toFinite.rounds.length = shape.cubeVariables :=
  certificate.toTranscript.toFinite_rounds_length

end Certificate

structure ReplayInput
    (Field : Type uField)
    (State : Type uState)
    (shape : Shape) where
  statement : Statement Field State shape
  rounds : FiatShamir.Certificate Field shape

structure ReplayAuthority
    (Field : Type uField)
    (State : Type uState)
    (shape : Shape) where
  priorState : State
  rounds : FiatShamir.Certificate Field shape

namespace ReplayAuthority

def derive
    {Field : Type uField}
    {State : Type uState}
    {shape : Shape}
    (oracle : Oracle Field State shape)
    (authority : ReplayAuthority Field State shape) :
    FiatShamir.DerivedCoins Field State shape :=
  let stateOracle : FiatShamir.Oracle Unit Field State shape :=
    { initialState := fun _ => authority.priorState
      absorbRound := oracle.transcript.absorbRound
      squeeze := oracle.transcript.squeeze }
  FiatShamir.derive stateOracle () authority.rounds

end ReplayAuthority

namespace ReplayInput

def authority
    {Field : Type uField}
    {State : Type uState}
    {shape : Shape}
    (input : ReplayInput Field State shape) :
    ReplayAuthority Field State shape where
  priorState := input.statement.priorState
  rounds := input.rounds

def derive
    {Field : Type uField}
    {State : Type uState}
    {shape : Shape}
    (oracle : Oracle Field State shape)
    (input : ReplayInput Field State shape) :
    FiatShamir.DerivedCoins Field State shape :=
  input.authority.derive oracle

theorem derive_eq_of_authority_eq
    {Field : Type uField}
    {State : Type uState}
    {shape : Shape}
    (oracle : Oracle Field State shape)
    (left right : ReplayInput Field State shape)
    (same : left.authority = right.authority) :
    left.derive oracle = right.derive oracle := by
  exact congrArg (ReplayAuthority.derive oracle) same

end ReplayInput

def TranscriptReplayCollision
    {Field : Type uField}
    {State : Type uState}
    {shape : Shape}
    (oracle : Oracle Field State shape)
    (left right : ReplayInput Field State shape) : Prop :=
  left.authority ≠ right.authority /\
    (left.derive oracle).alpha = (right.derive oracle).alpha /\
    (left.derive oracle).gamma = (right.derive oracle).gamma /\
    (left.derive oracle).roundPoint = (right.derive oracle).roundPoint

def TranscriptStateCollision
    {Field : Type uField}
    {State : Type uState}
    {shape : Shape}
    (oracle : Oracle Field State shape)
    (left right : ReplayInput Field State shape) : Prop :=
  left.authority ≠ right.authority /\
    (left.derive oracle).finalState = (right.derive oracle).finalState

theorem authority_eq_or_challenge_collision
    {Field : Type uField}
    {State : Type uState}
    {shape : Shape}
    (oracle : Oracle Field State shape)
    (left right : ReplayInput Field State shape)
    (alphaEqual : (left.derive oracle).alpha =
      (right.derive oracle).alpha)
    (gammaEqual : (left.derive oracle).gamma =
      (right.derive oracle).gamma)
    (roundPointEqual : (left.derive oracle).roundPoint =
      (right.derive oracle).roundPoint) :
    left.authority = right.authority ∨
      TranscriptReplayCollision oracle left right := by
  classical
  by_cases same : left.authority = right.authority
  · exact Or.inl same
  · exact Or.inr ⟨same, alphaEqual, gammaEqual, roundPointEqual⟩

theorem authority_eq_or_state_collision
    {Field : Type uField}
    {State : Type uState}
    {shape : Shape}
    (oracle : Oracle Field State shape)
    (left right : ReplayInput Field State shape)
    (finalStateEqual : (left.derive oracle).finalState =
      (right.derive oracle).finalState) :
    left.authority = right.authority ∨
      TranscriptStateCollision oracle left right := by
  classical
  by_cases same : left.authority = right.authority
  · exact Or.inl same
  · exact Or.inr ⟨same, finalStateEqual⟩

structure Derived
    (Field : Type uField)
    (State : Type uState)
    (shape : Shape) where
  coins : FiatShamir.DerivedCoins Field State shape

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
  { coins := coins }

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

end NightstreamFPrime.Spec.Folding.PiCCS.TranscriptReplay
