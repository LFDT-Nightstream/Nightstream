import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.FiatShamir
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ProtocolPolynomial

/-! Provenance: adapted from
`formal/nightstream-lean/Nightstream/SuperNeo/Folding/PiCCS/PaperJoint/ProtocolVerifier.lean`
at commit `fb7a8a99aefbb8ebb5474681ecf80f1b95a1b7a2`; retained only the exact
statement-and-round transcript replay and moved it to the canonical v1.1
authority namespace. -/

/-!
Verifier-owned transcript replay for canonical SuperNeo v1.1 `Pi_CCS`.

Protocol: SuperNeo `Pi_CCS` (Section 7.3 / Appendix D.4).
Phase: statement absorption, round-message absorption, and challenge
derivation through the final SumCheck round. This file emits no rows.

Owns: the typed PiCCS proof-message carrier, the exact projection of its round
messages into Fiat--Shamir replay, a statement carrier containing the prior
verifier state and complete public polynomial input, and derivation of every
alpha/gamma/SumCheck challenge from that statement and those rounds.

Does not own: hidden semantic assignments or image tables, construction of
production PiCCS inputs, executable acceptance, terminal checking,
post-SumCheck output absorption, semantic degree bounds, a concrete Poseidon2
encoding, random-oracle security, Pi_RLC handoff refinement, Rust, R1CS, or
counts.

Emits constraints: no.

Authority boundary: the certificate carries no challenges, point, terminal,
degree, or transcript state. The statement passed to transcript initialization
contains the prior state and the complete public polynomial input; there is no
arbitrary caller context beside it. `derive` projects only the round messages;
the typed output does not affect a challenge. The abstract transcript may
still ignore or collide on its arguments. Whole-replay challenge/state
collisions are therefore named below; Poseidon2 refinement and their security
bounds remain explicit obligations. The NIFS key separately owns the one
complete `y'` absorption.

| Protocol | Phase | Family | Mathematical obligation |
|---|---|---|---|
| `Pi_CCS` | pre-SumCheck | alpha / gamma | derived by `FiatShamir.derive` from the complete typed statement |
| `Pi_CCS` | statement | prior state / complete public polynomial input | `Statement`; passed together to transcript initialization |
| `Pi_CCS` | verifier input | structure / prior point / public claims / degree | `ProtocolPolynomial.VerifierInput`; degree derives from explicit terms |
| `Pi_CCS` | SumCheck rounds | messages / challenges | each message is absorbed before its challenge is squeezed |
| assurance | transcript binding boundary | statement/messages determine challenges and final round state | `TranscriptReplayCollision`, `TranscriptStateCollision` |
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.TranscriptReplay

open NightstreamFPrime.Spec.SumCheck
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

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

/-- The verifier-owned transcript schedule through the final SumCheck
challenge. The NIFS key owns post-SumCheck output absorption. -/
structure Oracle
    (Field : Type uField)
    (State : Type uState)
    (shape : Shape) where
  transcript : FiatShamir.Oracle (Statement Field State shape) Field State shape

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
round polynomial message. The NIFS key binds the separate complete output
after this replay. -/
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

/-- Verifier-derived coins from the exact statement and round schedule. -/
structure Derived
    (Field : Type uField)
    (State : Type uState)
    (shape : Shape) where
  coins : FiatShamir.DerivedCoins Field State shape

/-- Replay the complete challenge schedule. -/
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

/-- The challenge vector is computed from the complete statement and round
messages. -/
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
