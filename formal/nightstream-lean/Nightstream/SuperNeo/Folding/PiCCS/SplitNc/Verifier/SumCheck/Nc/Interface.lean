import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.Algebra
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Degree.Parameters
import Nightstream.SuperNeo.SumCheck.FixedPhase

/-!
Exact-width physical SumCheck interface for the independent Split-NC NC
polynomial.

Owns: the five-coefficient round carrier, exact raw decoding, typed
certificate codec, and verifier-visible claimed-chain checker.

Does not own: the NC polynomial, degree proof, semantic terminal binding,
completeness, soundness events, transcript derivation, Poseidon2, Rust, R1CS,
rows, removals, or costs.

Emits constraints: no.

Authority boundary: a prover supplies exactly five coefficients per round.
The verifier owns initial values, challenges, and terminals. Invalid widths
are rejected before they can enter the typed checker.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc.sumcheck.message.decode` | accept exactly five constant-first coefficients | checked | `RoundMessage.decode_isSome_iff` |
| `nifs.pi_ccs.nc.sumcheck.certificate.decode` | reject a certificate containing any malformed round | checked | `Certificate.ofRaw` |
| `nifs.pi_ccs.nc.sumcheck.chain` | replay one fixed-width polynomial per verifier challenge | checked | `Accepted`, `check` |
| `nifs.pi_ccs.nc.sumcheck.terminal` | final forwarded claim equals a verifier-owned terminal | checked | `Accepted` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.SumCheck.Finite

private abbrev ops := ConcreteCarrier.extensionOps
private abbrev RawMessage :=
  Nightstream.SuperNeo.SumCheck.Finite.Message K
private abbrev RawCertificate :=
  Nightstream.SuperNeo.SumCheck.Finite.Certificate K

/-- One NC round has exactly the five constant-first slots justified by the
independent degree-at-most-four theorem. The highest slot may be zero. -/
abbrev RoundMessage :=
  Nightstream.SuperNeo.SumCheck.Finite.FixedPolynomial K
    Polynomial.Nc.Degree.ncSumcheckDegreeBound

namespace RoundMessage

/-- Erase the static width for the serialized verifier boundary. -/
def toRaw (message : RoundMessage) : RawMessage :=
  message.toMessage

/-- Every serialized typed round has exactly five coefficient slots. -/
@[simp] theorem toRaw_coefficients_length
    (message : RoundMessage) :
    message.toRaw.coefficients.length = 5 := by
  simp [toRaw, Polynomial.Nc.Degree.ncSumcheckDegreeBound]

/-- The raw list-derived upper bound of every typed round is four. This does
not assert that the highest coefficient is nonzero. -/
@[simp] theorem toRaw_degreeUpperBound
    (message : RoundMessage) :
    message.toRaw.degreeUpperBound = 4 := by
  simp [toRaw, Polynomial.Nc.Degree.ncSumcheckDegreeBound]

/-- Parse one serialized round only when it contains exactly five slots. -/
def decode (raw : RawMessage) : Option RoundMessage :=
  if length : raw.coefficients.length = Polynomial.Nc.Degree.ncMessageWidth then
    some {
      coefficients := raw.coefficients
      coefficients_length := by
        simpa [Polynomial.Nc.Degree.ncMessageWidth] using length }
  else
    none

/-- Raw acceptance is exactly the protocol-specific five-slot language. -/
@[simp] theorem decode_isSome_iff
    (raw : RawMessage) :
    (decode raw).isSome = true ↔
      raw.coefficients.length = Polynomial.Nc.Degree.ncMessageWidth := by
  unfold decode
  split <;> simp_all

/-- Erasing and reparsing a typed round is lossless. -/
@[simp] theorem decode_toRaw
    (message : RoundMessage) :
    decode message.toRaw = some message := by
  simp only [decode, toRaw,
    Nightstream.SuperNeo.SumCheck.Finite.FixedPolynomial.toMessage]
  have length :
      message.coefficients.length = Polynomial.Nc.Degree.ncMessageWidth := by
    simpa [Polynomial.Nc.Degree.ncMessageWidth] using
      message.coefficients_length
  simp [length]

/-- Typed evaluation is the same constant-first Horner machine used at the
raw verifier boundary. -/
theorem evaluate_eq_raw
    (message : RoundMessage)
    (point : K) :
    message.evaluate ops.toOps point =
      message.toRaw.evaluate ops.toOps point :=
  rfl

end RoundMessage

/-- Typed NC certificate. Its only data is the generic fixed-phase list of
five-slot round polynomials. -/
abbrev Certificate :=
  FixedPhase.Certificate K Polynomial.Nc.Degree.ncSumcheckDegreeBound

namespace Certificate

private def decodeRounds : List RawMessage -> Option (List RoundMessage)
  | [] => some []
  | raw :: raws => do
      let round <- RoundMessage.decode raw
      let rounds <- decodeRounds raws
      pure (round :: rounds)

/-- Serialize a typed certificate without changing coefficient order. -/
def toRaw (certificate : Certificate) : RawCertificate where
  rounds := certificate.rounds.map RoundMessage.toRaw

/-- Parse a raw certificate, rejecting the whole value if any round is not
exactly five slots. -/
def ofRaw (certificate : RawCertificate) : Option Certificate := do
  let rounds <- decodeRounds certificate.rounds
  pure { rounds }

private theorem decodeRounds_map_toRaw
    (rounds : List RoundMessage) :
    decodeRounds (rounds.map RoundMessage.toRaw) = some rounds := by
  induction rounds with
  | nil => rfl
  | cons round rounds inductionHypothesis =>
      simp [decodeRounds, inductionHypothesis]

/-- Certificate serialization and exact-width parsing are inverse. -/
@[simp] theorem ofRaw_toRaw
    (certificate : Certificate) :
    ofRaw certificate.toRaw = some certificate := by
  cases certificate with
  | mk rounds =>
      simp [ofRaw, toRaw, decodeRounds_map_toRaw]

end Certificate

/-- Verifier-visible logical acceptance. The terminal is an explicit
verifier-owned value; no semantic source data enters this surface. -/
def Accepted
    (initial : K)
    (challenges : List K)
    (terminal : K)
    (certificate : Certificate) : Prop :=
  FixedPhase.Chain ops.toOps initial certificate.rounds challenges terminal

/-- Executable NC claimed-chain verifier. Width has already been checked by
the typed/raw boundary. -/
def check
    (initial : K)
    (challenges : List K)
    (terminal : K)
    (certificate : Certificate) : Bool :=
  FixedPhase.checkChain ops.toOps initial certificate.rounds challenges terminal

/-- The protocol-specific executable verifier is exactly its logical
verifier-visible relation. -/
theorem check_eq_true_iff_accepted
    (initial : K)
    (challenges : List K)
    (terminal : K)
    (certificate : Certificate) :
    check initial challenges terminal certificate = true ↔
      Accepted initial challenges terminal certificate :=
  FixedPhase.checkChain_eq_true_iff ops.toOps initial terminal
    certificate.rounds challenges

/-- Raw verifier boundary: reject malformed widths before protocol-specific
typed replay. -/
def checkRaw
    (initial : K)
    (challenges : List K)
    (terminal : K)
    (certificate : RawCertificate) : Bool :=
  match Certificate.ofRaw certificate with
  | none => false
  | some typed => check initial challenges terminal typed

/-- A well-typed certificate has identical raw and typed acceptance. -/
@[simp] theorem checkRaw_toRaw
    (initial : K)
    (challenges : List K)
    (terminal : K)
    (certificate : Certificate) :
    checkRaw initial challenges terminal certificate.toRaw =
      check initial challenges terminal certificate := by
  simp [checkRaw]

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc
