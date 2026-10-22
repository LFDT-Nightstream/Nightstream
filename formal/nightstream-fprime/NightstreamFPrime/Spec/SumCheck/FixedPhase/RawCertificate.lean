import NightstreamFPrime.Spec.SumCheck.FixedPhase

/-!
Exact-width transport for the paper SumCheck message relation.

Source: SuperNeo v1.2 Definition 11, Section 7.3, and Appendix B.2.  The paper
message is a polynomial of verifier-bounded degree; it does not require a
canonical variable-length coefficient serialization.

Owns: fail-closed decoding of raw constant-first coefficient lists at one
verifier-owned width, the corresponding executable chain checker, and exact
round trips between typed fixed-width certificates and raw messages.

Does not own: challenge generation, transcript encoding, Fiat--Shamir,
polynomial soundness, Rust, R1CS, artifacts, minimality, or costs.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.SumCheck.Finite.FixedPhase.RawCertificate

universe uField

def decodeMessage
    {Field : Type uField}
    (degree : Nat)
    (message : Message Field) :
    Option (FixedPolynomial Field degree) :=
  if exactWidth : message.coefficients.length = degree + 1 then
    some {
      coefficients := message.coefficients
      coefficients_length := exactWidth
    }
  else
    none

def decodeRounds
    {Field : Type uField}
    (degree : Nat) :
    List (Message Field) -> Option (List (FixedPolynomial Field degree))
  | [] => some []
  | message :: messages => do
      let polynomial <- decodeMessage degree message
      let polynomials <- decodeRounds degree messages
      pure (polynomial :: polynomials)

def decode
    {Field : Type uField}
    (degree : Nat)
    (certificate : NightstreamFPrime.Spec.SumCheck.Finite.Certificate Field) :
    Option (Certificate Field degree) :=
  (decodeRounds degree certificate.rounds).map fun rounds => { rounds }

theorem decodeRounds_eq_some_implies_length
    {Field : Type uField}
    (degree : Nat) :
    forall
      (messages : List (Message Field))
      (polynomials : List (FixedPolynomial Field degree)),
      decodeRounds degree messages = some polynomials ->
        messages.length = polynomials.length
  | [], polynomials, decoded => by
      simp [decodeRounds] at decoded
      subst polynomials
      rfl
  | message :: messages, polynomials, decoded => by
      cases messageDecoded : decodeMessage degree message with
      | none => simp [decodeRounds, messageDecoded] at decoded
      | some polynomial =>
          cases tailDecoded : decodeRounds degree messages with
          | none => simp [decodeRounds, messageDecoded, tailDecoded] at decoded
          | some tail =>
              simp [decodeRounds, messageDecoded, tailDecoded] at decoded
              subst polynomials
              simp [decodeRounds_eq_some_implies_length degree messages tail
                tailDecoded]

theorem decodeRounds_eq_some_get
    {Field : Type uField}
    (degree : Nat) :
    forall
      (messages : List (Message Field))
      (polynomials : List (FixedPolynomial Field degree))
      (decoded : decodeRounds degree messages = some polynomials)
      (index : Fin messages.length),
      decodeMessage degree (messages.get index) =
        some
          (polynomials.get
            (Fin.cast
              (decodeRounds_eq_some_implies_length degree messages
                polynomials decoded)
              index))
  | [], _, _, index => Fin.elim0 index
  | message :: messages, polynomials, decoded, index => by
      cases messageDecoded : decodeMessage degree message with
      | none => simp [decodeRounds, messageDecoded] at decoded
      | some polynomial =>
          cases tailDecoded : decodeRounds degree messages with
          | none => simp [decodeRounds, messageDecoded, tailDecoded] at decoded
          | some tail =>
              have polynomialsEqual :
                  polynomials = polynomial :: tail := by
                have decodedEqual :
                    some (polynomial :: tail) = some polynomials := by
                  simpa [decodeRounds, messageDecoded, tailDecoded] using
                    decoded
                exact (Option.some.inj decodedEqual).symm
              subst polynomials
              refine Fin.cases ?_ ?_ index
              · simpa using messageDecoded
              · intro tailIndex
                simpa using
                  decodeRounds_eq_some_get degree messages tail tailDecoded
                    tailIndex

theorem decode_eq_some_implies_rounds
    {Field : Type uField}
    (degree : Nat)
    (raw : NightstreamFPrime.Spec.SumCheck.Finite.Certificate Field)
    (fixed : Certificate Field degree)
    (decoded : decode degree raw = some fixed) :
    decodeRounds degree raw.rounds = some fixed.rounds := by
  unfold decode at decoded
  cases roundsDecoded : decodeRounds degree raw.rounds with
  | none => simp [roundsDecoded] at decoded
  | some rounds =>
      have fixedEqual : fixed = { rounds } := by
        have decodedEqual :
            some ({ rounds } : Certificate Field degree) = some fixed := by
          simpa [roundsDecoded] using decoded
        exact (Option.some.inj decodedEqual).symm
      subst fixed
      simp at roundsDecoded ⊢

def encode
    {Field : Type uField}
    {degree : Nat}
    (certificate : Certificate Field degree) :
    NightstreamFPrime.Spec.SumCheck.Finite.Certificate Field where
  rounds := certificate.rounds.map FixedPolynomial.toMessage

def check
    {Field : Type uField}
    [DecidableEq Field]
    (ops : Ops Field)
    (degree : Nat)
    (initial : Field)
    (challenges : List Field)
    (terminal : Field)
    (certificate : NightstreamFPrime.Spec.SumCheck.Finite.Certificate Field) :
    Bool :=
  match decode degree certificate with
  | none => false
  | some fixed =>
      FixedPhase.checkChain ops initial fixed.rounds challenges terminal

@[simp] theorem decodeMessage_toMessage
    {Field : Type uField}
    {degree : Nat}
    (polynomial : FixedPolynomial Field degree) :
    decodeMessage degree polynomial.toMessage = some polynomial := by
  unfold decodeMessage
  rw [dif_pos polynomial.toMessage_coefficients_length]
  congr

@[simp] theorem decodeRounds_map_toMessage
    {Field : Type uField}
    {degree : Nat}
    (rounds : List (FixedPolynomial Field degree)) :
    decodeRounds degree (rounds.map FixedPolynomial.toMessage) =
      some rounds := by
  induction rounds with
  | nil => rfl
  | cons polynomial rounds inductionHypothesis =>
      simp [decodeRounds, inductionHypothesis]

@[simp] theorem decode_encode
    {Field : Type uField}
    {degree : Nat}
    (certificate : Certificate Field degree) :
    decode degree (encode certificate) = some certificate := by
  cases certificate
  simp [decode, encode]

@[simp] theorem check_encode
    {Field : Type uField}
    {degree : Nat}
    [DecidableEq Field]
    (ops : Ops Field)
    (initial : Field)
    (challenges : List Field)
    (terminal : Field)
    (certificate : Certificate Field degree) :
    check ops degree initial challenges terminal (encode certificate) =
      FixedPhase.checkChain ops initial certificate.rounds challenges
        terminal := by
  simp [check]

theorem check_eq_true_iff
    {Field : Type uField}
    [DecidableEq Field]
    (ops : Ops Field)
    (degree : Nat)
    (initial : Field)
    (challenges : List Field)
    (terminal : Field)
    (certificate : NightstreamFPrime.Spec.SumCheck.Finite.Certificate Field) :
    check ops degree initial challenges terminal certificate = true <->
      exists fixed : Certificate Field degree,
        decode degree certificate = some fixed /\
          FixedPhase.Chain ops initial fixed.rounds challenges terminal := by
  unfold check
  cases decoded : decode degree certificate with
  | none => simp
  | some fixed =>
      rw [FixedPhase.checkChain_eq_true_iff]
      constructor
      · intro chain
        exact ⟨fixed, rfl, chain⟩
      · rintro ⟨other, equal, chain⟩
        cases equal
        exact chain

end NightstreamFPrime.Spec.SumCheck.Finite.FixedPhase.RawCertificate
