import NightstreamFPrime.Spec.SumCheck.FixedPhase
import NightstreamFPrime.Spec.SumCheck.FixedPolynomialCanonical
import NightstreamFPrime.Spec.SumCheck.VerifierCertificate

/-! Provenance: copied from `formal/nightstream-lean/Nightstream/SuperNeo/SumCheck/FixedPhase/Canonical.lean`
at commit `fb7a8a99aefbb8ebb5474681ecf80f1b95a1b7a2`; namespaces renamed, otherwise unchanged. -/

/-!
Canonical raw-certificate projection for fixed-width SumCheck.

Owns: mapping every typed fixed-width round to its canonical raw coefficient
message and preserving logical SumCheck acceptance exactly.

Does not own: honest-prover construction, challenges, transcripts, protocol
polynomials, Fiat--Shamir, generated artifacts, Rust, R1CS, or costs.

The projection never weakens raw `Message.Canonical`. High zero coefficients
are removed by `FixedPolynomial.canonicalMessage`; evaluation and degree
preservation come from its kernel-checked theorems.
-/

namespace NightstreamFPrime.Spec.SumCheck.Finite.FixedPhase.Canonical

universe uField

/-- Project a typed fixed-width certificate to raw canonical messages without
changing round order. -/
def toFinite
    {Field : Type uField}
    {degree : Nat}
    [DecidableEq Field]
    (ops : Ops Field)
    (certificate : FixedPhase.Certificate Field degree) :
    SumCheck.Finite.Certificate Field where
  rounds := certificate.rounds.map
    (FixedPolynomial.canonicalMessage ops)

private theorem chain_toFinite
    {Field : Type uField}
    {degree : Nat}
    [DecidableEq Field]
    (ops : Ops Field)
    (laws : FixedPolynomial.Laws ops)
    (current terminal : Field)
    (rounds : List (FixedPolynomial Field degree))
    (challenges : List Field)
    (chain : FixedPhase.Chain ops current rounds challenges terminal) :
    SumCheck.Finite.Chain ops degree current
      (rounds.map (FixedPolynomial.canonicalMessage ops))
      challenges terminal := by
  induction rounds generalizing current challenges with
  | nil =>
      cases challenges with
      | nil => exact chain
      | cons challenge challenges => simp [FixedPhase.Chain] at chain
  | cons polynomial polynomials inductionHypothesis =>
      cases challenges with
      | nil => simp [FixedPhase.Chain] at chain
      | cons challenge challenges =>
          simp only [FixedPhase.Chain] at chain
          simp only [List.map_cons, SumCheck.Finite.Chain]
          refine ⟨FixedPolynomial.canonicalMessage_canonical ops polynomial,
            FixedPolynomial.canonicalMessage_degreeUpperBound_le ops polynomial,
            ?_, ?_⟩
          · calc
              current = ops.add
                  (polynomial.evaluate ops ops.zero)
                  (polynomial.evaluate ops ops.one) := chain.1
              _ = ops.add
                  ((FixedPolynomial.canonicalMessage ops polynomial).evaluate
                    ops ops.zero)
                  ((FixedPolynomial.canonicalMessage ops polynomial).evaluate
                    ops ops.one) := by
                rw [FixedPolynomial.canonicalMessage_evaluate ops laws,
                  FixedPolynomial.canonicalMessage_evaluate ops laws]
          · rw [FixedPolynomial.canonicalMessage_evaluate ops laws]
            exact inductionHypothesis
              (current := polynomial.evaluate ops challenge)
              (challenges := challenges) chain.2

/-- Fixed-width logical acceptance implies raw logical acceptance with every
round independently checked for canonical shape and verifier-derived degree.
The terminal remains the same explicit `q(challenges)` value. -/
theorem accepted_toFinite
    {Field : Type uField}
    {degree : Nat}
    [DecidableEq Field]
    (ops : Ops Field)
    (laws : FixedPolynomial.Laws ops)
    (q : List Field -> Field)
    (initial : Field)
    (challenges : List Field)
    (certificate : FixedPhase.Certificate Field degree)
    (accepted : FixedPhase.Accepted ops q initial challenges certificate) :
    SumCheck.Finite.Accepted ops degree initial challenges (q challenges)
      (toFinite ops certificate) := by
  exact chain_toFinite ops laws initial (q challenges)
    certificate.rounds challenges accepted

end NightstreamFPrime.Spec.SumCheck.Finite.FixedPhase.Canonical
