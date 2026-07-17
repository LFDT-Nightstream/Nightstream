import Nightstream.SuperNeo.SumCheck.FixedPhase

/-!
Focused regressions for the ghost-free fixed-layout SumCheck verifier.

| Property | Failure caught |
|---|---|
| five coefficient slots remain verifier-visible | accidental canonical trimming or variable-width acceptance |
| executable and logical acceptance coincide | checker/specification drift |
| expected rounds come only from explicit `q` | prover-carried semantic ghosts |
| honest fixed rounds imply representability | hidden degree premise in soundness |
| representability yields honest acceptance | incomplete fixed-width construction |
| false accepted claim yields a bad challenge | lost deterministic soundness projection |
-/

namespace NightstreamTests.SumCheckFixedPhase

open Nightstream.SuperNeo.SumCheck.Finite
open Nightstream.SuperNeo.SumCheck.Finite.FixedPhase

private def natOps : Ops Nat where
  zero := 0
  one := 1
  add := Nat.add
  mul := Nat.mul

/-- One-coordinate polynomial `q(x) = x`. -/
private def coordinate : List Nat -> Nat
  | value :: _ => value
  | [] => 0

/-- The same affine polynomial in an exact quartic layout. Its three high zero
slots are intentional protocol data, not redundant encoding. -/
private def paddedCoordinate : FixedPolynomial Nat 4 where
  coefficients := [0, 1, 0, 0, 0]
  coefficients_length := rfl

private def honestCertificate : Certificate Nat 4 where
  rounds := [paddedCoordinate]

/-- Exact-width high zero slots are accepted without a canonical-list rule. -/
example : check natOps coordinate 1 [3] honestCertificate = true := by
  decide

/-- The executable result is exactly the logical fixed-phase predicate. -/
example : Accepted natOps coordinate 1 [3] honestCertificate := by
  exact (check_eq_true_iff_accepted natOps coordinate 1 [3]
    honestCertificate).1 (by decide)

/-- The expected affine round derived from `q(x) = x` is representable in the
quartic layout, including its fixed high zero slots. -/
example : ExpectedRoundsRepresentable natOps coordinate 4 [3] := by
  intro expected expectedIn
  simp [expectedRounds, HypercubeTruth.expectedPolynomials,
    HypercubeTruth.expectedPolynomialsFrom] at expectedIn
  subst expected
  refine ⟨paddedCoordinate, ?_⟩
  intro point
  simp [FixedPolynomial.evaluate, FixedPolynomial.toMessage,
    paddedCoordinate, Message.evaluate, Message.evaluateCoefficients,
    natOps, coordinate, HypercubeTruth.sumCompletions]

/-- Honesty is sufficient evidence for the fixed-degree semantic premise; the
soundness theorem does not need a second caller-supplied representability
witness. -/
example
    (honest : Honest natOps coordinate [3] honestCertificate) :
    ExpectedRoundsRepresentable natOps coordinate 4 [3] :=
  expectedRoundsRepresentable_of_honest natOps coordinate [3]
    honestCertificate honest

/-- The generic constructor supplies an honest accepted certificate whenever
all independently derived rounds are representable. -/
example :
    ∃ certificate : Certificate Nat 4,
      Honest natOps coordinate [3] certificate ∧
      Accepted natOps coordinate
        (semanticInitial natOps coordinate 1) [3] certificate := by
  apply exists_honest_accepted_certificate
  intro expected expectedIn
  simp [expectedRounds, HypercubeTruth.expectedPolynomials,
    HypercubeTruth.expectedPolynomialsFrom] at expectedIn
  subst expected
  refine ⟨paddedCoordinate, ?_⟩
  intro point
  simp [FixedPolynomial.evaluate, FixedPolynomial.toMessage,
    paddedCoordinate, Message.evaluate, Message.evaluateCoefficients,
    natOps, coordinate, HypercubeTruth.sumCompletions]

private def paddedZero : FixedPolynomial Nat 4 where
  coefficients := [0, 0, 0, 0, 0]
  coefficients_length := rfl

private def falseCertificate : Certificate Nat 4 where
  rounds := [paddedZero]

/-- At challenge zero, the false zero polynomial collides with the expected
`q(x) = x` round. The generic reduction finds that round deterministically. -/
example :
    ∃ round,
      BadChallenge natOps coordinate 4 17 0 [0] falseCertificate round := by
  apply false_acceptance_implies_bad_challenge natOps coordinate 17 0 [0]
    falseCertificate
  · intro expected expectedIn
    simp [expectedRounds, HypercubeTruth.expectedPolynomials,
      HypercubeTruth.expectedPolynomialsFrom] at expectedIn
    subst expected
    refine ⟨paddedCoordinate, ?_⟩
    intro point
    simp [FixedPolynomial.evaluate, FixedPolynomial.toMessage,
      paddedCoordinate, Message.evaluate, Message.evaluateCoefficients,
      natOps, coordinate, HypercubeTruth.sumCompletions]
  · exact (check_eq_true_iff_accepted natOps coordinate 0 [0]
      falseCertificate).1 (by decide)
  · decide

/-- The bounded collision exposes both exact-width polynomial witnesses, not
only two extensionally unequal functions. This is the interface consumed by a
later root-count theorem. -/
example :
    ∃ (round : Nightstream.SuperNeo.SumCheck.Round Nat Nat)
        (claimedPolynomial expectedPolynomial : FixedPolynomial Nat 4),
      AlgebraicBadChallenge natOps coordinate 4 17 0 [0]
          falseCertificate round ∧
        Represents natOps claimedPolynomial round.claimed ∧
        Represents natOps expectedPolynomial round.expected := by
  rcases false_acceptance_implies_bad_challenge natOps coordinate 17 0 [0]
      falseCertificate (by
        intro expected expectedIn
        simp [expectedRounds, HypercubeTruth.expectedPolynomials,
          HypercubeTruth.expectedPolynomialsFrom] at expectedIn
        subst expected
        refine ⟨paddedCoordinate, ?_⟩
        intro point
        simp [FixedPolynomial.evaluate, FixedPolynomial.toMessage,
          paddedCoordinate, Message.evaluate, Message.evaluateCoefficients,
          natOps, coordinate, HypercubeTruth.sumCompletions])
      ((check_eq_true_iff_accepted natOps coordinate 0 [0]
        falseCertificate).1 (by decide)) (by decide) with
    ⟨round, algebraicBad, claimedPolynomial, expectedPolynomial,
      claimedRepresentation, expectedRepresentation⟩
  exact ⟨round, claimedPolynomial, expectedPolynomial, algebraicBad,
    claimedRepresentation, expectedRepresentation⟩

end NightstreamTests.SumCheckFixedPhase
