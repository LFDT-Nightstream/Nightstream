import Mathlib.Algebra.Polynomial.Roots
import Mathlib.Algebra.QuadraticAlgebra.Defs
import Mathlib.Algebra.Ring.TransferInstance
import Mathlib.Data.NNRat.Lemmas
import NightstreamFPrime.Spec.GoldilocksExtension
import NightstreamFPrime.Spec.FieldTower
import NightstreamFPrime.Spec.SumCheck.FixedPhase

/-! Root counts for the verifier's actual coefficient lists over `K`.
The probability statement uses a uniform point from the stated finite set,
after both polynomials are fixed. It does not assert hash uniformity. -/

namespace NightstreamFPrime.Spec.SumCheck.Finite.GoldilocksRoots

private def carrierEquiv : K ≃ QuadraticAlgebra (ZMod goldilocksModulus) 7 0 where
  toFun value := ⟨value.c0, value.c1⟩
  invFun value := ⟨value.re, value.im⟩
  left_inv _ := rfl
  right_inv _ := rfl

local instance : CommRing K := carrierEquiv.commRing

private theorem zero_eq : (0 : K) = K.zero := rfl
private theorem add_eq (left right : K) : left + right = K.add left right := rfl
private theorem mul_eq (left right : K) : left * right = K.mul left right := by
  change K.mk _ (left.c0 * right.c1 + left.c1 * right.c0 + 0 * left.c1 * right.c1) = _
  simp only [Fin.zero_mul, Fin.add_zero]
  rfl

local instance : Nontrivial K := ⟨⟨K.zero, K.one, by
  intro same
  have : (0 : F) = 1 := congrArg K.c0 same
  exact (by decide : (0 : F) ≠ 1) this⟩⟩

local instance : NoZeroDivisors K where
  eq_zero_or_eq_zero_of_mul_eq_zero := by
    intro left right productZero
    exact GoldilocksExtension.extensionNoZeroDivisors left right
      (by simpa only [mul_eq, zero_eq] using productZero)

local instance : IsDomain K := NoZeroDivisors.to_isDomain K

/-- The operations used by the concrete extension-field verifier. -/
def ops : Ops K := Folding.PiCCS.PaperJoint.ConcreteCarrier.extensionOps.toOps

private noncomputable def polynomial : List K → Polynomial K
  | [] => 0
  | coefficient :: rest => Polynomial.C coefficient + Polynomial.X * polynomial rest

private theorem polynomial_eval (coefficients : List K) (point : K) :
    (polynomial coefficients).eval point =
      Message.evaluateCoefficients ops point coefficients := by
  induction coefficients with
  | nil => simp only [polynomial, Polynomial.eval_zero, zero_eq]; rfl
  | cons coefficient rest ih =>
    simp only [polynomial, Polynomial.eval_add, Polynomial.eval_C,
      Polynomial.eval_mul, Polynomial.eval_X, ih, add_eq, mul_eq]
    rfl

private theorem polynomial_degree (coefficients : List K) :
    (polynomial coefficients).natDegree ≤ coefficients.length - 1 := by
  induction coefficients with
  | nil => simp [polynomial]
  | cons coefficient rest ih =>
    cases rest with
    | nil => simp [polynomial]
    | cons next rest =>
      have bound := Polynomial.natDegree_add_le (Polynomial.C coefficient)
        (Polynomial.X * polynomial (next :: rest))
      have productBound := Polynomial.natDegree_mul_le
        (p := (Polynomial.X : Polynomial K)) (q := polynomial (next :: rest))
      simp only [Polynomial.natDegree_C, Polynomial.natDegree_X,
        List.length_cons] at bound productBound ih ⊢
      change (Polynomial.C coefficient + Polynomial.X * polynomial (next :: rest)).natDegree ≤ _
      omega

/-- Distinct polynomial functions of degree at most `degree` can agree on at
most `degree` points of any finite challenge set. Degree comes from the
verifier's coefficient width, not from a supplied annotation. -/
theorem agreement_count_le {degree : Nat}
    (claimed expected : FixedPolynomial K degree) (challenges : Finset K)
    (different : ∃ point, claimed.evaluate ops point ≠ expected.evaluate ops point) :
    (challenges.filter fun point =>
      claimed.evaluate ops point = expected.evaluate ops point).card ≤ degree := by
  let difference := polynomial claimed.coefficients - polynomial expected.coefficients
  have evalDifference (point : K) : difference.eval point =
      claimed.evaluate ops point - expected.evaluate ops point := by
    simp only [difference, Polynomial.eval_sub, polynomial_eval,
      FixedPolynomial.evaluate, FixedPolynomial.toMessage, Message.evaluate]
  have nonzero : difference ≠ 0 := by
    intro zero
    obtain ⟨point, different⟩ := different
    apply different
    exact sub_eq_zero.mp (by simpa only [zero, Polynomial.eval_zero] using
      (evalDifference point).symm)
  have degreeBound : difference.natDegree ≤ degree := by
    apply (Polynomial.natDegree_sub_le _ _).trans
    apply max_le
    · simpa only [claimed.coefficients_length, Nat.add_sub_cancel] using
        polynomial_degree claimed.coefficients
    · simpa only [expected.coefficients_length, Nat.add_sub_cancel] using
        polynomial_degree expected.coefficients
  apply (Polynomial.card_le_degree_of_subset_roots (p := difference) ?_).trans degreeBound
  intro point member
  apply (Polynomial.mem_roots nonzero).mpr
  change difference.eval point = 0
  rw [evalDifference, sub_eq_zero]
  exact (Finset.mem_filter.mp member).2

/-- Exact event frequency for uniform sampling from a nonempty finite set.
Both polynomial messages are fixed before this sample is drawn. -/
theorem uniform_agreement_probability_le {degree : Nat}
    (claimed expected : FixedPolynomial K degree) (challenges : Finset K)
    (_nonempty : challenges.Nonempty)
    (different : ∃ point, claimed.evaluate ops point ≠ expected.evaluate ops point) :
    ((challenges.filter fun point =>
      claimed.evaluate ops point = expected.evaluate ops point).card : ℚ≥0) /
      challenges.card ≤ (degree : ℚ≥0) / challenges.card := by
  apply div_le_div_of_nonneg_right _ zero_le
  exact_mod_cast agreement_count_le claimed expected challenges different

@[reducible] private noncomputable def kFintype : Fintype K := Fintype.ofEquiv (F × F) {
  toFun value := ⟨value.1, value.2⟩
  invFun value := (value.c0, value.c1)
  left_inv _ := rfl
  right_inv _ := rfl }

attribute [local instance] kFintype

/-- The whole extension field is the ideal uniform sampling space. -/
noncomputable def fullChallengeSet : Finset K := Finset.univ

theorem fullChallengeSet_card : fullChallengeSet.card = goldilocksModulus ^ 2 := by
  rw [fullChallengeSet, Finset.card_univ, ← Nat.card_eq_fintype_card]
  exact FieldTower.extension_cardinality

theorem uniform_fullField_agreement_probability_le {degree : Nat}
    (claimed expected : FixedPolynomial K degree)
    (different : ∃ point, claimed.evaluate ops point ≠ expected.evaluate ops point) :
    ((fullChallengeSet.filter fun point =>
      claimed.evaluate ops point = expected.evaluate ops point).card : ℚ≥0) /
      (goldilocksModulus ^ 2 : Nat) ≤ (degree : ℚ≥0) / (goldilocksModulus ^ 2 : Nat) := by
  rw [← fullChallengeSet_card]
  exact uniform_agreement_probability_le claimed expected fullChallengeSet
    (by exact Finset.univ_nonempty) different

/-- Connect the existing false-acceptance event to the root count, retaining
its independently proved expected-polynomial representation. -/
theorem badChallenge_count_le {degree challengeSetSize : Nat}
    (q : List K → K) (initial : K) (transcriptChallenges : List K)
    (certificate : FixedPhase.Certificate K degree) (round : SumCheck.Round K K)
    (bad : FixedPhase.BadChallenge ops q degree challengeSetSize initial
      transcriptChallenges certificate round) (sampleSpace : Finset K) :
    (sampleSpace.filter fun point => round.claimed point = round.expected point).card ≤ degree := by
  obtain ⟨algebraic, claimed, expected, claimedRepresents, expectedRepresents⟩ := bad
  dsimp only [FixedPhase.Represents] at claimedRepresents expectedRepresents
  have different : ∃ point, claimed.evaluate ops point ≠ expected.evaluate ops point := by
    by_contra same
    apply algebraic.2.2.1
    funext point
    have atPoint : claimed.evaluate ops point = expected.evaluate ops point := by
      by_contra unequal
      exact same ⟨point, unequal⟩
    rw [claimedRepresents point, expectedRepresents point] at atPoint
    exact atPoint
  have bound := agreement_count_le claimed expected sampleSpace different
  simpa only [claimedRepresents, expectedRepresents] using bound

end NightstreamFPrime.Spec.SumCheck.Finite.GoldilocksRoots
