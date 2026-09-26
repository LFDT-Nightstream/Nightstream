import Mathlib.LinearAlgebra.Lagrange
import NightstreamFPrime.Spec.GoldilocksPrime
import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.RingFPolynomial

/-!
Exact Phi81 product equations for SuperNeo section 7.4 over Goldilocks.
The 108 fixed evaluation points determine a polynomial of degree below 108.
A 54-coefficient quotient is private witness data; these equations preserve
the existing ring product. No random test or security assumption is added.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Phi81Relation.QuotientProduct

open NightstreamFPrime.Spec
open Polynomial
open EvaluationHomomorphism.RingFPolynomial

private abbrev Base := ZMod goldilocksModulus

local instance : Fact (Nat.Prime goldilocksModulus) :=
  ⟨GoldilocksPrime.goldilocks_natPrime⟩

local instance : Field F := inferInstanceAs (Field Base)

/-- Executable evaluation of the existing coefficient carrier. -/
def evaluate (value : RingF) (point : F) : F :=
  (List.ofFn fun index : Fin ringDegree =>
    value index * point ^ index.val).sum

def node (index : Fin 108) : F :=
  ⟨index.val, Nat.lt_trans index.isLt (by decide)⟩

def modulusValue (point : F) : F :=
  let base : Base := point
  show F from (base ^ ringDegree + (base ^ ringMiddleDegree + 1) : Base)

/-- All fixed points are checked; the quotient is never treated as authority. -/
def Equations (left right output quotient : RingF) : Prop :=
  ∀ index : Fin 108,
    evaluate left (node index) * evaluate right (node index) =
      evaluate output (node index) +
        modulusValue (node index) * evaluate quotient (node index)

theorem evaluate_eq_polynomial (value : RingF) (point : F) :
    evaluate value point = (toPolynomial value).eval (show Base from point) := by
  simp only [evaluate, toPolynomial, List.sum_ofFn, eval_finsetSum,
    eval_mul, eval_C, eval_pow, eval_X]
  rfl

theorem evaluate_sub (left right : RingF) (point : F) :
    evaluate (fun index => left index - right index) point =
      evaluate left point - evaluate right point := by
  simp [evaluate, List.sum_ofFn, sub_mul, Finset.sum_sub_distrib]

theorem evaluate_add (left right : RingF) (point : F) :
    evaluate (ringFAdd left right) point =
      evaluate left point + evaluate right point := by
  simp [evaluate, ringFAdd, List.sum_ofFn, add_mul, Finset.sum_add_distrib]

theorem modulusValue_eq_polynomial (point : F) :
    modulusValue point = modulus.eval (show Base from point) := by
  simp only [modulusValue, modulus, eval_add, eval_pow, eval_X, eval_one]

private theorem node_injective : Function.Injective node := by
  intro left right equal
  apply Fin.ext
  exact congrArg (fun value : F => value.val) equal

private theorem polynomial_degree (value : RingF) :
    (toPolynomial value).degree < (54 : WithBot Nat) := by
  exact degree_sum_fin_lt (R := Base) value

private theorem polynomial_natDegree (value : RingF) :
    (toPolynomial value).natDegree ≤ 53 := by
  by_cases zero : toPolynomial value = 0
  · simp [zero]
  · have bound : (toPolynomial value).natDegree < 54 :=
      (natDegree_lt_iff_degree_lt zero).mpr
      (polynomial_degree value)
    omega

private theorem modulus_natDegree : modulus.natDegree = 54 := by
  exact natDegree_eq_of_degree_eq_some degree_modulus

private theorem product_natDegree (left right : RingF) :
    (toPolynomial left * toPolynomial right).natDegree ≤ 106 := by
  have bound : (toPolynomial left * toPolynomial right).natDegree ≤
      (toPolynomial left).natDegree + (toPolynomial right).natDegree :=
    natDegree_mul_le
  have leftBound := polynomial_natDegree left
  have rightBound := polynomial_natDegree right
  omega

/-- The bounded evaluation equations are an exact polynomial identity. -/
theorem equations_iff_identity (left right output quotient : RingF) :
    Equations left right output quotient ↔
      toPolynomial left * toPolynomial right =
        toPolynomial output + modulus * toPolynomial quotient := by
  constructor
  · intro equations
    apply Polynomial.eq_of_degrees_lt_of_eval_index_eq
      (Finset.univ : Finset (Fin 108))
      (v := fun index => (node index : Base))
      (fun left _ right _ equal => node_injective equal)
    · simp only [Finset.card_univ, Fintype.card_fin]
      exact (degree_le_of_natDegree_le
        (product_natDegree left right)).trans_lt (by norm_num)
    · simp only [Finset.card_univ, Fintype.card_fin]
      have multiplied : (modulus * toPolynomial quotient).natDegree ≤
          modulus.natDegree + (toPolynomial quotient).natDegree := natDegree_mul_le
      have summed := natDegree_add_le (toPolynomial output)
        (modulus * toPolynomial quotient)
      have outputBound := polynomial_natDegree output
      have quotientBound := polynomial_natDegree quotient
      rw [modulus_natDegree] at multiplied
      have bounded :
          (toPolynomial output + modulus * toPolynomial quotient).natDegree ≤ 107 := by
        apply summed.trans
        apply max_le <;> omega
      exact (degree_le_of_natDegree_le bounded).trans_lt (by norm_num)
    · intro index _
      have atIndex := equations index
      simp only [evaluate_eq_polynomial, modulusValue_eq_polynomial] at atIndex
      change (toPolynomial left * toPolynomial right).eval
          (show Base from node index) =
        (toPolynomial output + modulus * toPolynomial quotient).eval
          (show Base from node index)
      rw [eval_mul, eval_add, eval_mul]
      exact atIndex
  · intro identity index
    have evaluated := congrArg (fun polynomial : Polynomial Base =>
      polynomial.eval (show Base from node index)) identity
    simp only [evaluate_eq_polynomial, modulusValue_eq_polynomial]
    simpa only [eval_mul, eval_add] using! evaluated

/-- Every accepted quotient assignment gives the unchanged ring product. -/
theorem sound (left right output quotient : RingF)
    (equations : Equations left right output quotient) :
    output = ringFMul left right := by
  have identity := (equations_iff_identity left right output quotient).mp equations
  have remainder := (div_modByMonic_unique (toPolynomial quotient)
    (toPolynomial output) modulus_monic
    ⟨identity.symm, by simpa only [degree_modulus] using! polynomial_degree output⟩).2
  apply toPolynomial_injective
  exact remainder.symm.trans (toPolynomial_ringFMul_mod left right).symm

/-- The PiRLC accumulator form preserves its previous value. -/
theorem sound_add (left right prior output quotient : RingF)
    (equations : ∀ index : Fin 108,
      evaluate left (node index) * evaluate right (node index) =
        evaluate output (node index) - evaluate prior (node index) +
          modulusValue (node index) * evaluate quotient (node index)) :
    output = ringFAdd prior (ringFMul left right) := by
  have difference : Equations left right (fun index => output index - prior index)
      quotient := by
    intro index
    rw [evaluate_sub]
    exact equations index
  have product := sound left right _ quotient difference
  funext index
  have coefficient := congrFun product index
  change output index = prior index + ringFMul left right index
  calc
    output index = prior index + (output index - prior index) := by abel
    _ = prior index + ringFMul left right index := by rw [coefficient]

/-- Mathematical quotient witness; executable witness generation has its
own correspondence obligation. The final coefficient is padding. -/
noncomputable def quotient (left right : RingF) : RingF :=
  fun index => ((toPolynomial left * toPolynomial right) /ₘ modulus).coeff index.val

private theorem quotient_natDegree (left right : RingF) :
    ((toPolynomial left * toPolynomial right) /ₘ modulus).natDegree ≤ 52 := by
  rw [natDegree_divByMonic _ modulus_monic, modulus_natDegree]
  have bound := product_natDegree left right
  omega

theorem quotient_toPolynomial (left right : RingF) :
    toPolynomial (quotient left right) =
      (toPolynomial left * toPolynomial right) /ₘ modulus := by
  ext index
  by_cases inside : index < ringDegree
  · exact coeff_toPolynomial (quotient left right) ⟨index, inside⟩
  · have outside : 54 ≤ index := Nat.le_of_not_gt inside
    rw [coeff_eq_zero_of_degree_lt
      ((polynomial_degree (quotient left right)).trans_le (by exact_mod_cast outside))]
    symm
    apply coeff_eq_zero_of_natDegree_lt
    have bound := quotient_natDegree left right
    omega

theorem quotient_last_zero (left right : RingF) :
    quotient left right ⟨53, by decide⟩ = 0 := by
  change ((toPolynomial left * toPolynomial right) /ₘ modulus).coeff 53 = 0
  apply coeff_eq_zero_of_natDegree_lt
  have bound := quotient_natDegree left right
  omega

/-- Executable quotient coefficient, including the zero padding at lane 53. -/
def quotientCoeff (left right : RingF) (lane : Fin ringDegree) : F :=
  rawMulCoeffF left right (lane.val + 54) -
    rawMulCoeffF left right (lane.val + 81)

private theorem polynomial_coeff (value : RingF) (index : Nat) :
    (toPolynomial value).coeff index = ringFCoeff value index := by
  by_cases inside : index < ringDegree
  · simpa only [ringFCoeff, dif_pos inside] using
      coeff_toPolynomial value ⟨index, inside⟩
  · rw [ringFCoeff, dif_neg inside]
    apply coeff_eq_zero_of_degree_lt
    exact (polynomial_degree value).trans_le (by
      exact_mod_cast Nat.le_of_not_gt inside)

private theorem raw_prefix (left right : RingF) (degree count : Nat) :
    (List.range count).foldl (fun accumulated index =>
      if index ≤ degree ∧ degree - index < ringDegree then
        accumulated + ringFCoeff left index * ringFCoeff right (degree - index)
      else accumulated) 0 =
      ∑ index ∈ Finset.range count,
        if index ≤ degree ∧ degree - index < ringDegree then
          ringFCoeff left index * ringFCoeff right (degree - index)
        else 0 := by
  induction count with
  | zero => simp
  | succ count ih =>
      simp only [List.range_succ, List.foldl_append, List.foldl_cons,
        List.foldl_nil, ih, Finset.sum_range_succ]
      split_ifs <;> simp

private theorem raw_eq_coefficient (left right : RingF) (degree : Nat) :
    rawMulCoeffF left right degree =
      (toPolynomial left * toPolynomial right).coeff degree := by
  rw [rawMulCoeffF, raw_prefix]
  rw [← Fin.sum_univ_eq_sum_range]
  conv_rhs => rw [toPolynomial, Finset.sum_mul, finsetSum_coeff]
  apply Finset.sum_congr rfl
  intro index _
  rw [mul_assoc, coeff_C_mul, coeff_X_pow_mul', polynomial_coeff]
  have inside := index.isLt
  by_cases below : index.val ≤ degree <;>
    by_cases supported : degree - index.val < ringDegree
  all_goals simp [below, supported, ringFCoeff, inside]
  · rfl
  · change 0 = (show Base from left index) * 0
    exact (mul_zero _).symm

private theorem high_coefficient (left right : RingF) (index : Nat)
    (high : 54 ≤ index) :
    (toPolynomial left * toPolynomial right).coeff index =
      ((toPolynomial left * toPolynomial right) /ₘ modulus).coeff (index - 54) +
      ((toPolynomial left * toPolynomial right) /ₘ modulus).coeff (index - 27) := by
  let product := toPolynomial left * toPolynomial right
  have identity := congrArg (fun polynomial : Polynomial Base => polynomial.coeff index)
    (modByMonic_add_div product modulus)
  have remainderZero : (product %ₘ modulus).coeff index = 0 := by
    apply coeff_eq_zero_of_degree_lt
    have bounded := degree_modByMonic_lt product modulus_monic
    rw [degree_modulus] at bounded
    exact bounded.trans_le (by exact_mod_cast high)
  have quotientZero : (product /ₘ modulus).coeff index = 0 := by
    apply coeff_eq_zero_of_natDegree_lt
    have bounded := quotient_natDegree left right
    dsimp only [product]
    omega
  have middle : 27 ≤ index := by omega
  rw [coeff_add, remainderZero, zero_add] at identity
  nth_rw 1 [modulus] at identity
  simp only [add_mul, one_mul, coeff_add, coeff_X_pow_mul',
    show ringDegree ≤ index from high, show ringMiddleDegree ≤ index from middle,
    if_true, quotientZero, add_zero] at identity
  exact identity.symm

theorem quotientCoeff_eq_quotient (left right : RingF) (lane : Fin ringDegree) :
    quotientCoeff left right lane = quotient left right lane := by
  rw [quotientCoeff, raw_eq_coefficient, raw_eq_coefficient,
    high_coefficient left right (lane.val + 54) (by omega),
    high_coefficient left right (lane.val + 81) (by omega)]
  have zero :
      ((toPolynomial left * toPolynomial right) /ₘ modulus).coeff (lane.val + 54) = 0 := by
    apply coeff_eq_zero_of_natDegree_lt
    have bounded := quotient_natDegree left right
    omega
  have low : lane.val + 54 - 54 = lane.val := by omega
  have middle : lane.val + 54 - 27 = lane.val + 27 := by omega
  have high : lane.val + 81 - 54 = lane.val + 27 := by omega
  have highest : lane.val + 81 - 27 = lane.val + 54 := by omega
  rw [low, middle, high, highest, zero, add_zero]
  change ((toPolynomial left * toPolynomial right) /ₘ modulus).coeff lane.val +
      ((toPolynomial left * toPolynomial right) /ₘ modulus).coeff (lane.val + 27) -
      ((toPolynomial left * toPolynomial right) /ₘ modulus).coeff (lane.val + 27) =
    ((toPolynomial left * toPolynomial right) /ₘ modulus).coeff lane.val
  exact add_sub_cancel_right _ _

/-- Honest products always have a quotient satisfying every fixed check. -/
theorem complete (left right : RingF) :
    Equations left right (ringFMul left right) (quotient left right) := by
  apply (equations_iff_identity left right _ _).mpr
  rw [quotient_toPolynomial, toPolynomial_ringFMul_mod]
  exact (modByMonic_add_div (toPolynomial left * toPolynomial right) modulus).symm

theorem complete_add (left right prior : RingF) (index : Fin 108) :
    evaluate left (node index) * evaluate right (node index) =
      evaluate (ringFAdd prior (ringFMul left right)) (node index) -
        evaluate prior (node index) +
          modulusValue (node index) * evaluate (quotient left right) (node index) := by
  rw [evaluate_add, add_sub_cancel_left]
  exact complete left right index

theorem exists_quotient_iff (left right output : RingF) :
    (∃ witness, Equations left right output witness) ↔
      output = ringFMul left right := by
  constructor
  · rintro ⟨witness, equations⟩
    exact sound left right output witness equations
  · intro equal
    subst output
    exact ⟨quotient left right, complete left right⟩

end NightstreamFPrime.Spec.Phi81Relation.QuotientProduct
