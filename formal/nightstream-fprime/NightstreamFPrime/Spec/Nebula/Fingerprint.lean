import Mathlib.Algebra.MvPolynomial.CommRing
import Mathlib.Algebra.MvPolynomial.SchwartzZippel
import Mathlib.Algebra.Polynomial.Roots
import Mathlib.Algebra.CharP.Basic
import NightstreamFPrime.Spec.Nebula.Records

/-! Owns the fingerprint of spec §8.2 and security note Lemma 3 part 1: for
fixed multisets `A ≠ B` of small tuples, at most `2·m·|E|` challenge pairs make
the two fingerprint products equal. The carrier `E` is any finite domain of
characteristic `q`; the Goldilocks instance is in `GoldilocksExtensionRing`.
It does not own how the multisets arise (Lemma 4) or the retry argument. -/

namespace NightstreamFPrime.Spec.Nebula

variable {E : Type} [CommRing E]

/-- Spec §8.2: `f_η(t, g, v) = g + η1 · v + η1² · t − η2`. -/
def fingerprint (η : E × E) (τ : Tuple) : E :=
  (τ.2.1 : E) + η.1 * (τ.2.2 : E) + η.1 ^ 2 * (τ.1 : E) - η.2

/-- The fingerprint product of a multiset. -/
def product (η : E × E) (A : Multiset Tuple) : E := (A.map (fingerprint η)).prod

/-- The close product equation of spec §11.2. -/
def Multisets.ProductEq (η : E × E) (m : Multisets) : Prop :=
  product η m.initial * product η m.write = product η m.read * product η m.final

theorem product_add (η : E × E) (A B : Multiset Tuple) :
    product η (A + B) = product η A * product η B := by
  simp only [product, Multiset.map_add, Multiset.prod_add]

theorem productEq_iff (η : E × E) (m : Multisets) :
    m.ProductEq η ↔ product η (m.initial + m.write) = product η (m.read + m.final) := by
  rw [Multisets.ProductEq, product_add, product_add]

/-- Completeness of the test: balanced multisets pass for every challenge. -/
theorem productEq_of_balanced (η : E × E) {m : Multisets} (balanced : m.Balanced) :
    m.ProductEq η :=
  (productEq_iff η m).mpr (congrArg (product η) balanced)

/-- Every component below `q`, so the field encoding is injective (spec §4.2
rules 3 and 4). -/
def Tuple.Small (τ : Tuple) : Prop :=
  τ.1 < goldilocksModulus ∧ τ.2.1 < goldilocksModulus ∧ τ.2.2 < goldilocksModulus

section Polynomial

open MvPolynomial

/-- The fingerprint factor as a polynomial: variable 0 is `η2`, variable 1 is `η1`. -/
private noncomputable def factor (τ : Tuple) : MvPolynomial (Fin 2) E :=
  C (τ.2.1 : E) + C (τ.2.2 : E) * X 1 + C (τ.1 : E) * X 1 ^ 2 - X 0

/-- The root of `factor τ` in `η2`, as a polynomial in `η1`. -/
private noncomputable def root (τ : Tuple) : MvPolynomial (Fin 1) E :=
  C (τ.2.1 : E) + C (τ.2.2 : E) * X 0 + C (τ.1 : E) * X 0 ^ 2

private theorem eval_factor (η : E × E) (τ : Tuple) :
    eval ![η.2, η.1] (factor τ) = fingerprint η τ := by
  simp [factor, fingerprint]
  ring

private theorem finSuccEquiv_factor (τ : Tuple) :
    finSuccEquiv E 1 (factor τ) = Polynomial.C (root τ) - Polynomial.X := by
  have one : (X 1 : MvPolynomial (Fin 2) E) = X (Fin.succ 0) := rfl
  have constant (a : E) : finSuccEquiv E 1 (C a) = Polynomial.C (C a) := by
    simp [finSuccEquiv_apply]
  simp only [factor, root, one, map_sub, map_add, map_mul, map_pow, constant,
    finSuccEquiv_X_zero, finSuccEquiv_X_succ]

private theorem totalDegree_factor [Nontrivial E] (τ : Tuple) :
    (factor (E := E) τ).totalDegree ≤ 2 := by
  have linear (a : E) : (C a * X 1 : MvPolynomial (Fin 2) E).totalDegree ≤ 2 :=
    (totalDegree_mul _ _).trans (by simp only [totalDegree_X, totalDegree_C]; omega)
  have quadratic (a : E) : (C a * X 1 ^ 2 : MvPolynomial (Fin 2) E).totalDegree ≤ 2 :=
    (totalDegree_mul _ _).trans (by simp only [totalDegree_X_pow, totalDegree_C]; omega)
  refine (totalDegree_sub _ _).trans (max_le ?_ (by simp only [totalDegree_X]; omega))
  refine (totalDegree_add _ _).trans (max_le ?_ (quadratic _))
  refine (totalDegree_add _ _).trans (max_le ?_ (linear _))
  simp only [totalDegree_C]; omega

private theorem totalDegree_prod_factor [Nontrivial E] {A : Multiset Tuple} {m : ℕ}
    (size : Multiset.card A ≤ m) : ((A.map (factor (E := E))).prod).totalDegree ≤ 2 * m := by
  refine (totalDegree_multiset_prod _).trans ?_
  refine (Multiset.sum_le_card_nsmul _ 2 ?_).trans ?_
  · intro degree member
    obtain ⟨P, memberP, rfl⟩ := Multiset.mem_map.mp member
    obtain ⟨τ, -, rfl⟩ := Multiset.mem_map.mp memberP
    exact totalDegree_factor τ
  · rw [Multiset.card_map, Multiset.card_map, smul_eq_mul]
    omega

private theorem roots_prod [IsDomain E] (A : Multiset Tuple) :
    (A.map fun τ => Polynomial.C (root (E := E) τ) - Polynomial.X).prod.roots = A.map root := by
  have single (a : MvPolynomial (Fin 1) E) : (Polynomial.C a - Polynomial.X).roots = {a} := by
    rw [← neg_sub, Polynomial.roots_neg, Polynomial.roots_X_sub_C]
  rw [Polynomial.roots_multiset_prod, Multiset.bind_map]
  · simp only [single, Multiset.bind_singleton]
  · intro zero
    obtain ⟨τ, -, isZero⟩ := Multiset.mem_map.mp zero
    exact Polynomial.X_sub_C_ne_zero (root τ) (neg_eq_zero.mp ((neg_sub _ _).trans isZero))

private theorem root_injective [CharP E goldilocksModulus] {τ σ : Tuple} (τSmall : τ.Small)
    (σSmall : σ.Small) (same : root (E := E) τ = root σ) : τ = σ := by
  have cast {a b : ℕ} (aSmall : a < goldilocksModulus) (bSmall : b < goldilocksModulus)
      (equal : (a : E) = b) : a = b :=
    CharP.natCast_injOn_Iio E goldilocksModulus aSmall bSmall equal
  have constant := congrArg (coeff 0) same
  have linear := congrArg (coeff (Finsupp.single 0 1)) same
  have quadratic := congrArg (coeff (Finsupp.single 0 2)) same
  simp only [root, coeff_add, coeff_C, coeff_C_mul, coeff_X, coeff_X_pow]
    at constant linear quadratic
  simp [@eq_comm (Fin 1 →₀ ℕ) 0] at constant linear quadratic
  exact Prod.ext (cast τSmall.1 σSmall.1 quadratic)
    (Prod.ext (cast τSmall.2.1 σSmall.2.1 constant) (cast τSmall.2.2 σSmall.2.2 linear))

private theorem eq_of_map_root_eq [CharP E goldilocksModulus] {A B : Multiset Tuple}
    (small : ∀ τ ∈ A + B, τ.Small) (same : A.map (root (E := E)) = B.map root) : A = B := by
  have lift (D : Multiset Tuple) (smallD : ∀ τ ∈ D, τ.Small) :
      (D.pmap Subtype.mk smallD).map Subtype.val = D := by
    simp [Multiset.map_pmap, Multiset.pmap_eq_map]
  have liftRoot (D : Multiset Tuple) (smallD : ∀ τ ∈ D, τ.Small) :
      (D.pmap Subtype.mk smallD).map (fun τ => root (E := E) τ.1) = D.map root := by
    simp [Multiset.map_pmap, Multiset.pmap_eq_map]
  have smallA : ∀ τ ∈ A, τ.Small := fun τ member =>
    small τ (Multiset.mem_add.mpr (Or.inl member))
  have smallB : ∀ τ ∈ B, τ.Small := fun τ member =>
    small τ (Multiset.mem_add.mpr (Or.inr member))
  have injective : Function.Injective fun τ : {τ : Tuple // τ.Small} => root (E := E) τ.1 :=
    fun τ σ equal => Subtype.ext (root_injective τ.2 σ.2 equal)
  have lifted : A.pmap Subtype.mk smallA = B.pmap Subtype.mk smallB :=
    Multiset.map_injective injective (by rw [liftRoot, liftRoot, same])
  rw [← lift A smallA, ← lift B smallB, lifted]

private theorem difference_ne_zero [IsDomain E] [CharP E goldilocksModulus]
    {A B : Multiset Tuple} (different : A ≠ B) (small : ∀ τ ∈ A + B, τ.Small) :
    (A.map factor).prod - (B.map factor).prod ≠ (0 : MvPolynomial (Fin 2) E) := by
  intro zero
  apply different
  apply eq_of_map_root_eq (E := E) small
  have same := congrArg (fun P => (finSuccEquiv E 1 P).roots) (sub_eq_zero.mp zero)
  simpa only [map_multiset_prod, Multiset.map_map, Function.comp_def, finSuccEquiv_factor,
    roots_prod] using same

end Polynomial

section Count

variable [IsDomain E] [Fintype E] [DecidableEq E] [CharP E goldilocksModulus]

/-- The challenge pairs on which the products of `A` and `B` agree. -/
def BadChallenges (A B : Multiset Tuple) : Finset (E × E) :=
  Finset.univ.filter fun η => product η A = product η B

/-- Security note Lemma 3 part 1, as a count. -/
theorem badChallenges_card {A B : Multiset Tuple} {m : ℕ} (different : A ≠ B)
    (small : ∀ τ ∈ A + B, τ.Small) (sizeA : Multiset.card A ≤ m)
    (sizeB : Multiset.card B ≤ m) :
    (BadChallenges (E := E) A B).card ≤ 2 * m * Fintype.card E := by
  let G : MvPolynomial (Fin 2) E := (A.map factor).prod - (B.map factor).prod
  have degree : G.totalDegree ≤ 2 * m := (MvPolynomial.totalDegree_sub _ _).trans
    (max_le (totalDegree_prod_factor sizeA) (totalDegree_prod_factor sizeB))
  have evalG (η : E × E) : MvPolynomial.eval ![η.2, η.1] G = product η A - product η B := by
    simp only [G, map_sub, map_multiset_prod, Multiset.map_map, Function.comp_def, eval_factor,
      product]
  have toZeros : (BadChallenges (E := E) A B).card ≤
      (Finset.univ.filter fun x : Fin 2 → E => MvPolynomial.eval x G = 0).card := by
    refine Finset.card_le_card_of_injOn (fun η => ![η.2, η.1]) ?_ ?_
    · intro η member
      simp only [BadChallenges, Finset.coe_filter, Finset.mem_univ, true_and,
        Set.mem_setOf_eq] at member ⊢
      rw [evalG, member, sub_self]
    · intro η _ σ _ same
      have first := congrFun same 0
      have second := congrFun same 1
      simp at first second
      exact Prod.ext second first
  have schwartzZippel :=
    MvPolynomial.schwartz_zippel_totalDegree (difference_ne_zero (E := E) different small)
      (Finset.univ : Finset E)
  rw [Fintype.piFinset_univ, Finset.card_univ] at schwartzZippel
  have positive : (0 : ℚ≥0) < Fintype.card E := by exact_mod_cast Fintype.card_pos
  rw [div_le_div_iff₀ (pow_pos positive 2) positive, pow_two, ← mul_assoc] at schwartzZippel
  have bound : (Finset.univ.filter fun x : Fin 2 → E => MvPolynomial.eval x G = 0).card ≤
      G.totalDegree * Fintype.card E := by
    exact_mod_cast le_of_mul_le_mul_right schwartzZippel positive
  exact toZeros.trans (bound.trans (Nat.mul_le_mul_right _ degree))

/-- Security note Lemma 3 part 1, as a frequency: `ε_test ≤ 2·m / |E|`. -/
theorem badChallenges_frequency {A B : Multiset Tuple} {m : ℕ} (different : A ≠ B)
    (small : ∀ τ ∈ A + B, τ.Small) (sizeA : Multiset.card A ≤ m)
    (sizeB : Multiset.card B ≤ m) :
    ((BadChallenges (E := E) A B).card : ℚ≥0) / (Fintype.card (E × E) : ℚ≥0) ≤
      2 * m / Fintype.card E := by
  have positive : (0 : ℚ≥0) < Fintype.card E := by exact_mod_cast Fintype.card_pos
  have count : ((BadChallenges (E := E) A B).card : ℚ≥0) ≤ 2 * m * Fintype.card E := by
    exact_mod_cast badChallenges_card different small sizeA sizeB
  rw [Fintype.card_prod, Nat.cast_mul, div_le_div_iff₀ (mul_pos positive positive) positive]
  calc ((BadChallenges (E := E) A B).card : ℚ≥0) * Fintype.card E
      ≤ 2 * m * Fintype.card E * Fintype.card E := mul_le_mul_of_nonneg_right count zero_le
    _ = 2 * m * (Fintype.card E * Fintype.card E) := mul_assoc _ _ _

end Count

end NightstreamFPrime.Spec.Nebula
