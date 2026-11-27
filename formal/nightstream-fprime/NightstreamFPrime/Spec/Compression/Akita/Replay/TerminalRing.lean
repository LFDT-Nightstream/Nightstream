import Mathlib.FieldTheory.Finite.Basic
import NightstreamFPrime.Spec.FieldPrime
import NightstreamFPrime.Spec.Compression.Akita.Replay.Relation
import NightstreamFPrime.Spec.Compression.Lib.Cube

/-! Computable arithmetic of `R_{q,d} = Z_q[X]/(X^d + 1)` on the coefficient arrays of the
pinned Akita verifier (Akita revision `d50f700`, `akita-algebra/src/ring/cyclotomic/`), for the
terminal checks (`akita-verifier/src/terminal/direct.rs`), and its relation to `Rq d` of the
specification.

- A ring element is the array of its `d` coefficients, and `Poly.toRq d` is its element of
  `Rq d`. `Poly.mul` is the schoolbook negacyclic product, `Poly.shift s` the product with
  `X^s`, `Poly.subMul` the product with an element of `E` in the fixed subfield
  (`add_subfield_product` of `akita-types/src/proof/batch/subfield.rs`), and `Poly.ofSparse` the
  dense form of a sparse challenge (`sparse_challenge_dot` with the value 1). Each has its
  equation in `Rq d` (`toRq_mul`, `toRq_shift`, `toRq_subMul`, `toRq_ofSparse`).
- `Poly.inverse n` is the tower inverse of `inverse.rs` for `d = 2^n`:
  `c^{-1} = c(−X)·(c·c(−X))^{-1}` with the product in `Z_q[Y]/(Y^{d/2} + 1)`, and one inversion
  in `Z_q` at the bottom. `inverse_mul`: a result is the inverse. `inverse_isSome`: a unit has a
  result. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita
open NightstreamFPrime.Spec.Compression.Lib (eqAt)

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

namespace Fq

def ofInt (n : ℤ) : Fq := if 0 ≤ n then ofNat n.toNat else neg (ofNat (-n).toNat)

theorem toZq_ofInt (n : ℤ) : (ofInt n).toZq = (n : Zq) := by
  unfold ofInt
  split
  · rename_i nonneg
    rw [toZq_ofNat]
    conv_rhs => rw [← Int.toNat_of_nonneg nonneg]
    rw [Int.cast_natCast]
  · rename_i negative
    have cast : (((-n).toNat : ℕ) : Zq) = ((-n : ℤ) : Zq) := by
      conv_rhs => rw [← Int.toNat_of_nonneg (by omega : 0 ≤ -n)]
      rw [Int.cast_natCast]
    rw [toZq_neg, toZq_ofNat, cast, Int.cast_neg, neg_neg]

/-- `a^n` by squares. -/
def pow (a : Fq) (n : ℕ) : Fq :=
  if h : n = 0 then one else
    if n % 2 = 1 then mul (pow (mul a a) (n / 2)) a else pow (mul a a) (n / 2)
termination_by n
decreasing_by all_goals omega

theorem toZq_pow (a : Fq) (n : ℕ) : (pow a n).toZq = a.toZq ^ n := by
  induction n using Nat.strong_induction_on generalizing a with
  | _ n ih =>
      rw [pow]
      by_cases zero : n = 0
      · subst zero
        simp [toZq_one]
      · have split : n = 2 * (n / 2) + n % 2 := (Nat.div_add_mod n 2).symm
        rw [dite_eq_right zero]
        by_cases odd : n % 2 = 1
        · rw [ite_eq_left odd, toZq_mul, ih (n / 2) (by omega), toZq_mul, ← pow_two, ← pow_mul,
            ← pow_succ]
          exact congrArg _ (by omega)
        · rw [ite_eq_right odd, ih (n / 2) (by omega), toZq_mul, ← pow_two, ← pow_mul]
          exact congrArg _ (by omega)

theorem toZq_eq_zero_iff (a : Fq) : a.toZq = 0 ↔ a.word.toNat % fieldModulus = 0 := by
  rw [← ZMod.val_eq_zero, toZq, ZMod.val_natCast]

/-- The inverse in `Z_q` (`Field::inverse`): `a^{q−2}`, and `none` at 0. -/
def inv? (a : Fq) : Option Fq :=
  if a.word.toNat % fieldModulus = 0 then none else some (pow a (fieldModulus - 2))

theorem inv?_mul {a b : Fq} (found : inv? a = some b) : b.toZq * a.toZq = 1 := by
  unfold inv? at found
  split at found
  · cases found
  · rename_i nonzero
    cases found
    rw [toZq_pow, ← pow_succ, show fieldModulus - 2 + 1 = fieldModulus - 1 by decide]
    exact ZMod.pow_card_sub_one_eq_one fun zero => nonzero ((toZq_eq_zero_iff a).mp zero)

theorem inv?_isSome {a : Fq} (nonzero : a.toZq ≠ 0) : (inv? a).isSome := by
  unfold inv?
  rw [ite_eq_right fun zero => nonzero ((toZq_eq_zero_iff a).mpr zero)]
  rfl

end Fq

/-- `Σ_{i<n} f i` in `Z_q`. -/
def sumFq (f : ℕ → Fq) : ℕ → Fq
  | 0 => Fq.zero
  | n + 1 => (sumFq f n).add (f n)

theorem toZq_sumFq (f : ℕ → Fq) : ∀ n, (sumFq f n).toZq = ∑ i ∈ Finset.range n, (f i).toZq
  | 0 => by simp [sumFq, Fq.toZq_zero]
  | n + 1 => by rw [sumFq, Fq.toZq_add, toZq_sumFq f n, Finset.sum_range_succ]

namespace Poly

/-- Coefficient `i`, and 0 beyond the array. -/
def get (a : Array Fq) (i : ℕ) : Fq := a.getD i Fq.zero

/-- The array of `f 0, …, f (d − 1)`. -/
def ofFn (d : ℕ) (f : ℕ → Fq) : Array Fq := Array.ofFn (n := d) fun i => f i.val

theorem get_ofFn {d i : ℕ} (f : ℕ → Fq) (small : i < d) : get (ofFn d f) i = f i := by
  simp [get, ofFn, Array.getD, small]

theorem size_ofFn (d : ℕ) (f : ℕ → Fq) : (ofFn d f).size = d := by simp [ofFn]

variable (d : ℕ)

/-- The element of `Rq d` with the first `d` coefficients of the array. -/
noncomputable def toRq [NeZero d] (a : Array Fq) : Rq d := ofCoeffs fun i => (get a i).toZq

variable {d}

theorem coeff_toRq [NeZero d] (a : Array Fq) {i : ℕ} (small : i < d) :
    coeff (toRq d a) i = (get a i).toZq :=
  coeff_ofCoeffs _ small

theorem toRq_ext [NeZero d] {a b : Array Fq} (same : ∀ i < d, (get a i).toZq = (get b i).toZq) :
    toRq d a = toRq d b :=
  ext_coeff fun i small => by rw [coeff_toRq _ small, coeff_toRq _ small, same i small]

variable (d)

def zero : Array Fq := ofFn d fun _ => Fq.zero

def add (a b : Array Fq) : Array Fq := ofFn d fun k => (get a k).add (get b k)

def sub (a b : Array Fq) : Array Fq := ofFn d fun k => (get a k).sub (get b k)

def scale (s : Fq) (a : Array Fq) : Array Fq := ofFn d fun k => s.mul (get a k)

/-- The product with `X^s` for `s ≤ d`. -/
def shift (s : ℕ) (a : Array Fq) : Array Fq :=
  ofFn d fun k => if k < s then (get a (d + k - s)).neg else get a (k - s)

/-- The schoolbook product in `Z_q[X]/(X^d + 1)` (`negacyclic_mul` of `inverse.rs`). -/
def mul (a b : Array Fq) : Array Fq :=
  ofFn d fun k => sumFq (fun i =>
    (get a i).mul (if k < i then (get b (d + k - i)).neg else get b (k - i))) d

/-- `Σ_{i<n} f i` of ring elements. -/
def sum (f : ℕ → Array Fq) : ℕ → Array Fq
  | 0 => zero d
  | n + 1 => add d (sum f n) (f n)

variable {d} [NeZero d]

theorem toRq_zero : toRq d (zero d) = 0 :=
  ext_coeff fun i small => by
    rw [coeff_toRq _ small, zero, get_ofFn _ small, Fq.toZq_zero, coeff_zero]

theorem toRq_add (a b : Array Fq) : toRq d (add d a b) = toRq d a + toRq d b :=
  ext_coeff fun i small => by
    rw [coeff_toRq _ small, add, get_ofFn _ small, Fq.toZq_add, coeff_add, coeff_toRq _ small,
      coeff_toRq _ small]

theorem toRq_sub (a b : Array Fq) : toRq d (sub d a b) = toRq d a - toRq d b :=
  ext_coeff fun i small => by
    rw [coeff_toRq _ small, sub, get_ofFn _ small, Fq.toZq_sub, coeff_sub, coeff_toRq _ small,
      coeff_toRq _ small]

theorem toRq_scale (s : Fq) (a : Array Fq) : toRq d (scale d s a) = s.toZq • toRq d a :=
  ext_coeff fun i small => by
    rw [coeff_toRq _ small, scale, get_ofFn _ small, Fq.toZq_mul, coeff_smul, coeff_toRq _ small]

theorem toRq_shift {s : ℕ} (le : s ≤ d) (a : Array Fq) :
    toRq d (shift d s a) = AdjoinRoot.root (modulus d) ^ s * toRq d a :=
  ext_coeff fun i small => by
    rw [coeff_toRq _ small, shift, get_ofFn _ small, coeff_root_pow_mul le _ small]
    split
    · rw [Fq.toZq_neg, coeff_toRq _ (by omega)]
    · rw [coeff_toRq _ (by omega)]

theorem toRq_mul (a b : Array Fq) : toRq d (mul d a b) = toRq d a * toRq d b := by
  refine ext_coeff fun k small => ?_
  have expanded : toRq d a =
      ∑ i ∈ Finset.range d, (get a i).toZq • AdjoinRoot.root (modulus d) ^ i :=
    ofCoeffs_eq_sum _
  rw [coeff_toRq _ small, mul, get_ofFn _ small, toZq_sumFq, expanded, Finset.sum_mul, coeff_sum]
  refine Finset.sum_congr rfl fun i member => ?_
  have inside := Finset.mem_range.mp member
  rw [smul_mul_assoc, coeff_smul, coeff_root_pow_mul inside.le _ small, Fq.toZq_mul]
  split
  · rw [Fq.toZq_neg, coeff_toRq _ (by omega)]
  · rw [coeff_toRq _ (by omega)]

theorem toRq_sum (f : ℕ → Array Fq) :
    ∀ n, toRq d (sum d f n) = ∑ i ∈ Finset.range n, toRq d (f i)
  | 0 => by rw [sum, toRq_zero, Finset.sum_range_zero]
  | n + 1 => by rw [sum, toRq_add, toRq_sum f n, Finset.sum_range_succ]

theorem one_ne_zero' : (1 : Rq d) ≠ 0 := fun same => by
  have value := congrArg (fun x => coeff x 0) same
  simp only [coeff_one, ite_true, coeff_zero] at value
  exact one_ne_zero value

end Poly

section Extension

variable {E : Type} [CommRing E] [Algebra Zq E] [Degree2 E]

/-- The test `a = b` in `E`; the words need not be below `q`. -/
def Ext.equals (a b : Ext) : Bool := decide (a - b = Ext.zero)

theorem Ext.equals_iff (a b : Ext) : a.equals b = true ↔ (a.into : E) = b.into := by
  rw [Ext.equals, decide_eq_true_iff]
  constructor
  · intro zero
    have value := congrArg Ext.toE zero
    rw [Ext.toE_sub, Ext.toE_zero] at value
    simp only [Ext.into, sub_eq_zero.mp value]
  · intro equal
    have same : a.toE = b.toE := (Degree2.coords (E := E)).symm.injective equal
    refine Ext.toE_injective (Fq.ofNat_canonical _) (Fq.ofNat_canonical _) (by decide)
      (by decide) ?_
    rw [Ext.toE_sub, Ext.toE_zero, same, sub_self]

/-- `eqIndex` (`EqPolynomial::evals`). -/
abbrev eqWeight : List Ext → ℕ → Ext := eqIndex

namespace Poly

variable {d : ℕ}

/-- The product with the element `c` of `E` in the fixed subfield of `R_{q,d}`:
`c_0·w + c_1·(X^m − X^{3m})·w` (`add_subfield_product`). -/
def subMul (d m : ℕ) (c : Ext) (w : Array Fq) : Array Fq :=
  add d (scale d c.c0 w) (scale d c.c1 (sub d (shift d m w) (shift d (3 * m) w)))

theorem toRq_subMul [NeZero d] {m : ℕ} (quarter : 4 * m = d) (c : Ext) (w : Array Fq) :
    toRq d (subMul d m c w) = subfield quarter (c.into : E) * toRq d w := by
  rw [subMul, toRq_add, toRq_scale, toRq_scale, toRq_sub, toRq_shift (by omega),
    toRq_shift (by omega), subfield_apply, re_into, im_into, fixed, add_mul,
    smul_mul_assoc, one_mul, smul_mul_assoc, sub_mul]

/-- The dense form of a sparse challenge: coefficient `k` is the sum of the coefficients at the
position `k` (`sparse_challenge_dot` of one challenge with the value 1). -/
def ofSparse (d : ℕ) (c : List (ℕ × ℤ)) : Array Fq :=
  ofFn d fun k => c.foldr (fun x acc => if x.1 = k then (Fq.ofInt x.2).add acc else acc) Fq.zero

theorem toRq_ofSparse [NeZero d] : ∀ c : List (ℕ × ℤ), (∀ x ∈ c, x.1 < d) →
    toRq d (ofSparse d c) =
      (c.map fun x => (x.2 : Zq) • AdjoinRoot.root (modulus d) ^ x.1).sum
  | [], _ => by
      rw [List.map_nil, List.sum_nil]
      exact toRq_zero
  | x :: c, inside => ext_coeff fun k small => by
      have rest : coeff (toRq d (ofSparse d c)) k = coeff (c.map fun x =>
          (x.2 : Zq) • AdjoinRoot.root (modulus d) ^ x.1).sum k := by
        rw [toRq_ofSparse c fun y member => inside y (List.mem_cons_of_mem _ member)]
      rw [List.map_cons, List.sum_cons, coeff_add, ← rest, coeff_smul,
        coeff_root_pow (inside x List.mem_cons_self) small, coeff_toRq _ small,
        coeff_toRq _ small, ofSparse, ofSparse, get_ofFn _ small, get_ofFn _ small,
        List.foldr_cons]
      by_cases at_ : x.1 = k
      · rw [ite_eq_left at_, ite_eq_left at_.symm, Fq.toZq_add, Fq.toZq_ofInt, mul_one]
      · rw [ite_eq_right at_, ite_eq_right (Ne.symm at_), mul_zero, zero_add]

end Poly

end Extension

section Inverse

/-- `σ : X ↦ −X` of `R_{q,d}` for an even `d`. -/
noncomputable def negX (d : ℕ) (even : Even d) : Rq d →ₐ[Zq] Rq d :=
  AdjoinRoot.liftAlgHom (modulus d) (Algebra.ofId Zq (Rq d)) (-AdjoinRoot.root (modulus d)) (by
    show Polynomial.eval₂ _ _ (Polynomial.X ^ d + 1) = 0
    rw [Polynomial.eval₂_add, Polynomial.eval₂_pow, Polynomial.eval₂_X, Polynomial.eval₂_one,
      even.neg_pow]
    exact root_pow_add_one)

variable {d : ℕ} (even : Even d)

theorem negX_root : negX d even (AdjoinRoot.root (modulus d)) = -AdjoinRoot.root (modulus d) :=
  AdjoinRoot.liftAlgHom_root _ _ _ _

theorem negX_negX (x : Rq d) : negX d even (negX d even x) = x := by
  have both : (negX d even).comp (negX d even) = AlgHom.id Zq (Rq d) :=
    AdjoinRoot.algHom_ext (by
      rw [AlgHom.comp_apply, negX_root, map_neg, negX_root, neg_neg, AlgHom.id_apply])
  exact AlgHom.congr_fun both x

variable [NeZero d]

theorem coeff_negX (x : Rq d) {i : ℕ} (small : i < d) :
    coeff (negX d even x) i = (-1) ^ i * coeff x i := by
  have expanded : negX d even x = ofCoeffs fun j => (-1) ^ j * coeff x j := by
    conv_lhs => rw [← ofCoeffs_coeff x, ofCoeffs_eq_sum]
    rw [ofCoeffs_eq_sum, map_sum]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [map_smul, map_pow, negX_root, neg_pow, mul_comm ((-1 : Zq) ^ j), ← smul_smul]
    congr 1
    rw [Algebra.smul_def, map_pow, map_neg, map_one]
  rw [expanded, coeff_ofCoeffs _ small]

theorem coeff_eq_zero_of_fixed {w : Rq d} (fixed : negX d even w = w) {i : ℕ} (small : i < d)
    (odd : i % 2 = 1) : coeff w i = 0 := by
  have value := coeff_negX even w small
  rw [fixed, (Nat.odd_iff.mpr odd).neg_one_pow, neg_one_mul] at value
  have two : (2 : Zq) * coeff w i = 0 := by linear_combination value
  exact (mul_eq_zero.mp two).resolve_left (by decide)

variable {h : ℕ} [NeZero h] (tower : 2 * h = d)

theorem eq_embed_plane {w : Rq d} (odd : ∀ i < d, i % 2 = 1 → coeff w i = 0) :
    w = embed 2 tower (plane h 2 0 w) := by
  refine ext_coeff fun i small => ?_
  rw [coeff_embed]
  split
  · rename_i two
    obtain ⟨j, rfl⟩ := two
    rw [Nat.mul_div_cancel_left _ (by norm_num), coeff_plane _ _ _ (by omega), Nat.zero_add]
  · rename_i notTwo
    exact odd i small (by omega)

theorem plane_one : plane h 2 0 (1 : Rq d) = 1 :=
  ext_coeff fun j small => by
    rw [coeff_plane _ _ _ small, coeff_one, coeff_one, Nat.zero_add]
    by_cases zero : j = 0
    · simp [zero]
    · rw [ite_eq_right (by omega), ite_eq_right zero]

include tower in
theorem isUnit_of_embed {N : Rq h} (unit : IsUnit (embed 2 tower N)) : IsUnit N := by
  obtain ⟨u, inverse⟩ := unit.exists_right_inv
  have value := congrArg (plane h 2 0) inverse
  rw [plane_embed_mul tower (by norm_num) N u, plane_one] at value
  exact ⟨⟨N, plane h 2 0 u, value, (mul_comm _ _).trans value⟩, rfl⟩

end Inverse

namespace Poly

/-- `c(−X)`. -/
def conj (d : ℕ) (a : Array Fq) : Array Fq :=
  ofFn d fun i => if i % 2 = 0 then get a i else (get a i).neg

/-- The coefficients at the even positions. -/
def evens (h : ℕ) (a : Array Fq) : Array Fq := ofFn h fun j => get a (2 * j)

/-- `ι(w)`: coefficient `j` of `w` at the position `2·j`. -/
def lift (d : ℕ) (w : Array Fq) : Array Fq :=
  ofFn d fun i => if i % 2 = 0 then get w (i / 2) else Fq.zero

/-- The inverse in `Z_q[X]/(X^{2^n} + 1)` (`negacyclic_inverse` of `inverse.rs`), or `none`. -/
def inverse : ℕ → Array Fq → Option (Array Fq)
  | 0, a => (Fq.inv? (get a 0)).map fun x => #[x]
  | n + 1, a =>
      (inverse n (evens (2 ^ n) (mul (2 ^ (n + 1)) a (conj (2 ^ (n + 1)) a)))).map fun w =>
        mul (2 ^ (n + 1)) (lift (2 ^ (n + 1)) w) (conj (2 ^ (n + 1)) a)

variable {d : ℕ} [NeZero d]

theorem toRq_conj (even : Even d) (a : Array Fq) : toRq d (conj d a) = negX d even (toRq d a) :=
  ext_coeff fun i small => by
    rw [coeff_toRq _ small, conj, get_ofFn _ small, coeff_negX even _ small, coeff_toRq _ small]
    split
    · rename_i zero
      rw [(Nat.even_iff.mpr zero).neg_one_pow, one_mul]
    · rename_i one
      rw [Fq.toZq_neg, (Nat.odd_iff.mpr (by omega)).neg_one_pow, neg_one_mul]

variable {h : ℕ} [NeZero h] (tower : 2 * h = d)

include tower in
theorem toRq_evens (a : Array Fq) : toRq h (evens h a) = plane h 2 0 (toRq d a) :=
  ext_coeff fun j small => by
    rw [coeff_toRq _ small, evens, get_ofFn _ small, coeff_plane _ _ _ small, Nat.zero_add,
      coeff_toRq _ (by omega)]

theorem toRq_lift (w : Array Fq) : toRq d (lift d w) = embed 2 tower (toRq h w) :=
  ext_coeff fun i small => by
    rw [coeff_toRq _ small, lift, get_ofFn _ small, coeff_embed]
    by_cases zero : i % 2 = 0
    · rw [ite_eq_left zero, ite_eq_left (Nat.dvd_of_mod_eq_zero zero), coeff_toRq _ (by omega)]
    · rw [ite_eq_right zero, ite_eq_right fun two => zero (Nat.mod_eq_zero_of_dvd two),
        Fq.toZq_zero]

theorem two_pow_tower (n : ℕ) : 2 * 2 ^ n = 2 ^ (n + 1) := by rw [pow_succ]; ring

theorem two_pow_even (n : ℕ) : Even (2 ^ (n + 1)) := ⟨2 ^ n, by rw [pow_succ]; ring⟩

/-- A result of `inverse` is the inverse. -/
theorem inverse_mul : ∀ (n : ℕ) (a b : Array Fq), inverse n a = some b →
    toRq (2 ^ n) b * toRq (2 ^ n) a = 1
  | 0, a, b, found => by
      simp only [inverse, Option.map_eq_some_iff] at found
      obtain ⟨x, inverted, rfl⟩ := found
      rw [← toRq_mul]
      refine ext_coeff fun i small => ?_
      obtain rfl : i = 0 := by simpa using small
      rw [coeff_toRq _ small, mul, get_ofFn _ small, toZq_sumFq,
        show Finset.range (2 ^ 0) = {0} by decide, Finset.sum_singleton,
        ite_eq_right (Nat.lt_irrefl 0), Fq.toZq_mul, coeff_one, ite_eq_left rfl]
      exact Fq.inv?_mul inverted
  | n + 1, a, b, found => by
      simp only [inverse, Option.map_eq_some_iff] at found
      obtain ⟨w, inner, rfl⟩ := found
      have product := inverse_mul n _ w inner
      have even := two_pow_even n
      rw [toRq_evens (two_pow_tower n), toRq_mul, toRq_conj even] at product
      have fixedW : negX _ even (toRq (2 ^ (n + 1)) a * negX _ even (toRq (2 ^ (n + 1)) a)) =
          toRq (2 ^ (n + 1)) a * negX _ even (toRq (2 ^ (n + 1)) a) := by
        rw [map_mul, negX_negX, mul_comm]
      have embedW := eq_embed_plane (two_pow_tower n)
        fun i small odd => coeff_eq_zero_of_fixed even fixedW small odd
      rw [toRq_mul, toRq_lift (two_pow_tower n), toRq_conj even, mul_assoc,
        mul_comm (negX _ even _), embedW, ← map_mul, product, map_one]

/-- A unit of `Rq (2^n)` has an inverse. -/
theorem inverse_isSome : ∀ (n : ℕ) (a : Array Fq), IsUnit (toRq (2 ^ n) a) →
    (inverse n a).isSome
  | 0, a, unit => by
      simp only [inverse, Option.isSome_map]
      refine Fq.inv?_isSome fun zero => ?_
      have vanishes : toRq (2 ^ 0) a = 0 := ext_coeff fun i small => by
        obtain rfl : i = 0 := by simpa using small
        rw [coeff_toRq _ small, zero, coeff_zero]
      rw [vanishes, isUnit_zero_iff] at unit
      exact one_ne_zero' unit.symm
  | n + 1, a, unit => by
      simp only [inverse, Option.isSome_map]
      refine inverse_isSome n _ ?_
      have even := two_pow_even n
      rw [toRq_evens (two_pow_tower n), toRq_mul, toRq_conj even]
      have fixedW : negX _ even (toRq (2 ^ (n + 1)) a * negX _ even (toRq (2 ^ (n + 1)) a)) =
          toRq (2 ^ (n + 1)) a * negX _ even (toRq (2 ^ (n + 1)) a) := by
        rw [map_mul, negX_negX, mul_comm]
      have embedW := eq_embed_plane (two_pow_tower n)
        fun i small odd => coeff_eq_zero_of_fixed even fixedW small odd
      have both := unit.mul (unit.map (negX _ even))
      rw [embedW] at both
      exact isUnit_of_embed (two_pow_tower n) both

end Poly

end NightstreamFPrime.Spec.Compression.Akita.Replay
