import Mathlib.Data.Matrix.Mul
import Mathlib.Data.ZMod.ValMinAbs
import Mathlib.RingTheory.AdjoinRoot
import NightstreamFPrime.Spec.Algebra

/-! The rings and the norms of Akita (ePrint 2026/1983, revision of 2026-10-15, Section 3.1), and
Module-SIS (Akita Definition 3.2).

- `Rq d` is `Z_q[X]/(X^d + 1)` for `q = 2^64 − 59` and a ring degree `d`. `coeff x i` is
  coefficient `i` of the representative of `x` of degree below `d`.
- `magnitude a` is the absolute value of the representative of `a` in `(−q/2, q/2]`. `normInf` and
  `normOne` are the norms `‖·‖_∞` and `‖·‖_1` of the coefficients of a ring element, and
  `vecNormInf` is `‖·‖_∞` of a vector.
- `normInf_mul_le` is Akita Lemma 3.1, inequality (18): `‖c·s‖_∞ ≤ ‖c‖_1 · ‖s‖_∞`.
- `ShortKernel M β` is a solution of `MSIS_{q,d}(n, m, β)` for the matrix `M` (Definition 3.2 with
  `p = ∞`), and `ShortKernel.ofCollision?` is the difference of two inputs of `M` when it is one. -/

namespace NightstreamFPrime.Spec.Compression.Akita

open Polynomial

/-- `Z_q` for `q = 2^64 − 59`. -/
abbrev Zq : Type := ZMod fieldModulus

scoped instance : Fact (1 < fieldModulus) := ⟨by decide⟩

/-- `X^d + 1`. -/
noncomputable def modulus (d : ℕ) : Zq[X] := X ^ d + 1

/-- `R_{q,d} = Z_q[X]/(X^d + 1)` (Akita Section 3.1). -/
abbrev Rq (d : ℕ) : Type := AdjoinRoot (modulus d)

section Coefficients

variable {d : ℕ} [NeZero d]

theorem modulus_monic (d : ℕ) [NeZero d] : (modulus d).Monic := by
  have monic := monic_X_pow_add_C (1 : Zq) (NeZero.ne d)
  rwa [C_1] at monic

theorem degree_modulus (d : ℕ) [NeZero d] : (modulus d).degree = d := by
  have degree := degree_X_pow_add_C (NeZero.pos d) (1 : Zq)
  rwa [C_1] at degree

/-- The representative of `x` of degree below `d`. -/
noncomputable def rep (x : Rq d) : Zq[X] := AdjoinRoot.modByMonicHom (modulus_monic d) x

/-- Coefficient `i` of `x`. -/
noncomputable def coeff (x : Rq d) (i : ℕ) : Zq := (rep x).coeff i

theorem mk_rep (x : Rq d) : AdjoinRoot.mk (modulus d) (rep x) = x :=
  AdjoinRoot.mk_leftInverse (modulus_monic d) x

theorem rep_mk (p : Zq[X]) : rep (AdjoinRoot.mk (modulus d) p) = p %ₘ modulus d :=
  AdjoinRoot.modByMonicHom_mk (modulus_monic d) p

theorem degree_rep_lt (x : Rq d) : (rep x).degree < d := by
  obtain ⟨p, rfl⟩ := AdjoinRoot.mk_surjective x
  have below := degree_modByMonic_lt p (modulus_monic d)
  rwa [degree_modulus, ← rep_mk] at below

theorem rep_mk_of_degree_lt {p : Zq[X]} (below : p.degree < d) :
    rep (AdjoinRoot.mk (modulus d) p) = p := by
  rw [rep_mk, modByMonic_eq_self_iff (modulus_monic d), degree_modulus]
  exact below

theorem coeff_eq_zero (x : Rq d) {i : ℕ} (large : d ≤ i) : coeff x i = 0 :=
  (degree_lt_iff_coeff_zero _ _).mp (degree_rep_lt x) i large

/-- Two ring elements with the same `d` coefficients are equal. -/
theorem ext_coeff {x y : Rq d} (same : ∀ i < d, coeff x i = coeff y i) : x = y := by
  rw [← mk_rep x, ← mk_rep y]
  congr 1
  ext i
  by_cases small : i < d
  · exact same i small
  · exact (coeff_eq_zero x (Nat.le_of_not_lt small)).trans
      (coeff_eq_zero y (Nat.le_of_not_lt small)).symm

@[simp] theorem coeff_zero (i : ℕ) : coeff (0 : Rq d) i = 0 := by
  simp [coeff, rep]

theorem coeff_add (x y : Rq d) (i : ℕ) : coeff (x + y) i = coeff x i + coeff y i := by
  simp [coeff, rep]

theorem coeff_sub (x y : Rq d) (i : ℕ) : coeff (x - y) i = coeff x i - coeff y i := by
  simp [coeff, rep]

theorem coeff_smul (a : Zq) (x : Rq d) (i : ℕ) : coeff (a • x) i = a * coeff x i := by
  simp [coeff, rep]

/-- The ring element with the coefficients `c 0, …, c (d − 1)`. -/
noncomputable def ofCoeffs (c : ℕ → Zq) : Rq d :=
  AdjoinRoot.mk (modulus d) (∑ i ∈ Finset.range d, monomial i (c i))

theorem coeff_ofCoeffs (c : ℕ → Zq) {i : ℕ} (small : i < d) :
    coeff (ofCoeffs (d := d) c) i = c i := by
  have below : (∑ j ∈ Finset.range d, monomial j (c j)).degree < d := by
    refine (degree_sum_le _ _).trans_lt ((Finset.sup_lt_iff (WithBot.bot_lt_coe d)).mpr ?_)
    intro j member
    exact (degree_monomial_le j (c j)).trans_lt
      (WithBot.coe_lt_coe.mpr (Finset.mem_range.mp member))
  rw [coeff, ofCoeffs, rep_mk_of_degree_lt below, finsetSum_coeff]
  simp [coeff_monomial, small]

theorem ofCoeffs_coeff (x : Rq d) : ofCoeffs (coeff x) = x :=
  ext_coeff fun _ small => coeff_ofCoeffs _ small

omit [NeZero d] in
theorem ofCoeffs_eq_sum (c : ℕ → Zq) :
    ofCoeffs (d := d) c = ∑ i ∈ Finset.range d, c i • AdjoinRoot.root (modulus d) ^ i := by
  rw [ofCoeffs, map_sum]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [← smul_X_eq_monomial, ← AdjoinRoot.smul_mk, map_pow, AdjoinRoot.mk_X]

/-- The product with `X` moves each coefficient up by one position and puts the negation of the
last coefficient at position 0. -/
theorem coeff_root_mul (x : Rq d) {i : ℕ} (small : i < d) :
    coeff (AdjoinRoot.root (modulus d) * x) i =
      if i = 0 then -coeff x (d - 1) else coeff x (i - 1) := by
  have positive := NeZero.pos d
  have reduced : (X * rep x) %ₘ modulus d =
      X * rep x - C ((rep x).coeff (d - 1)) * modulus d := by
    refine (div_modByMonic_unique (C ((rep x).coeff (d - 1))) _ (modulus_monic d)
      ⟨by ring, ?_⟩).2
    rw [degree_modulus, degree_lt_iff_coeff_zero]
    intro m large
    obtain ⟨k, rfl⟩ : ∃ k, m = k + 1 := ⟨m - 1, by omega⟩
    rw [Polynomial.coeff_sub, coeff_X_mul, modulus, coeff_C_mul, Polynomial.coeff_add,
      coeff_X_pow, coeff_one]
    by_cases top : k + 1 = d
    · subst top
      simp
    · have zero := (degree_lt_iff_coeff_zero _ _).mp (degree_rep_lt x) k (by omega)
      simp [top, zero]
  have product : AdjoinRoot.root (modulus d) * x = AdjoinRoot.mk (modulus d) (X * rep x) := by
    rw [map_mul, AdjoinRoot.mk_X, mk_rep]
  rw [coeff, product, rep_mk, reduced, Polynomial.coeff_sub, modulus, coeff_C_mul,
    Polynomial.coeff_add, coeff_X_pow, coeff_one]
  rcases i with _ | k
  · simp [coeff, positive.ne]
  · have low : k + 1 ≠ d := Nat.ne_of_lt small
    simp [coeff, low]

end Coefficients

section Magnitude

/-- The absolute value of the representative of `a` in `(−q/2, q/2]`. -/
def magnitude (a : Zq) : ℕ := a.valMinAbs.natAbs

@[simp] theorem magnitude_zero : magnitude 0 = 0 := by
  simp [magnitude]

theorem magnitude_neg (a : Zq) : magnitude (-a) = magnitude a :=
  ZMod.natAbs_valMinAbs_neg a

theorem magnitude_add_le (a b : Zq) : magnitude (a + b) ≤ magnitude a + magnitude b :=
  (ZMod.natAbs_valMinAbs_add_le a b).trans (Int.natAbs_add_le _ _)

theorem magnitude_sub_le (a b : Zq) : magnitude (a - b) ≤ magnitude a + magnitude b := by
  rw [sub_eq_add_neg, ← magnitude_neg b]
  exact magnitude_add_le a (-b)

/-- The magnitude of the residue of an integer is at most its absolute value. -/
theorem magnitude_intCast_le (y : ℤ) : magnitude (y : Zq) ≤ y.natAbs :=
  ZMod.natAbs_min_of_le_div_two fieldModulus _ y (ZMod.coe_valMinAbs _)
    (ZMod.natAbs_valMinAbs_le _)

theorem magnitude_mul_le (a b : Zq) : magnitude (a * b) ≤ magnitude a * magnitude b := by
  have product : a * b = ((a.valMinAbs * b.valMinAbs : ℤ) : Zq) := by
    rw [Int.cast_mul, ZMod.coe_valMinAbs, ZMod.coe_valMinAbs]
  rw [product]
  exact (magnitude_intCast_le _).trans (Int.natAbs_mul _ _).le

/-- `magnitude` is `centeredMagnitude` (`Spec/Algebra.lean`). -/
theorem magnitude_eq_min (a : Zq) : magnitude a = min a.val (fieldModulus - a.val) :=
  ZMod.valMinAbs_natAbs_eq_min a

end Magnitude

section Norms

variable {d : ℕ} [NeZero d]

/-- `‖x‖_∞`, the largest magnitude of a coefficient of `x` (Akita Section 3.1). -/
noncomputable def normInf (x : Rq d) : ℕ := (Finset.range d).sup fun i => magnitude (coeff x i)

/-- `‖x‖_1`, the sum of the magnitudes of the coefficients of `x` (Akita Section 3.1). -/
noncomputable def normOne (x : Rq d) : ℕ := ∑ i ∈ Finset.range d, magnitude (coeff x i)

/-- `‖v‖_∞` of a vector, over all coefficients of its entries (Akita Section 3.1). -/
noncomputable def vecNormInf {ι : Type} [Fintype ι] (v : ι → Rq d) : ℕ :=
  Finset.univ.sup fun j => normInf (v j)

theorem normInf_le_iff (x : Rq d) (bound : ℕ) :
    normInf x ≤ bound ↔ ∀ i < d, magnitude (coeff x i) ≤ bound := by
  simp [normInf, Finset.sup_le_iff]

theorem vecNormInf_le_iff {ι : Type} [Fintype ι] (v : ι → Rq d) (bound : ℕ) :
    vecNormInf v ≤ bound ↔ ∀ j, normInf (v j) ≤ bound := by
  simp [vecNormInf, Finset.sup_le_iff]

theorem normInf_le_vecNormInf {ι : Type} [Fintype ι] (v : ι → Rq d) (j : ι) :
    normInf (v j) ≤ vecNormInf v :=
  (vecNormInf_le_iff v _).mp le_rfl j

@[simp] theorem normInf_zero : normInf (0 : Rq d) = 0 :=
  Nat.le_zero.mp ((normInf_le_iff _ _).mpr fun i _ => by simp)

theorem normInf_add_le (x y : Rq d) : normInf (x + y) ≤ normInf x + normInf y := by
  refine (normInf_le_iff _ _).mpr fun i small => ?_
  rw [coeff_add]
  exact (magnitude_add_le _ _).trans
    (Nat.add_le_add ((normInf_le_iff x _).mp le_rfl i small) ((normInf_le_iff y _).mp le_rfl i small))

/-- `‖x − y‖_∞ ≤ ‖x‖_∞ + ‖y‖_∞`. -/
theorem normInf_sub_le (x y : Rq d) : normInf (x - y) ≤ normInf x + normInf y := by
  refine (normInf_le_iff _ _).mpr fun i small => ?_
  rw [coeff_sub]
  exact (magnitude_sub_le _ _).trans
    (Nat.add_le_add ((normInf_le_iff x _).mp le_rfl i small) ((normInf_le_iff y _).mp le_rfl i small))

theorem normInf_sum_le {ι : Type} (s : Finset ι) (f : ι → Rq d) :
    normInf (∑ i ∈ s, f i) ≤ ∑ i ∈ s, normInf (f i) :=
  Finset.le_sum_of_subadditive normInf normInf_zero.le normInf_add_le s f

theorem normInf_smul_le (a : Zq) (x : Rq d) : normInf (a • x) ≤ magnitude a * normInf x := by
  refine (normInf_le_iff _ _).mpr fun i small => ?_
  rw [coeff_smul]
  exact (magnitude_mul_le _ _).trans
    (Nat.mul_le_mul_left _ ((normInf_le_iff x _).mp le_rfl i small))

theorem normInf_root_mul_le (x : Rq d) : normInf (AdjoinRoot.root (modulus d) * x) ≤ normInf x := by
  refine (normInf_le_iff _ _).mpr fun i small => ?_
  have positive := NeZero.pos d
  rw [coeff_root_mul x small]
  split
  · rw [magnitude_neg]
    exact (normInf_le_iff x _).mp le_rfl _ (by omega)
  · exact (normInf_le_iff x _).mp le_rfl _ (by omega)

theorem normInf_root_pow_mul_le (k : ℕ) (x : Rq d) :
    normInf (AdjoinRoot.root (modulus d) ^ k * x) ≤ normInf x := by
  induction k with
  | zero => rw [pow_zero, one_mul]
  | succ k step =>
      rw [pow_succ', mul_assoc]
      exact (normInf_root_mul_le _).trans step

/-- Akita Lemma 3.1, inequality (18): `‖c·s‖_∞ ≤ ‖c‖_1 · ‖s‖_∞`. -/
theorem normInf_mul_le (c s : Rq d) : normInf (c * s) ≤ normOne c * normInf s := by
  have expanded : c * s =
      ∑ i ∈ Finset.range d, coeff c i • (AdjoinRoot.root (modulus d) ^ i * s) := by
    conv_lhs => rw [← ofCoeffs_coeff c, ofCoeffs_eq_sum, Finset.sum_mul]
    exact Finset.sum_congr rfl fun i _ => smul_mul_assoc _ _ _
  rw [expanded, normOne, Finset.sum_mul]
  refine (normInf_sum_le _ _).trans (Finset.sum_le_sum fun i _ => ?_)
  exact (normInf_smul_le _ _).trans (Nat.mul_le_mul_left _ (normInf_root_pow_mul_le i s))

/-- Inequality (18) for a vector: `‖c·v‖_∞ ≤ ‖c‖_1 · ‖v‖_∞`. -/
theorem vecNormInf_smul_le {ι : Type} [Fintype ι] (c : Rq d) (v : ι → Rq d) :
    vecNormInf (c • v) ≤ normOne c * vecNormInf v :=
  (vecNormInf_le_iff _ _).mpr fun j => (normInf_mul_le c (v j)).trans
    (Nat.mul_le_mul_left _ (normInf_le_vecNormInf v j))

/-- `‖v − w‖_∞ ≤ ‖v‖_∞ + ‖w‖_∞` for vectors. -/
theorem vecNormInf_sub_le {ι : Type} [Fintype ι] (v w : ι → Rq d) :
    vecNormInf (v - w) ≤ vecNormInf v + vecNormInf w :=
  (vecNormInf_le_iff _ _).mpr fun j => (normInf_sub_le (v j) (w j)).trans
    (Nat.add_le_add (normInf_le_vecNormInf v j) (normInf_le_vecNormInf w j))

end Norms

section ModuleSis

variable {d : ℕ} [NeZero d] {rows cols : Type} [Fintype cols]

/-- A solution of `MSIS_{q,d}(n, m, β)` for the matrix `M` (Akita Definition 3.2 with `p = ∞`): a
vector `z ≠ 0` with `M z = 0` and `‖z‖_∞ ≤ β`. -/
structure ShortKernel (M : Matrix rows cols (Rq d)) (bound : ℕ) where
  vector : cols → Rq d
  nonzero : vector ≠ 0
  kernel : M.mulVec vector = 0
  short : vecNormInf vector ≤ bound

/-- `x` and `x'` are two inputs of `M` with one image, and `‖x − x'‖_∞ ≤ β`. -/
def Collides (M : Matrix rows cols (Rq d)) (bound : ℕ) (x x' : cols → Rq d) : Prop :=
  x ≠ x' ∧ M.mulVec x = M.mulVec x' ∧ vecNormInf (x - x') ≤ bound

open Classical in
/-- The solution `x − x'` of Module-SIS for `M`, when `x` and `x'` collide. -/
noncomputable def ShortKernel.ofCollision? (M : Matrix rows cols (Rq d)) (bound : ℕ)
    (x x' : cols → Rq d) : Option (ShortKernel M bound) :=
  if collides : Collides M bound x x' then
    some ⟨x - x', sub_ne_zero.mpr collides.1,
      by rw [Matrix.mulVec_sub, collides.2.1, sub_self], collides.2.2⟩
  else none

theorem ShortKernel.ofCollision?_isSome {M : Matrix rows cols (Rq d)} {bound : ℕ}
    {x x' : cols → Rq d} (collides : Collides M bound x x') :
    (ShortKernel.ofCollision? M bound x x').isSome := by
  unfold ofCollision?
  split
  · rfl
  · contradiction

end ModuleSis

end NightstreamFPrime.Spec.Compression.Akita
