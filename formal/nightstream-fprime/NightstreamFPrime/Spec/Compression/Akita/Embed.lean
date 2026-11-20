import Mathlib.Algebra.Polynomial.Expand
import NightstreamFPrime.Spec.Compression.Akita.Ring

/-! The embedding of the challenge subring in the A ring (Akita, ePrint 2026/1983, revision of
2026-10-15, Section 4.4).

- `embed stride tower` is `ι : S → R`, `Y ↦ X^stride`, of Akita (38), for `S = R_{q,d_car}`,
  `R = R_{q,d}` and `stride · d_car = d`. A packing fold has `stride = k·h_car`.
- `coeff_embed` gives its coefficients. `isUnit_embed`, `normOne_embed` and `normInf_embed` are
  items 1 and 2 of Akita Lemma 4.2 for `p ∈ {1, ∞}`, and `normInf_embed_mul_le` is
  `‖ι(c)·u‖_∞ ≤ ‖c‖_1 · ‖u‖_∞` of Section 4.4.
- `plane d_car stride a` gives coordinate `a` of an element of `R` in the basis `(X^a)_a` of `R`
  over `ι(S)` (Akita (39)). `eq_of_planes` states that these coordinates fix the element, and
  `plane_embed_mul` states that each of them is linear over `S`, which is the content of
  Akita (61). -/

namespace NightstreamFPrime.Spec.Compression.Akita

open Polynomial

variable {d : ℕ} [NeZero d]

/-- The product with `X^s` for `s ≤ d` moves each coefficient up by `s` positions and puts the
negations of the last `s` coefficients at the positions below `s`. -/
theorem coeff_root_pow_mul {s : ℕ} (le : s ≤ d) (x : Rq d) {i : ℕ} (small : i < d) :
    coeff (AdjoinRoot.root (modulus d) ^ s * x) i =
      if i < s then -coeff x (d + i - s) else coeff x (i - s) := by
  induction s generalizing i with
  | zero => simp
  | succ s step =>
      rw [pow_succ', mul_assoc, coeff_root_mul _ small]
      by_cases first : i = 0
      · subst first
        rw [ite_eq_left rfl, step (Nat.le_of_succ_le le) (by omega : d - 1 < d),
          ite_eq_right (by omega), ite_eq_left (by omega)]
        exact congrArg (fun k => -coeff x k) (by omega)
      · rw [ite_eq_right first, step (Nat.le_of_succ_le le) (by omega : i - 1 < d)]
        by_cases low : i < s + 1
        · rw [ite_eq_left (by omega), ite_eq_left low]
          exact congrArg (fun k => -coeff x k) (by omega)
        · rw [ite_eq_right (by omega), ite_eq_right low]
          exact congrArg (coeff x) (by omega)

omit [NeZero d] in
theorem root_pow_add_one : AdjoinRoot.root (modulus d) ^ d + 1 = 0 := by
  have zero := AdjoinRoot.mk_self (f := modulus d)
  rwa [modulus, map_add, map_pow, AdjoinRoot.mk_X, map_one] at zero

variable {dCar : ℕ} [NeZero dCar] {stride : ℕ}

omit [NeZero dCar] in
theorem stride_pos (tower : stride * dCar = d) : 0 < stride :=
  Nat.pos_of_ne_zero fun zero => NeZero.ne d (by rw [← tower, zero, Nat.zero_mul])

/-- `ι : S → R`, `Y ↦ X^stride` (Akita (38)), for `stride · d_car = d`. -/
noncomputable def embed (stride : ℕ) (tower : stride * dCar = d) : Rq dCar →ₐ[Zq] Rq d :=
  AdjoinRoot.liftAlgHom (modulus dCar) (Algebra.ofId Zq (Rq d))
    (AdjoinRoot.root (modulus d) ^ stride) (by
      show eval₂ _ _ (X ^ dCar + 1) = 0
      rw [eval₂_add, eval₂_pow, eval₂_X, eval₂_one, ← pow_mul, tower]
      exact root_pow_add_one)

omit [NeZero d] [NeZero dCar] in
theorem embed_root (tower : stride * dCar = d) :
    embed stride tower (AdjoinRoot.root (modulus dCar)) = AdjoinRoot.root (modulus d) ^ stride :=
  AdjoinRoot.liftAlgHom_root _ _ _ _

omit [NeZero d] [NeZero dCar] in
theorem embed_mk (tower : stride * dCar = d) (p : Zq[X]) :
    embed stride tower (AdjoinRoot.mk (modulus dCar) p) =
      AdjoinRoot.mk (modulus d) (expand Zq stride p) := by
  rw [embed, AdjoinRoot.liftAlgHom_mk, expand_eq_comp_X_pow, comp, hom_eval₂, map_pow,
    AdjoinRoot.mk_X]
  rfl

/-- `ι` puts coefficient `j` of `c` at position `stride · j` and puts 0 at each other position
(Akita Lemma 4.2, proof). -/
theorem coeff_embed (tower : stride * dCar = d) (c : Rq dCar) (i : ℕ) :
    coeff (embed stride tower c) i = if stride ∣ i then coeff c (i / stride) else 0 := by
  have positive := stride_pos tower
  have below : (expand Zq stride (rep c)).degree < d := by
    rw [degree_lt_iff_coeff_zero]
    intro m large
    rw [coeff_expand positive]
    split
    · exact coeff_eq_zero c ((Nat.le_div_iff_mul_le positive).mpr
        (by rw [Nat.mul_comm, tower]; exact large))
    · rfl
  conv_lhs => rw [← mk_rep c]
  rw [embed_mk, coeff, rep_mk_of_degree_lt below, coeff_expand positive]
  rfl

theorem coeff_embed_mul (tower : stride * dCar = d) (c : Rq dCar) (j : ℕ) :
    coeff (embed stride tower c) (stride * j) = coeff c j := by
  rw [coeff_embed, ite_eq_left (Nat.dvd_mul_right _ _), Nat.mul_div_cancel_left _ (stride_pos tower)]

omit [NeZero d] [NeZero dCar] in
/-- Akita Lemma 4.2, item 1: `ι` sends a unit of `S` to a unit of `R`. -/
theorem isUnit_embed (tower : stride * dCar = d) {c : Rq dCar} (unit : IsUnit c) :
    IsUnit (embed stride tower c) :=
  unit.map _

/-- Akita Lemma 4.2, item 2, for `p = ∞`: `‖ι(c)‖_∞ = ‖c‖_∞`. -/
theorem normInf_embed (tower : stride * dCar = d) (c : Rq dCar) :
    normInf (embed stride tower c) = normInf c := by
  have positive := stride_pos tower
  refine le_antisymm ((normInf_le_iff _ _).mpr fun i small => ?_)
    ((normInf_le_iff _ _).mpr fun j small => ?_)
  · rw [coeff_embed]
    split
    · exact (normInf_le_iff c _).mp le_rfl _ (Nat.div_lt_of_lt_mul (by rwa [tower]))
    · simp
  · rw [← coeff_embed_mul tower c j]
    exact (normInf_le_iff _ _).mp le_rfl _
      (by rw [← tower]; exact (Nat.mul_lt_mul_left positive).mpr small)

/-- Akita Lemma 4.2, item 2, for `p = 1`: `‖ι(c)‖_1 = ‖c‖_1`. -/
theorem normOne_embed (tower : stride * dCar = d) (c : Rq dCar) :
    normOne (embed stride tower c) = normOne c := by
  have positive := stride_pos tower
  have image : ∑ j ∈ Finset.range dCar, magnitude (coeff c j) =
      ∑ i ∈ (Finset.range dCar).image (stride * ·),
        magnitude (coeff (embed stride tower c) i) := by
    rw [Finset.sum_image fun _ _ _ _ same => Nat.eq_of_mul_eq_mul_left positive same]
    exact Finset.sum_congr rfl fun j _ => by rw [coeff_embed_mul]
  rw [normOne, normOne, image]
  symm
  refine Finset.sum_subset ?_ ?_
  · intro i member
    obtain ⟨j, small, rfl⟩ := Finset.mem_image.mp member
    exact Finset.mem_range.mpr
      (by rw [← tower]; exact (Nat.mul_lt_mul_left positive).mpr (Finset.mem_range.mp small))
  · intro i inside outside
    rw [coeff_embed]
    by_cases divides : stride ∣ i
    · exact absurd (Finset.mem_image.mpr ⟨i / stride, Finset.mem_range.mpr
        (Nat.div_lt_of_lt_mul (by rw [tower]; exact Finset.mem_range.mp inside)),
        Nat.mul_div_cancel' divides⟩) outside
    · rw [ite_eq_right divides, magnitude_zero]

/-- `‖ι(c)·u‖_∞ ≤ ‖c‖_1 · ‖u‖_∞` for a packing challenge `c` and an element `u` of the A ring
(Akita Section 4.4). -/
theorem normInf_embed_mul_le (tower : stride * dCar = d) (c : Rq dCar) (u : Rq d) :
    normInf (embed stride tower c * u) ≤ normOne c * normInf u :=
  (normInf_mul_le _ u).trans_eq (by rw [normOne_embed])

/-- Coordinate `a` of an element of `R` in the decomposition `R = ⊕_a X^a · ι(S)` of Akita (39):
coefficient `j` of the coordinate is coefficient `a + stride · j` of the element. -/
noncomputable def plane (dCar : ℕ) [NeZero dCar] (stride a : ℕ) : Rq d →ₗ[Zq] Rq dCar where
  toFun x := ofCoeffs fun j => coeff x (a + stride * j)
  map_add' x y := ext_coeff fun j small => by
    rw [coeff_add, coeff_ofCoeffs _ small, coeff_ofCoeffs _ small, coeff_ofCoeffs _ small,
      coeff_add]
  map_smul' r x := ext_coeff fun j small => by
    rw [RingHom.id_apply, coeff_smul, coeff_ofCoeffs _ small, coeff_ofCoeffs _ small, coeff_smul]

theorem coeff_plane (stride a : ℕ) (x : Rq d) {j : ℕ} (small : j < dCar) :
    coeff (plane dCar stride a x) j = coeff x (a + stride * j) :=
  coeff_ofCoeffs _ small

omit [NeZero d] [NeZero dCar] in
theorem plane_address_lt (tower : stride * dCar = d) {a j : ℕ} (low : a < stride)
    (small : j < dCar) : a + stride * j < d :=
  calc a + stride * j < stride * (j + 1) := by rw [Nat.mul_succ]; omega
    _ ≤ stride * dCar := Nat.mul_le_mul_left _ small
    _ = d := tower

/-- Akita (39): an element of `R` is fixed by its `stride` coordinates. -/
theorem eq_of_planes (tower : stride * dCar = d) {x y : Rq d}
    (same : ∀ a < stride, plane dCar stride a x = plane dCar stride a y) : x = y := by
  refine ext_coeff fun i small => ?_
  have quotient : i / stride < dCar := Nat.div_lt_of_lt_mul (by rw [tower]; exact small)
  have entry := congrArg (fun z => coeff z (i / stride))
    (same (i % stride) (Nat.mod_lt i (stride_pos tower)))
  simp only [coeff_plane _ _ _ quotient, Nat.mod_add_div] at entry
  exact entry

/-- The product with `ι(Y) = X^stride` in `R` is the product with `Y` on each coordinate: a wrap
at `X^d` gives the same sign as a wrap at `Y^{d_car}`. -/
theorem plane_root_pow_mul (tower : stride * dCar = d) {a : ℕ} (low : a < stride) (x : Rq d) :
    plane dCar stride a (AdjoinRoot.root (modulus d) ^ stride * x) =
      AdjoinRoot.root (modulus dCar) * plane dCar stride a x := by
  have positive := NeZero.pos dCar
  have top : stride * dCar = stride * (dCar - 1) + stride := by
    rw [← Nat.mul_succ]
    exact congrArg (stride * ·) (by omega)
  refine ext_coeff fun j small => ?_
  rw [coeff_plane _ _ _ small, coeff_root_pow_mul (tower ▸ Nat.le_mul_of_pos_right _ positive) _
    (plane_address_lt tower low small), coeff_root_mul _ small]
  rcases j with _ | n
  · rw [ite_eq_left (by omega), ite_eq_left rfl, coeff_plane _ _ _ (by omega : dCar - 1 < dCar)]
    exact congrArg (fun k => -coeff x k) (by omega)
  · have step : stride * (n + 1) = stride * n + stride := Nat.mul_succ _ _
    rw [ite_eq_right (by omega), ite_eq_right (by omega), coeff_plane _ _ _ (by omega : n + 1 - 1 < dCar)]
    exact congrArg (coeff x) (by rw [Nat.add_sub_cancel]; omega)

/-- Each coordinate is linear over `S`: `plane a (ι(c)·x) = c · plane a x`. This is the content
of Akita (61). -/
theorem plane_embed_mul (tower : stride * dCar = d) {a : ℕ} (low : a < stride) (c : Rq dCar)
    (x : Rq d) : plane dCar stride a (embed stride tower c * x) = c * plane dCar stride a x := by
  have power (j : ℕ) : plane dCar stride a ((AdjoinRoot.root (modulus d) ^ stride) ^ j * x) =
      AdjoinRoot.root (modulus dCar) ^ j * plane dCar stride a x := by
    induction j with
    | zero => rw [pow_zero, pow_zero, one_mul, one_mul]
    | succ j step =>
        rw [pow_succ', mul_assoc, plane_root_pow_mul tower low, step, ← mul_assoc, ← pow_succ']
  have expanded : c =
      ∑ j ∈ Finset.range dCar, coeff c j • AdjoinRoot.root (modulus dCar) ^ j := by
    rw [← ofCoeffs_eq_sum, ofCoeffs_coeff]
  rw [expanded]
  simp only [map_sum, map_smul, map_pow, embed_root, Finset.sum_mul, smul_mul_assoc, power]

end NightstreamFPrime.Spec.Compression.Akita
