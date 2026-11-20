import Mathlib.Algebra.QuadraticAlgebra.Basic
import NightstreamFPrime.Spec.Compression.Akita.Commit
import NightstreamFPrime.Spec.Compression.Akita.Embed

/-! The extension field in the ring, the packing map `ψ` and the trace identity of Akita for
`k = 2` (ePrint 2026/1983, revision of 2026-10-15, Section 3.1 with (21), and Section 7.1 with
(96) and (97)).

The tower is `d = 2·k·m` with `k = 2`, so `d = 4·m`. The extension field is
`Quadratic = Z_q[u]/(u² − 2)`, which `FieldExtensionRing.carrierEquiv` identifies with the
challenge field `K` of the package.

- `conj` is `σ_{−1} : X ↦ X^{−1}`, and `coeff_mul_conj` is the trace identity for `k = 1`: the
  constant coefficient of `a·σ_{−1}(b)` is `Σ_i a_i·b_i`.
- `fixed d m` is `e_1 = X^m + X^{−m} = X^m − X^{d−m}`, with `e_1² = 2` (`fixed_mul_self`) and
  `σ_{−1}(e_1) = e_1` (`conj_fixed`). `subfield` is the embedding `u ↦ e_1` of the extension field
  in `R_{q,d}`, with the basis `(1, e_1)` of the fixed subfield.
- `psi` is the packing map `ψ` of Akita (21): entry `t < m` of the first bank is at `X^t`, and
  entry `t` of the second bank, which is entry `m + t` of the input, is at `X^{t + d/2}`.
- `traceForm d m` is `Tr_H` in the coordinates of the basis `(1, e_1)`:
  `Tr_H(w) = (d/2)·w_0 + (d/4)·(e_1·w)_0·u`. The file has no proof that this map is the sum of
  the automorphisms of `H = ⟨σ_{−1}, σ_9⟩`. `traceForm_subfield_mul` states that it is linear over
  the extension field.
- `traceForm_psi_mul_conj` is the trace identity of Akita Section 3.1 for `k = 2`:
  `Tr_H(ψ(a)·σ_{−1}(ψ(b))) = (d/k)·⟨a, b⟩`.
- `traceFunctional` is `(d/k)·T_{ρ_pack}` of Akita (97), and `traceFunctional_psi` gives its value
  `(d/k)·Σ_u f_u·ẽq(ρ_pack, u)` on a packed element, which is the content of the row (98). -/

namespace NightstreamFPrime.Spec.Compression.Akita

open Polynomial

/-- The extension field `E = Z_q[u]/(u² − 2)` of degree `k = 2`. -/
abbrev Quadratic : Type := QuadraticAlgebra Zq 2 0

section Fixed

variable {d m : ℕ}

theorem root_pow_degree : AdjoinRoot.root (modulus d) ^ d = -1 :=
  eq_neg_of_add_eq_zero_left root_pow_add_one

/-- `e_1 = X^m + X^{−m} = X^m − X^{d−m}` for `d = 4·m` (Akita Section 3.1, the basis of the
fixed subfield for `k = 2`). -/
noncomputable def fixed (d m : ℕ) : Rq d :=
  AdjoinRoot.root (modulus d) ^ m - AdjoinRoot.root (modulus d) ^ (3 * m)

/-- `e_1² = 2`. -/
theorem fixed_mul_self (quarter : 4 * m = d) :
    fixed d m * fixed d m = (2 : Zq) • 1 + (0 : Zq) • fixed d m := by
  have top : AdjoinRoot.root (modulus d) ^ (4 * m) = -1 := by
    rw [quarter]
    exact root_pow_degree
  rw [zero_smul, add_zero, Algebra.smul_def, mul_one, map_ofNat, fixed]
  linear_combination (AdjoinRoot.root (modulus d) ^ (2 * m) - 2) * top

/-- The embedding of the extension field in `R_{q,d}` as the fixed subfield: `u ↦ e_1`
(Akita Section 3.1). -/
noncomputable def subfield (quarter : 4 * m = d) : Quadratic →ₐ[Zq] Rq d :=
  QuadraticAlgebra.lift ⟨fixed d m, fixed_mul_self quarter⟩

theorem subfield_apply (quarter : 4 * m = d) (z : Quadratic) :
    subfield quarter z = z.re • 1 + z.im • fixed d m :=
  rfl

/-- The position in `R_{q,d}` of entry `u` of the input of `ψ` (Akita (21)): `X^u` for the first
bank `u < m`, and `X^{t + d/2}` for entry `u = m + t` of the second bank. -/
def slot (m u : ℕ) : ℕ := if u < m then u else u + m

theorem slot_range {u : ℕ} (small : u < 2 * m) :
    slot m u < m ∨ 2 * m ≤ slot m u ∧ slot m u < 3 * m := by
  unfold slot
  split <;> omega

theorem slot_injective {u v : ℕ} (same : slot m u = slot m v) : u = v := by
  unfold slot at same
  split at same <;> split at same <;> omega

/-- The packing map `ψ` of Akita (21) for `k = 2`, on the `d/k = 2·m` entries of its input:
`ψ(a) = Σ_{t<m} X^t·(a_t + X^{d/2}·a_{m+t})`. -/
noncomputable def psi (quarter : 4 * m = d) (a : ℕ → Quadratic) : Rq d :=
  ∑ u ∈ Finset.range (2 * m), AdjoinRoot.root (modulus d) ^ slot m u * subfield quarter (a u)

end Fixed

variable {d : ℕ} [NeZero d] {m : ℕ}

theorem coeff_root_pow {n : ℕ} (small : n < d) {i : ℕ} (inside : i < d) :
    coeff (AdjoinRoot.root (modulus d) ^ n) i = if i = n then 1 else 0 := by
  have product := coeff_root_pow_mul small.le (1 : Rq d) inside
  rw [mul_one, coeff_one, coeff_one] at product
  rw [product]
  by_cases low : i < n
  · rw [ite_eq_left low, ite_eq_right (by omega : ¬ d + i - n = 0), neg_zero,
      ite_eq_right (by omega : ¬ i = n)]
  · by_cases same : i = n
    · rw [ite_eq_right low, ite_eq_left (by omega : i - n = 0), ite_eq_left same]
    · rw [ite_eq_right low, ite_eq_right (by omega : ¬ i - n = 0), ite_eq_right same]

theorem coeff_root_pow_of_ne {n i : ℕ} (small : n < d) (inside : i < d) (other : i ≠ n) :
    coeff (AdjoinRoot.root (modulus d) ^ n) i = 0 := by
  rw [coeff_root_pow small inside, ite_eq_right other]

/-- `σ_{−1} : X ↦ X^{−1} = −X^{d−1}` (Akita Section 3.1). -/
noncomputable def conj : Rq d →ₐ[Zq] Rq d :=
  AdjoinRoot.liftAlgHom (modulus d) (Algebra.ofId Zq (Rq d))
    (-AdjoinRoot.root (modulus d) ^ (d - 1)) (by
      have positive := NeZero.pos d
      show eval₂ _ _ (X ^ d + 1) = 0
      rw [eval₂_add, eval₂_pow, eval₂_X, eval₂_one, neg_pow, ← pow_mul, Nat.mul_comm, pow_mul,
        root_pow_degree, ← pow_add, Odd.neg_one_pow ⟨d - 1, by omega⟩, neg_add_cancel])

theorem conj_root :
    conj (AdjoinRoot.root (modulus d)) = -AdjoinRoot.root (modulus d) ^ (d - 1) :=
  AdjoinRoot.liftAlgHom_root _ _ _ _

theorem root_mul_conj_root :
    AdjoinRoot.root (modulus d) * conj (AdjoinRoot.root (modulus d)) = 1 := by
  rw [conj_root, mul_neg, ← pow_succ', Nat.sub_add_cancel (NeZero.pos d), root_pow_degree,
    neg_neg]

/-- `σ_{−1}(X^i) = −X^j` for `i + j = d`. -/
theorem conj_root_pow {i j : ℕ} (sum : i + j = d) :
    conj (AdjoinRoot.root (modulus d) ^ i) = -AdjoinRoot.root (modulus d) ^ j := by
  rw [map_pow]
  refine left_inv_eq_right_inv (a := AdjoinRoot.root (modulus d) ^ i) ?_ ?_
  · rw [← mul_pow, mul_comm, root_mul_conj_root, one_pow]
  · rw [mul_neg, ← pow_add, sum, root_pow_degree, neg_neg]

theorem coeff_conj_root_pow_mul {i : ℕ} (small : i < d) (a : Rq d) :
    coeff (conj (AdjoinRoot.root (modulus d)) ^ i * a) 0 = coeff a i := by
  have back : a = AdjoinRoot.root (modulus d) ^ i *
      (conj (AdjoinRoot.root (modulus d)) ^ i * a) := by
    rw [← mul_assoc, ← mul_pow, root_mul_conj_root, one_pow, one_mul]
  conv_rhs =>
    rw [back, coeff_root_pow_mul small.le _ small, ite_eq_right (Nat.lt_irrefl i), Nat.sub_self]

/-- The trace identity of Akita Section 3.1 for `k = 1`: the constant coefficient of
`a·σ_{−1}(b)` is `Σ_i a_i·b_i`. -/
theorem coeff_mul_conj (a b : Rq d) :
    coeff (a * conj b) 0 = ∑ i ∈ Finset.range d, coeff a i * coeff b i := by
  have expanded : conj b =
      ∑ i ∈ Finset.range d, coeff b i • conj (AdjoinRoot.root (modulus d)) ^ i := by
    conv_lhs => rw [← ofCoeffs_coeff b, ofCoeffs_eq_sum]
    simp only [map_sum, map_smul, map_pow]
  rw [expanded, Finset.mul_sum, coeff_sum]
  refine Finset.sum_congr rfl fun i member => ?_
  rw [mul_smul_comm, coeff_smul, mul_comm a,
    coeff_conj_root_pow_mul (Finset.mem_range.mp member), mul_comm]

theorem coeff_mul_conj_root_pow (a : Rq d) {n : ℕ} (small : n < d) :
    coeff (a * conj (AdjoinRoot.root (modulus d) ^ n)) 0 = coeff a n := by
  rw [coeff_mul_conj, Finset.sum_eq_single n]
  · rw [coeff_root_pow small small, ite_eq_left rfl, mul_one]
  · intro i member other
    rw [coeff_root_pow small (Finset.mem_range.mp member), ite_eq_right other, mul_zero]
  · intro absent
    exact absurd (Finset.mem_range.mpr small) absent

/-- `σ_{−1}(e_1) = e_1`. -/
theorem conj_fixed (quarter : 4 * m = d) : conj (fixed d m) = fixed d m := by
  rw [fixed, map_sub, conj_root_pow (by omega : m + 3 * m = d),
    conj_root_pow (by omega : 3 * m + m = d)]
  ring

/-- `σ_{−1}` fixes the embedded extension field. -/
theorem conj_subfield (quarter : 4 * m = d) (z : Quadratic) :
    conj (subfield quarter z) = subfield quarter z := by
  rw [subfield_apply, map_add, map_smul, map_smul, map_one, conj_fixed quarter]

/-- The product of `e_1` and a position of `ψ` has no coefficient at a position of `ψ`. -/
theorem coeff_fixed_mul_root_pow (quarter : 4 * m = d) {p r : ℕ}
    (source : p < m ∨ 2 * m ≤ p ∧ p < 3 * m) (target : r < m ∨ 2 * m ≤ r ∧ r < 3 * m) :
    coeff (fixed d m * AdjoinRoot.root (modulus d) ^ p) r = 0 := by
  have psmall : p < d := by omega
  have rsmall : r < d := by omega
  rw [fixed, sub_mul, coeff_sub, coeff_root_pow_mul (by omega) _ rsmall,
    coeff_root_pow_mul (by omega) _ rsmall, ite_eq_left (by omega : r < 3 * m)]
  by_cases low : r < m
  · rw [ite_eq_left low, coeff_root_pow_of_ne (i := d + r - m) psmall (by omega) (by omega),
      coeff_root_pow_of_ne (i := d + r - 3 * m) psmall (by omega) (by omega), neg_zero, sub_zero]
  · rw [ite_eq_right low, coeff_root_pow_of_ne (i := r - m) psmall (by omega) (by omega),
      coeff_root_pow_of_ne (i := d + r - 3 * m) psmall (by omega) (by omega), neg_zero, sub_zero]

/-- `Tr_H` for `k = 2` and `d = 4·m` in the coordinates of the basis `(1, e_1)` of the fixed
subfield: `Tr_H(w) = (d/2)·w_0 + (d/4)·(e_1·w)_0·u` (Akita Section 3.1). -/
noncomputable def traceForm (d m : ℕ) [NeZero d] : Rq d →ₗ[Zq] Quadratic where
  toFun w := ⟨((2 * m : ℕ) : Zq) * coeff w 0, ((m : ℕ) : Zq) * coeff (fixed d m * w) 0⟩
  map_add' x y := QuadraticAlgebra.ext
    (by simp only [QuadraticAlgebra.re_add, coeff_add]; ring)
    (by simp only [QuadraticAlgebra.im_add, mul_add, coeff_add])
  map_smul' r x := QuadraticAlgebra.ext
    (by simp only [QuadraticAlgebra.re_smul, coeff_smul, smul_eq_mul, RingHom.id_apply]; ring)
    (by simp only [QuadraticAlgebra.im_smul, mul_smul_comm, coeff_smul, smul_eq_mul,
      RingHom.id_apply]; ring)

/-- `Tr_H` is linear over the embedded extension field: `Tr_H(c·w) = c·Tr_H(w)`. -/
theorem traceForm_subfield_mul (quarter : 4 * m = d) (c : Quadratic) (w : Rq d) :
    traceForm d m (subfield quarter c * w) = c * traceForm d m w := by
  have square : fixed d m * (fixed d m * w) = (2 : Zq) • w := by
    rw [← mul_assoc, fixed_mul_self quarter, zero_smul, add_zero, smul_mul_assoc, one_mul]
  have product : subfield quarter c * w = c.re • w + c.im • (fixed d m * w) := by
    rw [subfield_apply, add_mul, smul_mul_assoc, one_mul, smul_mul_assoc]
  have shifted : fixed d m * (subfield quarter c * w) =
      c.re • (fixed d m * w) + (c.im * 2) • w := by
    rw [product, mul_add, mul_smul_comm, mul_smul_comm, square, smul_smul]
  refine QuadraticAlgebra.ext ?_ ?_
  · simp only [traceForm, LinearMap.coe_mk, AddHom.coe_mk, QuadraticAlgebra.re_mul]
    rw [product, coeff_add, coeff_smul, coeff_smul]
    push_cast
    ring
  · simp only [traceForm, LinearMap.coe_mk, AddHom.coe_mk, QuadraticAlgebra.im_mul]
    rw [shifted, coeff_add, coeff_smul, coeff_smul]
    push_cast
    ring

/-- `Tr_H(X^{p(u)}·σ_{−1}(X^{p(v)}))` is `d/k` for `u = v` and 0 for two different positions of
`ψ`. -/
theorem traceForm_slot (quarter : 4 * m = d) {u v : ℕ} (left : u < 2 * m) (right : v < 2 * m) :
    traceForm d m (AdjoinRoot.root (modulus d) ^ slot m u *
        conj (AdjoinRoot.root (modulus d) ^ slot m v)) =
      if u = v then ((2 * m : ℕ) : Quadratic) else 0 := by
  have source := slot_range left
  have target := slot_range right
  have usmall : slot m u < d := by omega
  have vsmall : slot m v < d := by omega
  refine QuadraticAlgebra.ext ?_ ?_
  · simp only [traceForm, LinearMap.coe_mk, AddHom.coe_mk]
    rw [coeff_mul_conj_root_pow _ vsmall, coeff_root_pow usmall vsmall]
    by_cases same : u = v
    · rw [ite_eq_left same, ite_eq_left (congrArg (slot m) same).symm, mul_one,
        QuadraticAlgebra.re_natCast]
    · rw [ite_eq_right same,
        ite_eq_right (fun equal => same (slot_injective equal.symm)), mul_zero,
        QuadraticAlgebra.re_zero]
  · simp only [traceForm, LinearMap.coe_mk, AddHom.coe_mk]
    rw [← mul_assoc, coeff_mul_conj_root_pow _ vsmall,
      coeff_fixed_mul_root_pow quarter source target, mul_zero]
    by_cases same : u = v
    · rw [ite_eq_left same, QuadraticAlgebra.im_natCast]
    · rw [ite_eq_right same, QuadraticAlgebra.im_zero]

/-- The trace identity of Akita Section 3.1 for `k = 2` and `d = 4·m`:
`Tr_H(ψ(a)·σ_{−1}(ψ(b))) = (d/k)·⟨a, b⟩` for two inputs of `d/k = 2·m` entries. -/
theorem traceForm_psi_mul_conj (quarter : 4 * m = d) (a b : ℕ → Quadratic) :
    traceForm d m (psi quarter a * conj (psi quarter b)) =
      (2 * m) • ∑ u ∈ Finset.range (2 * m), a u * b u := by
  have term (u v : ℕ) (left : u ∈ Finset.range (2 * m)) (right : v ∈ Finset.range (2 * m)) :
      traceForm d m (AdjoinRoot.root (modulus d) ^ slot m u * subfield quarter (a u) *
        conj (AdjoinRoot.root (modulus d) ^ slot m v * subfield quarter (b v))) =
        a u * b v * if u = v then ((2 * m : ℕ) : Quadratic) else 0 := by
    rw [← traceForm_slot quarter (Finset.mem_range.mp left) (Finset.mem_range.mp right),
      ← traceForm_subfield_mul quarter, map_mul (subfield quarter), map_mul (conj (d := d)),
      conj_subfield]
    congr 1
    ring
  rw [psi, psi, map_sum, Finset.sum_mul_sum, map_sum, Finset.smul_sum]
  refine Finset.sum_congr rfl fun u left => ?_
  rw [map_sum, Finset.sum_eq_single u]
  · rw [term u u left left, ite_eq_left rfl, nsmul_eq_mul, mul_comm]
  · intro v right other
    rw [term u v left right, ite_eq_right (Ne.symm other), mul_zero]
  · intro absent
    exact absurd left absent

/-- `(d/k)·T_{ρ_pack}` of Akita (97) for the packed weights `χ̌_pack = ψ(weights)` of (96):
`Z ↦ Tr_H(Z·σ_{−1}(χ̌_pack))`. -/
noncomputable def traceFunctional (quarter : 4 * m = d) (weights : ℕ → Quadratic) :
    Rq d →ₗ[Zq] Quadratic :=
  traceForm d m ∘ₗ LinearMap.mulRight Zq (conj (psi quarter weights))

/-- The value of the functional of Akita (97) on a packed element:
`(d/k)·T_{ρ_pack}(ψ(f)) = (d/k)·Σ_u f_u·ẽq(ρ_pack, u)`, which makes the row (98) a statement on
the evaluation of the packed table. -/
theorem traceFunctional_psi (quarter : 4 * m = d) (weights f : ℕ → Quadratic) :
    traceFunctional quarter weights (psi quarter f) =
      (2 * m) • ∑ u ∈ Finset.range (2 * m), f u * weights u :=
  traceForm_psi_mul_conj quarter f weights

end NightstreamFPrime.Spec.Compression.Akita
