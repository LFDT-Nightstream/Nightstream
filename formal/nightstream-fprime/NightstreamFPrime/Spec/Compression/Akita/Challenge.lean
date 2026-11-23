import NightstreamFPrime.Spec.Compression.Akita.Embed

/-! The sparse fold challenges of Akita (ePrint 2026/1983, revision of 2026-11-23, Section 3.2).

- `Sparse ω c_max c` has the two norms of a member `c` of a sparse challenge family:
  `‖c‖_1 = ω` and `‖c‖_∞ = c_max`. The family and its sampler are not in this file.
- `Sparse.normOne_sub_le` and `Sparse.normInf_sub_le` are the bounds `‖c − c'‖_1 ≤ 2ω` and
  `‖c − c'‖_∞ ≤ 2·c_max` on a difference of two challenges.
- `UnitDifferences C` is the pairwise-difference condition of Akita Definition 9.2, E1: each
  difference of two different members of `C` is a unit. Akita Section 3.2 gets it from the bound
  `2·c_max < q^{1/s_split}/√s_split` and a cited result (Lyubashevsky and Seiler, Corollary 1.2).
  This file has no proof of that result, so the condition is a hypothesis of each user.
- `Sparse.embed` and `UnitDifferences.isUnit_embed_sub` move the norms and the units to the A ring
  (Akita Lemma 4.2, and the remark after Akita (60)). -/

namespace NightstreamFPrime.Spec.Compression.Akita

variable {d : ℕ} [NeZero d]

/-- `‖x − y‖_1 ≤ ‖x‖_1 + ‖y‖_1`. -/
theorem normOne_sub_le (x y : Rq d) : normOne (x - y) ≤ normOne x + normOne y := by
  rw [normOne, normOne, normOne, ← Finset.sum_add_distrib]
  exact Finset.sum_le_sum fun i _ => by rw [coeff_sub]; exact magnitude_sub_le _ _

/-- The norms of a member of a sparse challenge family with `ω = Σ_a a·n_a` and largest magnitude
`c_max` (Akita Section 3.2): `‖c‖_1 = ω` and `‖c‖_∞ = c_max`. -/
structure Sparse (mass peak : ℕ) (c : Rq d) : Prop where
  oneNorm : normOne c = mass
  infNorm : normInf c = peak

/-- `‖c − c'‖_1 ≤ 2ω` for two challenges (Akita Section 3.2). -/
theorem Sparse.normOne_sub_le {mass peak : ℕ} {c c' : Rq d} (left : Sparse mass peak c)
    (right : Sparse mass peak c') : normOne (c - c') ≤ 2 * mass := by
  have bound := Akita.normOne_sub_le c c'
  rw [left.oneNorm, right.oneNorm] at bound
  omega

/-- `‖c − c'‖_∞ ≤ 2·c_max` for two challenges (Akita Section 3.2). -/
theorem Sparse.normInf_sub_le {mass peak : ℕ} {c c' : Rq d} (left : Sparse mass peak c)
    (right : Sparse mass peak c') : normInf (c - c') ≤ 2 * peak := by
  have bound := Akita.normInf_sub_le c c'
  rw [left.infNorm, right.infNorm] at bound
  omega

/-- The pairwise-difference condition of a challenge family (Akita Definition 9.2, E1): each
difference of two different members is a unit. -/
def UnitDifferences (family : Set (Rq d)) : Prop :=
  ∀ c ∈ family, ∀ c' ∈ family, c ≠ c' → IsUnit (c - c')

variable {dCar : ℕ} [NeZero dCar] {stride : ℕ}

/-- The embedded challenge `ι(c)` has the norms of `c` (Akita Lemma 4.2, item 2). -/
theorem Sparse.embed {mass peak : ℕ} {c : Rq dCar} (sparse : Sparse mass peak c)
    (tower : stride * dCar = d) : Sparse mass peak (Akita.embed stride tower c) :=
  ⟨(normOne_embed tower c).trans sparse.oneNorm, (normInf_embed tower c).trans sparse.infNorm⟩

omit [NeZero d] [NeZero dCar] in
/-- If `c − c'` is a unit in `S`, then `ι(c) − ι(c')` is a unit in `R` (Akita Lemma 4.2, item 1,
with Akita (60)). -/
theorem UnitDifferences.isUnit_embed_sub {family : Set (Rq dCar)} (units : UnitDifferences family)
    (tower : stride * dCar = d) {c c' : Rq dCar} (member : c ∈ family) (member' : c' ∈ family)
    (different : c ≠ c') : IsUnit (embed stride tower c - embed stride tower c') := by
  rw [← map_sub]
  exact isUnit_embed tower (units c member c' member' different)

end NightstreamFPrime.Spec.Compression.Akita
