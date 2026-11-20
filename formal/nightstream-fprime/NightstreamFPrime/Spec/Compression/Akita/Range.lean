import NightstreamFPrime.Spec.FieldPrime
import NightstreamFPrime.Spec.Compression.Lib.Density
import NightstreamFPrime.Spec.Compression.Akita.Euclid

/-! The norm checks of Akita for the next witness (ePrint 2026/1983, revision of 2026-10-15,
Section 6): the digit range check, the fused check of the compression digits, and the norm
certificate of the two routes.

- `rangePoly b⋆ w` is `Q_{b⋆}(w) = Π_{a ∈ A_{b⋆}} (w − a)` and `rangeSq b⋆ s` is `Q_sq(s)` of
  Akita (79). `rangePoly_eq_rangeSq` is `Q_{b⋆}(w) = Q_sq(w(w + 1))`, and `rangePoly_eq_zero_iff`
  states that a root of `Q_{b⋆}` is a member of `A_{b⋆}`, for `b⋆ < q`.
- `mul_add_one_eq_zero_iff` is the fused constraint `w(w + 1) = 0` of Akita Remark 6.2, which
  gives the alphabet `{−1, 0}`.
- `RangeHolds b⋆ μ w` is the digit range statement for a witness on the cube of `μ` variables.
  `rangeHolds_iff` gives the alphabet of each cell, `mle_rangeResidual_eq_zero` states that the
  sum (80) is zero at each point, and `density_rangeSum_le` bounds the points where the sum (80)
  is zero for a witness outside the alphabet.
- `balanced_iff_rangePoly` and `negativeBinary_iff` connect the two constraints to the alphabets
  `Balanced` and `NegativeBinary` of the weak opening. `vecNormInf_gadget_le_of_range` is the norm
  certificate of the coefficient route: the envelope (77).
- `intCoordinate` is the integer `z^Z_u` of Akita (81), and `energy` is `E_int(ẑ)` of Akita (82).
  `vecNormSq_gadget_le_energy` is (82), `directBound` is `U_dir` of Akita (84),
  `energy_eq_of_direct` is Akita Lemma 6.3, and `vecNormSq_gadget_le_of_direct` is the norm
  certificate of the Euclidean route: the first claim of Akita (184). -/

namespace NightstreamFPrime.Spec.Compression.Akita

open NightstreamFPrime.Spec.Compression.Lib

section Alphabet

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

/-- `Q_{b⋆}(w) = Π_{a ∈ A_{b⋆}} (w − a)` (Akita Section 6.1), with the members of
`A_{b⋆} = {−b⋆/2, …, b⋆/2 − 1}` in the pairs `j` and `−(j + 1)` for `j < b⋆/2`. -/
def rangePoly (b : ℕ) (w : Zq) : Zq :=
  ∏ j ∈ Finset.range (b / 2), ((w - (j : Zq)) * (w + ((j : Zq) + 1)))

/-- `Q_sq(s) = Π_{j < b⋆/2} (s − j(j + 1))` (Akita (79)). -/
def rangeSq (b : ℕ) (s : Zq) : Zq := ∏ j ∈ Finset.range (b / 2), (s - (j : Zq) * ((j : Zq) + 1))

/-- `Q_{b⋆}(w) = Q_sq(w(w + 1))` (Akita Section 6.1), from
`(w − j)(w + j + 1) = w(w + 1) − j(j + 1)`. -/
theorem rangePoly_eq_rangeSq (b : ℕ) (w : Zq) : rangePoly b w = rangeSq b (w * (w + 1)) :=
  Finset.prod_congr rfl fun j _ => by ring

/-- A value is a root of `Q_{b⋆}` when its representative in `(−q/2, q/2]` is in
`A_{b⋆} = {−b⋆/2, …, b⋆/2 − 1}`, and only then, for `b⋆ < q` (Akita Section 6.1). -/
theorem rangePoly_eq_zero_iff {b : ℕ} (small : b < fieldModulus) (w : Zq) :
    rangePoly b w = 0 ↔
      -((b / 2 : ℕ) : ℤ) ≤ w.valMinAbs ∧ w.valMinAbs ≤ ((b / 2 : ℕ) : ℤ) - 1 := by
  rw [rangePoly, Finset.prod_eq_zero_iff]
  constructor
  · rintro ⟨j, member, zero⟩
    have below := Finset.mem_range.mp member
    rcases mul_eq_zero.mp zero with nonneg | negative
    · have value : w.valMinAbs = (j : ℤ) :=
        (ZMod.valMinAbs_spec w j).mpr
          ⟨by rw [Int.cast_natCast]; exact sub_eq_zero.mp nonneg, by omega, by omega⟩
      omega
    · have value : w.valMinAbs = -((j : ℤ) + 1) :=
        (ZMod.valMinAbs_spec w _).mpr
          ⟨by push_cast; linear_combination negative, by omega, by omega⟩
      omega
  · rintro ⟨low, high⟩
    rcases le_or_gt 0 w.valMinAbs with nonneg | negative
    · refine ⟨w.valMinAbs.toNat, Finset.mem_range.mpr (by omega), mul_eq_zero.mpr (Or.inl ?_)⟩
      have cast := (ZMod.coe_valMinAbs w).symm
      rw [← Int.toNat_of_nonneg nonneg, Int.cast_natCast] at cast
      exact sub_eq_zero.mpr cast
    · refine ⟨(-w.valMinAbs - 1).toNat, Finset.mem_range.mpr (by omega),
        mul_eq_zero.mpr (Or.inr ?_)⟩
      have sum : w.valMinAbs + (((-w.valMinAbs - 1).toNat : ℕ) : ℤ) + 1 = 0 := by omega
      have cast := congrArg (Int.cast : ℤ → Zq) sum
      rw [Int.cast_add, Int.cast_add, ZMod.coe_valMinAbs, Int.cast_natCast, Int.cast_one,
        Int.cast_zero] at cast
      linear_combination cast

/-- The fused constraint `w(w + 1) = 0` of Akita Remark 6.2 holds when the representative of `w`
in `(−q/2, q/2]` is in `{−1, 0}`, and only then. -/
theorem mul_add_one_eq_zero_iff (w : Zq) :
    w * (w + 1) = 0 ↔ -1 ≤ w.valMinAbs ∧ w.valMinAbs ≤ 0 := by
  have range := rangePoly_eq_zero_iff (b := 2) (by decide) w
  rw [rangePoly, show 2 / 2 = 1 from rfl, Finset.prod_range_one, Nat.cast_zero, sub_zero,
    zero_add, Nat.cast_one, sub_self] at range
  exact range

variable {d : ℕ} [NeZero d] {ι : Type}

/-- The digit range check on each coefficient of `v` is the alphabet `A_{b⋆}` of Akita
Definition 10.22 (v). -/
theorem balanced_iff_rangePoly {b : ℕ} (small : b < fieldModulus) (v : ι → Rq d) :
    Balanced b v ↔ ∀ j, ∀ i < d, rangePoly b (coeff (v j) i) = 0 := by
  unfold Balanced Digits
  exact forall_congr' fun j => forall_congr' fun i => forall_congr' fun _ =>
    (rangePoly_eq_zero_iff small _).symm

/-- The fused constraint on each coefficient of `v` is the alphabet `{−1, 0}` of a compression
digit vector (Akita Remark 6.2). -/
theorem negativeBinary_iff (v : ι → Rq d) :
    NegativeBinary v ↔ ∀ j, ∀ i < d, coeff (v j) i * (coeff (v j) i + 1) = 0 := by
  unfold NegativeBinary Digits
  exact forall_congr' fun j => forall_congr' fun i => forall_congr' fun _ =>
    (mul_add_one_eq_zero_iff _).symm

/-- The norm certificate of the coefficient route (Akita Section 6, `Norm_h`): when each
coefficient of the response digits `ẑ` passes the digit range check for `b⋆`, the recomposed
response has the envelope (77): `‖G_{b,n}·ẑ‖_∞ ≤ (b⋆/2) · Σ_{u<δ} b^u`. -/
theorem vecNormInf_gadget_le_of_range [Fintype ι] {δ certified : ℕ} (b : ℕ)
    (small : certified < fieldModulus) {digits : ι × Fin δ → Rq d}
    (range : ∀ j, ∀ i < d, rangePoly certified (coeff (digits j) i) = 0) :
    vecNormInf (gadget b digits) ≤ certified / 2 * ∑ u ∈ Finset.range δ, b ^ u :=
  vecNormInf_gadget_le_of_balanced b ((balanced_iff_rangePoly small digits).mpr range)

/-- The digit range statement of Akita Section 6.1 for the witness `w` on the cube of `μ`
variables: `Q_sq(w(x)·(w(x) + 1)) = 0` at each index `x`. -/
def RangeHolds (b μ : ℕ) (w : Table) : Prop := ∀ x < 2 ^ μ, rangeSq b (w x * (w x + 1)) = 0

/-- The digit range statement puts each cell of the witness in `A_{b⋆}`. -/
theorem rangeHolds_iff {b : ℕ} (small : b < fieldModulus) (μ : ℕ) (w : Table) :
    RangeHolds b μ w ↔ ∀ x < 2 ^ μ,
      -((b / 2 : ℕ) : ℤ) ≤ (w x).valMinAbs ∧ (w x).valMinAbs ≤ ((b / 2 : ℕ) : ℤ) - 1 :=
  forall_congr' fun x => forall_congr' fun _ => by
    rw [← rangePoly_eq_rangeSq, rangePoly_eq_zero_iff small]

variable {E : Type} [Field E] [Algebra Zq E]

/-- The summand `Q_sq(s(x))` of Akita (80), in the field `E` of the range point. -/
def rangeResidual (E : Type) [Field E] [Algebra Zq E] (b : ℕ) (w : Table) (x : ℕ) : E :=
  algebraMap Zq E (rangeSq b (w x * (w x + 1)))

/-- Akita (80) holds at each range point `τ_0` for a witness with each cell in the alphabet. -/
theorem mle_rangeResidual_eq_zero {b μ : ℕ} {w : Table} (holds : RangeHolds b μ w)
    (point : List E) (length : point.length = μ) : mle point (rangeResidual E b w) = 0 := by
  rw [mle_congr point (u := fun _ => 0) fun x small => by
    rw [rangeResidual, holds x (length ▸ small), map_zero]]
  exact mle_const point 0

/-- Akita (80) for a witness with a cell outside the alphabet holds for at most the fraction
`μ / |E|` of the range points `τ_0`. -/
theorem density_rangeSum_le [Fintype E] [DecidableEq E] {b μ : ℕ} {w : Table}
    (fails : ¬ RangeHolds b μ w) :
    density (Finset.univ.filter fun point : Fin μ → E =>
      mle (List.ofFn point) (rangeResidual E b w) = 0) ≤ μ / Fintype.card E := by
  unfold RangeHolds at fails
  push Not at fails
  obtain ⟨x, small, nonzero⟩ := fails
  exact density_mle_zero_le μ _ ⟨x, small, fun zero =>
    nonzero ((algebraMap Zq E).injective (zero.trans (map_zero _).symm))⟩

end Alphabet

section Euclidean

variable {d : ℕ} [NeZero d] {ι : Type} {δ : ℕ}

/-- The signed integer response coordinate `z^Z_u = Σ_{a<δ_f} b^a · ẑ_{u,a}` of Akita (81), for
coefficient `i` of entry `j`, with each digit as its representative in `(−q/2, q/2]`. -/
noncomputable def intCoordinate (b : ℕ) (digits : ι × Fin δ → Rq d) (j : ι) (i : ℕ) : ℤ :=
  ∑ u : Fin δ, (b : ℤ) ^ (u : ℕ) * (coeff (digits (j, u)) i).valMinAbs

/-- `E_int(ẑ) = Σ_u (z^Z_u)²` of Akita (82), over each coefficient of each entry. -/
noncomputable def energy [Fintype ι] (b : ℕ) (digits : ι × Fin δ → Rq d) : ℤ :=
  ∑ j, ∑ i ∈ Finset.range d, intCoordinate b digits j i ^ 2

/-- The residue of `z^Z_u` modulo `q` is the coordinate `u` of `z = G_{b,m_A}·ẑ`
(Akita Section 6.2). -/
theorem coeff_gadget_eq_intCast (b : ℕ) (digits : ι × Fin δ → Rq d) (j : ι) (i : ℕ) :
    coeff (gadget b digits j) i = ((intCoordinate b digits j i : ℤ) : Zq) := by
  rw [gadget, coeff_sum, intCoordinate, Int.cast_sum]
  refine Finset.sum_congr rfl fun u _ => ?_
  rw [coeff_natCast_pow_mul, Int.cast_mul, Int.cast_pow, Int.cast_natCast, ZMod.coe_valMinAbs]

/-- Akita (82): `‖G_{b,m_A}·ẑ‖_2² ≤ E_int(ẑ)`. -/
theorem vecNormSq_gadget_le_energy [Fintype ι] (b : ℕ) (digits : ι × Fin δ → Rq d) :
    ((vecNormSq (gadget b digits) : ℕ) : ℤ) ≤ energy b digits := by
  rw [vecNormSq, energy, Nat.cast_sum]
  refine Finset.sum_le_sum fun j _ => ?_
  rw [normSq, Nat.cast_sum]
  refine Finset.sum_le_sum fun i _ => ?_
  rw [coeff_gadget_eq_intCast, Nat.cast_pow]
  exact (pow_le_pow_left₀ (Nat.cast_nonneg _) (Nat.cast_le.mpr (magnitude_intCast_le _)) 2).trans
    (le_of_eq (Int.natAbs_pow_two _))

/-- Each coefficient of the digit plane `a` has a magnitude of at most `B_dig,a`
(Akita Section 6.2). -/
def PlaneBounded (planeBound : Fin δ → ℕ) (digits : ι × Fin δ → Rq d) : Prop :=
  ∀ j u, ∀ i < d, magnitude (coeff (digits (j, u)) i) ≤ planeBound u

/-- The digit range check for `b⋆` gives `B_dig,a = b⋆/2` on each plane. -/
theorem PlaneBounded.of_balanced [Fintype ι] {certified : ℕ} {digits : ι × Fin δ → Rq d}
    (balanced : Balanced certified digits) : PlaneBounded (fun _ => certified / 2) digits :=
  fun j u _ small => (normInf_le_iff _ _).mp
    ((normInf_le_vecNormInf digits (j, u)).trans balanced.vecNormInf_le) _ small

/-- `U_dir = N_A · (Σ_{a<δ_f} b^a·B_dig,a)²` of Akita (84), for `N_A` coordinates. -/
def directBound (b : ℕ) (planeBound : Fin δ → ℕ) (coordinates : ℕ) : ℕ :=
  coordinates * (∑ u : Fin δ, b ^ (u : ℕ) * planeBound u) ^ 2

theorem abs_intCoordinate_le {b : ℕ} {planeBound : Fin δ → ℕ} {digits : ι × Fin δ → Rq d}
    (bounded : PlaneBounded planeBound digits) (j : ι) {i : ℕ} (small : i < d) :
    |intCoordinate b digits j i| ≤ ((∑ u : Fin δ, b ^ (u : ℕ) * planeBound u : ℕ) : ℤ) := by
  rw [intCoordinate, Nat.cast_sum]
  refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun u _ => ?_)
  rw [abs_mul, abs_pow, Nat.abs_cast, Nat.cast_mul, Nat.cast_pow, ← Int.natCast_natAbs]
  exact mul_le_mul_of_nonneg_left (Nat.cast_le.mpr (bounded j u i small)) (by positivity)

theorem energy_nonneg [Fintype ι] (b : ℕ) (digits : ι × Fin δ → Rq d) : 0 ≤ energy b digits :=
  Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => sq_nonneg _

/-- `E_int(ẑ) ≤ U_dir` for digits with the plane bounds `B_dig,a` (Akita Section 6.2). -/
theorem energy_le_directBound [Fintype ι] {b : ℕ} {planeBound : Fin δ → ℕ}
    {digits : ι × Fin δ → Rq d} (bounded : PlaneBounded planeBound digits) :
    energy b digits ≤ ((directBound b planeBound (Fintype.card ι * d) : ℕ) : ℤ) := by
  have each (j : ι) {i : ℕ} (small : i < d) : intCoordinate b digits j i ^ 2 ≤
      ((∑ u : Fin δ, b ^ (u : ℕ) * planeBound u : ℕ) : ℤ) ^ 2 := by
    rw [← sq_abs]
    exact pow_le_pow_left₀ (abs_nonneg _) (abs_intCoordinate_le bounded j small) 2
  refine (Finset.sum_le_sum fun j _ => Finset.sum_le_sum fun i member =>
    each j (Finset.mem_range.mp member)).trans (le_of_eq ?_)
  rw [Finset.sum_const, Finset.sum_const, Finset.card_univ, Finset.card_range, directBound]
  simp only [nsmul_eq_mul]
  push_cast
  ring

/-- Akita Lemma 6.3 (direct norm check). The digits have the plane bounds `B_dig,a`,
`max{U_dir, S_max} < q`, the claimed value `E_resp` is at most `S_max`, and the sum (85) of the
squares of the coordinates of `z = G_{b,m_A}·ẑ` is `E_resp` in `Z_q`. Then `E_int(ẑ) = E_resp` in
the integers. -/
theorem energy_eq_of_direct [Fintype ι] {b limit claimed : ℕ} {planeBound : Fin δ → ℕ}
    {digits : ι × Fin δ → Rq d} (bounded : PlaneBounded planeBound digits)
    (noWrap : max (directBound b planeBound (Fintype.card ι * d)) limit < fieldModulus)
    (accepted : claimed ≤ limit)
    (check : ∑ j, ∑ i ∈ Finset.range d, coeff (gadget b digits j) i ^ 2 = (claimed : Zq)) :
    energy b digits = claimed := by
  have cast : ((energy b digits : ℤ) : Zq) = ((claimed : ℤ) : Zq) := by
    rw [Int.cast_natCast, ← check, energy, Int.cast_sum]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [Int.cast_sum]
    exact Finset.sum_congr rfl fun i _ => by rw [Int.cast_pow, coeff_gadget_eq_intCast]
  have divides := (ZMod.intCast_eq_intCast_iff_dvd_sub _ _ _).mp cast
  have low := energy_nonneg b digits
  have high := energy_le_directBound (b := b) bounded
  have direct := (max_lt_iff.mp noWrap).1
  have scheduled := (max_lt_iff.mp noWrap).2
  have zero := Int.eq_zero_of_abs_lt_dvd divides (abs_lt.mpr ⟨by omega, by omega⟩)
  omega

/-- The norm certificate of the Euclidean route (Akita Section 6, `Norm_h`, and the first claim of
Akita (184)): under the conditions of Akita Lemma 6.3, `‖G_{b,m_A}·ẑ‖_2² ≤ E_resp ≤ S_max`. -/
theorem vecNormSq_gadget_le_of_direct [Fintype ι] {b limit claimed : ℕ}
    {planeBound : Fin δ → ℕ} {digits : ι × Fin δ → Rq d}
    (bounded : PlaneBounded planeBound digits)
    (noWrap : max (directBound b planeBound (Fintype.card ι * d)) limit < fieldModulus)
    (accepted : claimed ≤ limit)
    (check : ∑ j, ∑ i ∈ Finset.range d, coeff (gadget b digits j) i ^ 2 = (claimed : Zq)) :
    vecNormSq (gadget b digits) ≤ limit := by
  have bound := vecNormSq_gadget_le_energy b digits
  rw [energy_eq_of_direct bounded noWrap accepted check] at bound
  exact (Nat.cast_le.mp bound).trans accepted

end Euclidean

end NightstreamFPrime.Spec.Compression.Akita
