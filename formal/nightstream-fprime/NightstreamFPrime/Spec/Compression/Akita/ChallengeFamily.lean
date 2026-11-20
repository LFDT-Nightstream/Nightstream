import NightstreamFPrime.Spec.Compression.Akita.Challenge
import NightstreamFPrime.Spec.Compression.Akita.ShortUnit

/-! The sparse challenge families of Akita with the magnitudes 1 and 2 (ePrint 2026/1983,
revision of 2026-10-15, Section 3.2).

- `SparseFamily d n₁ n₂` is the family `C` for the ring degree `d` and the counts `n₁` and `n₂`
  of the coefficients `±1` and `±2`. The Akita code (revision `d50f700f`,
  `crates/akita-challenges/src/config.rs`) has the counts `(31, 10)` for the degree 64, `(31, 0)`
  for the degree 128, and `(31, 11)` for the base family of the filter at the degree 64.
- `SparseFamily.sparse` gives `‖c‖_1 = ω = n₁ + 2·n₂` and `‖c‖_∞ = c_max` (`peak`) for each
  member, and `SparseFamily.sparse_embed` gives them for its image in a larger ring.
- `card_sparseFamily` is `|C| = C(d, n₁)·C(d − n₁, n₂)·2^{n₁ + n₂}`.
- `SparseFamily.unitDifferences` is the condition E1 of Akita Definition 9.2 for a degree
  `2^{n+1}`: a difference of two members has `‖·‖_∞ ≤ 4`, and it is a unit by
  `isUnit_of_short`. -/

namespace NightstreamFPrime.Spec.Compression.Akita

variable {d : ℕ} [NeZero d]

theorem magnitude_natCast {n : ℕ} (small : n ≤ fieldModulus / 2) : magnitude (n : Zq) = n := by
  rw [magnitude, ZMod.valMinAbs_natCast_of_le_half small, Int.natAbs_natCast]

/-- An element of `Z_q` is its magnitude or the negation of its magnitude. -/
theorem eq_or_eq_neg_magnitude (a : Zq) : a = (magnitude a : Zq) ∨ a = -(magnitude a : Zq) := by
  have cast : ((a.valMinAbs : ℤ) : Zq) = a := ZMod.coe_valMinAbs a
  rcases Int.natAbs_eq a.valMinAbs with same | opposite
  · refine Or.inl ?_
    conv_lhs => rw [← cast, same, Int.cast_natCast]
    rfl
  · refine Or.inr ?_
    conv_lhs => rw [← cast, opposite, Int.cast_neg, Int.cast_natCast]
    rfl

/-- `−w ≠ w` in `Z_q` for a natural number `w` with `0 < 2·w < q`. -/
theorem neg_natCast_ne {w : ℕ} (positive : 0 < w) (small : w + w < fieldModulus) :
    -(w : Zq) ≠ w := by
  intro same
  have zero : ((w + w : ℕ) : Zq) = 0 := by
    push_cast
    linear_combination (-1 : Zq) * same
  have := Nat.le_of_dvd (by omega) ((ZMod.natCast_eq_zero_iff _ _).mp zero)
  omega

/-- The sparse challenge family `C` of Akita Section 3.2 for the magnitudes `{1, 2}`: the
elements of `R_{q,d}` with `ones` coefficients `±1`, `twos` coefficients `±2`, and the other
coefficients 0. -/
def SparseFamily (d ones twos : ℕ) [NeZero d] : Set (Rq d) :=
  {c | normInf c ≤ 2 ∧
    ((Finset.range d).filter fun i => magnitude (coeff c i) = 1).card = ones ∧
    ((Finset.range d).filter fun i => magnitude (coeff c i) = 2).card = twos}

/-- `c_max` of Akita Section 3.2: the largest magnitude with a positive count. -/
def peak (ones twos : ℕ) : ℕ := if twos = 0 then (if ones = 0 then 0 else 1) else 2

/-- The positions of the coefficients `±1`, of the coefficients `±2` and of the negative
coefficients of the members of `SparseFamily d ones twos`. -/
def patterns (d ones twos : ℕ) : Finset (Σ _ : Finset ℕ, Σ _ : Finset ℕ, Finset ℕ) :=
  (Finset.powersetCard ones (Finset.range d)).sigma fun first =>
    (Finset.powersetCard twos (Finset.range d \ first)).sigma fun second =>
      (first ∪ second).powerset

theorem card_patterns (d ones twos : ℕ) : (patterns d ones twos).card =
    d.choose ones * ((d - ones).choose twos * 2 ^ (ones + twos)) := by
  have inner : ∀ first ∈ Finset.powersetCard ones (Finset.range d),
      ((Finset.powersetCard twos (Finset.range d \ first)).sigma fun second =>
        (first ∪ second).powerset).card = (d - ones).choose twos * 2 ^ (ones + twos) := by
    intro first member
    obtain ⟨inside, size⟩ := Finset.mem_powersetCard.mp member
    have each : ∀ second ∈ Finset.powersetCard twos (Finset.range d \ first),
        (first ∪ second).powerset.card = 2 ^ (ones + twos) := by
      intro second member'
      obtain ⟨outside, size'⟩ := Finset.mem_powersetCard.mp member'
      rw [Finset.card_powerset, Finset.card_union_of_disjoint
        (Finset.disjoint_of_subset_right outside Finset.disjoint_sdiff), size, size']
    rw [Finset.card_sigma, Finset.sum_congr rfl each, Finset.sum_const,
      Finset.card_powersetCard, Finset.card_sdiff_of_subset inside, Finset.card_range, size,
      smul_eq_mul]
  rw [patterns, Finset.card_sigma, Finset.sum_congr rfl inner, Finset.sum_const,
    Finset.card_powersetCard, Finset.card_range, smul_eq_mul]

/-- The magnitude of the coefficient `i` of the member of a pattern. -/
def weight (p : Σ _ : Finset ℕ, Σ _ : Finset ℕ, Finset ℕ) (i : ℕ) : ℕ :=
  if i ∈ p.1 then 1 else if i ∈ p.2.1 then 2 else 0

/-- The member of a pattern: `±1` at the first positions, `±2` at the second positions, and the
sign `−` at the third positions. -/
noncomputable def member (p : Σ _ : Finset ℕ, Σ _ : Finset ℕ, Finset ℕ) : Rq d :=
  ofCoeffs fun i => (if i ∈ p.2.2 then -1 else 1) * (weight p i : Zq)

/-- The pattern of a ring element. -/
noncomputable def pattern (c : Rq d) : Σ _ : Finset ℕ, Σ _ : Finset ℕ, Finset ℕ :=
  ⟨(Finset.range d).filter fun i => magnitude (coeff c i) = 1,
    (Finset.range d).filter fun i => magnitude (coeff c i) = 2,
    (Finset.range d).filter fun i => coeff c i ≠ (magnitude (coeff c i) : Zq)⟩

theorem weight_le (p : Σ _ : Finset ℕ, Σ _ : Finset ℕ, Finset ℕ) (i : ℕ) : weight p i ≤ 2 := by
  unfold weight
  split_ifs <;> omega

theorem magnitude_coeff_member (p : Σ _ : Finset ℕ, Σ _ : Finset ℕ, Finset ℕ) {i : ℕ}
    (small : i < d) : magnitude (coeff (member (d := d) p) i) = weight p i := by
  have value : magnitude (weight p i : Zq) = weight p i :=
    magnitude_natCast ((weight_le p i).trans (by decide))
  rw [member, coeff_ofCoeffs _ small]
  split_ifs
  · rw [neg_one_mul, magnitude_neg, value]
  · rw [one_mul, value]

/-- A pattern of `patterns` is the pattern of its member. -/
theorem pattern_member {ones twos : ℕ} {p : Σ _ : Finset ℕ, Σ _ : Finset ℕ, Finset ℕ}
    (listed : p ∈ patterns d ones twos) : pattern (member (d := d) p) = p := by
  obtain ⟨first, second, signs⟩ := p
  simp only [patterns, Finset.mem_sigma, Finset.mem_powersetCard, Finset.mem_powerset] at listed
  obtain ⟨⟨inside, -⟩, ⟨outside, -⟩, signed⟩ := listed
  have apart : ∀ i ∈ second, i ∈ Finset.range d ∧ i ∉ first := fun i at_ =>
    Finset.mem_sdiff.mp (outside at_)
  have ones_ : ((Finset.range d).filter fun i =>
      magnitude (coeff (member (d := d) ⟨first, second, signs⟩) i) = 1) = first := by
    ext i
    rw [Finset.mem_filter, Finset.mem_range]
    constructor
    · rintro ⟨small, one⟩
      rw [magnitude_coeff_member _ small] at one
      by_contra missing
      simp only [weight, missing, ite_false] at one
      split_ifs at one
      omega
    · intro at_
      have small := Finset.mem_range.mp (inside at_)
      exact ⟨small, by rw [magnitude_coeff_member _ small]; simp [weight, at_]⟩
  have twos_ : ((Finset.range d).filter fun i =>
      magnitude (coeff (member (d := d) ⟨first, second, signs⟩) i) = 2) = second := by
    ext i
    rw [Finset.mem_filter, Finset.mem_range]
    constructor
    · rintro ⟨small, two⟩
      rw [magnitude_coeff_member _ small] at two
      by_contra missing
      simp only [weight, missing, ite_false] at two
      split_ifs at two
      omega
    · intro at_
      have small := Finset.mem_range.mp (apart i at_).1
      exact ⟨small, by rw [magnitude_coeff_member _ small]; simp [weight, at_, (apart i at_).2]⟩
  have signs_ : ((Finset.range d).filter fun i =>
      coeff (member (d := d) ⟨first, second, signs⟩) i ≠
        (magnitude (coeff (member (d := d) ⟨first, second, signs⟩) i) : Zq)) = signs := by
    ext i
    rw [Finset.mem_filter, Finset.mem_range]
    constructor
    · rintro ⟨small, differs⟩
      rw [magnitude_coeff_member _ small, member, coeff_ofCoeffs _ small] at differs
      by_contra missing
      exact differs (by simp [missing])
    · intro at_
      have positive : 0 < weight ⟨first, second, signs⟩ i := by
        rcases Finset.mem_union.mp (signed at_) with one | two
        · simp [weight, one]
        · simp [weight, two, (apart i two).2]
      have small : i < d := by
        rcases Finset.mem_union.mp (signed at_) with one | two
        · exact Finset.mem_range.mp (inside one)
        · exact Finset.mem_range.mp (apart i two).1
      refine ⟨small, ?_⟩
      rw [magnitude_coeff_member _ small, member, coeff_ofCoeffs _ small]
      simp only [at_, ite_true, neg_one_mul]
      exact neg_natCast_ne positive (by
        have := weight_le ⟨first, second, signs⟩ i
        have : 4 < fieldModulus := by decide
        omega)
  simp only [pattern, ones_, twos_, signs_]

variable {ones twos : ℕ} {c : Rq d}

/-- The member of a pattern of `patterns` is in the family. -/
theorem member_mem {p : Σ _ : Finset ℕ, Σ _ : Finset ℕ, Finset ℕ}
    (listed : p ∈ patterns d ones twos) : member p ∈ SparseFamily d ones twos := by
  have same := pattern_member listed
  simp only [patterns, Finset.mem_sigma, Finset.mem_powersetCard] at listed
  refine ⟨(normInf_le_iff _ _).mpr fun i small => ?_, ?_, ?_⟩
  · rw [magnitude_coeff_member _ small]
    exact weight_le p i
  · exact (congrArg (fun q => q.1.card) same).trans listed.1.2
  · exact (congrArg (fun q => q.2.1.card) same).trans listed.2.1.2

/-- A member of the family is the member of its pattern, and its pattern is in `patterns`. -/
theorem pattern_listed (inside : c ∈ SparseFamily d ones twos) :
    pattern c ∈ patterns d ones twos ∧ member (pattern c) = c := by
  obtain ⟨short, first, second⟩ := inside
  have each := (normInf_le_iff c 2).mp short
  constructor
  · simp only [patterns, pattern, Finset.mem_sigma, Finset.mem_powersetCard,
      Finset.mem_powerset]
    refine ⟨⟨Finset.filter_subset _ _, first⟩, ⟨fun i at_ => ?_, second⟩, fun i at_ => ?_⟩
    · obtain ⟨small, two⟩ := Finset.mem_filter.mp at_
      exact Finset.mem_sdiff.mpr ⟨small, fun one => by
        have := (Finset.mem_filter.mp one).2
        omega⟩
    · obtain ⟨small, differs⟩ := Finset.mem_filter.mp at_
      have bound := each i (Finset.mem_range.mp small)
      have positive : magnitude (coeff c i) ≠ 0 := fun zero => differs (by
        rcases eq_or_eq_neg_magnitude (coeff c i) with same | same <;>
          rw [zero] at same ⊢ <;> simpa using same)
      rw [Finset.mem_union, Finset.mem_filter, Finset.mem_filter]
      have : magnitude (coeff c i) = 1 ∨ magnitude (coeff c i) = 2 := by omega
      exact this.imp (fun one => ⟨small, one⟩) fun two => ⟨small, two⟩
  · refine ext_coeff fun i small => ?_
    have bound := each i small
    have value : weight (pattern c) i = magnitude (coeff c i) := by
      simp only [weight, pattern, Finset.mem_filter, Finset.mem_range, small, true_and]
      split_ifs <;> omega
    rw [member, coeff_ofCoeffs _ small, value]
    simp only [pattern, Finset.mem_filter, Finset.mem_range, small, true_and]
    split_ifs with differs
    · rw [neg_one_mul]
      exact ((eq_or_eq_neg_magnitude (coeff c i)).resolve_left differs).symm
    · rw [one_mul]
      exact (not_not.mp differs).symm

/-- `|C| = C(d, n₁)·C(d − n₁, n₂)·2^{n₁ + n₂}` (Akita Section 3.2: `C(d, s)·(s!/(n₁!·n₂!))·2^s`
for `s = n₁ + n₂`). -/
theorem card_sparseFamily (d ones twos : ℕ) [NeZero d] : Nat.card (SparseFamily d ones twos) =
    d.choose ones * ((d - ones).choose twos * 2 ^ (ones + twos)) := by
  have image : SparseFamily d ones twos = member '' (patterns d ones twos : Set _) := by
    ext c
    constructor
    · intro inside
      exact ⟨pattern c, (pattern_listed inside).1, (pattern_listed inside).2⟩
    · rintro ⟨p, listed, rfl⟩
      exact member_mem listed
  have injective : Set.InjOn (member (d := d)) (patterns d ones twos : Set _) :=
    fun p listed p' listed' same => by
      rw [← pattern_member listed, ← pattern_member listed', same]
  rw [Nat.card_coe_set_eq, image, injective.ncard_image, Set.ncard_coe_finset,
    card_patterns]

theorem normOne_of_mem (inside : c ∈ SparseFamily d ones twos) :
    normOne c = ones + 2 * twos := by
  obtain ⟨short, rfl, rfl⟩ := inside
  rw [normOne, Finset.card_filter, Finset.card_filter, Finset.mul_sum, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun i at_ => ?_
  have small := (normInf_le_iff c 2).mp short i (Finset.mem_range.mp at_)
  generalize magnitude (coeff c i) = m at small ⊢
  interval_cases m <;> rfl

theorem normInf_of_mem (inside : c ∈ SparseFamily d ones twos) :
    normInf c = peak ones twos := by
  obtain ⟨short, first, second⟩ := inside
  have each := (normInf_le_iff c 2).mp short
  have reached : ∀ i < d, magnitude (coeff c i) ≤ normInf c := (normInf_le_iff c _).mp le_rfl
  have none : ∀ {value count : ℕ},
      ((Finset.range d).filter fun i => magnitude (coeff c i) = value).card = count →
      count = 0 → ∀ i < d, magnitude (coeff c i) ≠ value := fun size zero i small =>
    Finset.filter_eq_empty_iff.mp (Finset.card_eq_zero.mp (size.trans zero))
      (Finset.mem_range.mpr small)
  have some : ∀ {value count : ℕ},
      ((Finset.range d).filter fun i => magnitude (coeff c i) = value).card = count →
      count ≠ 0 → value ≤ normInf c := fun size positive => by
    obtain ⟨i, at_⟩ := Finset.card_pos.mp (Nat.pos_of_ne_zero (size ▸ positive))
    obtain ⟨small, same⟩ := Finset.mem_filter.mp at_
    exact same ▸ reached i (Finset.mem_range.mp small)
  unfold peak
  split_ifs with noTwos noOnes
  · refine Nat.le_zero.mp ((normInf_le_iff c 0).mpr fun i small => ?_)
    have := each i small
    have := none first noOnes i small
    have := none second noTwos i small
    omega
  · refine le_antisymm ((normInf_le_iff c 1).mpr fun i small => ?_) (some first noOnes)
    have := each i small
    have := none second noTwos i small
    omega
  · exact le_antisymm short (some second noTwos)

/-- Each member of the family has `‖c‖_1 = ω = n₁ + 2·n₂` and `‖c‖_∞ = c_max` (Akita
Section 3.2). -/
theorem SparseFamily.sparse (inside : c ∈ SparseFamily d ones twos) :
    Sparse (ones + 2 * twos) (peak ones twos) c :=
  ⟨normOne_of_mem inside, normInf_of_mem inside⟩

/-- The image `ι(c)` of a member in a larger ring has the norms of the member (Akita Lemma 4.2,
item 2). -/
theorem SparseFamily.sparse_embed {dCar stride : ℕ} [NeZero dCar] {c : Rq dCar}
    (inside : c ∈ SparseFamily dCar ones twos) (tower : stride * dCar = d) :
    Sparse (ones + 2 * twos) (peak ones twos) (embed stride tower c) :=
  (SparseFamily.sparse inside).embed tower

/-- A difference of two members has `‖c − c'‖_∞ ≤ 4 = 2·c_max` (Akita Section 3.2). -/
theorem SparseFamily.normInf_sub_le {c' : Rq d} (inside : c ∈ SparseFamily d ones twos)
    (inside' : c' ∈ SparseFamily d ones twos) : normInf (c - c') ≤ 4 :=
  (Akita.normInf_sub_le c c').trans (Nat.add_le_add inside.1 inside'.1)

/-- The condition E1 of Akita Definition 9.2 for the family of a degree `2^{n+1}`: each
difference of two different members is a unit (`isUnit_of_short` with `2·4² < q`). -/
theorem SparseFamily.unitDifferences (n : ℕ) (power : 2 ^ (n + 1) = d) (ones twos : ℕ) :
    UnitDifferences (SparseFamily d ones twos) := fun c inside c' inside' different =>
  isUnit_of_short n power (sub_ne_zero.mpr different)
    (lt_of_le_of_lt (Nat.mul_le_mul_left 2
      (Nat.pow_le_pow_left (SparseFamily.normInf_sub_le inside inside') 2)) (by decide))

end NightstreamFPrime.Spec.Compression.Akita
