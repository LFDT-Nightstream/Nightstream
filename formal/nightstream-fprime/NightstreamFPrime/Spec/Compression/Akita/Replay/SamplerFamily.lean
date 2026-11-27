import NightstreamFPrime.Spec.Compression.Akita.Replay.SamplerRank
import NightstreamFPrime.Spec.Compression.Akita.ChallengeFamily

/-! The challenges of the sampler of the pinned Akita code (`Replay/Sampler.lean`) as elements of
`R_{q,d}`, against the family and the filter of the specification (Akita, ePrint 2026/1983,
revision of 2026-11-23, Section 3.2 and Appendix H.1).

- `Sparse.value c j` is the coefficient of `c` at `j`, and `Sparse.ring d c` the element of
  `R_{q,d}` with these coefficients. `Sparse.Valid d c`: distinct positions below `d`, magnitudes
  at most 2.
- `accepts_iff`: for a valid challenge, `Filter.accepts (Filter.of d)` is the predicate
  `TrigTable.codeAccepts d T` of the specification (`Akita/TrigBuild.lean`) on its element.
- `candidate_table`: the sampler with the table of `binomials` is the sampler with `Nat.choose`.
- `candidate_mem`: each challenge of a candidate is in `SparseFamily d pm1 pm2`.
- `candidate_injective`: two candidates with the same element of `R_{q,d}` are equal.
- `candidate_isSome`: a candidate whose rank is below `rankSpace` has a challenge. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita

namespace Sparse

/-- The coefficient of a challenge at the position `j`. -/
def value (c : Sparse) (j : ℕ) : ℤ := ((c.filter fun x => x.1 == j).map Prod.snd).sum

/-- The challenge as an element of `R_{q,d}`. -/
noncomputable def ring (d : ℕ) [NeZero d] (c : Sparse) : Rq d :=
  ofCoeffs fun j => (c.value j : Zq)

/-- Distinct positions below `d` and magnitudes at most 2. -/
structure Valid (d : ℕ) (c : Sparse) : Prop where
  distinct : (c.map Prod.fst).Nodup
  bounded : ∀ x ∈ c, x.1 < d
  small : ∀ x ∈ c, x.2.natAbs ≤ 2

theorem value_cons (x : ℕ × ℤ) (c : Sparse) (j : ℕ) :
    value (x :: c) j = (if x.1 = j then x.2 else 0) + value c j := by
  unfold value
  by_cases same : x.1 = j
  · simp [same]
  · simp [same]

theorem value_eq_zero {c : Sparse} {j : ℕ} (absent : j ∉ c.map Prod.fst) : value c j = 0 := by
  induction c with
  | nil => rfl
  | cons x c ih =>
      simp only [List.map_cons, List.mem_cons, not_or] at absent
      rw [value_cons, ite_eq_right fun same => absent.1 same.symm, ih absent.2, add_zero]

theorem value_mem {c : Sparse} (distinct : (c.map Prod.fst).Nodup) {x : ℕ × ℤ}
    (member : x ∈ c) : value c x.1 = x.2 := by
  induction c with
  | nil => simp at member
  | cons y c ih =>
      simp only [List.map_cons, List.nodup_cons] at distinct
      rw [value_cons]
      rcases List.mem_cons.mp member with rfl | member
      · rw [ite_eq_left rfl, value_eq_zero distinct.1, add_zero]
      · have differ : y.1 ≠ x.1 := fun same =>
          distinct.1 (by rw [same]; exact List.mem_map_of_mem member)
        rw [ite_eq_right differ, zero_add, ih distinct.2 member]

variable {d : ℕ}

/-- A sum over the coefficients of a challenge with distinct positions below `d` is the sum over
its pairs. -/
theorem sum_value {M : Type} [AddCommMonoid M] (f : ℕ → ℤ → M) (zero : ∀ j, f j 0 = 0)
    {c : Sparse} (distinct : (c.map Prod.fst).Nodup) (bounded : ∀ x ∈ c, x.1 < d) :
    ∑ j ∈ Finset.range d, f j (value c j) = (c.map fun x => f x.1 x.2).sum := by
  induction c with
  | nil => simp [value, zero]
  | cons x c ih =>
      simp only [List.map_cons, List.nodup_cons] at distinct
      have point (j : ℕ) : f j (value (x :: c) j) =
          (if x.1 = j then f x.1 x.2 else 0) + f j (value c j) := by
        rw [value_cons]
        by_cases same : x.1 = j
        · subst same
          rw [ite_eq_left rfl, ite_eq_left rfl, value_eq_zero distinct.1, add_zero, zero, add_zero]
        · rw [ite_eq_right same, ite_eq_right same, zero_add, zero_add]
      rw [Finset.sum_congr rfl fun j _ => point j, Finset.sum_add_distrib,
        Finset.sum_ite_eq_of_mem _ _ _ (Finset.mem_range.mpr (bounded x List.mem_cons_self)),
        ih distinct.2 fun y member => bounded y (List.mem_cons_of_mem _ member), List.map_cons,
        List.sum_cons]

theorem natAbs_value_le {c : Sparse} (valid : Valid d c) (j : ℕ) : (value c j).natAbs ≤ 2 := by
  by_cases present : j ∈ c.map Prod.fst
  · obtain ⟨x, member, rfl⟩ := List.mem_map.mp present
    rw [value_mem valid.distinct member]
    exact valid.small x member
  · rw [value_eq_zero present]
    simp

theorem valMinAbs_cast {v : ℤ} (small : v.natAbs ≤ 2) : (v : Zq).valMinAbs = v := by
  rw [ZMod.valMinAbs_spec]
  refine ⟨rfl, ?_⟩
  have modulus : (fieldModulus : ℤ) = 18446744073709551557 := rfl
  rw [Set.mem_Ioc, modulus]
  rcases Int.natAbs_eq v with same | same <;> omega

variable [NeZero d]

theorem coeff_ring (c : Sparse) {j : ℕ} (small : j < d) : coeff (ring d c) j = (value c j : Zq) :=
  coeff_ofCoeffs _ small

theorem valMinAbs_coeff {c : Sparse} (valid : Valid d c) {j : ℕ} (small : j < d) :
    (coeff (ring d c) j).valMinAbs = value c j := by
  rw [coeff_ring c small, valMinAbs_cast (natAbs_value_le valid j)]

theorem magnitude_coeff {c : Sparse} (valid : Valid d c) {j : ℕ} (small : j < d) :
    magnitude (coeff (ring d c) j) = (value c j).natAbs := by
  rw [magnitude, valMinAbs_coeff valid small]

theorem normOne_ring {c : Sparse} (valid : Valid d c) :
    normOne (ring d c) = (c.map fun x => x.2.natAbs).sum := by
  rw [normOne, Finset.sum_congr rfl fun j member =>
      magnitude_coeff valid (Finset.mem_range.mp member),
    sum_value (fun _ v => v.natAbs) (fun _ => rfl) valid.distinct valid.bounded]

/-- Two valid challenges with the same element have the same coefficients. -/
theorem value_eq_of_ring {c c' : Sparse} (valid : Valid d c) (valid' : Valid d c')
    (same : ring d c = ring d c') (j : ℕ) : value c j = value c' j := by
  by_cases small : j < d
  · rw [← valMinAbs_coeff valid small, ← valMinAbs_coeff valid' small, same]
  · rw [value_eq_zero fun member => small (by
        obtain ⟨x, inside, rfl⟩ := List.mem_map.mp member
        exact valid.bounded x inside),
      value_eq_zero fun member => small (by
        obtain ⟨x, inside, rfl⟩ := List.mem_map.mp member
        exact valid'.bounded x inside)]

end Sparse

open Sparse

theorem cos_of {d t : ℕ} (small : t < 2 * d) :
    (Filter.of d).cos[t]?.getD 0 = (TrigTable.code d 48).cos t := by
  simp [Filter.of, TrigTable.code, small]

theorem sin_of {d t : ℕ} (small : t < 2 * d) :
    (Filter.of d).sin[t]?.getD 0 = (TrigTable.code d 48).sin t := by
  simp [Filter.of, TrigTable.code, small]

variable {d : ℕ} [NeZero d]

theorem real_ring {c : Sparse} (valid : Valid d c) (k : ℕ) :
    (TrigTable.code d 48).real (ring d c) k = (Filter.of d).sum (Filter.of d).cos c k := by
  have index (p : ℕ) : (2 * k + 1) * p % (2 * d) < 2 * d :=
    Nat.mod_lt _ (by have := NeZero.pos d; omega)
  rw [TrigTable.real, Finset.sum_congr rfl fun j member => by
      rw [valMinAbs_coeff valid (Finset.mem_range.mp member)],
    sum_value (fun j v => v * (TrigTable.code d 48).cos ((2 * k + 1) * j % (2 * d)))
      (fun _ => zero_mul _) valid.distinct valid.bounded, Filter.sum]
  exact congrArg List.sum (List.map_congr_left fun x _ =>
    congrArg (x.2 * ·) (cos_of (index x.1)).symm)

theorem imaginary_ring {c : Sparse} (valid : Valid d c) (k : ℕ) :
    (TrigTable.code d 48).imaginary (ring d c) k = (Filter.of d).sum (Filter.of d).sin c k := by
  have index (p : ℕ) : (2 * k + 1) * p % (2 * d) < 2 * d :=
    Nat.mod_lt _ (by have := NeZero.pos d; omega)
  rw [TrigTable.imaginary, Finset.sum_congr rfl fun j member => by
      rw [valMinAbs_coeff valid (Finset.mem_range.mp member)],
    sum_value (fun j v => v * (TrigTable.code d 48).sin ((2 * k + 1) * j % (2 * d)))
      (fun _ => zero_mul _) valid.distinct valid.bounded, Filter.sum]
  exact congrArg List.sum (List.map_congr_left fun x _ =>
    congrArg (x.2 * ·) (sin_of (index x.1)).symm)

/-- The filter of the replay is the predicate `TrigTable.codeAccepts` of the specification. -/
theorem accepts_iff {c : Sparse} (valid : Valid d c) :
    (Filter.of d).accepts c = true ↔
      TrigTable.codeAccepts d (FoldGroup.threshold d) (ring d c) := by
  simp only [Filter.accepts, List.all_eq_true, List.mem_range, decide_eq_true_eq,
    TrigTable.codeAccepts, TrigTable.Accepts, TrigTable.upper, real_ring valid,
    imaginary_ring valid, normOne_ring valid]
  rfl

end NightstreamFPrime.Spec.Compression.Akita.Replay
