import NightstreamFPrime.Spec.Compression.Akita.Replay.WeightPacking
import NightstreamFPrime.Spec.Compression.Akita.Replay.WeightSpec

/-! The building blocks of the weight at `r_2` of a packing fold (`Replay/WeightPacking.lean`)
against their sums.

- `dotFq_into`: `dotFq p a d = Σ_{i<d} a_i·p_i` (the value `A(α)` of a column of a setup matrix
  with `p = powersOf α`, `powersOf_getD`).
- `lowLevel_into`: the recursion `L^{(m+1)}_y = (1 − r_m)·L^{(m)}_{2y} + r_m·L^{(m)}_{2y+1}` gives
  `L^{(m)}_y = Σ_{c<2^m} w_{y·2^m+c}·ẽq(r[..m], c)`.
- `Tables.run_into`: `Tables.run` is `Σ_{c<2^m} w_{y·2^m+c}·ẽq(r_2, b + c)` for each start `b`: one
  product `L^{(m)}_y·ẽq(r_2[m..], b/2^m)` when `2^m ∣ b` (`eqAt_split`), else the two halves. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita NightstreamFPrime.Spec.Compression.Lib

variable {E : Type} [CommRing E] [Algebra Zq E] [Degree2 E]

theorem dotFrom_toE (p : Array Ext) (a : Array Fq) : ∀ n,
    Ext.toE ⟨Fq.ofNat (dotFrom p a n).1, Fq.ofNat (dotFrom p a n).2⟩ =
      (sumExt (fun i => Ext.scale (Poly.get a i) (p.getD i Ext.zero)) n).toE
  | 0 => by ext <;> simp [dotFrom, sumExt, Ext.toE, Ext.zero, Fq.toZq_ofNat, Fq.toZq_zero]
  | n + 1 => by
      rw [sumExt, Ext.toE_add, ← dotFrom_toE p a n, dotFrom.eq_2]
      split
      · rename_i zero
        have nil : (Poly.get a n).toZq = 0 := by simp [Fq.toZq, zero]
        ext <;> simp [Ext.toE, Ext.scale, Fq.toZq_mul, nil]
      · ext <;> simp [Ext.toE, Ext.scale, Fq.toZq_mul, Fq.toZq_ofNat] <;> simp [Fq.toZq]

/-- `dotFq p a d = Σ_{i<d} a_i·p_i`. -/
theorem dotFq_into (p : Array Ext) (a : Array Fq) (d : ℕ) :
    ((dotFq p a d).into : E) = ∑ i ∈ Finset.range d,
      algebraMap Zq E (Poly.get a i).toZq * (p.getD i Ext.zero).into := by
  have same : ((dotFq p a d).into : E) =
      (sumExt (fun i => Ext.scale (Poly.get a i) (p.getD i Ext.zero)) d).into := by
    simp only [Ext.into]
    rw [← dotFrom_toE]
    rfl
  rw [same, sumExt_into]
  exact Finset.sum_congr rfl fun i _ => Ext.into_scale _ _

/-- `powersOf α n` has `α^k` at `k < n`. -/
theorem powersOf_getD (α : Ext) {n k : ℕ} (small : k < n) :
    (((powersOf α n).getD k Ext.zero).into : E) = α.into ^ k := by
  rw [show (powersOf α n).getD k Ext.zero = Ext.pow α k by simp [powersOf, Array.getD, small],
    Ext.into_pow]

theorem lowLevel_size (r : List Ext) (w : Array Ext) : ∀ m, (lowLevel r w m).size = w.size / 2 ^ m
  | 0 => by simp [lowLevel]
  | m + 1 => by
      simp only [lowLevel, Array.size_ofFn, lowLevel_size r w m, pow_succ, Nat.div_div_eq_div_mul]

/-- `ẽq(r[..m+1], ·)` on the two halves of the cube of `m + 1` coordinates. -/
theorem eqAt_take_succ (r : List Ext) {m : ℕ} (inside : m < r.length) {c : ℕ} (small : c < 2 ^ m) :
    eqAt ((r.take (m + 1)).map (Ext.into (E := E))) c =
        eqAt ((r.take m).map Ext.into) c * (1 - (r.getD m Ext.zero).into) ∧
      eqAt ((r.take (m + 1)).map (Ext.into (E := E))) (2 ^ m + c) =
        eqAt ((r.take m).map Ext.into) c * (r.getD m Ext.zero).into := by
  have take : ((r.take (m + 1)).map (Ext.into (E := E))).take m = (r.take m).map Ext.into := by
    rw [← List.map_take, List.take_take, min_eq_left (Nat.le_succ m)]
  have drop : ((r.take (m + 1)).map (Ext.into (E := E))).drop m = [(r.getD m Ext.zero).into] := by
    rw [← List.map_drop, List.drop_take, show m + 1 - m = 1 by omega,
      List.take_one_drop_eq_of_lt_length inside, List.map_singleton]
    simp only [List.getD, List.getElem?_eq_getElem inside, Option.getD_some, List.get_eq_getElem]
  constructor
  · rw [eqAt_split m, take, drop, Nat.mod_eq_of_lt small, Nat.div_eq_of_lt small]
    simp [eqAt]
  · rw [eqAt_split m, take, drop, Nat.add_mod_left, Nat.mod_eq_of_lt small,
      Nat.add_div_left _ (Nat.two_pow_pos m), Nat.div_eq_of_lt small]
    simp [eqAt]

/-- `L^{(m)}_y = Σ_{c<2^m} w_{y·2^m+c}·ẽq(r[..m], c)`. -/
theorem lowLevel_into (r : List Ext) (w : Array Ext) : ∀ m, m ≤ r.length → ∀ y,
    (y + 1) * 2 ^ m ≤ w.size →
    (((lowLevel r w m).getD y Ext.zero).into : E) = ∑ c ∈ Finset.range (2 ^ m),
      ((w.getD (y * 2 ^ m + c) Ext.zero).into : E) * eqAt ((r.take m).map Ext.into) c
  | 0, _, y, _ => by simp [lowLevel, eqAt]
  | m + 1, le, y, fits => by
      have inside : m < r.length := by omega
      have below : y < (lowLevel r w m).size / 2 := by
        rw [lowLevel_size, Nat.div_div_eq_div_mul, ← pow_succ]
        exact Nat.lt_of_lt_of_le (Nat.lt_succ_self y)
          ((Nat.le_div_iff_mul_le (Nat.two_pow_pos _)).mpr fits)
      have entry : (lowLevel r w (m + 1)).getD y Ext.zero =
          (Ext.one - r.getD m Ext.zero) * (lowLevel r w m).getD (2 * y) Ext.zero +
            r.getD m Ext.zero * (lowLevel r w m).getD (2 * y + 1) Ext.zero := by
        simp only [lowLevel]
        simp [Array.getD, below]
      have two : 2 ^ (m + 1) = 2 ^ m + 2 ^ m := by rw [pow_succ]; ring
      rw [entry, Ext.into_add, Ext.into_mul, Ext.into_mul, Ext.into_sub, Ext.into_one,
        lowLevel_into r w m (by omega) (2 * y) (by rw [two] at fits; nlinarith),
        lowLevel_into r w m (by omega) (2 * y + 1) (by rw [two] at fits; nlinarith), two,
        Finset.sum_range_add, Finset.mul_sum, Finset.mul_sum]
      congr 1
      · refine Finset.sum_congr rfl fun c member => ?_
        rw [(eqAt_take_succ r inside (Finset.mem_range.mp member)).1,
          show y * (2 ^ m + 2 ^ m) + c = 2 * y * 2 ^ m + c by ring]
        ring
      · refine Finset.sum_congr rfl fun c member => ?_
        rw [(eqAt_take_succ r inside (Finset.mem_range.mp member)).2,
          show y * (2 ^ m + 2 ^ m) + (2 ^ m + c) = (2 * y + 1) * 2 ^ m + c by ring]
        ring

theorem lowSums_getD (r : List Ext) (w : Array Ext) {bits m : ℕ} (le : m ≤ bits) :
    (lowSums r bits w).getD m #[] = lowLevel r w m := by
  simp [lowSums, Array.getD, Nat.lt_succ_of_le le]

theorem Tables.of_high (α : Ext) (r2 : List Ext) (bits top : ℕ) {m : ℕ} (le : m ≤ bits) :
    (Tables.of α r2 bits top).high.getD m (Tables.of α r2 bits top).eq = CellEq.half (r2.drop m) := by
  simp [Tables.of, Array.getD, Nat.lt_succ_of_le le]

/-- `Tables.run` is `Σ_{c<2^m} w_{y·2^m+c}·ẽq(r_2, b + c)` for the low sums of `w` with `B ≥ m`
levels and each start `b` of a run in the cube. -/
theorem Tables.run_into (α : Ext) (r2 : List Ext) (bits top : ℕ) (w : Array Ext) {B : ℕ}
    (levels : B ≤ bits) (short : bits ≤ r2.length) : ∀ m y b, m ≤ B → (y + 1) * 2 ^ m ≤ w.size →
    b + 2 ^ m ≤ 2 ^ r2.length →
    (((Tables.of α r2 bits top).run (lowSums r2 B w) m y b).into : E) =
      ∑ c ∈ Finset.range (2 ^ m), ((w.getD (y * 2 ^ m + c) Ext.zero).into : E) *
        eqAt (r2.map Ext.into) (b + c)
  | 0, y, b, _, fits, inside => by
      simp only [Tables.run, lowSums_getD r2 w (Nat.zero_le B), lowLevel, Ext.into_mul]
      rw [show (Tables.of α r2 bits top).eq = CellEq.half r2 from rfl,
        CellEq.get_half r2 (by rw [pow_zero] at inside; omega)]
      simp
  | m + 1, y, b, le, fits, inside => by
      have two : 2 ^ (m + 1) = 2 ^ m + 2 ^ m := by rw [pow_succ]; ring
      unfold Tables.run
      split
      · rename_i aligned
        obtain ⟨q, rfl⟩ := Nat.dvd_of_mod_eq_zero aligned
        have long : m + 1 ≤ r2.length := by omega
        rw [Ext.into_mul, lowSums_getD r2 w le, lowLevel_into r2 w (m + 1) long y fits,
          Tables.of_high α r2 bits top (by omega), Nat.mul_div_cancel_left _ (Nat.two_pow_pos _),
          CellEq.get_half]
        · rw [Finset.sum_mul]
          refine Finset.sum_congr rfl fun c member => ?_
          have small := Finset.mem_range.mp member
          rw [eqAt_split (m + 1) (r2.map Ext.into) (2 ^ (m + 1) * q + c), ← List.map_take,
            ← List.map_drop, show (2 ^ (m + 1) * q + c) % 2 ^ (m + 1) = c by
              rw [Nat.mul_add_mod, Nat.mod_eq_of_lt small],
            show (2 ^ (m + 1) * q + c) / 2 ^ (m + 1) = q by
              rw [Nat.mul_add_div (Nat.two_pow_pos _), Nat.div_eq_of_lt small, add_zero]]
          ring
        · rw [List.length_drop]
          have split : 2 ^ r2.length = 2 ^ (m + 1) * 2 ^ (r2.length - (m + 1)) := by
            rw [← pow_add, Nat.add_sub_cancel' long]
          rw [split] at inside
          have scaled : 2 ^ (m + 1) * (q + 1) ≤ 2 ^ (m + 1) * 2 ^ (r2.length - (m + 1)) := by
            rw [Nat.mul_add, mul_one]
            exact inside
          have := Nat.le_of_mul_le_mul_left scaled (Nat.two_pow_pos _)
          omega
      · rw [Ext.into_add, Tables.run_into α r2 bits top w levels short m (2 * y) b (by omega)
          (by rw [two] at fits; nlinarith) (by rw [two] at inside; omega),
          Tables.run_into α r2 bits top w levels short m (2 * y + 1) (b + 2 ^ m) (by omega)
          (by rw [two] at fits; nlinarith) (by rw [two] at inside; omega), two,
          Finset.sum_range_add]
        congr 1
        · refine Finset.sum_congr rfl fun c _ => ?_
          rw [show y * (2 ^ m + 2 ^ m) + c = 2 * y * 2 ^ m + c by ring]
        · refine Finset.sum_congr rfl fun c _ => ?_
          rw [show y * (2 ^ m + 2 ^ m) + (2 ^ m + c) = (2 * y + 1) * 2 ^ m + c by ring,
            show b + (2 ^ m + c) = b + 2 ^ m + c by ring]

end NightstreamFPrime.Spec.Compression.Akita.Replay
