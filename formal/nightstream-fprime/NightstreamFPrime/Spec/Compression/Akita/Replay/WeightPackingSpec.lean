import NightstreamFPrime.Spec.Compression.Akita.Replay.WeightPacking
import NightstreamFPrime.Spec.Compression.Akita.Replay.WeightTraceSpec

/-! The building blocks of the weight at `r_2` of a packing fold (`Replay/WeightPacking.lean`)
against their sums.

- `dotFq_into`: `dotFq p a d = Σ_{i<d} a_i·p_i` (the value `A(α)` of a column of a setup matrix
  with `p = powersOf α`, `powersOf_getD`).
- `lowLevel_into`: the recursion `L^{(m+1)}_y = (1 − r_m)·L^{(m)}_{2y} + r_m·L^{(m)}_{2y+1}` gives
  `L^{(m)}_y = Σ_{c<2^m} w_{y·2^m+c}·ẽq(r[..m], c)`.
- `Tables.run_into`: `Tables.run` is `Σ_{c<2^m} w_{y·2^m+c}·ẽq(r_2, b + c)` for each start `b`: one
  product `L^{(m)}_y·ẽq(r_2[m..], b/2^m)` when `2^m ∣ b` (`eqAt_split`), else the two halves.
- `Runs` states that an address map is a run of cells in each chunk of `2^m` positions.
  `Tables.read_into` and `Tables.liftRead_into` give `Σ_k w_k·ẽq(r_2, addr k)` for such a map, and
  `Lift.value_into` gives the value (112) of a lift:
  `(Σ_{(s,addr)} s·Σ_j α^j·ẽq(r_2, addr j))·Σ_{(c,A)} c·A(α)`. -/

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

/-- The address map is a run of cells in each of the first `n` chunks of `2^m` positions, and each
run ends at `bound` or before. -/
structure Runs (m n bound : ℕ) (addr : ℕ → Option ℕ) : Prop where
  run : ∀ y < n, ∀ c < 2 ^ m, addr (y * 2 ^ m + c) = (addr (y * 2 ^ m)).map (· + c)
  inside : ∀ y < n, ∀ b, addr (y * 2 ^ m) = some b → b + 2 ^ m ≤ bound

/-- `Tables.read` is `Σ_{k<n·2^m} w_k·ẽq(r_2, addr k)` for an address map that is a run of cells in
each chunk of `2^m` positions. -/
theorem Tables.read_into (α : Ext) (r2 : List Ext) (bits top : ℕ) (w : Array Ext) {B m n : ℕ}
    (levels : B ≤ bits) (short : bits ≤ r2.length) (le : m ≤ B) (size : n * 2 ^ m ≤ w.size)
    {addr : ℕ → Option ℕ} (runs : Runs m n (2 ^ r2.length) addr) :
    (((Tables.of α r2 bits top).read (lowSums r2 B w) m n addr).into : E) =
      ∑ k ∈ Finset.range (n * 2 ^ m), ((w.getD k Ext.zero).into : E) *
        (addr k).elim 0 (eqAt (r2.map Ext.into)) := by
  unfold Tables.read
  rw [sumList_into, sum_list_range, ← sum_range_mul]
  refine Finset.sum_congr rfl fun y member => ?_
  have small := Finset.mem_range.mp member
  have fits : (y + 1) * 2 ^ m ≤ w.size :=
    (Nat.mul_le_mul_right _ (Nat.succ_le_of_lt small)).trans size
  cases start : addr (y * 2 ^ m) with
  | none =>
      rw [Ext.into_zero]
      refine (Finset.sum_eq_zero fun c inside => ?_).symm
      rw [runs.run y small c (Finset.mem_range.mp inside), start]
      simp
  | some b =>
      rw [Tables.run_into α r2 bits top w levels short m y b le fits (runs.inside y small b start)]
      refine Finset.sum_congr rfl fun c inside => ?_
      rw [runs.run y small c (Finset.mem_range.mp inside), start]
      rfl

/-- `Tables.liftRead` is `Σ_{j<d} α^j·ẽq(r_2, addr j)` for an address map that is a run of cells
in each chunk of `2^m` positions, with `2^m ∣ d` and `d` at most the powers of the tables. -/
theorem Tables.liftRead_into (α : Ext) (r2 : List Ext) (bits top : ℕ) {m d : ℕ}
    (short : bits ≤ r2.length) (le : m ≤ bits) (divides : 2 ^ m ∣ d) (degree : d ≤ top)
    {addr : ℕ → Option ℕ} (runs : Runs m (d / 2 ^ m) (2 ^ r2.length) addr) :
    (((Tables.of α r2 bits top).liftRead m d addr).into : E) =
      ∑ j ∈ Finset.range d, (α.into : E) ^ j * (addr j).elim 0 (eqAt (r2.map Ext.into)) := by
  have total : d / 2 ^ m * 2 ^ m = d := Nat.div_mul_cancel divides
  have size : d / 2 ^ m * 2 ^ m ≤ (powersOf α top).size := by
    rw [total, powersOf, Array.size_ofFn]
    exact degree
  rw [Tables.liftRead, show (Tables.of α r2 bits top).powerLows = lowSums r2 bits (powersOf α top)
    from rfl, Tables.read_into α r2 bits top _ le_rfl short le size runs, total]
  exact Finset.sum_congr rfl fun j member => by
    rw [powersOf_getD α ((Finset.mem_range.mp member).trans_le degree)]

omit [Degree2 E] in
/-- The value at `α` of the element of `R_{q,d}` of an array: `Σ_{i<d} a_i·α^i`. -/
theorem aeval_toRq (d : ℕ) [NeZero d] (a : Array Fq) (α : E) :
    Polynomial.aeval α (rep (Poly.toRq d a)) =
      ∑ i ∈ Finset.range d, algebraMap Zq E (Poly.get a i).toZq * α ^ i := by
  rw [aeval_rep]
  exact Finset.sum_congr rfl fun i member => by
    rw [Poly.coeff_toRq _ (Finset.mem_range.mp member), Algebra.smul_def]

/-- The value of a lift: `(Σ_{(s,addr)} s·Σ_j α^j·ẽq(r_2, addr j))·Σ_{(c,A)} c·A(α)` (Akita (112)),
for reads that are runs of cells in the chunks of the lift. -/
theorem Lift.value_into (α : Ext) (r2 : List Ext) (bits top : ℕ) (t : Lift) [NeZero t.degree]
    (short : bits ≤ r2.length) (le : t.bits ≤ bits) (divides : 2 ^ t.bits ∣ t.degree)
    (degree : t.degree ≤ top)
    (runs : ∀ r ∈ t.reads, Runs t.bits (t.degree / 2 ^ t.bits) (2 ^ r2.length) r.2) :
    ((t.value (Tables.of α r2 bits top)).into : E) =
      (t.reads.map fun r => algebraMap Zq E r.1.toZq * ∑ j ∈ Finset.range t.degree,
        (α.into : E) ^ j * (r.2 j).elim 0 (eqAt (r2.map Ext.into))).sum *
      (t.muls.map fun m => (m.1.into : E) *
        Polynomial.aeval (α.into : E) (rep (Poly.toRq t.degree m.2))).sum := by
  rw [Lift.value, Ext.into_mul, sumList_into, sumList_into]
  congr 1
  · refine congrArg List.sum (List.map_congr_left fun r member => ?_)
    rw [Ext.into_scale, Tables.liftRead_into α r2 bits top short le divides degree (runs r member)]
  · refine congrArg List.sum (List.map_congr_left fun m _ => ?_)
    rw [Ext.into_mul, aeval_toRq, show (Tables.of α r2 bits top).powers = powersOf α top from rfl,
      dotFq_into]
    exact congrArg _ (Finset.sum_congr rfl fun i member => by
      rw [powersOf_getD α ((Finset.mem_range.mp member).trans_le degree)])

end NightstreamFPrime.Spec.Compression.Akita.Replay
