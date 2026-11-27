import NightstreamFPrime.Spec.Compression.Akita.Replay.Fused
import NightstreamFPrime.Spec.Compression.Akita.Replay.TerminalSpec

/-! The computable weight at `r_2` and the end check of the fused sum-check against the
specification (ePrint 2026/1983, Lemma 7.3, (117) to (125); `Akita/RingCheck.lean`,
`Akita/RowBatch.lean`, `Akita/FoldCheck.lean`).

- `wrap α d k` is `(X^k mod (X^d + 1))(α)` for `k < 2d` (`reducedEval_root_pow`), so the residue
  kernel of Akita (117) is `κ_{A,α}(j) = Σ_i A_i·wrap(i + j)` (`residueKernel_eq_wrap`).
- `kernelSum α d e i = Σ_j e_j·wrap(i + j)` has the recurrence of `kernel` (`kernelSum_succ`), and
  `kernel_into` states that `kernel` computes it. `dotKernel_into`: `Σ_i A_i·K_i` is
  `Σ_j e_j·κ_{A,α}(j)` (Lemma 7.3, read in the other order).
- `termValue_into` is the value of a `Term`: `Σ_{(c,A)} c·Σ_{(s,addr)} s·Σ_j κ_{A,α}(j)·eq(addr j)`.
- `CellEq.get_of`: the split table is `ẽq(r, x)`.
- `mle_batchWeight`: `m_{τ_1}(r_2) = Σ_i ẽq(τ_1, i)·M_i(r_2)` for the rows of a field relation, and
  `mle_freeWeight`: the weight of one ring row of the quotient-free reduction at `r_2` is
  `Σ_c Σ_j κ_{A_c,α}(j)·ẽq(r_2, addr_c(j))`.
- `FusedEnd.holds_iff_fused`: the check of the code is the final check of `fusedCheck` (with the
  weight `m_{τ_1} + η·K_{r_virt}` of `foldCheckSq`) when the four evaluations are the values at
  `r_2` of the weights of the specification. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita NightstreamFPrime.Spec.Compression.Lib

section Wrap

variable {E : Type} [CommRing E] [Algebra Zq E]

/-- `σ(k) = α^k` for `k < d` and `σ(k) = −α^{k−d}` for `d ≤ k`. -/
def wrap (α : E) (d k : ℕ) : E := if k < d then α ^ k else -α ^ (k - d)

variable {d : ℕ} [NeZero d]

theorem reducedEval_root_pow_lt (α : E) {k : ℕ} (small : k < d) :
    reducedEval α (AdjoinRoot.root (modulus d) ^ k) = α ^ k := by
  rw [reducedEval_apply, aeval_rep, Finset.sum_eq_single k]
  · rw [coeff_root_pow small small, ite_eq_left rfl, one_smul]
  · intro i member other
    rw [coeff_root_pow small (Finset.mem_range.mp member), ite_eq_right other, zero_smul]
  · intro absent
    exact absurd (Finset.mem_range.mpr small) absent

/-- `(X^k mod (X^d + 1))(α) = σ(k)` for `k < 2d`. -/
theorem reducedEval_root_pow (α : E) {k : ℕ} (small : k < 2 * d) :
    reducedEval α (AdjoinRoot.root (modulus d) ^ k) = wrap α d k := by
  unfold wrap
  split
  · exact reducedEval_root_pow_lt α ‹_›
  · obtain ⟨m, rfl⟩ : ∃ m, k = d + m := ⟨k - d, by omega⟩
    rw [pow_add, root_pow_degree, neg_one_mul, map_neg, reducedEval_root_pow_lt α (by omega),
      Nat.add_sub_cancel_left]

/-- Akita (118): `κ_{A,α}(j) = Σ_i A_i·σ(i + j)`. -/
theorem residueKernel_eq_wrap (A : Rq d) (α : E) {j : ℕ} (small : j < d) :
    residueKernel A α j = ∑ i ∈ Finset.range d, coeff A i • wrap α d (i + j) := by
  rw [residueKernel]
  conv_lhs => rw [← ofCoeffs_coeff A, ofCoeffs_eq_sum, Finset.sum_mul]
  rw [map_sum]
  refine Finset.sum_congr rfl fun i member => ?_
  rw [smul_mul_assoc, ← pow_add, map_smul,
    reducedEval_root_pow α (by have := Finset.mem_range.mp member; omega)]

omit [NeZero d] [Algebra Zq E] in
theorem wrap_succ (α : E) {i : ℕ} (j : ℕ) (hi : i < d) :
    wrap α d (i + 1 + j) = α * wrap α d (i + j) - (if j = d - 1 - i then α ^ d + 1 else 0) := by
  unfold wrap
  by_cases low : i + j + 1 < d
  · rw [ite_eq_left (by omega), ite_eq_left (by omega), ite_eq_right (by omega), sub_zero,
      show i + 1 + j = i + j + 1 by omega, pow_succ]
    ring
  · by_cases edge : i + j + 1 = d
    · rw [ite_eq_right (by omega), ite_eq_left (by omega), ite_eq_left (by omega),
        show i + 1 + j - d = 0 by omega, pow_zero, ← edge, pow_succ]
      ring
    · rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega), sub_zero,
        show i + 1 + j - d = i + j - d + 1 by omega, pow_succ]
      ring

/-- The kernels `K_i = Σ_{j<d} e_j·σ(i + j)` of a read vector `e`. -/
def kernelSum (α : E) (d : ℕ) (e : ℕ → E) (i : ℕ) : E :=
  ∑ j ∈ Finset.range d, e j * wrap α d (i + j)

omit [NeZero d] [Algebra Zq E] in
theorem kernelSum_zero (α : E) (e : ℕ → E) :
    kernelSum α d e 0 = ∑ j ∈ Finset.range d, e j * α ^ j :=
  Finset.sum_congr rfl fun j member => by
    rw [zero_add, wrap, ite_eq_left (Finset.mem_range.mp member)]

omit [NeZero d] [Algebra Zq E] in
/-- `K_{i+1} = α·K_i − e_{d−1−i}·(α^d + 1)`. -/
theorem kernelSum_succ (α : E) (e : ℕ → E) {i : ℕ} (hi : i < d) :
    kernelSum α d e (i + 1) = α * kernelSum α d e i - e (d - 1 - i) * (α ^ d + 1) := by
  unfold kernelSum
  have terms : ∑ j ∈ Finset.range d, e j * wrap α d (i + 1 + j) =
      ∑ j ∈ Finset.range d, (α * (e j * wrap α d (i + j)) -
        (if j = d - 1 - i then e j * (α ^ d + 1) else 0)) := by
    refine Finset.sum_congr rfl fun j member => ?_
    rw [wrap_succ α j hi]
    split <;> ring
  rw [terms, Finset.sum_sub_distrib, Finset.sum_ite_eq', ite_eq_left (Finset.mem_range.mpr (by omega)),
    Finset.mul_sum]

omit [NeZero d] in
/-- Lemma 7.3 in the other order: `Σ_i A_i·K_i = Σ_j e_j·Σ_i A_i·σ(i + j)`. -/
theorem sum_kernelSum (α : E) (a : ℕ → Zq) (e : ℕ → E) :
    ∑ i ∈ Finset.range d, a i • kernelSum α d e i =
      ∑ j ∈ Finset.range d, e j * ∑ i ∈ Finset.range d, a i • wrap α d (i + j) := by
  simp only [kernelSum, Finset.smul_sum, Finset.mul_sum]
  rw [Finset.sum_comm]
  exact Finset.sum_congr rfl fun j _ => Finset.sum_congr rfl fun i _ =>
    (mul_smul_comm _ _ _).symm

end Wrap

section Computable

variable {E : Type} [CommRing E] [Algebra Zq E] [Degree2 E]

theorem Ext.toE_scale (s : Fq) (z : Ext) : (Ext.scale s z).toE = (Ext.ofBase s * z).toE := by
  show _ = (Ext.mul _ _).toE
  ext <;> simp [Ext.toE, Ext.scale, Ext.mul, Ext.ofBase, Fq.toZq_add, Fq.toZq_mul, Fq.toZq_zero]

theorem Ext.into_scale (s : Fq) (z : Ext) :
    ((Ext.scale s z).into : E) = algebraMap Zq E s.toZq * z.into := by
  rw [← Ext.into_ofBase, ← Ext.into_mul]
  simp only [Ext.into, Ext.toE_scale]

theorem Ext.into_neg (z : Ext) : ((-z).into : E) = -z.into := by
  have zero : (Ext.zero - z).toE = (-z).toE := by
    show (Ext.sub _ _).toE = (Ext.neg _).toE
    ext <;> simp [Ext.toE, Ext.sub, Ext.neg, Ext.zero, Fq.toZq_sub, Fq.toZq_neg, Fq.toZq_zero]
  have value := (Ext.into_sub (E := E) Ext.zero z)
  rw [Ext.into_zero, zero_sub] at value
  rw [← value]
  simp only [Ext.into, zero]

theorem Ext.into_pow (z : Ext) (n : ℕ) : ((Ext.pow z n).into : E) = z.into ^ n := by
  induction n using Nat.strong_induction_on generalizing z with
  | _ n ih =>
      rw [Ext.pow]
      split
      · rename_i zero
        rw [zero, pow_zero, Ext.into_one]
      · split
        · rename_i _ odd
          rw [Ext.into_mul, ih _ (by omega), Ext.into_mul, ← pow_two, ← pow_mul, ← pow_succ]
          congr 1
          omega
        · rename_i _ even
          rw [ih _ (by omega), Ext.into_mul, ← pow_two, ← pow_mul]
          congr 1
          omega

theorem sumList_into {α : Type} (l : List α) (f : α → Ext) :
    ((sumList l f).into : E) = (l.map fun x => ((f x).into : E)).sum := by
  induction l with
  | nil => simp [sumList, Ext.into_zero]
  | cons x rest ih =>
      simp only [sumList, List.foldr_cons, List.map_cons, List.sum_cons] at ih ⊢
      rw [Ext.into_add, ih]

theorem hornerFrom_into (α : Ext) (e : Array Ext) : ∀ n i,
    ((hornerFrom α e i n).into : E) =
      ∑ k ∈ Finset.range n, ((e.getD (i + k) Ext.zero).into : E) * (α.into : E) ^ k
  | 0, i => by simp [hornerFrom, Ext.into_zero]
  | n + 1, i => by
      rw [hornerFrom, Ext.into_add, Ext.into_mul, hornerFrom_into α e n (i + 1),
        Finset.sum_range_succ', Finset.mul_sum, add_comm]
      simp only [add_zero, pow_zero, mul_one]
      congr 1
      refine Finset.sum_congr rfl fun k _ => ?_
      rw [show i + 1 + k = i + (k + 1) by omega, pow_succ]
      ring

theorem kernelFrom_getD (d : ℕ) (α αd1 : Ext) (e : Array Ext) (K : ℕ → E)
    (step : ∀ i < d, K (i + 1) =
      (α.into : E) * K i - ((e.getD (d - 1 - i) Ext.zero).into : E) * αd1.into) :
    ∀ n i (k : Ext), i + n ≤ d → (k.into : E) = K i → ∀ m < n,
      (((kernelFrom d α αd1 e i n k).getD m Ext.zero).into : E) = K (i + m)
  | 0, _, _, _, _, m, small => absurd small (Nat.not_lt_zero m)
  | n + 1, i, k, fits, start, m, small => by
      cases m with
      | zero => simpa [kernelFrom] using start
      | succ m =>
          simp only [kernelFrom, List.getD_cons_succ]
          rw [kernelFrom_getD d α αd1 e K step n (i + 1) _ (by omega) ?_ m (by omega)]
          · congr 1
            omega
          · rw [Ext.into_sub, Ext.into_mul, Ext.into_mul, start, step i (by omega)]

theorem toArray_getD (l : List Ext) (m : ℕ) : l.toArray.getD m Ext.zero = l.getD m Ext.zero := by
  unfold Array.getD List.getD
  split
  · rename_i h
    simp only [List.size_toArray] at h
    simp [List.getElem?_eq_getElem h]
  · rename_i h
    simp only [List.size_toArray] at h
    simp [List.getElem?_eq_none (Nat.le_of_not_lt h)]

/-- `kernel` computes the kernels `K_i` of the read vector. -/
theorem kernel_into (d : ℕ) (α : Ext) (e : Array Ext) {i : ℕ} (small : i < d) :
    (((kernel d α (Ext.pow α d + Ext.one) e).getD i Ext.zero).into : E) =
      kernelSum (α.into : E) d (fun j => ((e.getD j Ext.zero).into : E)) i := by
  rw [kernel, toArray_getD]
  have start : ((hornerFrom α e 0 d).into : E) =
      kernelSum (α.into : E) d (fun j => ((e.getD j Ext.zero).into : E)) 0 := by
    rw [hornerFrom_into, kernelSum_zero]
    simp only [zero_add]
  have value := kernelFrom_getD (E := E) d α (Ext.pow α d + Ext.one) e
    (kernelSum (α.into : E) d (fun j => ((e.getD j Ext.zero).into : E)))
    (fun i small => by rw [kernelSum_succ _ _ small, Ext.into_add, Ext.into_pow, Ext.into_one])
    d 0 _ (by omega) start i small
  rwa [zero_add] at value

/-- `Σ_i A_i·K_i = Σ_j e_j·κ_{A,α}(j)` for the kernels that `kernel` computes. -/
theorem dotKernel_into (d : ℕ) [NeZero d] (a : Array Fq) (α : Ext) (e : Array Ext) :
    ((dotKernel d a (kernel d α (Ext.pow α d + Ext.one) e)).into : E) =
      ∑ j ∈ Finset.range d, ((e.getD j Ext.zero).into : E) *
        residueKernel (Poly.toRq d a) (α.into : E) j := by
  rw [dotKernel, sumExt_into]
  have each : ∀ i ∈ Finset.range d,
      ((Ext.scale (Poly.get a i) ((kernel d α (Ext.pow α d + Ext.one) e).getD i Ext.zero)).into : E) =
        (Poly.get a i).toZq • kernelSum (α.into : E) d (fun j => ((e.getD j Ext.zero).into : E)) i :=
    fun i member => by
      rw [Ext.into_scale, kernel_into d α e (Finset.mem_range.mp member), Algebra.smul_def]
  rw [Finset.sum_congr rfl each, sum_kernelSum]
  refine Finset.sum_congr rfl fun j member => ?_
  rw [residueKernel_eq_wrap _ _ (Finset.mem_range.mp member)]
  exact congrArg _ (Finset.sum_congr rfl fun i inside => by
    rw [Poly.coeff_toRq _ (Finset.mem_range.mp inside)])

theorem readAt_into (eq : ℕ → Ext) (reads : List Read) (j : ℕ) :
    ((readAt eq reads j).into : E) = (reads.map fun r => algebraMap Zq E r.1.toZq *
      (r.2 j).elim 0 fun x => ((eq x).into : E)).sum := by
  induction reads with
  | nil => simp [readAt, Ext.into_zero]
  | cons r rest ih =>
      simp only [readAt, List.foldr_cons, List.map_cons, List.sum_cons] at ih ⊢
      cases h : r.2 j with
      | none => simp only [Option.elim_none, mul_zero, zero_add]; exact ih
      | some x => rw [Ext.into_add, Ext.into_scale, ih]; rfl

/-- The value of a term: `Σ_{(c,A)} c·Σ_{(s,addr)} s·Σ_j κ_{A,α}(j)·eq(addr j)`. -/
theorem termValue_into (t : Term) [NeZero t.degree] (α : Ext) (eq : ℕ → Ext) :
    ((termValue α eq t).into : E) = (t.muls.map fun m => (m.1.into : E) *
      ∑ j ∈ Finset.range t.degree, residueKernel (Poly.toRq t.degree m.2) (α.into : E) j *
        (t.reads.map fun r => algebraMap Zq E r.1.toZq *
          (r.2 j).elim 0 fun x => ((eq x).into : E)).sum).sum := by
  rw [termValue, sumList_into]
  refine congrArg List.sum (List.map_congr_left fun m _ => ?_)
  rw [Ext.into_mul, dotKernel_into]
  congr 1
  refine Finset.sum_congr rfl fun j member => ?_
  have small := Finset.mem_range.mp member
  rw [mul_comm, show (Array.ofFn (n := t.degree) fun j => readAt eq t.reads j.val).getD j Ext.zero =
    readAt eq t.reads j by simp [Array.getD, small], readAt_into]

end Computable

section Eq

theorem eqAt_split {K : Type} [CommRing K] : ∀ (b : ℕ) (r : List K) (x : ℕ),
    eqAt r x = eqAt (r.take b) (x % 2 ^ b) * eqAt (r.drop b) (x / 2 ^ b)
  | 0, r, x => by simp [eqAt]
  | b + 1, [], x => by simp [eqAt]
  | b + 1, r0 :: rest, x => by
      have h1 : x % 2 ^ (b + 1) % 2 = x % 2 :=
        Nat.mod_mod_of_dvd x (dvd_pow_self 2 (Nat.succ_ne_zero b))
      have h2 : x % 2 ^ (b + 1) / 2 = x / 2 % 2 ^ b := by
        rw [pow_succ', Nat.mod_mul_right_div_self]
      have h3 : x / 2 ^ (b + 1) = x / 2 / 2 ^ b := by
        rw [pow_succ', ← Nat.div_div_eq_div_mul]
      rw [List.take_succ_cons, List.drop_succ_cons, eqAt, eqAt, h1, h2, h3,
        eqAt_split b rest (x / 2), mul_assoc]

theorem eqCells_size : ∀ r : List Ext, (eqCells r).size = 2 ^ r.length
  | [] => rfl
  | r :: rest => by
      simp only [eqCells, Array.size_ofFn, eqCells_size rest, List.length_cons, pow_succ]
      ring

variable {E : Type} [CommRing E] [Algebra Zq E] [Degree2 E]

theorem eqCells_getD : ∀ (r : List Ext) {i : ℕ}, i < 2 ^ r.length →
    (((eqCells r).getD i Ext.zero).into : E) = eqAt (r.map Ext.into) i
  | [], i, small => by
      obtain rfl : i = 0 := by simpa using small
      simp [eqCells, eqAt, Ext.into_one]
  | r :: rest, i, small => by
      have size := eqCells_size rest
      have inside : i < 2 * (eqCells rest).size := by
        rw [size]
        simpa [pow_succ, mul_comm] using small
      have half : i / 2 < 2 ^ rest.length := by
        simp only [List.length_cons, pow_succ] at small
        omega
      have entry : (eqCells (r :: rest)).getD i Ext.zero =
          (if i % 2 = 1 then r else Ext.one - r) * (eqCells rest).getD (i / 2) Ext.zero := by
        conv_lhs => rw [eqCells]
        have : i < (Array.ofFn (n := 2 * (eqCells rest).size) fun i =>
            (if i.val % 2 = 1 then r else Ext.one - r) *
              (eqCells rest).getD (i.val / 2) Ext.zero).size := by
          rw [Array.size_ofFn]; exact inside
        rw [Array.getD, dite_eq_left this]
        exact Array.getElem_ofFn this
      rw [entry, Ext.into_mul, List.map_cons, eqAt, eqCells_getD rest half]
      split
      · rfl
      · rw [Ext.into_sub, Ext.into_one]

/-- The split table is `ẽq(r, x)` on the cube of `r`. -/
theorem CellEq.get_of (r : List Ext) {b x : ℕ} (le : b ≤ r.length) (small : x < 2 ^ r.length) :
    (((CellEq.of r b).get x).into : E) = eqAt (r.map Ext.into) x := by
  have low : x % 2 ^ b < 2 ^ (r.take b).length := by
    rw [List.length_take, min_eq_left le]
    exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have high : x / 2 ^ b < 2 ^ (r.drop b).length := by
    rw [List.length_drop, Nat.div_lt_iff_lt_mul (Nat.two_pow_pos _), ← pow_add,
      Nat.sub_add_cancel le]
    exact small
  rw [CellEq.get, Ext.into_mul]
  show (((eqCells (r.take b)).getD (x % 2 ^ b) Ext.zero).into : E) *
    ((eqCells (r.drop b)).getD (x / 2 ^ b) Ext.zero).into = _
  rw [eqCells_getD _ low, eqCells_getD _ high, eqAt_split b (r.map Ext.into) x, List.map_take,
    List.map_drop]

theorem CellEq.get_half (r : List Ext) {x : ℕ} (small : x < 2 ^ r.length) :
    (((CellEq.half r).get x).into : E) = eqAt (r.map Ext.into) x :=
  CellEq.get_of r (Nat.div_le_self _ _) small

theorem eqPoints_into : ∀ a b : List Ext,
    ((eqPoints a b).into : E) = eqEval (a.map Ext.into) (b.map Ext.into)
  | [], _ => by simp [eqPoints, eqEval, Ext.into_one]
  | _ :: _, [] => by simp [eqPoints, eqEval, Ext.into_one]
  | x :: xs, y :: ys => by
      simp only [eqPoints, eqEval, List.map_cons]
      rw [Ext.into_mul, eqPoints_into xs ys, Ext.into_add, Ext.into_mul, Ext.into_mul,
        Ext.into_sub, Ext.into_sub, Ext.into_one]

end Eq

section Rows

variable {E : Type} [Field E] [Algebra Zq E]

omit [Algebra Zq E] in
/-- `m_{τ_1}(r_2) = Σ_i ẽq(τ_1, i)·M_i(r_2)` (Akita (123) at `r_2`). -/
theorem mle_batchWeight (rows : FieldRows E) (τ r : List E) (fits : rows.count ≤ 2 ^ τ.length) :
    mle r (rows.batchWeight τ) = ∑ i : Fin rows.count, eqAt τ i * mle r (rows.weight i) := by
  have expand : rows.batchWeight τ =
      fun u => ∑ i ∈ Finset.range (2 ^ τ.length), eqAt τ i * rows.weightAt i u := by
    funext u
    rw [FieldRows.batchWeight, mle_eq_cubeSum, cubeSum_eq_sum_range]
  rw [expand, mle_finset_sum]
  simp only [mle_mul_left]
  rw [← Finset.sum_range_add_sum_Ico _ fits, Finset.sum_eq_zero (s := Finset.Ico _ _), add_zero,
    ← Fin.sum_univ_eq_sum_range]
  · refine Finset.sum_congr rfl fun i _ => ?_
    congr 1
    exact congrArg (mle r) (funext fun u => by simp [FieldRows.weightAt])
  · intro i member
    have large : rows.count ≤ i := (Finset.mem_Ico.mp member).1
    have zero : rows.weightAt i = fun _ => 0 :=
      funext fun u => by simp [FieldRows.weightAt, Nat.not_lt.mpr large]
    rw [zero, mle_const, mul_zero]

/-- The weight of one ring row of the quotient-free reduction at `r_2`:
`Σ_c Σ_j κ_{A_c,α}(j)·ẽq(r_2, addr_c(j))`. -/
theorem mle_freeWeight (row : RingRow) (α : E) (r : List E)
    (inside : ∀ c j x, j < row.degree → row.address c j = some x → x < 2 ^ r.length) :
    mle r (row.freeWeight α) = ∑ c, ∑ j ∈ Finset.range row.degree,
      residueKernel (row.multiplier c) α j * (row.address c j).elim 0 (eqAt r) := by
  rw [mle_eq_cubeSum, cubeSum_eq_sum_range]
  simp only [RingRow.freeWeight, Finset.mul_sum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun c _ => ?_
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun j member => ?_
  simp only [mul_ite, mul_zero]
  cases read : row.address c j with
  | none => simp
  | some x =>
      simp only [Option.some.injEq, Option.elim_some]
      rw [Finset.sum_ite_eq, ite_eq_left
        (Finset.mem_range.mpr (inside c j x (Finset.mem_range.mp member) read))]
      exact mul_comm _ _

end Rows

section End

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E]

/-- The check of the code in `E`. -/
theorem FusedEnd.holds_iff (f : FusedEnd) :
    f.holds = true ↔ (f.output.into : E) =
      f.value.into * (f.weight.into + f.norm.into * f.normWeight.into) +
        (f.batch.into * f.virt.into + f.binary.into * f.support.into) *
          (f.value.into * (f.value.into + 1)) := by
  rw [FusedEnd.holds, Ext.equals_iff (E := E), FusedEnd.expected]
  simp only [Ext.into_add, Ext.into_mul, Ext.into_one]

/-- The check of the code is the final check of `fusedCheck` for the weight
`m + η·K` of `foldCheckSq` (`η = 0` at a level without the norm check) when the four
evaluations of the values are the values at `r_2` of the weight `m`, of the norm weight `K`, of
`ẽq(r_virt, ·)` and of `1_{I_bin}·ẽq(r_virt, ·)`. -/
theorem FusedEnd.holds_iff_fused (f : FusedEnd) (support : ℕ → Prop) [DecidablePred support]
    (weight normWeight : ℕ → E) (r r2 : List E) (weighted : (f.weight.into : E) = mle r2 weight)
    (normed : (f.normWeight.into : E) = mle r2 normWeight)
    (virtual : (f.virt.into : E) = mle r2 (eqAt r))
    (supported : (f.support.into : E) = mle r2 fun x => if support x then eqAt r x else 0) :
    f.holds = true ↔ (f.output.into : E) =
      f.value.into * mle r2 (fun x => weight x + f.norm.into * normWeight x) +
        mle r2 (virtWeight support r f.batch.into f.binary.into) *
          (f.value.into * (f.value.into + 1)) := by
  have split : mle r2 (virtWeight support r f.batch.into f.binary.into) =
      (f.batch.into : E) * mle r2 (eqAt r) +
        f.binary.into * mle r2 (fun x => if support x then eqAt r x else 0) := by
    rw [show virtWeight support r f.batch.into f.binary.into = fun x =>
      (f.batch.into : E) * eqAt r x + f.binary.into * (if support x then eqAt r x else 0) from rfl,
      mle_add, mle_mul_left, mle_mul_left]
  rw [FusedEnd.holds_iff (E := E), mle_add, mle_mul_left, split, ← weighted, ← normed, ← virtual,
    ← supported]

end End

end NightstreamFPrime.Spec.Compression.Akita.Replay
