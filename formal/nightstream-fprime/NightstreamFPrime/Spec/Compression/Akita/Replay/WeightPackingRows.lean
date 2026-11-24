import NightstreamFPrime.Spec.Compression.Akita.Replay.WeightPackingSpec
import NightstreamFPrime.Spec.Compression.Akita.Extract.SetupFactor

/-! The public weight at `r_2` of a quotient-lift reduction as sums over its ring rows, and the
runs of cells of the address maps of the flat witness (ePrint 2026/1983, (111) to (113) and
(123); `Replay/WeightPacking.lean`).

- `RingRow.mle_liftWeight`: the weights (112) of a ring row at `r_2` are its linear form
  `RingRow.scan` with `ẽq(r_2, ·)` minus `(α^{d_r} + 1)` times the weights of its quotient digits.
- `LiftRelation.mle_lift`: `m_{τ_1}(r_2)` of `LiftRelation.lift` is the sum over the ring rows
  with the coefficients `ẽq(τ_1, slot r)·basis r` of the join, and the scalar row.
- `Runs.moved_of` moves a run of cells by a map that is a shift on the segment of the run, and
  `Gaps.place_shift` states that `Gaps.place` is a shift on each of its three segments.
- `run_of_start`, `OpeningShape.innerCell_eq` and `OpeningShape.partialPosition_eq` give the
  runs of the address maps of `FlatWitness.lean`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita NightstreamFPrime.Spec.Compression.Lib

section Lift

variable {E : Type} [Field E] [Algebra Zq E]

/-- The weights (112) of a ring row at `r_2`: its linear form `Σ_c Σ_j A_c(α)·α^j·ẽq(r_2, addr_c j)`
minus `(α^{d_r} + 1)·Σ_j Σ_a α^j·s_a·ẽq(r_2, q(j, a))` for its quotient digits. -/
theorem RingRow.mle_liftWeight (row : RingRow) (q : QuotientRead) (α : E) (r : List E)
    (inside : ∀ c j x, j < row.degree → row.address c j = some x → x < 2 ^ r.length)
    (quotientInside : ∀ j a, j < row.degree → q.address j a < 2 ^ r.length) :
    mle r (row.liftWeight q α) = row.scan α (fun o => o.elim 0 (eqAt r)) -
      (α ^ row.degree + 1) * ∑ j ∈ Finset.range row.degree, ∑ a,
        α ^ j * algebraMap Zq E (q.scale a) * eqAt r (q.address j a) := by
  rw [mle_eq_cubeSum, cubeSum_eq_sum_range]
  have read : ∑ u ∈ Finset.range (2 ^ r.length), eqAt r u * row.readWeight α u =
      row.scan α (fun o => o.elim 0 (eqAt r)) := by
    simp only [RingRow.readWeight, RingRow.scan, Finset.mul_sum]
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun c _ => ?_
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun j member => ?_
    simp only [mul_ite, mul_zero]
    cases h : row.address c j with
    | none => simp
    | some x =>
        simp only [Option.some.injEq, Option.elim_some]
        rw [Finset.sum_ite_eq, ite_eq_left (Finset.mem_range.mpr
          (inside c j x (Finset.mem_range.mp member) h))]
        ring
  have quotient : ∑ u ∈ Finset.range (2 ^ r.length), eqAt r u * q.weight row.degree α u =
      ∑ j ∈ Finset.range row.degree, ∑ a,
        α ^ j * algebraMap Zq E (q.scale a) * eqAt r (q.address j a) := by
    simp only [QuotientRead.weight, Finset.mul_sum]
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun j member => ?_
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun a _ => ?_
    simp only [mul_ite, mul_zero]
    rw [Finset.sum_ite_eq, ite_eq_left (Finset.mem_range.mpr
      (quotientInside j a (Finset.mem_range.mp member)))]
    ring
  unfold RingRow.liftWeight
  simp only [mul_sub, Finset.sum_sub_distrib]
  rw [read, ← quotient, Finset.mul_sum]
  congr 1
  exact Finset.sum_congr rfl fun u _ => by ring

theorem RingRow.scan_relocate (row : RingRow) (move : ℕ → ℕ) (α : E) (φ : Option ℕ → E) :
    (row.relocate move).scan α φ = row.scan α fun o => φ (o.map move) :=
  rfl

variable [Fintype E] in
/-- `m_{τ_1}(r_2)` of a quotient-lift reduction (Akita (123) at `r_2`): each ring row with the
coefficient `ẽq(τ_1, slot r)·basis r` of its row over `E`, and the scalar row. -/
theorem LiftRelation.mle_lift (rel : LiftRelation E) (largest batch : ℕ)
    (degrees : ∀ r, (rel.row r).degree ≤ largest) (fits : rel.join.count + 1 ≤ 2 ^ batch)
    (α : E) (τ r : List E) (tau : τ.length = batch) :
    mle r (((rel.lift largest batch degrees fits).rows α).batchWeight τ) =
      ∑ q, eqAt τ (rel.join.slot q) * (rel.join.basis q *
        mle r ((rel.row q).liftWeight (rel.quotient q) α)) +
        eqAt τ rel.join.count * mle r rel.weight := by
  rw [mle_batchWeight ((rel.lift largest batch degrees fits).rows α) τ r (by rw [tau]; exact fits)]
  show ∑ i : Fin (rel.join.count + 1), eqAt τ i * mle r ((rel.liftRows α).weight i) = _
  rw [Fin.sum_univ_castSucc]
  have each : ∀ k : Fin rel.join.count, mle r ((rel.liftRows α).weight (Fin.castSucc k)) =
      ∑ q, if rel.join.slot q = k then
        rel.join.basis q * mle r ((rel.row q).liftWeight (rel.quotient q) α) else 0 := by
    intro k
    simp only [LiftRelation.liftRows, Fin.lastCases_castSucc]
    rw [mle_finset_sum]
    refine Finset.sum_congr rfl fun q _ => ?_
    split_ifs
    · exact mle_mul_left _ _ _
    · exact mle_const _ _
  simp only [each, Finset.mul_sum, mul_ite, mul_zero, Fin.val_castSucc]
  rw [Finset.sum_comm]
  congr 1
  · refine Finset.sum_congr rfl fun q _ => ?_
    rw [Finset.sum_ite_eq, ite_eq_left (Finset.mem_univ _)]
  · simp only [LiftRelation.liftRows, Fin.lastCases_last, Fin.val_last]

end Lift

section Runs

/-- A run of cells of `f` moves to a run of cells by a map `move` that is a shift on the segment
`[lo, hi)` of the cells of `f`, when `move` puts each cell of `f` below `bound`. -/
theorem Runs.moved_of {m n bound lo hi : ℕ} {f : ℕ → Option ℕ} {move : ℕ → ℕ}
    (shift : ∀ x c, lo ≤ x → x + c < hi → move (x + c) = move x + c)
    (run : ∀ y < n, ∀ c < 2 ^ m, f (y * 2 ^ m + c) = (f (y * 2 ^ m)).map (· + c))
    (within : ∀ j x, f j = some x → lo ≤ x ∧ x < hi)
    (bounded : ∀ j x, f j = some x → move x < bound) :
    Runs m n bound (moved move f) := by
  refine ⟨fun y small c inside => ?_, fun y small b start => ?_⟩
  · unfold moved
    rw [run y small c inside]
    cases h : f (y * 2 ^ m) with
    | none => rfl
    | some x =>
        have last := within _ _ (show f (y * 2 ^ m + c) = some (x + c) by
          rw [run y small c inside, h]
          rfl)
        show some (move (x + c)) = some (move x + c)
        rw [shift x c (within _ _ h).1 last.2]
  · unfold moved at start
    cases h : f (y * 2 ^ m) with
    | none =>
        rw [h] at start
        cases start
    | some x =>
        rw [h] at start
        have same : move x = b := Option.some.inj start
        have positive := Nat.two_pow_pos m
        have read : f (y * 2 ^ m + (2 ^ m - 1)) = some (x + (2 ^ m - 1)) := by
          rw [run y small (2 ^ m - 1) (by omega), h]
          rfl
        have moved := bounded _ _ read
        rw [shift x _ (within _ _ h).1 (within _ _ read).2] at moved
        omega

/-- `Gaps.place` is a shift on a segment that is in one of its three segments. -/
theorem Gaps.place_shift (G : Gaps) {lo hi : ℕ}
    (segment : hi ≤ G.main ∨ (G.main ≤ lo ∧ hi ≤ G.middle) ∨ G.middle ≤ lo) :
    ∀ x c, lo ≤ x → x + c < hi → G.place (x + c) = G.place x + c := by
  intro x c low high
  have ordered := G.ordered
  unfold Gaps.place
  split_ifs <;> omega

/-- A map that reads the cells from `s` on its first `d` positions is a run in each chunk of
`2^m` positions before `n·2^m ≤ d`. -/
theorem run_of_start {f : ℕ → Option ℕ} {m n s d : ℕ} (form : ∀ k < d, f k = some (s + k))
    (fits : n * 2 ^ m ≤ d) :
    ∀ y < n, ∀ c < 2 ^ m, f (y * 2 ^ m + c) = (f (y * 2 ^ m)).map (· + c) := by
  intro y small c inside
  have step := Nat.mul_le_mul_right (2 ^ m) (Nat.succ_le_of_lt small)
  rw [Nat.succ_mul] at step
  rw [form (y * 2 ^ m + c) (by omega), form (y * 2 ^ m) (by omega)]
  exact congrArg some (Nat.add_assoc _ _ _).symm

/-- A map with no cell is a run. -/
theorem run_of_none {f : ℕ → Option ℕ} {m n : ℕ} (none : ∀ k, f k = Option.none) :
    ∀ y < n, ∀ c < 2 ^ m, f (y * 2 ^ m + c) = (f (y * 2 ^ m)).map (· + c) := by
  intro y _ c _
  rw [none, none]
  rfl

/-- `a + c < b` for multiples `a < b` of `k` and `c < k`. -/
theorem add_lt_of_dvd {k a b c : ℕ} (left : k ∣ a) (right : k ∣ b) (lt : a < b) (small : c < k) :
    a + c < b := by
  obtain ⟨a, rfl⟩ := left
  obtain ⟨b, rfl⟩ := right
  have below : a < b := Nat.lt_of_mul_lt_mul_left lt
  have step : k * (a + 1) ≤ k * b := Nat.mul_le_mul_left _ below
  rw [Nat.mul_add, mul_one] at step
  omega

end Runs

section Cells

variable {S : Shape} (O : OpeningShape S) (fold : Descriptor)

theorem Shape.responseCell_eq (col : (Fin S.blockLength × Fin S.sourceDigits) ×
    Fin fold.responseDigits) {k : ℕ} (small : k < S.innerDegree) :
    S.responseCell fold col k = some (S.responseCode fold col * S.innerDegree + k) :=
  ite_eq_left small

theorem OpeningShape.partialCell_eq (col : O.OpenColumn) {k : ℕ} (small : k < O.openDegree) :
    O.partialCell fold col k =
      some (S.responseCells fold + O.partialCode col * O.openDegree + k) := by
  unfold OpeningShape.partialCell
  rw [ite_eq_left small, Nat.add_assoc]

theorem OpeningShape.outerCell_eq (i : Fin S.blocks) (col : S.OuterColumn) {k : ℕ}
    (small : k < S.outerDegree) :
    O.outerCell fold i col k =
      some (O.innerStart fold + S.innerCode (i, col) * S.outerDegree + k) := by
  unfold OpeningShape.outerCell
  rw [ite_eq_left small, Nat.add_assoc]

/-- Coefficient `y·d_B + c` of digit `u` of row `a` of `t̂_i` is coefficient `c` of the subcolumn
`y`. -/
theorem OpeningShape.innerCell_eq (i : Fin S.blocks) (a : Fin S.innerRank)
    (u : Fin S.openingDigits) {y c : ℕ} (small : y < S.ratio) (inside : c < S.outerDegree) :
    O.innerCell fold i a u (y * S.outerDegree + c) =
      some (O.innerStart fold + S.innerCode (i, (a, ⟨y, small⟩, u)) * S.outerDegree + c) := by
  have div : (y * S.outerDegree + c) / S.outerDegree = y := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ (NeZero.pos _), Nat.div_eq_of_lt inside, zero_add]
  have mod : (y * S.outerDegree + c) % S.outerDegree = c := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt inside]
  unfold OpeningShape.innerCell
  split
  · rename_i inside'
    have same : (⟨_, inside'⟩ : Fin S.ratio) = ⟨y, small⟩ := Fin.ext div
    rw [same, mod]
    exact OpeningShape.outerCell_eq O fold i _ inside
  · rename_i outside
    exact absurd (by rw [div]; exact small) outside

/-- The coordinate `p + c` of a partial, for `p mod d_D + c < d_D`, is in the subcolumn of `p`. -/
theorem OpeningShape.partialPosition_eq (i : Fin S.blocks) (u : Fin S.openingDigits) {p c : ℕ}
    (small : p / O.openDegree < O.subcolumns) (aligned : p % O.openDegree + c < O.openDegree) :
    O.partialPosition fold i u (p + c) =
      some (S.responseCells fold + O.partialCode (i, ⟨p / O.openDegree, small⟩, u) *
        O.openDegree + (p % O.openDegree + c)) := by
  have decomp : p + c = (p % O.openDegree + c) + O.openDegree * (p / O.openDegree) := by
    have := Nat.mod_add_div p O.openDegree
    omega
  have div : (p + c) / O.openDegree = p / O.openDegree := by
    rw [decomp, Nat.add_mul_div_left _ _ (NeZero.pos _), Nat.div_eq_of_lt aligned, zero_add]
  have mod : (p + c) % O.openDegree = p % O.openDegree + c := by
    rw [decomp, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt aligned]
  unfold OpeningShape.partialPosition
  split
  · rename_i inside'
    have same : (⟨_, inside'⟩ : Fin O.subcolumns) = ⟨p / O.openDegree, small⟩ := Fin.ext div
    rw [same, mod]
    exact OpeningShape.partialCell_eq O fold _ aligned
  · rename_i outside
    exact absurd (by rw [div]; exact small) outside

theorem OpeningShape.partialPosition_none (i : Fin S.blocks) (u : Fin S.openingDigits) {p c : ℕ}
    (large : O.subcolumns ≤ p / O.openDegree) (aligned : p % O.openDegree + c < O.openDegree) :
    O.partialPosition fold i u (p + c) = none := by
  have decomp : p + c = (p % O.openDegree + c) + O.openDegree * (p / O.openDegree) := by
    have := Nat.mod_add_div p O.openDegree
    omega
  have div : (p + c) / O.openDegree = p / O.openDegree := by
    rw [decomp, Nat.add_mul_div_left _ _ (NeZero.pos _), Nat.div_eq_of_lt aligned, zero_add]
  unfold OpeningShape.partialPosition
  rw [dite_eq_right (by rw [div]; omega)]

/-- The coordinates `t·d_car + j` of a partial are runs in each chunk of `2^m` positions that
divides `d_car` and `d_D`. -/
theorem OpeningShape.partialPosition_run (i : Fin S.blocks) (u : Fin S.openingDigits) (t : ℕ)
    {m : ℕ} (carrier : 2 ^ m ∣ O.carrierDegree) (opened : 2 ^ m ∣ O.openDegree) :
    ∀ y < O.carrierDegree / 2 ^ m, ∀ c < 2 ^ m,
      O.partialPosition fold i u (t * O.carrierDegree + (y * 2 ^ m + c)) =
        (O.partialPosition fold i u (t * O.carrierDegree + y * 2 ^ m)).map (· + c) := by
  intro y _ c inside
  have aligned : (t * O.carrierDegree + y * 2 ^ m) % O.openDegree + c < O.openDegree :=
    add_lt_of_dvd ((Nat.dvd_mod_iff opened).mpr (Nat.dvd_add (Dvd.dvd.mul_left carrier _)
      (Dvd.intro_left _ rfl))) opened (Nat.mod_lt _ (NeZero.pos _)) inside
  rw [← Nat.add_assoc]
  by_cases small : (t * O.carrierDegree + y * 2 ^ m) / O.openDegree < O.subcolumns
  · have zero := OpeningShape.partialPosition_eq O fold i u small (c := 0) (by omega)
    simp only [Nat.add_zero] at zero
    rw [OpeningShape.partialPosition_eq O fold i u small aligned, zero]
    exact congrArg some (Nat.add_assoc _ _ _).symm
  · have zero := OpeningShape.partialPosition_none O fold i u (Nat.le_of_not_lt small) (c := 0) (by omega)
    simp only [Nat.add_zero] at zero
    rw [OpeningShape.partialPosition_none O fold i u (Nat.le_of_not_lt small) aligned, zero]
    rfl

end Cells

section Scans

section Values

variable {E : Type} [CommRing E] [Algebra Zq E] {d : ℕ} [NeZero d] (α : E)

theorem aeval_rep_one : Polynomial.aeval α (rep (1 : Rq d)) = 1 := by
  rw [aeval_rep, Finset.sum_eq_single 0 (fun i _ other => by rw [coeff_one, ite_eq_right other,
    zero_smul]) (fun absent => absurd (Finset.mem_range.mpr (NeZero.pos d)) absent), coeff_one,
    ite_eq_left rfl, one_smul, pow_zero]

theorem aeval_rep_smul (z : Zq) (x : Rq d) :
    Polynomial.aeval α (rep (z • x)) = algebraMap Zq E z * Polynomial.aeval α (rep x) := by
  rw [← reducedEval_apply, ← reducedEval_apply, map_smul, Algebra.smul_def]

theorem aeval_rep_mul_natCast (x : Rq d) (n : ℕ) :
    Polynomial.aeval α (rep (x * (n : Rq d))) = (n : E) * Polynomial.aeval α (rep x) := by
  rw [← reducedEval_apply, ← reducedEval_apply, mul_comm, ← nsmul_eq_mul, map_nsmul, nsmul_eq_mul]

theorem aeval_rep_natCast (n : ℕ) : Polynomial.aeval α (rep (n : Rq d)) = (n : E) := by
  rw [← one_mul (n : Rq d), aeval_rep_mul_natCast, aeval_rep_one, mul_one]

theorem aeval_rep_algebraMap (z : Zq) :
    Polynomial.aeval α (rep (algebraMap Zq (Rq d) z)) = algebraMap Zq E z := by
  rw [Algebra.algebraMap_eq_smul_one, aeval_rep_smul, aeval_rep_one, mul_one]

theorem aeval_rep_neg' (x : Rq d) :
    Polynomial.aeval α (rep (-x)) = -Polynomial.aeval α (rep x) := by
  rw [← reducedEval_apply, ← reducedEval_apply, map_neg]

end Values

variable {E : Type} [Field E] [Algebra Zq E] {S : Shape} (O : OpeningShape S) (fold : Descriptor)
  (α : E) (φ : Option ℕ → E)

/-- The linear form of plane `t` of the left side of (102). -/
theorem partialForm_scan (challenge : Fin S.blocks → Rq O.carrierDegree) (t : Fin O.planes) :
    (partialForm O fold challenge t).scan α φ =
      ∑ i : Fin S.blocks, ∑ u : Fin S.openingDigits, ((S.openingBase ^ (u : ℕ) : ℕ) : E) *
        Polynomial.aeval α (rep (challenge i)) * ∑ j ∈ Finset.range O.carrierDegree,
          α ^ j * φ (O.partialPosition fold i u (t * O.carrierDegree + j)) := by
  show ∑ c : Fin S.blocks × Fin S.openingDigits, ∑ j ∈ Finset.range O.carrierDegree,
    Polynomial.aeval α (rep (challenge c.1 * (S.openingBase : Rq O.carrierDegree) ^ (c.2 : ℕ))) *
      α ^ j * φ (O.partialPosition fold c.1 c.2 (t * O.carrierDegree + j)) = _
  rw [Fintype.sum_prod_type]
  refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun u _ => ?_
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [← Nat.cast_pow, aeval_rep_mul_natCast]
  ring

/-- The linear form of plane `t` of the right side of (102). -/
theorem contractForm_scan (β : Module.Basis (Fin O.planes) Zq E) (point : Point O E)
    (t : Fin O.planes) :
    (contractForm O fold β point t).scan α φ =
      ∑ x : Fin S.blockLength, ∑ a : Fin O.stride, ∑ u : Fin S.sourceDigits,
        ∑ f : Fin fold.responseDigits,
          algebraMap Zq E (β.repr (point.position x * point.pack a) t) *
            ((S.sourceBase ^ (u : ℕ) * fold.responseBase ^ (f : ℕ) : ℕ) : E) *
              ∑ j ∈ Finset.range O.carrierDegree,
                α ^ j * φ (S.responseCell fold ((x, u), f) (a + O.stride * j)) := by
  show ∑ c : Fin S.blockLength × Fin O.stride × Fin S.sourceDigits × Fin fold.responseDigits,
    ∑ j ∈ Finset.range O.carrierDegree,
      Polynomial.aeval α (rep (algebraMap Zq (Rq O.carrierDegree)
        (β.repr (point.position c.1 * point.pack c.2.1) t) *
          ((S.sourceBase : Rq O.carrierDegree) ^ (c.2.2.1 : ℕ) *
            (fold.responseBase : Rq O.carrierDegree) ^ (c.2.2.2 : ℕ)))) * α ^ j *
        φ (S.responseCell fold ((c.1, c.2.2.1), c.2.2.2) (c.2.1 + O.stride * j)) = _
  simp only [Fintype.sum_prod_type]
  refine Finset.sum_congr rfl fun x _ => Finset.sum_congr rfl fun a _ =>
    Finset.sum_congr rfl fun u _ => Finset.sum_congr rfl fun f _ => ?_
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [← Nat.cast_pow, ← Nat.cast_pow, ← Nat.cast_mul, aeval_rep_mul_natCast, aeval_rep_algebraMap]
  ring

/-- The linear form of row `a` of the left side of (104). -/
theorem imageForm_scan (challenge : Fin S.blocks → Rq O.carrierDegree) (a : Fin S.innerRank) :
    (imageForm O fold challenge a).scan α φ =
      ∑ i : Fin S.blocks, ∑ u : Fin S.openingDigits, ((S.openingBase ^ (u : ℕ) : ℕ) : E) *
        Polynomial.aeval α (rep (embed O.stride O.stride_tower (challenge i))) *
          ∑ j ∈ Finset.range S.innerDegree, α ^ j * φ (O.innerCell fold i a u j) := by
  show ∑ c : Fin S.blocks × Fin S.openingDigits, ∑ j ∈ Finset.range S.innerDegree,
    Polynomial.aeval α (rep (embed O.stride O.stride_tower (challenge c.1) *
      (S.openingBase : Rq S.innerDegree) ^ (c.2 : ℕ))) * α ^ j *
        φ (O.innerCell fold c.1 a c.2 j) = _
  rw [Fintype.sum_prod_type]
  refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun u _ => ?_
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [← Nat.cast_pow, aeval_rep_mul_natCast]
  ring

/-- The linear form of the recomposition `G·ξ̂` of entry `entry` of an image. -/
theorem recompose_scan (d n : ℕ) [NeZero d] (cell : ℕ → Option ℕ) (entry : Fin n) :
    (Form.recompose d n cell entry).scan α φ =
      ∑ u : Fin binaryDigits, ((2 ^ (u : ℕ) : ℕ) : E) *
        ∑ k ∈ Finset.range d, α ^ k * φ (cell (entry * d + k + n * d * u)) := by
  show ∑ u : Fin binaryDigits, ∑ k ∈ Finset.range d,
    Polynomial.aeval α (rep (((2 : ℕ) : Rq d) ^ (u : ℕ))) * α ^ k *
      φ (cell (entry * d + k + n * d * u)) = _
  refine Finset.sum_congr rfl fun u _ => ?_
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [← Nat.cast_pow, aeval_rep_natCast]
  ring

end Scans

end NightstreamFPrime.Spec.Compression.Akita.Replay
