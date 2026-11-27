import NightstreamFPrime.Spec.Compression.Akita.Replay.WeightPackingBatch

/-! The weight of the replay at a fold with opening groups (`foldParts` of
`Replay/WeightPacking.lean` for the groups of `packGroupsIn` in the order `groupOrder`).

- `batchGroups` is the list of the groups of a fold with opening groups in the order of the code
  (`PackShape.slots` of `Schedules.ofBatch`): the group `groupOrder n k` at the index `k`, with its
  fold challenges, its point and its claim coefficient.
- `before_finRange` and `zipIdx_finRange`: the sums of the replay over a list of groups are sums
  over `Fin n` with the offsets `offset` of the groups before.
- `Placement.move_batch`: the cell of the code of the position `x` of the flat witness of the
  group at the index `k` is `Gaps.place (Batch.relocate (groupOrder n k) x)`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita NightstreamFPrime.Spec.Compression.Lib

section Lists

variable {n : ℕ}

theorem take_finRange_sum (h : Fin n → ℕ) (k : ℕ) :
    (((List.finRange n).map h).take k).sum = offset h k := by
  induction n generalizing k with
  | zero => simp [offset]
  | succ n ih =>
      cases k with
      | zero => exact (offset_zero h).symm
      | succ k =>
          rw [List.finRange_succ, List.map_cons, List.take_succ_cons, List.sum_cons, List.map_map,
            ih]
          unfold offset
          rw [Fin.sum_univ_succ, ite_eq_left (by simp)]
          congr 1
          exact Finset.sum_congr rfl fun i _ => by
            simp only [Function.comp_apply, Fin.val_succ, Nat.add_lt_add_iff_right]

theorem before_finRange {α : Type} (G : Fin n → α) (f : α → ℕ) (k : ℕ) :
    before ((List.finRange n).map G) f k = offset (fun j => f (G j)) k := by
  unfold before
  rw [List.map_take, List.map_map]
  exact take_finRange_sum _ k

theorem zipIdx_finRange {α : Type} (G : Fin n → α) :
    ((List.finRange n).map G).zipIdx = (List.finRange n).map fun k => (G k, (k : ℕ)) := by
  refine List.ext_getElem (by simp) fun i small _ => ?_
  simp [List.getElem_zipIdx]

theorem getElem?_finRange_map {α : Type} (G : Fin n → α) (k : Fin n) :
    ((List.finRange n).map G)[(k : ℕ)]? = some (G k) := by
  simp

theorem sum_finRange_order (f : Fin n → ℕ) :
    ((List.finRange n).map fun k => f (groupOrder n k)).sum = ∑ g, f g := by
  rw [sum_finRange]
  exact Equiv.sum_comp (groupOrder n) f

theorem offset_order (f : Fin n → ℕ) (k : Fin n) :
    offset (fun j => f (groupOrder n j)) k = codeOffset f (groupOrder n k) := by
  unfold codeOffset
  rw [Equiv.symm_apply_apply]

end Lists

section Groups

variable {n : ℕ} (B : Batch (Fin n)) (cs amb : Fin n → List (Array Fq)) (pts : Fin n → List Ext)
  (w : Fin n → Ext)

/-- The group at the index `k` of a fold with opening groups: the group `groupOrder n k`. -/
def batchGroup (k : Fin n) : PackGroup :=
  ⟨B.shape (groupOrder n k), B.opening (groupOrder n k), B.fold (groupOrder n k), cs k, amb k, pts k,
    w k⟩

/-- The groups of a fold with opening groups in the order of the code. -/
def batchGroups : List PackGroup := (List.finRange n).map (batchGroup B cs amb pts w)

theorem batchGroups_length : (batchGroups B cs amb pts w).length = n := by
  simp [batchGroups]

theorem batchGroups_sum (f : PackGroup → ℕ) (F : Fin n → ℕ)
    (same : ∀ k, f (batchGroup B cs amb pts w k) = F (groupOrder n k)) :
    ((batchGroups B cs amb pts w).map f).sum = ∑ g, F g := by
  unfold batchGroups
  rw [List.map_map, ← sum_finRange_order]
  exact congrArg List.sum (List.map_congr_left fun k _ => same k)

theorem batchGroups_before (f : PackGroup → ℕ) (F : Fin n → ℕ)
    (same : ∀ k, f (batchGroup B cs amb pts w k) = F (groupOrder n k)) (k : Fin n) :
    before (batchGroups B cs amb pts w) f k = codeOffset F (groupOrder n k) := by
  unfold batchGroups
  rw [before_finRange, ← offset_order]
  exact congrArg (offset · k) (funext same)

variable (digits : ℕ)

/-- The cells of the code of the group at the index `k` of a fold with opening groups and the
shared path of `D`: the positions `Gaps.place` of `Batch.relocate` of the group `groupOrder n k`
(Akita (76) with the groups in the order of the code). -/
theorem Placement.move_batch (k : Fin n) :
    (Placement.of (batchGroups B cs amb pts w) ⟨B.openDegree, B.openRank, B.chain⟩ digits).move
        (batchGroups B cs amb pts w) k {} =
      fun x => (B.gaps digits).place (B.relocate (groupOrder n k) x) := by
  funext x
  have main := batchGroups_before B cs amb pts w (·.mainCells) B.mainCells (fun _ => rfl) k
  have first := batchGroups_before B cs amb pts w (fun Q => firstCells Q.shape.chain)
    (fun g => firstCells (B.shape g).chain) (fun _ => rfl) k
  have second := batchGroups_before B cs amb pts w (fun Q => secondCells Q.shape.chain)
    (fun g => secondCells (B.shape g).chain) (fun _ => rfl) k
  have total := batchGroups_sum B cs amb pts w (·.mainCells) B.mainCells (fun _ => rfl)
  have degrees := batchGroups_sum B cs amb pts w (·.mainDegrees) B.groupDegrees (fun _ => rfl)
  have firsts := batchGroups_sum B cs amb pts w (fun Q => firstCells Q.shape.chain)
    (fun g => firstCells (B.shape g).chain) (fun _ => rfl)
  have firstRows := batchGroups_sum B cs amb pts w
    (fun Q => firstRank Q.shape.chain * firstDegree Q.shape.chain)
    (fun g => firstRank (B.shape g).chain * firstDegree (B.shape g).chain) (fun _ => rfl)
  have seconds := batchGroups_sum B cs amb pts w (fun Q => secondCells Q.shape.chain)
    (fun g => secondCells (B.shape g).chain) (fun _ => rfl)
  have bounds := B.bounds (groupOrder n k)
  have starts := B.starts (groupOrder n k)
  unfold Placement.move
  rw [show (batchGroups B cs amb pts w)[(k : ℕ)]? = some (batchGroup B cs amb pts w k) from
    getElem?_finRange_map _ k]
  simp only [beforeIn, main, first, second, Placement.of, total, degrees, firsts, firstRows,
    seconds]
  unfold batchGroup
  simp only
  unfold Batch.relocate Gaps.place Batch.gaps Batch.mainStart Batch.firstStart Batch.secondStartOf
  simp only
  unfold Batch.mainDegrees Batch.firstDegrees at *
  unfold Batch.cells Batch.openSecondStart Batch.secondStart Batch.openFirstStart Batch.mainCells
    Batch.mainStart Batch.firstStart Batch.secondStartOf at bounds
  rw [one_mul]
  have c0 : B.chainStart = ∑ g, B.mainCells g := rfl
  have c1 : B.openFirstStart = B.chainStart + ∑ g, firstCells (B.shape g).chain := rfl
  have c2 : B.secondStart = B.openFirstStart + firstCells B.chain := rfl
  have c3 : B.openSecondStart = B.secondStart + ∑ g, secondCells (B.shape g).chain := rfl
  split_ifs <;> omega

end Groups

section Values

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E] (F : FoldInputs) (move : ℕ → ℕ)
  (P : PackGroup) (α : Ext) (r2 : List Ext) (bits top : ℕ)

/-- The columns of a group in the rows (106) of `D`, from the column `column` of the width
`width`: for each column of the group and each row `r` of `D`, `ẽq(τ_1, dRow + r)` times the
linear form of the column at the cells of `move`. -/
theorem openLifts_columns (dRow column width : ℕ) (short : bits ≤ r2.length)
    (openPow : 2 ^ Nat.log 2 P.opening.openDegree = P.opening.openDegree)
    (le : Nat.log 2 P.opening.openDegree ≤ bits) (degree : P.opening.openDegree ≤ top)
    (shift : ∀ x c, x + c < P.opening.chainStart P.fold → move (x + c) = move x + c)
    (bounded : ∀ x < P.opening.cells P.fold, move x < 2 ^ r2.length) :
    ((openLifts F move P dRow column width).map
      fun l => ((l.value (Tables.of α r2 bits top)).into : E)).sum =
      ∑ c : P.opening.OpenColumn, ∑ r : Fin P.opening.openRank,
        ((eqIndex F.tau1 (dRow + r)).into : E) * ∑ j ∈ Finset.range P.opening.openDegree,
          Polynomial.aeval (α.into : E) (rep (Poly.toRq P.opening.openDegree
            (viewEntry F.setup P.opening.openDegree width r
              (column + (P.opening.openSetupCode c).val)))) * (α.into : E) ^ j *
            ((P.opening.partialCell P.fold c j).map move).elim 0 (eqAt (r2.map Ext.into)) := by
  unfold openLifts
  simp only [sum_flatMap, List.map_map, Function.comp_def, sum_finRange]
  rw [Fintype.sum_prod_type]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [Fintype.sum_prod_type]
  refine Finset.sum_congr rfl fun y _ => Finset.sum_congr rfl fun u _ => ?_
  rw [Lift.ofTerm_value_into α r2 bits top _ short openPow le degree
    (fun rd member => by
      rw [List.mem_singleton] at member
      subst member
      exact Runs.main_of shift bounded
        (run_of_start (s := P.shape.responseCells P.fold + P.opening.partialCode (i, y, u) *
          P.opening.openDegree) (d := P.opening.openDegree)
          (fun k small => OpeningShape.partialCell_eq P.opening P.fold _ small)
          (by rw [one_mul, openPow]))
        (fun j x read => partialCell_lt_chainStart P.opening P.fold _ read))]
  rw [reads_single, List.map_map, sum_finRange, Finset.mul_sum]
  refine Finset.sum_congr rfl fun r _ => ?_
  simp only [Function.comp_apply, moved]
  rw [Finset.sum_mul, Finset.mul_sum]
  refine Finset.sum_congr rfl fun k _ => ?_
  ring

/-- The scalar row (100) of a group at `r_2` for a map `move` that is a shift on the main
segments: the claim coefficient times `Σ_y ẽq(r_2, move y)·scalarWeight(y)`. -/
theorem scalarPart_moved (short : bits ≤ r2.length)
    (carrierPow : 2 ^ Nat.log 2 P.opening.carrierDegree = P.opening.carrierDegree)
    (openPow : 2 ^ Nat.log 2 P.opening.openDegree = P.opening.openDegree)
    (le : Nat.log 2 P.opening.carrierDegree ≤ bits)
    (shift : ∀ x c, x + c < P.opening.chainStart P.fold → move (x + c) = move x + c)
    (bounded : ∀ x < P.opening.cells P.fold, move x < 2 ^ r2.length)
    (fits : P.opening.cells P.fold ≤ 2 ^ r2.length)
    (β : Module.Basis (Fin P.opening.planes) Zq E)
    (based : ∀ t, β t = (extBasis.getD t Ext.zero).into) (point : Point P.opening E)
    (blocked : ∀ i, point.block i = eqAt (P.segments.2.2.2.map Ext.into) i)
    (tailed : ∀ j, point.tail j = eqAt (P.segments.2.1.map Ext.into) j) :
    ((scalarPart move P (Tables.of α r2 bits top)).into : E) = (P.claim.into : E) *
      ∑ y ∈ Finset.range (2 ^ r2.length), eqAt (r2.map Ext.into) (move y) *
        scalarWeight P.opening P.fold β point y := by
  have divides : 2 ^ min (Nat.log 2 P.opening.carrierDegree) (Nat.log 2 P.opening.openDegree) ∣
      P.opening.carrierDegree := (Nat.pow_dvd_pow 2 (min_le_left _ _)).trans (dvd_of_eq carrierPow)
  have opened : 2 ^ min (Nat.log 2 P.opening.carrierDegree) (Nat.log 2 P.opening.openDegree) ∣
      P.opening.openDegree := (Nat.pow_dvd_pow 2 (min_le_right _ _)).trans (dvd_of_eq openPow)
  have total : P.opening.carrierDegree /
      2 ^ min (Nat.log 2 P.opening.carrierDegree) (Nat.log 2 P.opening.openDegree) *
        2 ^ min (Nat.log 2 P.opening.carrierDegree) (Nat.log 2 P.opening.openDegree) =
      P.opening.carrierDegree := Nat.div_mul_cancel divides
  unfold scalarWeight
  rw [sum_mul_indicator _ _ _ (fun y => eqAt (r2.map Ext.into) (move y)) fun entry x read =>
    (P.opening.partialPosition_lt P.fold _ _ _ read).trans_le fits]
  simp only [Fintype.sum_prod_type]
  unfold scalarPart
  simp only [Ext.into_mul]
  congr 1
  simp only [Ext.into_mul, sumList_into, sum_finRange, eqIndex_into, sum_range_list, Ext.into_scale,
    natFq, Fq.toZq_ofNat, map_natCast, ← blocked]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [Finset.mul_sum, Finset.sum_comm]
  refine Finset.sum_congr rfl fun t _ => ?_
  rw [show (Tables.of α r2 bits top).lowsOf
      (min (Nat.log 2 P.opening.carrierDegree) (Nat.log 2 P.opening.openDegree))
      P.opening.carrierDegree (fun j => eqIndex P.segments.2.1 j) =
    lowSums r2 (min (Nat.log 2 P.opening.carrierDegree) (Nat.log 2 P.opening.openDegree))
      (Array.ofFn (n := P.opening.carrierDegree) fun k => eqIndex P.segments.2.1 k.val) from rfl]
  have below : ∀ (u : Fin P.shape.openingDigits) j x,
      P.opening.partialPosition P.fold i u (t * P.opening.carrierDegree + j) = some x →
        x < P.opening.chainStart P.fold := fun u j x read =>
    (P.opening.partialPosition_lt_innerStart P.fold i u _ read).trans_le (Nat.le_add_right _ _)
  have keep : ∀ (u : Fin P.shape.openingDigits) j,
      ((P.opening.partialPosition P.fold i u (t * P.opening.carrierDegree + j)).map move).elim 0
          (eqAt (r2.map (Ext.into (E := E)))) =
        (P.opening.partialPosition P.fold i u (t * P.opening.carrierDegree + j)).elim 0
          (fun y => eqAt (r2.map Ext.into) (move y)) := fun u j => by
    cases P.opening.partialPosition P.fold i u (t * P.opening.carrierDegree + j) <;> rfl
  conv_rhs => rw [Finset.sum_comm]
  rw [Finset.mul_sum, Finset.mul_sum]
  refine Finset.sum_congr rfl fun u _ => ?_
  rw [Tables.read_into α r2 bits top _ ((min_le_left _ _).trans le) short le_rfl
    (by rw [total, Array.size_ofFn])
    (Runs.moved_of (lo := 0) (hi := P.opening.chainStart P.fold) (fun x c _ high => shift x c high)
      (OpeningShape.partialPosition_run P.opening P.fold i u t divides opened)
      (fun j x read => ⟨Nat.zero_le _, below u j x read⟩)
      (fun j x read => bounded x ((below u j x read).trans_le (P.opening.chainStart_le_cells P.fold)))),
    total, Finset.mul_sum, Finset.mul_sum, Finset.mul_sum, Finset.sum_range]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [ofFn_getD (fun k => eqIndex P.segments.2.1 k) j.isLt, eqIndex_into, ← tailed, ← based]
  simp only [moved]
  rw [keep, Algebra.smul_def, map_pow, map_natCast, Nat.cast_pow]
  ring

end Values

section Scalar

variable {n : ℕ} (B : Batch (Fin n)) {E : Type} [Field E] [Algebra Zq E]

/-- The scalar row of a fold with opening groups at `r_2`: for each group `ϑ_g` times
`Σ_y ẽq(r_2, relocate_g y)·scalarWeight_g(y)`. -/
theorem mle_batchScalar (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E) (st : B.Statement E)
    {μ : ℕ} (fits : B.cells ≤ 2 ^ μ) {r : List E} (length : r.length = μ) :
    mle r (B.scalarWeight β st μ) = ∑ g, st.weight g * ∑ y ∈ Finset.range (2 ^ μ),
      eqAt r (B.relocate g y) * Akita.scalarWeight (B.opening g) (B.fold g) (β g) (st.point g) y := by
  rw [mle_eq_cubeSum, cubeSum_eq_sum_range, length]
  unfold Batch.scalarWeight
  simp only [Finset.mul_sum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun g _ => ?_
  have moved := sum_mul_relocate (2 ^ μ) (B.relocate g)
    (Akita.scalarWeight (B.opening g) (B.fold g) (β g) (st.point g)) (eqAt r) fun y _ nonzero => by
      by_contra outside
      refine nonzero (Akita.scalarWeight_of_le (B.opening g) (B.fold g) (β g) (st.point g) ?_)
      by_contra small
      have below := B.relocate_lt_chainStart g (Nat.lt_of_not_le small)
      have ends := B.chainStart_le_cells
      omega
  rw [← Finset.mul_sum, ← moved]
  simp only [Finset.mul_sum]
  refine Finset.sum_congr rfl fun x _ => Finset.sum_congr rfl fun y _ => ?_
  ring

end Scalar

section Starts

variable {M : Type} [AddCommMonoid M]

theorem offset_succ_val {n : ℕ} (h : Fin (n + 1) → ℕ) (k : Fin n) :
    offset h (k.succ : ℕ) = h 0 + offset (fun j => h j.succ) k := by
  unfold offset
  rw [Fin.sum_univ_succ, ite_eq_left (by simp)]
  congr 1
  exact Finset.sum_congr rfl fun i _ => by simp only [Fin.val_succ, Nat.add_lt_add_iff_right]

/-- The rows of `rowStarts` for the rows of the groups and a last row family, as sums over the
groups. -/
theorem rowStarts_groups (f : (ℕ × ℕ) × ℕ → M) (digits : ℕ) : ∀ {n : ℕ} (rows : Fin n → ℕ × ℕ)
    (last : ℕ × ℕ) (s c : ℕ),
    (((rowStarts digits s ((List.finRange n).map rows ++ [last])).zipIdx c).map f).sum =
      ∑ k, ∑ x : Fin (rows k).2, f (((rows k).1, s + digits * offset (fun j => (rows j).2 * (rows j).1)
        k + x * (digits * (rows k).1)), c + offset (fun j => (rows j).2) k + x) +
      ∑ x : Fin last.2, f ((last.1, s + digits * ∑ j, (rows j).2 * (rows j).1 +
        x * (digits * last.1)), c + ∑ j, (rows j).2 + x)
  | 0, rows, last, s, c => by
      simp only [List.finRange_zero, List.map_nil, List.nil_append, Finset.univ_eq_empty,
        Finset.sum_empty, zero_add, mul_zero, add_zero]
      rw [rowStarts_sum, rowStarts_nil, add_zero]
  | n + 1, rows, last, s, c => by
      rw [List.finRange_succ, List.map_cons, List.cons_append, rowStarts_sum, List.map_map,
        rowStarts_groups f digits (rows ∘ Fin.succ) last _ _, Fin.sum_univ_succ]
      refine (congrArg₂ (· + ·) ?_ (congrArg₂ (· + ·) ?_ ?_)).trans (add_assoc _ _ _).symm
      · refine Finset.sum_congr rfl fun x _ => congrArg f ?_
        rw [Fin.val_zero, offset_zero, offset_zero, mul_zero, add_zero, add_zero]
      · refine Finset.sum_congr rfl fun k _ => Finset.sum_congr rfl fun x _ => congrArg f ?_
        rw [offset_succ_val, offset_succ_val]
        exact Prod.ext (Prod.ext rfl (by simp only [Function.comp_apply]; ring))
          (by simp only [Function.comp_apply]; ring)
      · refine Finset.sum_congr rfl fun x _ => congrArg f ?_
        rw [Fin.sum_univ_succ (fun j => (rows j).2 * (rows j).1), Fin.sum_univ_succ (fun j => (rows j).2)]
        exact Prod.ext (Prod.ext rfl (by simp only [Function.comp_apply]; ring))
          (by simp only [Function.comp_apply]; ring)

end Starts

end NightstreamFPrime.Spec.Compression.Akita.Replay
