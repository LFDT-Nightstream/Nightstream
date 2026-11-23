import NightstreamFPrime.Spec.Compression.Akita.Replay.Relation
import NightstreamFPrime.Spec.Compression.Akita.Extract.LiftSource
import NightstreamFPrime.Spec.Compression.Akita.Extract.TraceSource

/-! The public rows of the relation of one fold of the pinned Akita verifier, and the targets of
the reductions of the claims at these rows (Akita revision `d50f700`: `RelationRhsLayout` and
`build_relation_rhs_layout` of `akita-params/src/layout/relation_rhs_layout.rs`,
`assemble_relation_rhs` and `assemble_compressed_relation_rhs` of `akita-types`; ePrint
2026/1983, Figure 5 and Section 7.3).

- `relation` has the runs of the public rows of `y` of one fold, with the groups `shape k` in the
  order of the rows, and the index of the opening row: each group has its consistency row, the
  rows of `A` and the rows of `B` of each slice, then `D` has its rows, then each compression map
  has the rows of each group and of `H`. The public rows are those of `B` and `D` without
  compression maps and those of `F_2` and `H_2` with them (`groupRun`, `openRun`).
- `place_symm_val` is the index of a row of one family with the groups in the order `groupOrder`.
- `spread` is a table with the values of a run of rows, and `joinTargets` the targets of the
  field rows of a join.
- `traceRows_targetAt`, `packingRows_targetAt` and `batchRows_targetAt`: the target of each field
  row of `traceRelation`, `flatRelation` and `Batch.flatRelation` is the value that
  `rhsEntries` lists at this row, when the payloads of the statement are the words of the runs
  (`payloadOf`), and the scalar row is the opening row with the value of the statement.
- `trace_fused`, `packing_fused` and `batch_fused`: for `traceReduction`, `liftedReduction` and
  `Batch.liftedReduction`, the reductions of the claims at a trace level, a packing level and a
  level with opening groups, the fused claim of `stage2` is `V_{α,τ_1}` of the field rows plus
  `rest` (Akita (123), `FieldRows.batchTarget`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita NightstreamFPrime.Spec.Compression.Lib

section Runs

/-- `Σ_{j<k} f j`. -/
def rowsBefore {n : ℕ} (f : Fin n → ℕ) (k : Fin n) : ℕ := ∑ j : Fin k, f (Fin.castLE k.2.le j)

/-- The public rows of a path: without compression maps the `count` image rows of degree
`degree` from the row `image`, and with them the rows of the second map from the row `second`. -/
def pathRun (chain : Option ChainShape) (image second degree count : ℕ) (source : RhsSource) :
    RhsRun :=
  match chain with
  | none => ⟨image, degree, count, source⟩
  | some c => ⟨second, c.secondDegree, c.secondRank, source⟩

/-- The rows of one group before the rows of `D`: its consistency row, the rows of `A` and the
rows of `B` of each slice. -/
def groupRows (S : Shape) : ℕ := 1 + (S.innerRank + S.slices * S.outerRank)

variable {n : ℕ} (shape : Fin n → Shape) (openRank openDegree : ℕ)
  (openChain : Option ChainShape)

def mainRows : ℕ := ∑ k, groupRows (shape k)

/-- The first row of the second compression maps. -/
def secondStart : ℕ :=
  mainRows shape + 1 * openRank + ∑ k, firstRank (shape k).chain + firstRank openChain

/-- The count of rows of `y`: the index of the opening row. -/
def rowTotal : ℕ :=
  secondStart shape openRank openChain + ∑ k, secondRank (shape k).chain + secondRank openChain

def groupRun (source : Fin n → RhsSource) (k : Fin n) : RhsRun :=
  pathRun (shape k).chain (rowsBefore (fun j => groupRows (shape j)) k + 1 + (shape k).innerRank)
    (secondStart shape openRank openChain + rowsBefore (fun j => secondRank (shape j).chain) k)
    (shape k).outerDegree ((shape k).slices * (shape k).outerRank) (source k)

def openRun : RhsRun :=
  pathRun openChain (mainRows shape)
    (secondStart shape openRank openChain + ∑ k, secondRank (shape k).chain) openDegree
    (1 * openRank) .payload

/-- The runs of the public rows of `y` of one fold and the index of the opening row
(`RelationRhsLayout::row_families`, `scalar_opening_row_index`, `evaluation_trace_row_index`). -/
def relation (source : Fin n → RhsSource) : List RhsRun × ℕ :=
  (List.ofFn (groupRun shape openRank openChain source) ++
      [openRun shape openRank openDegree openChain],
    rowTotal shape openRank openChain)

theorem rowsBefore_add_le (f : Fin n → ℕ) (k : Fin n) : rowsBefore f k + f k ≤ ∑ j, f j := by
  let g : ℕ → ℕ := fun j => if h : j < n then f ⟨j, h⟩ else 0
  have head : rowsBefore f k = ∑ j ∈ Finset.range k, g j := by
    rw [rowsBefore, ← Fin.sum_univ_eq_sum_range]
    exact Finset.sum_congr rfl fun j _ => by
      simp only [g]
      rw [dite_eq_left (j.2.trans k.2)]
      rfl
  have total : ∑ j, f j = ∑ j ∈ Finset.range n, g j := by
    rw [← Fin.sum_univ_eq_sum_range]
    exact Finset.sum_congr rfl fun j _ => by
      simp only [g]
      rw [dite_eq_left j.2]
  have last : f k = g k := by
    simp only [g]
    rw [dite_eq_left k.2]
  rw [head, total, last, ← Finset.sum_range_succ]
  exact Finset.sum_le_sum_of_subset (Finset.range_subset_range.mpr k.2)

/-- Each run of public rows ends at or before the opening row. -/
theorem relation_below (source : Fin n → RhsSource) :
    ∀ run ∈ (relation shape openRank openDegree openChain source).1,
      run.row + run.count ≤ (relation shape openRank openDegree openChain source).2 := by
  have start : mainRows shape + 1 * openRank ≤ secondStart shape openRank openChain := by
    unfold secondStart
    omega
  have ends : secondStart shape openRank openChain + ∑ k, secondRank (shape k).chain ≤
      rowTotal shape openRank openChain := Nat.le_add_right _ _
  intro run member
  simp only [relation, List.mem_append, List.mem_ofFn, List.mem_singleton] at member
  rcases member with ⟨k, rfl⟩ | rfl
  · have group :
        rowsBefore (fun j => groupRows (shape j)) k + groupRows (shape k) ≤ mainRows shape :=
      rowsBefore_add_le _ k
    have second : rowsBefore (fun j => secondRank (shape j).chain) k + secondRank (shape k).chain ≤
        ∑ j, secondRank (shape j).chain := rowsBefore_add_le _ k
    have rows : groupRows (shape k) =
        1 + ((shape k).innerRank + (shape k).slices * (shape k).outerRank) := rfl
    simp only [relation, groupRun, pathRun]
    cases chain : (shape k).chain with
    | none =>
        dsimp only
        omega
    | some c =>
        rw [chain] at second
        exact (Nat.add_assoc _ _ _).trans_le ((Nat.add_le_add_left second _).trans ends)
  · simp only [relation, openRun, pathRun]
    cases openChain with
    | none =>
        dsimp only
        omega
    | some c => exact Nat.le_of_eq rfl

end Runs

section Spread

variable {E : Type} [AddCommMonoid E]

/-- The table with `f r` at the row `start + r` for each `r < count`. -/
def spread (start count : ℕ) (f : ℕ → E) (i : ℕ) : E :=
  ∑ r ∈ Finset.range count, if start + r = i then f r else 0

/-- The entries of a vector as a table. -/
def tableOf {m : ℕ} (v : Fin m → E) (r : ℕ) : E := if h : r < m then v ⟨r, h⟩ else 0

theorem spread_congr {s m : ℕ} {f g : ℕ → E} (same : ∀ r < m, f r = g r) (i : ℕ) :
    spread s m f i = spread s m g i :=
  Finset.sum_congr rfl fun r member => by rw [same r (Finset.mem_range.mp member)]

theorem spread_eq_zero {s m : ℕ} {f : ℕ → E} (zero : ∀ r < m, f r = 0) (i : ℕ) :
    spread s m f i = 0 :=
  Finset.sum_eq_zero fun r member => by rw [zero r (Finset.mem_range.mp member), ite_self]

theorem tableOf_eq_spread {m : ℕ} (v : Fin m → E) (i : ℕ) :
    tableOf v i = spread 0 m (tableOf v) i := by
  unfold spread
  simp only [zero_add, Finset.sum_ite_eq', Finset.mem_range]
  by_cases small : i < m
  · rw [ite_eq_left small]
  · rw [ite_eq_right small, tableOf, dite_eq_right small]

theorem spread_fin {m : ℕ} (s : ℕ) (v : Fin m → E) (i : ℕ) :
    spread s m (tableOf v) i = ∑ r : Fin m, if s + r = i then v r else 0 := by
  unfold spread
  rw [← Fin.sum_univ_eq_sum_range (fun r => if s + r = i then tableOf v r else 0)]
  exact Finset.sum_congr rfl fun r _ => by simp only [tableOf, dite_eq_left r.2]

theorem tableOf_append_left {m k : ℕ} (a : Fin m → E) (b : Fin k → E) {r : ℕ} (small : r < m) :
    tableOf (Fin.append a b) r = tableOf a r := by
  rw [tableOf, tableOf, dite_eq_left small, dite_eq_left (Nat.lt_add_right k small)]
  exact Fin.append_left a b ⟨r, small⟩

theorem tableOf_append_right {m k : ℕ} (a : Fin m → E) (b : Fin k → E) (r : ℕ) :
    tableOf (Fin.append a b) (m + r) = tableOf b r := by
  unfold tableOf
  by_cases small : r < k
  · rw [dite_eq_left small, dite_eq_left (Nat.add_lt_add_left small m)]
    exact Fin.append_right a b ⟨r, small⟩
  · rw [dite_eq_right small, dite_eq_right (by omega)]

theorem spread_append {m k : ℕ} (s : ℕ) (a : Fin m → E) (b : Fin k → E) (i : ℕ) :
    spread s (m + k) (tableOf (Fin.append a b)) i =
      spread s m (tableOf a) i + spread (s + m) k (tableOf b) i := by
  unfold spread
  rw [Finset.sum_range_add]
  congr 1
  · exact Finset.sum_congr rfl fun r member => by
      rw [tableOf_append_left a b (Finset.mem_range.mp member)]
  · exact Finset.sum_congr rfl fun r _ => by rw [tableOf_append_right, Nat.add_assoc]

theorem tableOf_lastCases_lt {m : ℕ} (x : E) (w : Fin m → E) {r : ℕ} (small : r < m) :
    tableOf (Fin.lastCases x w : Fin (m + 1) → E) r = tableOf w r := by
  rw [tableOf, tableOf, dite_eq_left small, dite_eq_left (Nat.lt_succ_of_lt small)]
  exact Fin.lastCases_castSucc (motive := fun _ => E) ⟨r, small⟩

theorem tableOf_lastCases_last {m : ℕ} (x : E) (w : Fin m → E) :
    tableOf (Fin.lastCases x w : Fin (m + 1) → E) m = x := by
  rw [tableOf, dite_eq_left (Nat.lt_succ_self m)]
  exact Fin.lastCases_last (motive := fun _ => E)

theorem spread_lastCases {m : ℕ} (s : ℕ) (x : E) (w : Fin m → E) (i : ℕ) :
    spread s (m + 1) (tableOf (Fin.lastCases x w : Fin (m + 1) → E)) i =
      spread s m (tableOf w) i + if s + m = i then x else 0 := by
  unfold spread
  rw [Finset.sum_range_succ, tableOf_lastCases_last]
  congr 1
  exact Finset.sum_congr rfl fun r member => by
    rw [tableOf_lastCases_lt x w (Finset.mem_range.mp member)]

/-- The index of the row `i` of the group `groupOrder n k` in a family whose groups are in the
order `groupOrder`: the rows of the groups before it, then `i`. -/
theorem place_symm_val {n : ℕ} (count : Fin n → ℕ) (k : Fin n)
    (i : Fin (count (groupOrder n k))) :
    ((place count).symm ⟨groupOrder n k, i⟩ : ℕ) =
      rowsBefore (fun j => count (groupOrder n j)) k + i := by
  have moved : (place count).symm ⟨groupOrder n k, i⟩ =
      finCongr (Equiv.sum_comp (groupOrder n) count)
        (finSigmaFinEquiv (n := fun k => count (groupOrder n k)) ⟨k, i⟩) := by
    rw [Equiv.symm_apply_eq]
    simp [place]
  rw [moved, finCongr_apply, Fin.val_cast, finSigmaFinEquiv_apply]
  rfl

theorem spread_gather {n : ℕ} (count : Fin n → ℕ) (s : ℕ) (F : (Σ g : Fin n, Fin (count g)) → E)
    (i : ℕ) :
    spread s (∑ g, count g) (tableOf fun r => F (place count r)) i =
      ∑ k, spread (s + rowsBefore (fun j => count (groupOrder n j)) k) (count (groupOrder n k))
        (tableOf fun j => F ⟨groupOrder n k, j⟩) i := by
  simp only [spread_fin]
  refine (Fintype.sum_equiv (place count) _
    (fun x => if s + ((place count).symm x : ℕ) = i then F x else 0) fun r => by
      simp only [Equiv.symm_apply_apply]).trans ?_
  rw [Fintype.sum_sigma]
  refine (Fintype.sum_equiv (groupOrder n)
    (fun k => ∑ j : Fin (count (groupOrder n k)),
      if s + ((place count).symm ⟨groupOrder n k, j⟩ : ℕ) = i then F ⟨groupOrder n k, j⟩ else 0)
    _ fun k => rfl).symm.trans ?_
  exact Finset.sum_congr rfl fun k _ => Finset.sum_congr rfl fun j _ => by
    rw [place_symm_val, Nat.add_assoc]

theorem list_sum_range (m : ℕ) (h : ℕ → E) :
    ((List.range m).map h).sum = ∑ r ∈ Finset.range m, h r := by
  induction m with
  | zero => simp
  | succ m ih => rw [List.range_succ, List.map_append, List.sum_append, ih, Finset.sum_range_succ,
      List.map_singleton, List.sum_singleton]

end Spread

section Targets

variable {E : Type} [CommRing E]

/-- The target of each field row of a join, from the targets `t` of its ring rows. -/
def joinTargets {ring : ℕ} (J : Join E ring) (t : Fin ring → E) : Fin J.count → E :=
  fun k => ∑ r, if J.slot r = k then J.basis r * t r else 0

theorem plain_targets (m : ℕ) (t : Fin m → E) : joinTargets (Join.plain E m) t = t :=
  funext fun k => by
    simp only [joinTargets, Join.plain, id, one_mul]
    exact (Finset.sum_ite_eq' Finset.univ k t).trans (ite_eq_left (Finset.mem_univ _))

theorem all_targets {ring : ℕ} (β : Fin ring → E) (t : Fin ring → E) :
    joinTargets (Join.all β) t = fun _ => ∑ r, β r * t r :=
  funext fun _ => Finset.sum_congr rfl fun _ _ => ite_eq_left (Subsingleton.elim (α := Fin 1) _ _)

theorem append_targets {m k : ℕ} (a : Join E m) (b : Join E k) (ta : Fin m → E)
    (tb : Fin k → E) :
    joinTargets (a.append b) (Fin.append ta tb) =
      Fin.append (joinTargets a ta) (joinTargets b tb) := by
  have apart : ∀ (x : Fin a.count) (y : Fin b.count),
      Fin.natAdd a.count y ≠ Fin.castAdd b.count x := fun x y same => by
    have values := congrArg Fin.val same
    rw [Fin.val_natAdd, Fin.val_castAdd] at values
    omega
  funext q
  refine Fin.addCases (fun x => ?_) (fun y => ?_) q
  · rw [Fin.append_left]
    unfold joinTargets Join.append
    rw [Fin.sum_univ_add]
    simp only [Fin.addCases_left, Fin.addCases_right, Fin.append_left, Fin.append_right,
      Fin.castAdd_inj, apart, ite_false, Finset.sum_const_zero, add_zero]
  · rw [Fin.append_right]
    unfold joinTargets Join.append
    rw [Fin.sum_univ_add]
    simp only [Fin.addCases_left, Fin.addCases_right, Fin.append_left, Fin.append_right,
      Fin.natAdd_inj, fun x y => (apart x y).symm, ite_false, Finset.sum_const_zero, zero_add]

theorem gather_targets {n : ℕ} {size : Fin n → ℕ} (f : ∀ g, Join E (size g))
    (t : (Σ g, Fin (size g)) → E) (k : Fin (∑ g, (f g).count)) :
    joinTargets (Join.gather f) (fun r => t (place size r)) k =
      joinTargets (f (place (fun g => (f g).count) k).1)
        (fun i => t ⟨(place (fun g => (f g).count) k).1, i⟩)
        (place (fun g => (f g).count) k).2 := by
  obtain ⟨⟨h, k⟩, rfl⟩ := (place fun g => (f g).count).symm.surjective k
  rw [Equiv.apply_symm_apply]
  unfold joinTargets Join.gather
  refine (Fintype.sum_equiv (place size) _ (fun x =>
    if (place fun g => (f g).count).symm ⟨x.1, (f x.1).slot x.2⟩ =
        (place fun g => (f g).count).symm ⟨h, k⟩ then (f x.1).basis x.2 * t x else 0)
    fun r => rfl).trans ?_
  rw [Fintype.sum_sigma]
  refine (Finset.sum_eq_single h (fun g _ other => Finset.sum_eq_zero fun i _ => ?_)
    fun absent => absurd (Finset.mem_univ h) absent).trans ?_
  · exact ite_eq_right fun same => other (Sigma.mk.inj_iff.mp ((place _).symm.injective same)).1
  · refine Finset.sum_congr rfl fun i _ => ?_
    exact if_congr ⟨fun same =>
      eq_of_heq (Sigma.mk.inj_iff.mp ((place _).symm.injective same)).2,
        fun same => by rw [same]⟩ rfl rfl

variable [Algebra Zq E]

/-- The values at `α` of the targets of ring rows. -/
noncomputable def targetsOf (α : E) {m : ℕ} (rows : Fin m → RingRow) : Fin m → E :=
  fun r => reducedEval α (rows r).target

theorem targetsOf_append (α : E) {m k : ℕ} (a : Fin m → RingRow) (b : Fin k → RingRow) :
    targetsOf α (Fin.append a b) = Fin.append (targetsOf α a) (targetsOf α b) := by
  funext r
  refine Fin.addCases (fun x => ?_) (fun y => ?_) r
  · rw [Fin.append_left]
    exact congrArg (fun row : RingRow => reducedEval α row.target) (Fin.append_left a b x)
  · rw [Fin.append_right]
    exact congrArg (fun row : RingRow => reducedEval α row.target) (Fin.append_right a b y)

theorem gather_rows_targets {n : ℕ} {size : Fin n → ℕ} (f : ∀ g, Join E (size g))
    (rows : (g : Fin n) → Fin (size g) → RingRow) (α : E) :
    joinTargets (Join.gather f) (targetsOf α (Akita.gather rows)) = fun k =>
      joinTargets (f (place (fun g => (f g).count) k).1)
        (targetsOf α (rows (place (fun g => (f g).count) k).1))
        (place (fun g => (f g).count) k).2 :=
  funext (gather_targets f fun x => reducedEval α (rows x.1 x.2).target)

theorem firstRow_target (chain : Option ChainShape) (key : PathKey chain)
    (first second : ℕ → Option ℕ) (r : Fin (firstRank chain)) :
    (firstRow chain key first second r).target = 0 := by
  cases chain with
  | none => exact r.elim0
  | some _ => rfl

end Targets

section Values

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E]

/-- The ring element `r` of degree `d` of the words `data`. -/
noncomputable def ringOf (d : ℕ) (data : Array Fq) (r : ℕ) : Rq d :=
  ofCoeffs fun j => ((chunk d data r).getD j Fq.zero).toZq

/-- The payload of a path of the words `data`: `count` elements of degree `d`, or the elements of
the second compression map. -/
noncomputable def payloadOf : (chain : Option ChainShape) → (d count : ℕ) → Array Fq →
    Payload d count chain
  | none, d, _, data => fun j => ringOf d data j
  | some c, _, _, data => fun r => ringOf c.secondDegree data r

/-- The value `y(α)_i` of the row `run.row + r` of a run. -/
def runValues (α : Ext) (data : RhsRun → Array Fq) (run : RhsRun) (r : ℕ) : E :=
  (ringEval α (chunk run.degree (data run) r)).into

theorem chunk_length (d : ℕ) (data : Array Fq) (r : ℕ) : (chunk d data r).length ≤ d := by
  unfold chunk
  rw [Array.length_toList]
  refine Array.size_extract_le.trans (Nat.le_of_eq ?_)
  rw [Nat.mul_add_one, Nat.add_sub_cancel_left]

theorem into_ringOf (α : Ext) (d : ℕ) [NeZero d] (data : Array Fq) (r : ℕ) :
    reducedEval (α.into : E) (ringOf d data r) = (ringEval α (chunk d data r)).into :=
  (ringEval_into α _ (chunk_length d data r)).symm

/-- The image rows and the rows of the second compression map of a path have the values of its
public run at their rows. -/
theorem path_spread {d m : ℕ} [NeZero d] (chain : Option ChainShape) (key : PathKey chain)
    (form : Fin m → Akita.Form d) (first second : ℕ → Option ℕ) (image start : ℕ)
    (source : RhsSource) (α : Ext) (data : RhsRun → Array Fq) (payload : Payload d m chain)
    (reads : payload = payloadOf chain d m (data (pathRun chain image start d m source)))
    (i : ℕ) :
    spread image m (tableOf (targetsOf (α.into : E) (imageRow chain form first payload))) i +
      spread start (secondRank chain) (tableOf (targetsOf (α.into : E)
        (secondRow chain key second payload))) i =
      spread (pathRun chain image start d m source).row (pathRun chain image start d m source).count
        (runValues α data (pathRun chain image start d m source)) i := by
  subst reads
  cases chain with
  | none =>
      rw [show spread start (secondRank none) _ i = (0 : E) from Finset.sum_range_zero _, add_zero]
      exact spread_congr (fun r small => by
        rw [tableOf, dite_eq_left small]
        exact into_ringOf α d _ r) i
  | some c =>
      rw [spread_eq_zero (fun r small => by
        rw [tableOf, dite_eq_left small]
        exact map_zero (reducedEval (d := d) (α.into : E))) i, zero_add]
      exact spread_congr (fun r small => by
        rw [tableOf, dite_eq_left small]
        exact into_ringOf α c.secondDegree _ r) i

omit [Degree2 E] in
/-- Ring rows with the target 0 are zero in the table of the targets. -/
theorem spread_targets_zero {m : ℕ} (α : E) (s : ℕ) (rows : Fin m → RingRow)
    (zero : ∀ r, (rows r).target = 0) (i : ℕ) : spread s m (tableOf (targetsOf α rows)) i = 0 :=
  spread_eq_zero (fun r small => by
    rw [tableOf, dite_eq_left small]
    exact (congrArg _ (zero _)).trans (map_zero _)) i

/-- The sum of the values of the runs at the row `i`, and the value at the opening row. -/
def runsAt (α : Ext) (data : RhsRun → Array Fq) (runs : List RhsRun) (opening : ℕ) (value : E)
    (i : ℕ) : E :=
  (runs.map fun run => spread run.row run.count (runValues α data run) i).sum +
    if opening = i then value else 0

theorem entries_at (α : Ext) (data : RhsRun → Array Fq) (runs : List RhsRun) (i : ℕ) :
    (((rhsEntries α data runs).filter (·.1 == i)).map (·.2.into)).sum =
      (runs.map fun run => spread run.row run.count (runValues (E := E) α data run) i).sum := by
  have one : ∀ (s : ℕ) (g : ℕ → Ext) (l : List ℕ),
      (((l.map fun r => (s + r, g r)).filter (·.1 == i)).map (·.2.into)).sum =
        (l.map fun r => if s + r = i then ((g r).into : E) else 0).sum := by
    intro s g l
    induction l with
    | nil => rfl
    | cons r rest ih =>
        rw [List.map_cons, List.filter_cons]
        by_cases same : s + r = i
        · rw [ite_eq_left (by simpa using same), List.map_cons, List.sum_cons, ih, List.map_cons,
            List.sum_cons, ite_eq_left same]
        · rw [ite_eq_right (by simpa using same), ih, List.map_cons, List.sum_cons,
            ite_eq_right same,
            zero_add]
  induction runs with
  | nil => rfl
  | cons run rest ih =>
      unfold rhsEntries at ih ⊢
      rw [List.flatMap_cons, List.filter_append, List.map_append, List.sum_append, ih, one,
        list_sum_range, List.map_cons, List.sum_cons]
      rfl

/-- The fused claim of `stage2` is `V_{α,τ_1}` of field rows plus `rest`, when the target of
each field row is the value that the runs list at it and the opening row is the value
`Σ_i ρ_i·c_i` of the claims. -/
theorem fused_of_targets (rows : FieldRows E) (τ : List Ext) (α : Ext) (data : RhsRun → Array Fq)
    (runs : List RhsRun) (opening : ℕ) (coefficients claims : Array Ext) (rest : Ext)
    (below : ∀ run ∈ runs, run.row + run.count ≤ opening) (fits : opening < 2 ^ τ.length)
    (targets : ∀ i, rows.targetAt i =
      runsAt α data runs opening (dot coefficients claims).into i) :
    ((stage2 τ α data runs opening coefficients claims rest).fused.into : E) =
      rows.batchTarget (τ.map Ext.into) + rest.into := by
  rw [stage2_fused, batchEntries_batchTarget rows τ _ ?_ fun i => ?_]
  · intro e member
    rcases List.mem_append.mp member with entry | last
    · obtain ⟨run, inside, entry⟩ := List.mem_flatMap.mp entry
      obtain ⟨r, small, rfl⟩ := List.mem_map.mp entry
      exact (lt_of_lt_of_le (Nat.add_lt_add_left (List.mem_range.mp small) _)
        (below run inside)).trans fits
    · rw [List.mem_singleton.mp last]
      exact fits
  · rw [targets i, runsAt, List.filter_append, List.map_append, List.sum_append, entries_at]
    congr 1
    by_cases same : opening = i
    · simp [same]
    · simp [same]

end Values

section One

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E] {S : Shape} {O : OpeningShape S}
  {fold : Descriptor}

theorem groupRun_one (source : RhsSource) :
    groupRun (fun _ : Fin 1 => S) O.openRank O.chain (fun _ => source) 0 =
      pathRun S.chain (1 + S.innerRank)
        (groupRows S + 1 * O.openRank + firstRank S.chain + firstRank O.chain)
        S.outerDegree (S.slices * S.outerRank) source := by
  simp [groupRun, secondStart, mainRows, rowsBefore]

theorem openRun_one :
    openRun (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain =
      pathRun O.chain (groupRows S)
        (groupRows S + 1 * O.openRank + firstRank S.chain + firstRank O.chain +
          secondRank S.chain) O.openDegree (1 * O.openRank) .payload := by
  simp [openRun, secondStart, mainRows]

theorem rowTotal_one :
    rowTotal (fun _ : Fin 1 => S) O.openRank O.chain = O.traceRowCount := by
  simp only [rowTotal, secondStart, mainRows, Fin.sum_univ_one, groupRows,
    OpeningShape.traceRowCount, OpeningShape.carriedCount]
  ring

theorem runsAt_one (α : Ext) (data : RhsRun → Array Fq) (source : RhsSource) (value : E)
    (i : ℕ) :
    runsAt α data (relation (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain
        fun _ => source).1 O.traceRowCount value i =
      spread (groupRun (fun _ : Fin 1 => S) O.openRank O.chain (fun _ => source) 0).row
          (groupRun (fun _ : Fin 1 => S) O.openRank O.chain (fun _ => source) 0).count
          (runValues α data (groupRun (fun _ : Fin 1 => S) O.openRank O.chain (fun _ => source) 0))
          i +
        spread (openRun (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain).row
          (openRun (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain).count
          (runValues α data (openRun (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain)) i +
        if O.traceRowCount = i then value else 0 := by
  simp only [runsAt, relation, List.ofFn_succ, List.ofFn_zero, List.cons_append, List.nil_append,
    List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero, add_assoc]

/-- The rows of one fold without a fold challenge (`carriedRows`) have the values of the two
public runs of the code at their rows, from the row after the rows of `A`. -/
theorem carried_spread (key : Key S) (openKey : OpeningKey O) (payload : S.Payload)
    (openPayload : O.Payload) (α : Ext) (data : RhsRun → Array Fq) (source : RhsSource)
    (reads : payload = payloadOf S.chain _ _
      (data (groupRun (fun _ : Fin 1 => S) O.openRank O.chain (fun _ => source) 0)))
    (openReads : openPayload = payloadOf O.chain _ _
      (data (openRun (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain))) (i : ℕ) :
    spread (1 + S.innerRank) O.carriedCount
        (tableOf (targetsOf (α.into : E) (carriedRows O fold key openKey payload openPayload))) i =
      spread (groupRun (fun _ : Fin 1 => S) O.openRank O.chain (fun _ => source) 0).row
          (groupRun (fun _ : Fin 1 => S) O.openRank O.chain (fun _ => source) 0).count
          (runValues α data (groupRun (fun _ : Fin 1 => S) O.openRank O.chain (fun _ => source) 0))
          i +
        spread (openRun (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain).row
          (openRun (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain).count
          (runValues α data (openRun (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain)) i := by
  rw [groupRun_one] at reads ⊢
  rw [openRun_one] at openReads ⊢
  rw [← path_spread S.chain key.chain (outerForm O fold key) (O.firstCell fold) (O.secondCell fold)
      _ _ source α data payload reads i,
    ← path_spread O.chain openKey.chain (openForm O fold openKey) (O.openFirstCell fold)
      (O.openSecondCell fold) _ _ .payload α data openPayload openReads i]
  unfold carriedRows OpeningShape.carriedCount
  rw [targetsOf_append, targetsOf_append, targetsOf_append, targetsOf_append, targetsOf_append,
    spread_append, spread_append, spread_append, spread_append, spread_append,
    spread_targets_zero _ _ _ (firstRow_target _ _ _ _),
    spread_targets_zero _ _ _ (firstRow_target _ _ _ _),
    show 1 + S.innerRank + S.slices * S.outerRank = groupRows S from Nat.add_assoc _ _ _]
  simp only [zero_add]
  abel

omit [Degree2 E] in
theorem freeRows_targetAt (rel : FlatRelation E) (α : E) (i : ℕ) :
    (rel.freeRows α).targetAt i =
      tableOf (Fin.lastCases rel.value (joinTargets rel.join (targetsOf α rel.row)) :
        Fin (rel.join.count + 1) → E) i :=
  rfl

/-- The field rows of a trace fold (`traceRelation`) have at each row the value that the public
runs of the code list at it, and the value of the statement at the opening row. -/
theorem traceRows_targetAt (st : TraceStatement O E) (challenge : Fin S.blocks → Rq S.innerDegree)
    (μ : ℕ) (fits : O.cells fold ≤ 2 ^ μ) (α : Ext) (data : RhsRun → Array Fq)
    (source : RhsSource)
    (reads : st.payload = payloadOf S.chain _ _
      (data (groupRun (fun _ : Fin 1 => S) O.openRank O.chain (fun _ => source) 0)))
    (openReads : st.openPayload = payloadOf O.chain _ _
      (data (openRun (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain))) (i : ℕ) :
    ((traceRelation O fold st challenge μ fits).freeRows (α.into : E)).targetAt i =
      runsAt α data (relation (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain
        fun _ => source).1 O.traceRowCount st.value i := by
  have targets : joinTargets (traceRelation O fold st challenge μ fits).join
      (targetsOf (α.into : E) (traceRelation O fold st challenge μ fits).row) =
        Fin.append (targetsOf (α.into : E) fun _ : Fin 1 => traceEvaluationRow O fold st challenge)
          (Fin.append (targetsOf (α.into : E) (traceInnerRow O fold st.key challenge))
            (targetsOf (α.into : E)
              (carriedRows O fold st.key st.openKey st.payload st.openPayload))) := by
    show joinTargets (Join.plain E (1 + (S.innerRank + O.carriedCount)))
      (targetsOf (α.into : E) (Fin.append (fun _ : Fin 1 => traceEvaluationRow O fold st challenge)
        (Fin.append (traceInnerRow O fold st.key challenge)
          (carriedRows O fold st.key st.openKey st.payload st.openPayload)))) = _
    rw [plain_targets, targetsOf_append, targetsOf_append]
  rw [freeRows_targetAt, targets]
  show tableOf (Fin.lastCases st.value
    (Fin.append (targetsOf (α.into : E) fun _ : Fin 1 => traceEvaluationRow O fold st challenge)
      (Fin.append (targetsOf (α.into : E) (traceInnerRow O fold st.key challenge))
        (targetsOf (α.into : E)
          (carriedRows O fold st.key st.openKey st.payload st.openPayload)))) :
          Fin (1 + (S.innerRank + O.carriedCount) + 1) → E) i = _
  rw [tableOf_eq_spread, spread_lastCases, spread_append, spread_append,
    spread_targets_zero _ _ (fun _ : Fin 1 => traceEvaluationRow O fold st challenge)
      (fun _ => rfl),
    spread_targets_zero _ _ (traceInnerRow O fold st.key challenge) (fun _ => rfl)]
  simp only [zero_add]
  rw [carried_spread st.key st.openKey st.payload st.openPayload α data source reads openReads i,
    runsAt_one]
  rfl

/-- The field rows of a packing fold (`flatRelation`) have at each row the value that the public
runs of the code list at it, and the value of the statement at the opening row. -/
theorem packingRows_targetAt (β : Module.Basis (Fin O.planes) Zq E) (st : Akita.Statement O E)
    (challenge : Fin S.blocks → Rq O.carrierDegree) (μ : ℕ) (fits : O.cells fold ≤ 2 ^ μ)
    (α : Ext) (data : RhsRun → Array Fq) (source : RhsSource)
    (reads : st.payload = payloadOf S.chain _ _
      (data (groupRun (fun _ : Fin 1 => S) O.openRank O.chain (fun _ => source) 0)))
    (openReads : st.openPayload = payloadOf O.chain _ _
      (data (openRun (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain))) (i : ℕ) :
    ((flatRelation O fold β st challenge μ fits).freeRows (α.into : E)).targetAt i =
      runsAt α data (relation (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain
        fun _ => source).1 O.traceRowCount st.value i := by
  have targets : joinTargets (flatRelation O fold β st challenge μ fits).join
      (targetsOf (α.into : E) (flatRelation O fold β st challenge μ fits).row) =
        Fin.append (fun _ : Fin 1 => ∑ t : Fin O.planes, β t *
            targetsOf (α.into : E) (evaluationRow O fold β st.point challenge) t)
          (Fin.append (targetsOf (α.into : E) (innerRow O fold st.key challenge))
            (targetsOf (α.into : E)
              (carriedRows O fold st.key st.openKey st.payload st.openPayload))) := by
    show joinTargets ((Join.all β).append (Join.plain E (S.innerRank + O.carriedCount)))
      (targetsOf (α.into : E) (Fin.append (evaluationRow O fold β st.point challenge)
        (Fin.append (innerRow O fold st.key challenge)
          (carriedRows O fold st.key st.openKey st.payload st.openPayload)))) = _
    rw [targetsOf_append, append_targets, all_targets, plain_targets, targetsOf_append]
    rfl
  rw [freeRows_targetAt, targets]
  show tableOf (Fin.lastCases st.value
    (Fin.append (fun _ : Fin 1 => ∑ t : Fin O.planes, β t *
        targetsOf (α.into : E) (evaluationRow O fold β st.point challenge) t)
      (Fin.append (targetsOf (α.into : E) (innerRow O fold st.key challenge))
        (targetsOf (α.into : E)
          (carriedRows O fold st.key st.openKey st.payload st.openPayload)))) :
          Fin (1 + (S.innerRank + O.carriedCount) + 1) → E) i = _
  rw [tableOf_eq_spread, spread_lastCases, spread_append, spread_append,
    spread_eq_zero (fun r small => by
      rw [tableOf, dite_eq_left small]
      exact Finset.sum_eq_zero fun t _ => by
        rw [show targetsOf (α.into : E) (evaluationRow O fold β st.point challenge) t = 0 from
          map_zero _, mul_zero]) i,
    spread_targets_zero _ _ (innerRow O fold st.key challenge) (fun _ => rfl)]
  simp only [zero_add]
  rw [carried_spread st.key st.openKey st.payload st.openPayload α data source reads openReads i,
    runsAt_one]
  rfl

variable [Fintype E]

/-- The fused claim of the replay at a trace level is `V_{α,τ_1}` of the field rows of
`traceReduction` plus `rest`, for a statement whose payloads are the words of the runs and whose
value is `Σ_i ρ_i·c_i`. -/
theorem trace_fused (st : TraceStatement O E) (challenge : Fin S.blocks → Rq S.innerDegree)
    (μ : ℕ) (fits : O.cells fold ≤ 2 ^ μ) (batch : ℕ) (rowsFit : O.traceRowCount + 1 ≤ 2 ^ batch)
    (τ : List Ext) (α : Ext) (data : RhsRun → Array Fq) (source : RhsSource)
    (coefficients claims : Array Ext) (rest : Ext) (length : τ.length = batch)
    (reads : st.payload = payloadOf S.chain _ _
      (data (groupRun (fun _ : Fin 1 => S) O.openRank O.chain (fun _ => source) 0)))
    (openReads : st.openPayload = payloadOf O.chain _ _
      (data (openRun (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain)))
    (value : st.value = (dot coefficients claims).into) :
    ((stage2 τ α data
        (relation (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain fun _ => source).1
        (relation (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain fun _ => source).2
        coefficients claims rest).fused.into : E) =
      ((traceReduction st challenge μ fits batch rowsFit).rows α.into).batchTarget
        (τ.map Ext.into) + rest.into := by
  have total : (relation (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain
      fun _ => source).2 = O.traceRowCount := rowTotal_one
  rw [total]
  refine fused_of_targets _ τ α data _ _ coefficients claims rest
    (fun run member => (relation_below _ _ _ _ _ run member).trans_eq total)
    (length ▸ rowsFit) fun i => ?_
  rw [← value]
  exact traceRows_targetAt st challenge μ fits α data source reads openReads i

/-- The fused claim of the replay at a packing level is `V_{α,τ_1}` of the field rows of
`liftedReduction` plus `rest`, for a statement whose payloads are the words of the runs and whose
value is `Σ_i ρ_i·c_i`. -/
theorem packing_fused (β : Module.Basis (Fin O.planes) Zq E) (st : Akita.Statement O E)
    (challenge : Fin S.blocks → Rq O.carrierDegree) (μ : ℕ) (fits : O.cells fold ≤ 2 ^ μ)
    (digits : ℕ) (room : O.quotientRoom fold digits ≤ 2 ^ μ) (batch : ℕ)
    (rowsFit : O.rowCount + 1 ≤ 2 ^ batch) (τ : List Ext) (α : Ext) (data : RhsRun → Array Fq)
    (source : RhsSource) (coefficients claims : Array Ext) (rest : Ext)
    (length : τ.length = batch)
    (reads : st.payload = payloadOf S.chain _ _
      (data (groupRun (fun _ : Fin 1 => S) O.openRank O.chain (fun _ => source) 0)))
    (openReads : st.openPayload = payloadOf O.chain _ _
      (data (openRun (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain)))
    (value : st.value = (dot coefficients claims).into) :
    ((stage2 τ α data
        (relation (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain fun _ => source).1
        (relation (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain fun _ => source).2
        coefficients claims rest).fused.into : E) =
      ((liftedReduction β st challenge μ fits digits room batch rowsFit).rows α.into).batchTarget
        (τ.map Ext.into) + rest.into := by
  have total : (relation (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain
      fun _ => source).2 = O.traceRowCount := rowTotal_one
  rw [total]
  refine fused_of_targets _ τ α data _ _ coefficients claims rest
    (fun run member => (relation_below _ _ _ _ _ run member).trans_eq total)
    (length ▸ rowsFit) fun i => ?_
  rw [← value]
  exact packingRows_targetAt β st challenge μ fits α data source reads openReads i

end One

section Groups

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E]

theorem path_spread_eq {d m : ℕ} [NeZero d] (chain : Option ChainShape) (key : PathKey chain)
    (form : Fin m → Akita.Form d) (first second : ℕ → Option ℕ) (image start : ℕ)
    (source : RhsSource) (α : Ext) (data : RhsRun → Array Fq) (payload : Payload d m chain)
    (run : RhsRun) (same : run = pathRun chain image start d m source)
    (reads : payload = payloadOf chain d m (data run)) (i : ℕ) :
    spread image m (tableOf (targetsOf (α.into : E) (imageRow chain form first payload))) i +
      spread start (secondRank chain) (tableOf (targetsOf (α.into : E)
        (secondRow chain key second payload))) i =
      spread run.row run.count (runValues α data run) i := by
  subst same
  exact path_spread chain key form first second image start source α data payload reads i

omit [Degree2 E] in
theorem spread_gather_rows {n : ℕ} (count : Fin n → ℕ) (s : ℕ) (α : E)
    (rows : (g : Fin n) → Fin (count g) → RingRow) (i : ℕ) :
    spread s (∑ g, count g) (tableOf (targetsOf α (Akita.gather rows))) i =
      ∑ k, spread (s + rowsBefore (fun j => count (groupOrder n j)) k) (count (groupOrder n k))
        (tableOf (targetsOf α (rows (groupOrder n k)))) i :=
  spread_gather count s (fun x => reducedEval α (rows x.1 x.2).target) i

theorem runsAt_relation {n : ℕ} (shape : Fin n → Shape) (openRank openDegree : ℕ)
    (openChain : Option ChainShape) (source : Fin n → RhsSource) (α : Ext)
    (data : RhsRun → Array Fq) (opening : ℕ) (value : E) (i : ℕ) :
    runsAt α data (relation shape openRank openDegree openChain source).1 opening value i =
      ∑ k, spread (groupRun shape openRank openChain source k).row
          (groupRun shape openRank openChain source k).count
          (runValues α data (groupRun shape openRank openChain source k)) i +
        spread (openRun shape openRank openDegree openChain).row
          (openRun shape openRank openDegree openChain).count
          (runValues α data (openRun shape openRank openDegree openChain)) i +
        if opening = i then value else 0 := by
  simp only [runsAt, relation, List.map_append, List.sum_append, List.map_ofFn, List.sum_ofFn,
    List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero, Function.comp_apply]

omit [Field E] [Algebra Zq E] [Degree2 E] in
theorem sum_regroup {M : Type} [AddCommMonoid M] {n : ℕ} (b f : Fin n → M) (d h x : M) :
    ∑ k, b k + (d + (∑ k, f k + h)) + x = ∑ k, (b k + f k) + (d + h) + x := by
  rw [Finset.sum_add_distrib]
  abel

variable {n : ℕ} {B : Batch (Fin n)}

omit [Field E] [Algebra Zq E] [Degree2 E] in
theorem mainRows_order :
    mainRows (fun k => B.shape (groupOrder n k)) = ∑ g, B.groupCount g :=
  Equiv.sum_comp (groupOrder n) B.groupCount

omit [Field E] [Algebra Zq E] [Degree2 E] in
theorem secondStart_order :
    secondStart (fun k => B.shape (groupOrder n k)) B.openRank B.chain =
      ∑ g, B.groupCount g + 1 * B.openRank + ∑ g, firstRank (B.shape g).chain +
        firstRank B.chain := by
  rw [secondStart, mainRows_order,
    show ∑ k, firstRank (B.shape (groupOrder n k)).chain = ∑ g, firstRank (B.shape g).chain from
      Equiv.sum_comp (groupOrder n) fun g => firstRank (B.shape g).chain]

omit [Field E] [Algebra Zq E] [Degree2 E] in
theorem secondSum_order :
    ∑ k, secondRank (B.shape (groupOrder n k)).chain = ∑ g, secondRank (B.shape g).chain :=
  Equiv.sum_comp (groupOrder n) fun g => secondRank (B.shape g).chain

omit [Field E] [Algebra Zq E] [Degree2 E] in
theorem rowTotal_order :
    rowTotal (fun k => B.shape (groupOrder n k)) B.openRank B.chain = B.rowCount := by
  rw [rowTotal, secondStart_order, secondSum_order]
  unfold Batch.rowCount Batch.restCount
  omega

/-- The rows of `F_1` of the groups of a fold with opening groups. -/
noncomputable def batchFirst (B : Batch (Fin n)) (st : B.Statement E) :
    (g : Fin n) → Fin (firstRank (B.shape g).chain) → RingRow := fun g r =>
  (firstRow (B.shape g).chain (st.key g).chain ((B.opening g).firstCell (B.fold g))
    ((B.opening g).secondCell (B.fold g)) r).relocate (B.relocate g)

/-- The rows of `F_2` of the groups of a fold with opening groups. -/
noncomputable def batchSecond (B : Batch (Fin n)) (st : B.Statement E) :
    (g : Fin n) → Fin (secondRank (B.shape g).chain) → RingRow := fun g r =>
  (secondRow (B.shape g).chain (st.key g).chain ((B.opening g).secondCell (B.fold g))
    (st.payload g) r).relocate (B.relocate g)

/-- The targets of the field rows of the group `x.1` of a fold with opening groups. -/
noncomputable def groupTargets (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E)
    (st : B.Statement E) (challenge : B.Challenge) (α : E)
    (x : Σ g : Fin n, Fin (B.groupCount g)) : E :=
  joinTargets ((Join.all (β x.1)).append (Join.plain E (B.groupTail x.1)))
    (targetsOf α (B.groupRows β st challenge x.1)) x.2

omit [Degree2 E] in
/-- The field rows of one group before the rows of `D` have the targets of its rows of `B`, after
its consistency row and its rows of `A`. -/
theorem group_spread (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E) (st : B.Statement E)
    (challenge : B.Challenge) (α : E) (g : Fin n) (s i : ℕ) :
    spread s (B.groupCount g) (tableOf fun j => groupTargets β st challenge α ⟨g, j⟩) i =
      spread (s + 1 + (B.shape g).innerRank) ((B.shape g).slices * (B.shape g).outerRank)
        (tableOf (targetsOf α (imageRow (B.shape g).chain
          (outerForm (B.opening g) (B.fold g) (st.key g))
          ((B.opening g).firstCell (B.fold g)) (st.payload g)))) i := by
  have split : (fun j => groupTargets β st challenge α ⟨g, j⟩) =
      Fin.append (fun _ : Fin 1 => ∑ t : Fin (B.planes g), β g t * targetsOf α (fun t =>
          (evaluationRow (B.opening g) (B.fold g) (β g) (st.point g) (challenge g) t).relocate
            (B.relocate g)) t)
        (Fin.append (targetsOf α fun a =>
            (innerRow (B.opening g) (B.fold g) (st.key g) (challenge g) a).relocate (B.relocate g))
          (targetsOf α fun j => (imageRow (B.shape g).chain (outerForm (B.opening g) (B.fold g)
            (st.key g)) ((B.opening g).firstCell (B.fold g)) (st.payload g) j).relocate
              (B.relocate g))) := by
    show joinTargets ((Join.all (β g)).append (Join.plain E
        ((B.shape g).innerRank + (B.shape g).slices * (B.shape g).outerRank)))
      (targetsOf α (Fin.append (fun t =>
          (evaluationRow (B.opening g) (B.fold g) (β g) (st.point g) (challenge g) t).relocate
            (B.relocate g))
        (Fin.append (fun a =>
            (innerRow (B.opening g) (B.fold g) (st.key g) (challenge g) a).relocate (B.relocate g))
          (fun j => (imageRow (B.shape g).chain (outerForm (B.opening g) (B.fold g) (st.key g))
            ((B.opening g).firstCell (B.fold g)) (st.payload g) j).relocate (B.relocate g))))) = _
    rw [targetsOf_append, append_targets, all_targets, plain_targets, targetsOf_append]
    rfl
  rw [split]
  show spread s (1 + ((B.shape g).innerRank + (B.shape g).slices * (B.shape g).outerRank))
    (@tableOf E _ (1 + ((B.shape g).innerRank + (B.shape g).slices * (B.shape g).outerRank)) _)
      i = _
  rw [spread_append, spread_append,
    spread_eq_zero (fun r small => by
      rw [tableOf, dite_eq_left small]
      exact Finset.sum_eq_zero fun t _ => by
        rw [show targetsOf α (fun t => (evaluationRow (B.opening g) (B.fold g) (β g) (st.point g)
          (challenge g) t).relocate (B.relocate g)) t = 0 from map_zero _, mul_zero]) i,
    spread_targets_zero _ _ _ (fun _ => rfl), zero_add, zero_add]
  rfl

/-- The field rows of a fold with opening groups (`Batch.flatRelation`) have at each row the
value that the public runs of the code list at it, with the groups in the order `groupOrder`,
and the value of the statement at the opening row. -/
theorem batchRows_targetAt (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E) (st : B.Statement E)
    (challenge : B.Challenge) (μ : ℕ) (fits : B.cells ≤ 2 ^ μ) (α : Ext)
    (data : RhsRun → Array Fq) (source : Fin n → RhsSource)
    (reads : ∀ k, st.payload (groupOrder n k) = payloadOf (B.shape (groupOrder n k)).chain _ _
      (data (groupRun (fun k => B.shape (groupOrder n k)) B.openRank B.chain source k)))
    (openReads : st.openPayload = payloadOf B.chain _ _
      (data (openRun (fun k => B.shape (groupOrder n k)) B.openRank B.openDegree B.chain)))
    (i : ℕ) :
    ((B.flatRelation β st challenge μ fits).freeRows (α.into : E)).targetAt i =
      runsAt α data (relation (fun k => B.shape (groupOrder n k)) B.openRank B.openDegree B.chain
        source).1 B.rowCount st.value i := by
  have targets : joinTargets (B.flatRelation β st challenge μ fits).join
      (targetsOf (α.into : E) (B.flatRelation β st challenge μ fits).row) =
        Fin.append (fun k => groupTargets β st challenge (α.into : E) (place B.groupCount k))
          (Fin.append (targetsOf (α.into : E)
              (imageRow B.chain (B.openForm st.openKey) B.openFirstCell st.openPayload))
            (Fin.append (targetsOf (α.into : E) (Akita.gather (batchFirst B st)))
              (Fin.append (targetsOf (α.into : E)
                  (firstRow B.chain st.openKey.chain B.openFirstCell B.openSecondCell))
                (Fin.append (targetsOf (α.into : E) (Akita.gather (batchSecond B st)))
                  (targetsOf (α.into : E)
                    (secondRow B.chain st.openKey.chain B.openSecondCell st.openPayload)))))) := by
    show joinTargets
      ((Join.gather fun g => (Join.all (β g)).append (Join.plain E (B.groupTail g))).append
        (Join.plain E (1 * B.openRank + (∑ g, firstRank (B.shape g).chain + (firstRank B.chain +
          (∑ g, secondRank (B.shape g).chain + secondRank B.chain))))))
      (targetsOf (α.into : E) (Fin.append
        (Akita.gather (count := fun g => B.planes g + B.groupTail g) (B.groupRows β st challenge))
        (Fin.append (imageRow B.chain (B.openForm st.openKey) B.openFirstCell st.openPayload)
          (Fin.append (Akita.gather (batchFirst B st))
            (Fin.append (firstRow B.chain st.openKey.chain B.openFirstCell B.openSecondCell)
              (Fin.append (Akita.gather (batchSecond B st))
                (secondRow B.chain st.openKey.chain B.openSecondCell st.openPayload))))))) = _
    rw [targetsOf_append, append_targets,
      gather_rows_targets (size := fun g => B.planes g + B.groupTail g)
        (fun g => (Join.all (β g)).append (Join.plain E (B.groupTail g)))
        (B.groupRows β st challenge),
      plain_targets, targetsOf_append, targetsOf_append, targetsOf_append, targetsOf_append]
    rfl
  rw [freeRows_targetAt, targets]
  show @tableOf E _ (∑ g, B.groupCount g + (1 * B.openRank + (∑ g, firstRank (B.shape g).chain +
      (firstRank B.chain + (∑ g, secondRank (B.shape g).chain + secondRank B.chain)))) + 1)
    (Fin.lastCases st.value (Fin.append
      (fun k => groupTargets β st challenge (α.into : E) (place B.groupCount k)) _)) i = _
  rw [tableOf_eq_spread, spread_lastCases, spread_append, spread_gather B.groupCount 0,
    spread_append, spread_append, spread_append, spread_append,
    spread_targets_zero _ _ (Akita.gather (batchFirst B st)) (fun _ => firstRow_target _ _ _ _ _),
    spread_targets_zero _ _ _ (firstRow_target _ _ _ _),
    spread_gather_rows (fun g => secondRank (B.shape g).chain)]
  simp only [zero_add, group_spread]
  rw [runsAt_relation, sum_regroup]
  refine congrArg₂ (· + ·) (congrArg₂ (· + ·) (Finset.sum_congr rfl fun k _ => ?_) ?_) rfl
  · refine path_spread_eq _ _ _ _ _ _ _ (source k) α data _ _ ?_ (reads k) i
    simp only [groupRun]
    rw [secondStart_order]
    rfl
  · refine path_spread_eq _ _ _ _ _ _ _ .payload α data _ _ ?_ openReads i
    simp only [openRun]
    rw [mainRows_order, secondStart_order, secondSum_order]

variable [Fintype E]

/-- The fused claim of the replay at a level with opening groups is `V_{α,τ_1}` of the field rows
of `Batch.liftedReduction` plus `rest`, for a statement whose payloads are the words of the runs
and whose value is `Σ_i ρ_i·c_i`. -/
theorem batch_fused (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E) (st : B.Statement E)
    (challenge : B.Challenge) (μ : ℕ) (fits : B.cells ≤ 2 ^ μ) (base digits : ℕ)
    (room : B.quotientRoom digits ≤ 2 ^ μ) (batch : ℕ) (rowsFit : B.rowCount + 1 ≤ 2 ^ batch)
    (τ : List Ext) (α : Ext) (data : RhsRun → Array Fq) (source : Fin n → RhsSource)
    (coefficients claims : Array Ext) (rest : Ext) (length : τ.length = batch)
    (reads : ∀ k, st.payload (groupOrder n k) = payloadOf (B.shape (groupOrder n k)).chain _ _
      (data (groupRun (fun k => B.shape (groupOrder n k)) B.openRank B.chain source k)))
    (openReads : st.openPayload = payloadOf B.chain _ _
      (data (openRun (fun k => B.shape (groupOrder n k)) B.openRank B.openDegree B.chain)))
    (value : st.value = (dot coefficients claims).into) :
    ((stage2 τ α data
        (relation (fun k => B.shape (groupOrder n k)) B.openRank B.openDegree B.chain source).1
        (relation (fun k => B.shape (groupOrder n k)) B.openRank B.openDegree B.chain source).2
        coefficients claims rest).fused.into : E) =
      ((Batch.liftedReduction β st challenge μ fits base digits room batch rowsFit).rows
        α.into).batchTarget (τ.map Ext.into) + rest.into := by
  have total : (relation (fun k => B.shape (groupOrder n k)) B.openRank B.openDegree B.chain
      source).2 = B.rowCount := rowTotal_order
  rw [total]
  refine fused_of_targets _ τ α data _ _ coefficients claims rest
    (fun run member => (relation_below _ _ _ _ _ run member).trans_eq total)
    (length ▸ rowsFit) fun i => ?_
  rw [← value]
  exact batchRows_targetAt β st challenge μ fits α data source reads openReads i

/-- `batch_fused` for the quotient-free reduction `Batch.flatReduction`. -/
theorem batch_flat_fused (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E) (st : B.Statement E)
    (challenge : B.Challenge) (μ : ℕ) (fits : B.cells ≤ 2 ^ μ) (batch : ℕ)
    (rowsFit : B.rowCount + 1 ≤ 2 ^ batch) (τ : List Ext) (α : Ext) (data : RhsRun → Array Fq)
    (source : Fin n → RhsSource) (coefficients claims : Array Ext) (rest : Ext)
    (length : τ.length = batch)
    (reads : ∀ k, st.payload (groupOrder n k) = payloadOf (B.shape (groupOrder n k)).chain _ _
      (data (groupRun (fun k => B.shape (groupOrder n k)) B.openRank B.chain source k)))
    (openReads : st.openPayload = payloadOf B.chain _ _
      (data (openRun (fun k => B.shape (groupOrder n k)) B.openRank B.openDegree B.chain)))
    (value : st.value = (dot coefficients claims).into) :
    ((stage2 τ α data
        (relation (fun k => B.shape (groupOrder n k)) B.openRank B.openDegree B.chain source).1
        (relation (fun k => B.shape (groupOrder n k)) B.openRank B.openDegree B.chain source).2
        coefficients claims rest).fused.into : E) =
      ((Batch.flatReduction β st challenge μ fits batch rowsFit).rows α.into).batchTarget
        (τ.map Ext.into) + rest.into := by
  have total : (relation (fun k => B.shape (groupOrder n k)) B.openRank B.openDegree B.chain
      source).2 = B.rowCount := rowTotal_order
  rw [total]
  refine fused_of_targets _ τ α data _ _ coefficients claims rest
    (fun run member => (relation_below _ _ _ _ _ run member).trans_eq total)
    (length ▸ rowsFit) fun i => ?_
  rw [← value]
  exact batchRows_targetAt β st challenge μ fits α data source reads openReads i

end Groups

end NightstreamFPrime.Spec.Compression.Akita.Replay
