import CompressionLean.Circuit.Block

/-!
Owns what a witness of a run's CCS says about the run: every glue row holds
(`Run.rows_hold`) and the cells of every block satisfy the template
(`Run.blocks_hold`), at the wire values `Run.wireValue`.

- The rows do not overlap (`layout.rs:116-151`). The block parts end where
  the glue rows begin, and distinct template rows of distinct blocks have
  distinct slots (`slot_inj`). So each CCS row is one glue row or one
  template row of one block.
- A glue row has no `X` entry, so its constraint is
  `Σ_i (A_i·z)·(B_i·z) - C·z` (`GlueRow.Holds`).
- The value of a wire is `z` at its column (`Layout.column`): the constant
  one is 1, and statement word `i + 1` is word `i` of the run.
-/

namespace CompressionLean.Lowering

open CompressionLean.Shrink

/-- `Σ coefficient·v(wire)` over a term list. -/
def termsValue {F : Type} [CommRing F] (v : Wire → F) (terms : List (Wire × F)) : F :=
  (terms.map fun t => t.2 * v t.1).sum

/-- A glue row holds at the wire values `v`: `Σ_i (A_i·v)·(B_i·v) = C·v`
(`Trace::row_value`, `record.rs:586-596`). -/
def _root_.CompressionLean.Shrink.GlueRow.Holds {F : Type} [CommRing F] (v : Wire → F) (row : GlueRow F) : Prop :=
  termsValue v (row.side 0) * termsValue v (row.side 1) + termsValue v (row.side 2) * termsValue v (row.side 3) +
      termsValue v (row.side 4) * termsValue v (row.side 5) =
    termsValue v (row.side 6)

/-- Two products and sums with remainders below `w` are equal only termwise. -/
private theorem mul_add_inj {w b b' k k' : ℕ} (hk : k < w) (hk' : k' < w) (h : w * b + k = w * b' + k') :
    b = b' ∧ k = k' := by
  rcases lt_trichotomy b b' with lt | rfl | gt
  · have step : w * (b + 1) ≤ w * b' := Nat.mul_le_mul_left _ lt
    rw [Nat.mul_succ] at step
    omega
  · omega
  · have step : w * (b' + 1) ≤ w * b := Nat.mul_le_mul_left _ gt
    rw [Nat.mul_succ] at step
    omega

/-- Distinct indices of distinct blocks sit in distinct slots. -/
theorem slot_inj {size count b b' k k' : ℕ} (hb : b < count) (hb' : b' < count) (hk : k < size) (hk' : k' < size)
    (h : slot size count b k = slot size count b' k') : b = b' ∧ k = k' := by
  have rest := rest_le_narrow size
  unfold slot at h
  split_ifs at h with low low'
  · exact mul_add_inj low low' h
  · have past : wide size * (b + 1) ≤ wide size * count := Nat.mul_le_mul_left _ hb
    rw [Nat.mul_succ] at past
    omega
  · have past : wide size * (b' + 1) ≤ wide size * count := Nat.mul_le_mul_left _ hb'
    rw [Nat.mul_succ] at past
    omega
  · have eq : narrow size * b + (k - wide size) = narrow size * b' + (k' - wide size) := by omega
    obtain ⟨same, shift⟩ := mul_add_inj (w := narrow size) (by omega) (by omega) eq
    exact ⟨same, by omega⟩

section Rows

variable {F : Type} [Field F]

theorem rowValue_append (es es' : List (Entry F)) (z : ℕ → F) (j : Fin 8) (r : ℕ) :
    rowValue (es ++ es') z j r = rowValue es z j r + rowValue es' z j r := by
  simp [rowValue, List.map_append, List.sum_append]

theorem rowValue_eq_zero {es : List (Entry F)} {r : ℕ} (h : ∀ e ∈ es, e.row ≠ r) (z : ℕ → F) (j : Fin 8) :
    rowValue es z j r = 0 := by
  unfold rowValue
  apply List.sum_eq_zero
  intro x hx
  obtain ⟨e, he, rfl⟩ := List.mem_map.mp hx
  simp [h e he]

theorem rowValue_flatMap {α : Type} (l : List α) (f : α → List (Entry F)) (z : ℕ → F) (j : Fin 8) (r : ℕ) :
    rowValue (l.flatMap f) z j r = (l.map fun a => rowValue (f a) z j r).sum := by
  induction l with
  | nil => simp [rowValue]
  | cons a l ih => rw [List.flatMap_cons, rowValue_append, ih, List.map_cons, List.sum_cons]

/-- A sum over `0, …, n - 1` with one nonzero term. -/
private theorem sum_range_single {n b : ℕ} (g : ℕ → F) (hb : b < n) (zero : ∀ i < n, i ≠ b → g i = 0) :
    ((List.range n).map g).sum = g b := by
  rw [← List.sum_toFinset g List.nodup_range, List.toFinset_range]
  exact Finset.sum_eq_single b (fun i hi ne => zero i (Finset.mem_range.mp hi) ne)
    fun out => absurd (Finset.mem_range.mpr hb) out

variable (T : Template F) (s : Counts)

omit [Field F] in
/-- Every entry of glue row `g` sits on CCS row `glueRow g`. -/
theorem row_of_mem_rowEntries {g : ℕ} {row : GlueRow F} {e : Entry F} (he : e ∈ Layout.rowEntries T s g row) :
    e.row = Layout.glueRow T s g := by
  simp only [Layout.rowEntries, List.mem_flatMap, List.mem_map] at he
  obtain ⟨_, -, _, -, rfl⟩ := he
  rfl

/-- Side `k` of glue row `g` is matrix `k + 1` on its row. -/
theorem rowValue_rowEntries (z : ℕ → F) (g : ℕ) (row : GlueRow F) (k : Fin 7) :
    rowValue (Layout.rowEntries T s g row) z k.succ (Layout.glueRow T s g) =
      termsValue (fun x => z (Layout.column T s x)) (row.side k) := by
  have side : ∀ side : Fin 7,
      rowValue ((row.side side).map fun t =>
          (⟨side.succ, Layout.glueRow T s g, Layout.column T s t.1, t.2⟩ : Entry F))
          z k.succ (Layout.glueRow T s g) =
        if side = k then termsValue (fun x => z (Layout.column T s x)) (row.side side) else 0 := by
    intro side
    unfold rowValue termsValue
    rw [List.map_map]
    by_cases same : side = k
    · subst same
      rw [if_pos rfl]
      congr 1
      apply List.map_congr_left
      intro t _
      simp
    · rw [if_neg same]
      apply List.sum_eq_zero
      intro x hx
      obtain ⟨t, -, rfl⟩ := List.mem_map.mp hx
      simp [Fin.succ_inj, same]
  rw [Layout.rowEntries, rowValue_flatMap]
  simp only [side]
  rw [← List.ofFn_eq_map, List.sum_ofFn]
  simp

/-- Glue rows have no `X` entry. -/
theorem rowValue_rowEntries_zero (z : ℕ → F) (g : ℕ) (row : GlueRow F) (r : ℕ) :
    rowValue (Layout.rowEntries T s g row) z 0 r = 0 := by
  unfold rowValue
  apply List.sum_eq_zero
  intro x hx
  simp only [Layout.rowEntries, List.mem_map, List.mem_flatMap] at hx
  obtain ⟨e, ⟨side, -, t, -, rfl⟩, rfl⟩ := hx
  simp [(Fin.succ_ne_zero side)]

/-- On CCS row `glueRow (g + k)`, the glue entries from row number `g` on are
those of the `k`-th row. -/
theorem rowValue_glueEntries (z : ℕ → F) (j : Fin 8) :
    ∀ (g k : ℕ) (rows : List (GlueRow F)) (row : GlueRow F), rows[k]? = some row →
      rowValue (Layout.glueEntries T s g rows) z j (Layout.glueRow T s (g + k)) =
        rowValue (Layout.rowEntries T s (g + k) row) z j (Layout.glueRow T s (g + k))
  | _, _, [], _, h => by simp at h
  | g, 0, first :: rest, row, h => by
      simp only [List.getElem?_cons_zero, Option.some.injEq] at h
      subst h
      rw [Layout.glueEntries, rowValue_append, Nat.add_zero]
      have later : rowValue (Layout.glueEntries T s (g + 1) rest) z j (Layout.glueRow T s g) = 0 := by
        apply rowValue_eq_zero
        intro e he
        obtain ⟨k, -, rowEq, -⟩ := mem_glueEntries T s (g + 1) rest he
        rw [rowEq]
        simp only [Layout.glueRow]
        omega
      rw [later, add_zero]
  | g, k + 1, first :: rest, row, h => by
      simp only [List.getElem?_cons_succ] at h
      rw [Layout.glueEntries, rowValue_append]
      have here : rowValue (Layout.rowEntries T s g first) z j (Layout.glueRow T s (g + (k + 1))) = 0 := by
        apply rowValue_eq_zero
        intro e he
        rw [row_of_mem_rowEntries T s he]
        simp only [Layout.glueRow]
        omega
      rw [here, zero_add, show g + (k + 1) = g + 1 + k by omega]
      exact rowValue_glueEntries z j (g + 1) k rest row h

/-- On the CCS row of template row `k` of block `b`, the block entries are the
template's entries on the cells of block `b`. -/
theorem rowValue_blockEntries (wf : T.WellFormed) (z : ℕ → F) (one : z (Layout.one T s) = 1) (j : Fin 8)
    {b k : ℕ} (hb : b < s.blocks) (hk : k < T.rows) :
    rowValue (Layout.blockEntries T s) z j (Layout.blockRow T s b k) =
      T.rowValue (fun c => z (Layout.blockCell T s b c)) j k := by
  rw [Layout.blockEntries, rowValue_flatMap]
  rw [sum_range_single _ hb]
  · unfold rowValue Template.rowValue
    rw [List.map_map]
    congr 1
    apply List.map_congr_left
    intro e he
    have hrow := wf.row_lt e he
    have same : Layout.blockRow T s b e.row = Layout.blockRow T s b k ↔ e.row = k :=
      ⟨fun h => (slot_inj hb hb hrow hk h).2, fun h => by rw [h]⟩
    simp only [Function.comp_apply, same]
    split_ifs
    · cases cell : e.cell with
      | none => simp [TemplateEntry.value, cell, Layout.entryColumn, one]
      | some c => simp [TemplateEntry.value, cell, Layout.entryColumn]
    · rfl
  · intro b' hb' ne
    apply rowValue_eq_zero
    intro e he
    obtain ⟨te, tm, rfl⟩ := List.mem_map.mp he
    exact fun h => ne (slot_inj hb' hb (wf.row_lt te tm) hk h).1

end Rows

section Witness

variable {F : Type} [Field F] (T : Template F) (s : Counts) (run : Run F) (w : ℕ → F)

/-- The value of each wire under the witness `w`: `z` at the wire's column. -/
noncomputable def _root_.CompressionLean.Shrink.Run.wireValue (x : Wire) : F := (run.ccs T s).z w (Layout.column T s x)

/-- The constant one is 1. -/
theorem _root_.CompressionLean.Shrink.Run.wireValue_one : run.wireValue T s w (.pub 0) = 1 := by
  simp [Run.wireValue, Ccs.z, Run.ccs, Layout.column, Layout.one, Ccs.publicTable]

/-- Statement wire `i + 1` is word `i` of the run. -/
theorem _root_.CompressionLean.Shrink.Run.wireValue_pub (i : ℕ) : run.wireValue T s w (.pub (i + 1)) = run.publics.getD i 0 := by
  simp [Run.wireValue, Ccs.z, Run.ccs, Layout.column, Layout.one, Ccs.publicTable]

variable {T s run w}

/-- **Every glue row holds** under a witness of the run's CCS. -/
theorem _root_.CompressionLean.Shrink.Run.rows_hold (wf : T.WellFormed) (counts : run.counts = s) (holds : (run.ccs T s).Relation w) :
    ∀ row ∈ run.rows, row.Holds (run.wireValue T s w) := by
  intro row member
  obtain ⟨g, hg, rfl⟩ := List.getElem_of_mem member
  have glueRows : run.rows.length = s.glueRows := congrArg Counts.glueRows counts
  have inside : Layout.glueRow T s g < 2 ^ (run.ccs T s).rowVars := by
    have := le_pow_variablesFor (partsEnd T.rows s.blocks + s.glueRows)
    show partsEnd T.rows s.blocks + g < 2 ^ Layout.rowVars T s
    unfold Layout.rowVars
    omega
  have value : ∀ j, rowValue (run.ccs T s).entries ((run.ccs T s).z w) j (Layout.glueRow T s g) =
      rowValue (Layout.rowEntries T s g run.rows[g]) ((run.ccs T s).z w) j (Layout.glueRow T s g) := by
    intro j
    show rowValue (Layout.blockEntries T s ++ Layout.glueEntries T s 0 run.rows) _ j _ = _
    rw [rowValue_append, rowValue_eq_zero, zero_add]
    · have := rowValue_glueEntries T s ((run.ccs T s).z w) j 0 g run.rows run.rows[g] (List.getElem?_eq_getElem hg)
      rwa [Nat.zero_add] at this
    · intro e he
      simp only [Layout.blockEntries, List.mem_flatMap, List.mem_range, List.mem_map] at he
      obtain ⟨b, hb, te, tm, rfl⟩ := he
      have hb' : b < s.blocks := hb
      have := slot_lt (size := T.rows) hb' (wf.row_lt te tm)
      simp only [Layout.blockRow, Layout.glueRow]
      omega
  have row := holds _ inside
  simp only [constraint, value] at row
  have side : ∀ k : Fin 7, rowValue (Layout.rowEntries T s g run.rows[g]) ((run.ccs T s).z w) k.succ
      (Layout.glueRow T s g) = termsValue (run.wireValue T s w) (run.rows[g].side k) :=
    rowValue_rowEntries T s _ g _
  have v1 : rowValue (Layout.rowEntries T s g run.rows[g]) ((run.ccs T s).z w) 1 (Layout.glueRow T s g) = _ := side 0
  have v2 : rowValue (Layout.rowEntries T s g run.rows[g]) ((run.ccs T s).z w) 2 (Layout.glueRow T s g) = _ := side 1
  have v3 : rowValue (Layout.rowEntries T s g run.rows[g]) ((run.ccs T s).z w) 3 (Layout.glueRow T s g) = _ := side 2
  have v4 : rowValue (Layout.rowEntries T s g run.rows[g]) ((run.ccs T s).z w) 4 (Layout.glueRow T s g) = _ := side 3
  have v5 : rowValue (Layout.rowEntries T s g run.rows[g]) ((run.ccs T s).z w) 5 (Layout.glueRow T s g) = _ := side 4
  have v6 : rowValue (Layout.rowEntries T s g run.rows[g]) ((run.ccs T s).z w) 6 (Layout.glueRow T s g) = _ := side 5
  have v7 : rowValue (Layout.rowEntries T s g run.rows[g]) ((run.ccs T s).z w) 7 (Layout.glueRow T s g) = _ := side 6
  rw [rowValue_rowEntries_zero, v1, v2, v3, v4, v5, v6, v7] at row
  rw [zero_pow (by norm_num), zero_add] at row
  exact sub_eq_zero.mp row

/-- **The cells of every block satisfy the template** under a witness of the
run's CCS. -/
theorem _root_.CompressionLean.Shrink.Run.blocks_hold (wf : T.WellFormed) (counts : run.counts = s) (holds : (run.ccs T s).Relation w) :
    ∀ b < run.blocks, T.Holds fun c => run.wireValue T s w (.block b c) := by
  intro b hb k hk
  have blocks : run.blocks = s.blocks := congrArg Counts.blocks counts
  have hb' : b < s.blocks := blocks ▸ hb
  have inParts := slot_lt (size := T.rows) hb' hk
  have inside : Layout.blockRow T s b k < 2 ^ (run.ccs T s).rowVars := by
    have := le_pow_variablesFor (partsEnd T.rows s.blocks + s.glueRows)
    show slot T.rows s.blocks b k < 2 ^ Layout.rowVars T s
    unfold Layout.rowVars
    omega
  have one : (run.ccs T s).z w (Layout.one T s) = 1 := run.wireValue_one T s w
  have value : ∀ j, rowValue (run.ccs T s).entries ((run.ccs T s).z w) j (Layout.blockRow T s b k) =
      T.rowValue (fun c => run.wireValue T s w (.block b c)) j k := by
    intro j
    show rowValue (Layout.blockEntries T s ++ Layout.glueEntries T s 0 run.rows) _ j _ = _
    rw [rowValue_append, rowValue_blockEntries T s wf _ one j hb' hk, rowValue_eq_zero, add_zero]
    · rfl
    · intro e he
      obtain ⟨g, -, rowEq, -⟩ := mem_glueEntries T s 0 run.rows he
      rw [rowEq]
      simp only [Layout.blockRow, Layout.glueRow]
      omega
  have row := holds _ inside
  simpa only [value] using row

end Witness

end CompressionLean.Lowering
