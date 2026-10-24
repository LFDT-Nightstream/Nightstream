import CompressionLean.Circuit.Package
import CompressionLean.Circuit.Exec
import NightstreamFPrime.Export.PackageExpansion

/-!
Owns the package-rows lemma (`system_holds_iff`): the system that `load` hands
to `Package::apply` holds, with the one column at one, exactly when F′'s rows
hold (`CircuitPackage.RowsHold`) under the package assignment of the system
values (`assignment`).

- Sides: `side` keeps the value of a side when the one column is one
  (`row_holds`).
- Rows: `orderedRows` holds every row of `CircuitPackage.expandedRows`,
  normalized, and no other row (`mem_orderedRows`).
- Columns: every package column below the column count that a row names has
  a system column (`Numbered.named`), with the source that `load` gives it
  (`SourceOf`). `reindex` reads that system column, as `assignment` does.

Does not own the program that reads the system (`Terminal.lean`).
-/

namespace CompressionLean.Circuit.Package

open NightstreamFPrime.Export.Package
open NightstreamFPrime.Layout
open CompressionLean.Shrink (R1csRow)
open CompressionLean.Replay (Word)
open Fin.CommRing

section Sides

variable (env : ℕ → NightstreamFPrime.Spec.F)

/-- The value of package terms. -/
def termsValue (terms : List (ℕ × NightstreamFPrime.Spec.F)) : NightstreamFPrime.Spec.F :=
  (terms.map fun t => t.2 * env t.1).sum

theorem termsValue_filter (terms : List (ℕ × NightstreamFPrime.Spec.F)) :
    termsValue env (terms.filter (·.2 ≠ 0)) = termsValue env terms := by
  induction terms with
  | nil => rfl
  | cons t rest ih =>
      by_cases zero : t.2 = 0
      · simp [zero, termsValue] at ih ⊢
        exact ih
      · simp [zero, termsValue] at ih ⊢
        rw [ih]

theorem termsValue_collect (terms : List (ℕ × NightstreamFPrime.Spec.F)) :
    termsValue env (collect terms) = termsValue env terms := by
  induction terms using collect.induct with
  | case1 x d y rest ih =>
      rw [collect.eq_1, if_pos rfl, ih]
      simp only [termsValue, List.map_cons, List.sum_cons]
      ring
  | case2 c x d y rest unequal ih =>
      rw [collect.eq_1, if_neg unequal]
      simp only [termsValue, List.map_cons, List.sum_cons] at ih ⊢
      rw [ih]
  | case3 terms short =>
      rw [collect.eq_2 terms short]

/-- Dropping zero sums and writing words keeps the value. -/
theorem words_value (terms : List (ℕ × NightstreamFPrime.Spec.F)) :
    ((terms.filterMap fun (c, x) => if x = 0 then none else some (c, Word.ofF x)).map
      fun t => env t.1 * Word.cast t.2).sum = termsValue env terms := by
  induction terms with
  | nil => rfl
  | cons t rest ih =>
      obtain ⟨c, x⟩ := t
      by_cases zero : x = 0
      · simp only [List.filterMap_cons, zero, if_true] at ih ⊢
        rw [ih]
        simp [termsValue]
      · simp only [List.filterMap_cons, zero, if_false, List.map_cons, List.sum_cons] at ih ⊢
        rw [ih, Word.cast_ofF]
        simp [termsValue, mul_comm]

/-- A side as `finish_row` writes it has the side's value when the one column
is one. -/
theorem side_value (constant : ℕ) (one : env constant = 1) (combination : R1CS.LinearCombination) :
    ((side constant combination).map fun t => env t.1 * Word.cast t.2).sum = combination.eval env := by
  unfold side
  rw [words_value, termsValue_collect, termsValue, ((List.mergeSort_perm _ _).map _).sum_eq,
    ← termsValue, termsValue_filter]
  simp [termsValue, R1CS.LinearCombination.eval, one]

/-- A row as `finish_row` writes it holds exactly when F′'s row holds, when
the one column is one. -/
theorem row_holds (constant : ℕ) (one : env constant = 1) (r : R1CS.Row) :
    R1csRow.Holds env (row constant r) ↔ r.Holds env := by
  simp only [R1csRow.Holds, row, R1CS.Row.Holds, side_value env constant one]

end Sides

section Rows

theorem foldl_push {α β : Type} (h : α → β) :
    ∀ (l : List α) (acc : Array β), (l.foldl (fun b a => b.push (h a)) acc).toList = acc.toList ++ l.map h
  | [], acc => by simp
  | a :: l, acc => by
      rw [List.foldl_cons, foldl_push h l]
      simp

/-- The producer loop of `orderedRows`: before each producer it moves the
assertions that start earlier, then it writes the producer's rows. -/
theorem scheduleFold_mem (f : R1CS.Row → R1csRow Word) (g : SparseRow → R1csRow Word)
    (rows : Producer → List R1CS.Row) (q : Producer → SparseRow → Bool) :
    ∀ (ps : List Producer) (acc : Array (R1csRow Word) × List SparseRow) (x : R1csRow Word),
      x ∈ (ps.foldl (fun b a => ((rows a).foldl (fun b r => b.push (f r))
          ((b.2.takeWhile (q a)).foldl (fun b s => b.push (g s)) b.1), b.2.dropWhile (q a))) acc).1.toList ++
        (ps.foldl (fun b a => ((rows a).foldl (fun b r => b.push (f r))
          ((b.2.takeWhile (q a)).foldl (fun b s => b.push (g s)) b.1), b.2.dropWhile (q a))) acc).2.map g ↔
      x ∈ acc.1.toList ++ acc.2.map g ∨ x ∈ (ps.flatMap rows).map f
  | [], acc, x => by simp
  | p :: ps, acc, x => by
      rw [List.foldl_cons, scheduleFold_mem f g rows q ps]
      simp only [foldl_push, List.flatMap_cons, List.map_append, List.mem_append]
      have split := congrArg (List.map g) (List.takeWhile_append_dropWhile (p := q p) (l := acc.2))
      rw [List.map_append] at split
      rw [← split]
      simp only [List.mem_append]
      tauto

theorem mem_scheduleRows (package : CircuitPackage) (r : R1CS.Row) :
    r ∈ (schedule package).flatMap (Producer.rows package) ↔
      r ∈ package.hashRows ∨ r ∈ package.permutationRows ∨ r ∈ package.compactRows ∨
        r ∈ package.instructionRows := by
  simp only [schedule, List.mem_flatMap, List.mem_mergeSort, List.mem_append, List.mem_map,
    CircuitPackage.hashRows, hashChainRows, CircuitPackage.permutationRows, CircuitPackage.compactRows,
    CircuitPackage.instructionRows, List.mem_range]
  constructor
  · rintro ⟨p, ((⟨chain, member, ordinal, below, rfl⟩ | ⟨invocation, member, rfl⟩) | ⟨invocation, member, rfl⟩) |
      ⟨instruction, member, rfl⟩, inside⟩
    · exact Or.inl ⟨chain, member, ordinal, below, by simpa [Producer.rows] using inside⟩
    · exact Or.inr (Or.inl ⟨invocation, member, inside⟩)
    · exact Or.inr (Or.inr (Or.inl ⟨invocation, member, inside⟩))
    · simp only [Producer.rows, List.mem_singleton] at inside
      exact Or.inr (Or.inr (Or.inr ⟨instruction, member, inside.symm⟩))
  · rintro (⟨chain, member, ordinal, below, inside⟩ | ⟨invocation, member, inside⟩ | ⟨invocation, member, inside⟩ |
      ⟨instruction, member, rfl⟩)
    · exact ⟨.hash chain ordinal, Or.inl (Or.inl (Or.inl ⟨chain, member, ordinal, below, rfl⟩)),
        by simpa [Producer.rows] using inside⟩
    · exact ⟨.explicit invocation, Or.inl (Or.inl (Or.inr ⟨invocation, member, rfl⟩)), inside⟩
    · exact ⟨.compact invocation, Or.inl (Or.inr ⟨invocation, member, rfl⟩), inside⟩
    · exact ⟨.generic instruction, Or.inr ⟨instruction, member, rfl⟩, by simp [Producer.rows]⟩

/-- **The ordered rows are F′'s rows**, each written as `finish_row` writes
it: the expanded rows of the package, in another order. -/
theorem mem_orderedRows (package : CircuitPackage) (x : R1csRow Word) :
    x ∈ orderedRows package ↔ ∃ r ∈ package.expandedRows, x = row package.layout.constantColumn r := by
  have ordered : x ∈ orderedRows package ↔
      x ∈ ((schedule package).flatMap (Producer.rows package)).map (row package.layout.constantColumn) ∨
        x ∈ package.assertionRows.map fun a => row package.layout.constantColumn a.toR1CS := by
    unfold orderedRows
    simp only [Id.run, List.forIn_pure_yield_eq_foldl, pure_bind]
    have fold := scheduleFold_mem (row package.layout.constantColumn)
      (fun a => row package.layout.constantColumn a.toR1CS) (Producer.rows package)
      (fun a x => decide (x.rowIndex < Producer.rowStart package a)) (schedule package)
      (#[], package.assertionRows) x
    show x ∈ (List.foldl (fun (b : Array (R1csRow Word)) (a : SparseRow) =>
      b.push (row package.layout.constantColumn a.toR1CS)) _ _) ↔ _
    rw [← Array.mem_toList_iff, foldl_push, fold]
    simp only [List.mem_append, List.not_mem_nil, false_or]
    tauto
  rw [ordered]
  simp only [List.mem_map, CircuitPackage.expandedRows, List.mem_append, mem_scheduleRows,
    CircuitPackage.assertionR1CSRows]
  constructor
  · rintro (⟨r, member, rfl⟩ | ⟨a, member, rfl⟩)
    · exact ⟨r, by tauto, rfl⟩
    · exact ⟨a.toR1CS, Or.inr ⟨a, member, rfl⟩, rfl⟩
  · rintro ⟨r, ((((hash | permutation) | compact) | instruction) | ⟨a, member, rfl⟩), rfl⟩
    · exact Or.inl ⟨r, Or.inl hash, rfl⟩
    · exact Or.inl ⟨r, Or.inr (Or.inl permutation), rfl⟩
    · exact Or.inl ⟨r, Or.inr (Or.inr (Or.inl compact)), rfl⟩
    · exact Or.inl ⟨r, Or.inr (Or.inr (Or.inr instruction)), rfl⟩
    · exact Or.inr ⟨a, member, rfl⟩

end Rows

section Named

theorem mark_size (used : Array Bool) (terms : List (ℕ × Word)) : (mark used terms).size = used.size := by
  induction terms generalizing used with
  | nil => rfl
  | cons t rest ih => rw [mark, List.foldl_cons, ← mark, ih]; simp [Array.set!]

theorem mark_mono (used : Array Bool) (terms : List (ℕ × Word)) (c : ℕ) (marked : used[c]? = some true) :
    (mark used terms)[c]? = some true := by
  induction terms generalizing used with
  | nil => exact marked
  | cons t rest ih =>
      rw [mark, List.foldl_cons, ← mark]
      apply ih
      simp only [Array.set!, Array.getElem?_setIfInBounds]
      split_ifs with h1 h2
      · rfl
      · subst h1; rw [Array.getElem?_eq_none (by omega)] at marked; cases marked
      · exact marked

theorem mark_sets (used : Array Bool) (terms : List (ℕ × Word)) (t : ℕ × Word) (member : t ∈ terms)
    (lt : t.1 < used.size) : (mark used terms)[t.1]? = some true := by
  induction terms generalizing used with
  | nil => cases member
  | cons u rest ih =>
      rw [mark, List.foldl_cons, ← mark]
      rcases List.mem_cons.mp member with rfl | inside
      · apply mark_mono
        simp [Array.set!, lt]
      · exact ih _ inside (by simp [Array.set!, lt])

/-- Mark the columns of one row. -/
abbrev markRow (used : Array Bool) (r : R1csRow Word) : Array Bool := mark (mark (mark used r.a) r.b) r.c

theorem markRows_size (rows : List (R1csRow Word)) (used : Array Bool) :
    (rows.foldl markRow used).size = used.size := by
  induction rows generalizing used with
  | nil => rfl
  | cons r rest ih => rw [List.foldl_cons, ih, mark_size, mark_size, mark_size]

theorem markRows_mono (rows : List (R1csRow Word)) (used : Array Bool) (c : ℕ) (marked : used[c]? = some true) :
    (rows.foldl markRow used)[c]? = some true := by
  induction rows generalizing used with
  | nil => exact marked
  | cons r rest ih => exact ih _ (mark_mono _ _ c (mark_mono _ _ c (mark_mono _ _ c marked)))

/-- A column below `total` that some row names is named. -/
theorem named_mem (total : ℕ) (rows : Array (R1csRow Word)) (r : R1csRow Word) (member : r ∈ rows)
    (t : ℕ × Word) (side : t ∈ r.a ∨ t ∈ r.b ∨ t ∈ r.c) (lt : t.1 < total) :
    (named total rows)[t.1]? = some true := by
  unfold named
  rw [← Array.foldl_toList]
  have marks : ∀ (l : List (R1csRow Word)) (used : Array Bool), r ∈ l → t.1 < used.size →
      (l.foldl markRow used)[t.1]? = some true := by
    intro l
    induction l with
    | nil => intro used inside; cases inside
    | cons u rest ih =>
        intro used inside lt
        rw [List.foldl_cons]
        rcases List.mem_cons.mp inside with rfl | inside
        · apply markRows_mono
          rcases side with a | b | c
          · exact mark_mono _ _ _ (mark_mono _ _ _ (mark_sets _ _ t a lt))
          · exact mark_mono _ _ _ (mark_sets _ _ t b (by rw [mark_size]; exact lt))
          · exact mark_sets _ _ t c (by rw [mark_size, mark_size]; exact lt)
        · exact ih _ inside (by rw [mark_size, mark_size, mark_size]; exact lt)
  exact marks _ _ (Array.mem_toList_iff.mpr member) (by simp [lt])

end Named

section Numbering

/-- The source that `load` gives package column `c`. -/
def SourceOf (layout : PhysicalLayout) (segments : List (Segment × ℕ)) (c : ℕ) : Source → Prop
  | .one => c = layout.privateColumnCount
  | .pub i => layout.privateColumnCount < c ∧ i = c - layout.privateColumnCount - 1
  | .priv i => c < layout.privateColumnCount ∧ privateInput segments c = some i
  | .computed _ => c < layout.privateColumnCount ∧ privateInput segments c = none

/-- What the numbering keeps once the columns below `j` are numbered: every
system column has the source of its package column, every named column below
`j` has a system column, and every computed index names a computed value. -/
structure Numbered (layout : PhysicalLayout) (segments : List (Segment × ℕ)) (used : Array Bool) (j : ℕ)
    (state : Numbering) : Prop where
  size : state.1.size = layout.totalColumnCount
  source : ∀ c k, state.1[c]? = some (some k) → ∃ h : k < state.2.1.size, SourceOf layout segments c state.2.1[k]
  named : ∀ c < j, used[c]! = true → ∃ k, state.1[c]? = some (some k)
  computed : ∀ k (h : k < state.2.1.size) i, state.2.1[k] = .computed i → i < state.2.2.size

theorem numbered_push {layout : PhysicalLayout} {segments : List (Segment × ℕ)} {used : Array Bool} {j : ℕ}
    {index : Array (Option ℕ)} {sources : Array Source} {computed computed' : Array ℕ} {src : Source}
    (numbered : Numbered layout segments used j (index, sources, computed)) (lt : j < layout.totalColumnCount)
    (source : SourceOf layout segments j src) (grows : computed.size ≤ computed'.size)
    (fresh : ∀ i, src = .computed i → i < computed'.size) :
    Numbered layout segments used (j + 1) (index.set! j (some sources.size), sources.push src, computed') := by
  have jlt : j < index.size := by rw [numbered.size]; exact lt
  refine ⟨by simp [Array.set!, numbered.size], ?_, ?_, ?_⟩
  · intro c k h
    simp only [Array.set!, Array.getElem?_setIfInBounds] at h
    by_cases same : j = c
    · subst same
      simp [jlt] at h
      subst h
      refine ⟨by simp, ?_⟩
      simpa using source
    · rw [if_neg same] at h
      obtain ⟨klt, sk⟩ := numbered.source c k h
      refine ⟨by simp only [Array.size_push]; exact Nat.lt_succ_of_lt klt, ?_⟩
      dsimp only at klt sk ⊢
      rw [Array.getElem_push_lt klt]
      exact sk
  · intro c clt usedc
    by_cases same : c = j
    · subst same
      exact ⟨sources.size, by simp [Array.set!, jlt]⟩
    · obtain ⟨k, hk⟩ := numbered.named c (by omega) usedc
      exact ⟨k, by simp only [Array.set!, Array.getElem?_setIfInBounds]; rw [if_neg (Ne.symm same)]; exact hk⟩
  · intro k klt i hi
    simp only [Array.size_push] at klt
    by_cases old : k < sources.size
    · rw [Array.getElem_push_lt old] at hi
      exact lt_of_lt_of_le (numbered.computed k old i hi) grows
    · have : k = sources.size := by omega
      subst this
      rw [Array.getElem_push_eq] at hi
      exact fresh i hi

theorem numberColumn_numbered {layout : PhysicalLayout} {segments : List (Segment × ℕ)} {used : Array Bool}
    {j : ℕ} {state : Numbering} (numbered : Numbered layout segments used j state)
    (lt : j < layout.totalColumnCount) :
    Numbered layout segments used (j + 1) (numberColumn layout segments used state j) := by
  obtain ⟨index, sources, computed⟩ := state
  unfold numberColumn
  dsimp only
  by_cases usedj : used[j]! = true
  · rw [if_pos usedj]
    by_cases one : j = layout.privateColumnCount
    · rw [if_pos one]
      exact numbered_push numbered lt one le_rfl (by simp)
    · rw [if_neg one]
      by_cases pub : j > layout.privateColumnCount
      · rw [if_pos pub]
        exact numbered_push numbered lt ⟨pub, rfl⟩ le_rfl (by simp)
      · rw [if_neg pub]
        have below : j < layout.privateColumnCount := by omega
        cases input : privateInput segments j with
        | some i => exact numbered_push numbered lt ⟨below, input⟩ le_rfl (by simp)
        | none =>
            exact numbered_push numbered lt ⟨below, input⟩ (by simp) (by
              intro i hi
              cases hi
              simp)
  · rw [if_neg usedj]
    refine ⟨numbered.size, numbered.source, ?_, numbered.computed⟩
    intro c clt usedc
    by_cases same : c = j
    · subst same; exact absurd usedc usedj
    · exact numbered.named c (by omega) usedc

theorem number_numbered (layout : PhysicalLayout) (segments : List (Segment × ℕ)) (used : Array Bool) :
    Numbered layout segments used layout.totalColumnCount (number layout segments used) := by
  unfold number
  simp only [Id.run, Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl, pure_bind]
  have size : [0:layout.totalColumnCount].size = layout.totalColumnCount := by simp [Std.Legacy.Range.size]
  rw [size]
  suffices ∀ j ≤ layout.totalColumnCount, Numbered layout segments used j
      ((List.range' 0 j).foldl (numberColumn layout segments used)
        (Array.replicate layout.totalColumnCount none, #[], #[])) from this _ le_rfl
  intro j
  induction j with
  | zero =>
      intro _
      refine ⟨by simp, ?_, fun c clt => absurd clt (Nat.not_lt_zero c), fun k klt => absurd klt (by simp)⟩
      intro c k h
      simp [Array.getElem?_replicate] at h
  | succ j ih =>
      intro le
      rw [List.range'_concat, List.foldl_append]
      simp only [List.foldl_cons, List.foldl_nil, zero_add, one_mul]
      exact numberColumn_numbered (ih (by omega)) (by omega)

/-- `load`'s numbering of a package's columns. -/
def numbering (package : CircuitPackage) : Numbering :=
  number package.layout (callerSegments package.layout)
    (named package.layout.totalColumnCount (orderedRows package))

theorem numbering_numbered (package : CircuitPackage) :
    Numbered package.layout (callerSegments package.layout)
      (named package.layout.totalColumnCount (orderedRows package))
      package.layout.totalColumnCount (numbering package) :=
  number_numbered _ _ _

/-- Every package column below the column count that an ordered row names has
a system column, and no column past the count does. -/
theorem numbering_named (package : CircuitPackage) (x : R1csRow Word) (member : x ∈ orderedRows package)
    (t : ℕ × Word) (side : t ∈ x.a ∨ t ∈ x.b ∨ t ∈ x.c) : (numbering package).1[t.1]? ≠ some none := by
  have numbered := numbering_numbered package
  by_cases lt : t.1 < package.layout.totalColumnCount
  · have marked := named_mem _ _ x member t side lt
    obtain ⟨k, hk⟩ := numbered.named t.1 lt (by
      rw [Array.getElem!_eq_getD, Array.getD_eq_getD_getElem?, marked]
      rfl)
    rw [hk]
    simp
  · rw [Array.getElem?_eq_none (by rw [numbered.size]; omega)]
    simp

end Numbering

section Load

theorem mapM_ok {α β ε : Type} (f : α → Except ε β) :
    ∀ (l : List α) (ys : List β), l.mapM f = .ok ys →
      ys.length = l.length ∧ ∀ i (hi : i < l.length), ∃ y, ys[i]? = some y ∧ f l[i] = .ok y
  | [], ys, h => by
      simp only [List.mapM_nil] at h
      cases h
      simp
  | a :: l, ys, h => by
      rw [List.mapM_cons] at h
      cases fa : f a with
      | error e => rw [fa] at h; cases h
      | ok y =>
          rw [fa] at h
          cases rest : l.mapM f with
          | error e => rw [rest] at h; cases h
          | ok ys' =>
              rw [rest] at h
              cases h
              obtain ⟨length, each⟩ := mapM_ok f l ys' rest
              refine ⟨by simp [length], ?_⟩
              intro i hi
              cases i with
              | zero => exact ⟨y, rfl, fa⟩
              | succ i => exact each i (by simpa using hi)

theorem map_ok {α β ε : Type} (g : α → β) (m : Except ε α) (b : β) (h : g <$> m = .ok b) :
    ∃ a, m = .ok a ∧ b = g a := by
  cases m with
  | error e => cases h
  | ok a => exact ⟨a, rfl, by cases h; rfl⟩

/-- An output list that `load` looks up: one system column per entry, each the
system column of a package column below the column count. -/
theorem outputColumns_ok {layout : PhysicalLayout} {index : Array (Option ℕ)}
    {list : List NightstreamFPrime.Circuit.Expr} {ks : Array ℕ} (h : outputColumns layout index list = .ok ks) :
    ks.size = list.length ∧ ∀ i (hi : i < list.length), ∃ c k, list[i] = .var c ∧
      c < layout.totalColumnCount ∧ index[c]! = some k ∧ ks[i]? = some k := by
  unfold outputColumns at h
  rw [Array.mapM_eq_mapM_toList, List.toList_toArray] at h
  obtain ⟨ys, mapped, rfl⟩ := map_ok _ _ _ h
  obtain ⟨length, each⟩ := mapM_ok _ list ys mapped
  refine ⟨by simp [length], ?_⟩
  intro i hi
  obtain ⟨y, hy, fy⟩ := each i hi
  revert fy
  generalize list[i] = e
  intro fy
  cases e with
  | var c =>
      simp only at fy
      split_ifs at fy with bounds
      · cases found : index[c]! with
        | none => rw [found] at fy; cases fy
        | some k =>
            rw [found] at fy
            cases fy
            exact ⟨c, y, rfl, bounds.1, found, by simpa using hy⟩
  | const _ => cases fy
  | add _ _ => cases fy
  | mul _ _ => cases fy

/-- What `load` gives when it succeeds: the ordered rows on system columns,
the numbering's sources and computed columns, and each output list looked up
in the numbering. -/
theorem load_ok {package : CircuitPackage} {map : Terminal.Outputs} {loaded : Loaded}
    (h : load package map = .ok loaded) :
    loaded.system = (orderedRows package).map (fun r =>
      ⟨reindex (numbering package).1 r.a, reindex (numbering package).1 r.b, reindex (numbering package).1 r.c⟩) ∧
    loaded.sources = (numbering package).2.1 ∧ loaded.computed = (numbering package).2.2 ∧
    outputColumns package.layout (numbering package).1 map.commitment = .ok loaded.outputs.commitment ∧
    outputColumns package.layout (numbering package).1 map.publicInput = .ok loaded.outputs.publicInput ∧
    outputColumns package.layout (numbering package).1 map.point = .ok loaded.outputs.point ∧
    outputColumns package.layout (numbering package).1 map.eval_K = .ok loaded.outputs.evalK ∧
    outputColumns package.layout (numbering package).1 map.eval_A = .ok loaded.outputs.evalA ∧
    outputColumns package.layout (numbering package).1 map.state = .ok loaded.outputs.state := by
  unfold load at h
  simp only [bind, Except.bind] at h
  split_ifs at h
  split at h
  · cases h
  rename_i commitment hc
  split at h
  · cases h
  rename_i publicInput hp
  split at h
  · cases h
  rename_i point hpt
  split at h
  · cases h
  rename_i evalK hk
  split at h
  · cases h
  rename_i evalA ha
  split at h
  · cases h
  rename_i state hs
  cases h
  exact ⟨rfl, rfl, rfl, hc, hp, hpt, hk, ha, hs⟩

end Load

section System

/-- The package assignment of system values `value`: a column with a system
column takes that column's value, read where `reindex` reads it, and a column
below the column count that no row names takes `rest`. -/
def assignment (index : Array (Option ℕ)) (value rest : ℕ → NightstreamFPrime.Spec.F) (c : ℕ) :
    NightstreamFPrime.Spec.F :=
  if index[c]? = some none then rest c else value ((index[c]!).getD 0)

theorem reindex_value (index : Array (Option ℕ)) (value rest : ℕ → NightstreamFPrime.Spec.F)
    (terms : List (ℕ × Word)) (named : ∀ t ∈ terms, index[t.1]? ≠ some none) :
    ((reindex index terms).map fun t => value t.1 * Word.cast t.2).sum =
      (terms.map fun t => assignment index value rest t.1 * Word.cast t.2).sum := by
  unfold reindex
  rw [List.map_map]
  congr 1
  apply List.map_congr_left
  intro t member
  simp only [Function.comp_apply, assignment, if_neg (named t member)]

/-- A row on system columns holds on the system values exactly when the row
holds on the package assignment, when every column it names has a system
column or lies past the column count. -/
theorem reindex_holds (index : Array (Option ℕ)) (value rest : ℕ → NightstreamFPrime.Spec.F) (r : R1csRow Word)
    (named : ∀ t, t ∈ r.a ∨ t ∈ r.b ∨ t ∈ r.c → index[t.1]? ≠ some none) :
    R1csRow.Holds value ⟨reindex index r.a, reindex index r.b, reindex index r.c⟩ ↔
      R1csRow.Holds (assignment index value rest) r := by
  simp only [R1csRow.Holds]
  rw [reindex_value index value rest r.a fun t m => named t (Or.inl m),
    reindex_value index value rest r.b fun t m => named t (Or.inr (Or.inl m)),
    reindex_value index value rest r.c fun t m => named t (Or.inr (Or.inr m))]

/-- **The package-rows lemma.** If `load` gives `loaded` and every compact
invocation of the package names a template, then the rows of `loaded.system`
hold on the system values `value`, with the one column at one, exactly when
F′'s rows hold on the package assignment of `value`. -/
theorem system_holds_iff {package : CircuitPackage} {map : Terminal.Outputs} {loaded : Loaded}
    (loads : load package map = .ok loaded) (valid : package.CompactTemplatesValid)
    (value rest : ℕ → NightstreamFPrime.Spec.F)
    (one : assignment (numbering package).1 value rest package.layout.constantColumn = 1) :
    (∀ r ∈ loaded.system, R1csRow.Holds value r) ↔
      package.RowsHold (assignment (numbering package).1 value rest) := by
  obtain ⟨system, -⟩ := load_ok loads
  rw [rowsHold_iff_expandedRows]
  constructor
  · intro holds
    refine ⟨valid, fun r member => ?_⟩
    have ordered := (mem_orderedRows package _).mpr ⟨r, member, rfl⟩
    have this := holds _ (by rw [system]; exact Array.mem_map_of_mem ordered)
    rw [reindex_holds _ value rest _ (numbering_named package _ ordered)] at this
    exact (row_holds _ _ one r).mp this
  · rintro ⟨-, holds⟩ r' member
    rw [system] at member
    obtain ⟨x, ordered, rfl⟩ := Array.mem_map.mp member
    obtain ⟨r, inside, rfl⟩ := (mem_orderedRows package x).mp ordered
    rw [reindex_holds _ value rest _ (numbering_named package _ ordered), row_holds _ _ one r]
    exact holds r inside

end System

end CompressionLean.Circuit.Package
