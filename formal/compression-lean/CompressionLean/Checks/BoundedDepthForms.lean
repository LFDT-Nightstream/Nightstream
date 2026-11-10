import CompressionLean.Spec.BoundedDepthMatrix

/-! The fast forms of `Spec.BoundedDepth.keyRunsFailure` are the key layout's forms (CW-6.5). Where the
check confirms a column's slot, `fastForm` is the column's form under `keyLayout` (`fastForm_eq`),
so a row that reads only confirmed columns compiles to the key relation's row
(`compileRow_fast`, `combination_fast`). -/

namespace CompressionLean.Checks.BoundedDepth

open CompressionLean.Spec.BoundedDepth
open NightstreamFPrime NightstreamFPrime.Spec NightstreamFPrime.Layout NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Export NightstreamFPrime.Export.Generic NightstreamFPrime.Export.BoundedDepth

/-! ### Widening -/

theorem widenForm_add {L L' : ℕ} (le : L ≤ L') (f g : SparseForm L) :
    widenForm le (.add f g) = .add (widenForm le f) (widenForm le g) := by
  simp [widenForm, SparseForm.add]

theorem widenForm_scale {L L' : ℕ} (le : L ≤ L') (k : F) (f : SparseForm L) :
    widenForm le (.scale k f) = .scale k (widenForm le f) := by
  simp [widenForm, SparseForm.scale]

theorem widenForm_recompose {L L' : ℕ} (le : L ≤ L') : ∀ forms : List (SparseForm L),
    widenForm le (RetainedSlot.recomposeForms forms) = RetainedSlot.recomposeForms (forms.map (widenForm le))
  | [] => rfl
  | head :: tail => by
      simp only [RetainedSlot.recomposeForms, List.map_cons, widenForm_add, widenForm_scale,
        widenForm_recompose le tail]

/-! ### The slot table -/

variable {width : ℕ}

theorem sources_lt {slots : Array (LowNormAssignment.Slot width)} (increase : sourcesIncrease slots = true)
    {i j : ℕ} (before : i < j) (inside : j < slots.size) : slots[i].source.val < slots[j].source.val := by
  have step : ∀ k (next : k + 1 < slots.size), slots[k].source.val < slots[k + 1].source.val := by
    intro k next
    have := List.all_eq_true.mp increase k (List.mem_range.mpr (by omega))
    simpa only [Array.getElem?_eq_getElem (show k < slots.size by omega), Array.getElem?_eq_getElem next,
      decide_eq_true_eq] using this
  induction j with
  | zero => omega
  | succ j ih =>
      rcases Nat.lt_succ_iff_lt_or_eq.mp before with lt | rfl
      · exact (ih lt (by omega) step).trans (step j inside)
      · exact step i inside

theorem slotAt_found {slots : Array (LowNormAssignment.Slot width)} {c k : ℕ}
    (found : slotAt slots c = some (some k)) : ∃ inside : k < slots.size, slots[k].source.val = c := by
  unfold slotAt at found
  dsimp only at found
  split at found
  · rename_i slot reads
    by_cases same : slot.source.val = c
    · rw [ite_eq_left same] at found
      obtain rfl := Option.some.inj (Option.some.inj found)
      obtain ⟨inside, rfl⟩ := Array.getElem?_eq_some_iff.mp reads
      exact ⟨inside, same⟩
    · rw [ite_eq_right same] at found
      split_ifs at found
      simp at found
  · split_ifs at found
    simp at found

theorem belowAt_lt {slots : Array (LowNormAssignment.Slot width)} (increase : sourcesIncrease slots = true)
    {c m : ℕ} (bracket : belowAt slots c m = true) {k : ℕ} (inside : k < slots.size) (before : k < m) :
    slots[k].source.val < c := by
  obtain ⟨m', rfl⟩ : ∃ m', m = m' + 1 := ⟨m - 1, by omega⟩
  simp only [belowAt] at bracket
  split at bracket
  · rename_i slot reads
    obtain ⟨inside', rfl⟩ := Array.getElem?_eq_some_iff.mp reads
    have last := of_decide_eq_true bracket
    rcases Nat.lt_succ_iff_lt_or_eq.mp before with lt | rfl
    · exact (sources_lt increase lt inside').trans last
    · exact last
  · simp at bracket

theorem slotAt_absent {slots : Array (LowNormAssignment.Slot width)} (increase : sourcesIncrease slots = true)
    {c : ℕ} (absent : slotAt slots c = some none) : ∀ k (inside : k < slots.size), slots[k].source.val ≠ c := by
  unfold slotAt at absent
  dsimp only at absent
  generalize lowerBound slots c = m at absent
  intro k inside same
  split at absent
  · rename_i slot reads
    obtain ⟨atM, rfl⟩ := Array.getElem?_eq_some_iff.mp reads
    by_cases hit : slots[m].source.val = c
    · rw [ite_eq_left hit] at absent
      simp at absent
    rw [ite_eq_right hit] at absent
    by_cases bracket : (belowAt slots c m && decide (c < slots[m].source.val)) = true
    · simp only [Bool.and_eq_true, decide_eq_true_eq] at bracket
      rcases lt_trichotomy k m with lt | rfl | gt
      · exact absurd same (Nat.ne_of_lt (belowAt_lt increase bracket.1 inside lt))
      · exact hit same
      · exact absurd same (Nat.ne_of_gt (lt_trans bracket.2 (sources_lt increase gt inside)))
    · rw [ite_eq_right bracket] at absent
      simp at absent
  · by_cases bracket : (belowAt slots c m && decide (m = slots.size)) = true
    · simp only [Bool.and_eq_true, decide_eq_true_eq] at bracket
      exact absurd same (Nat.ne_of_lt (belowAt_lt increase bracket.1 inside (by omega)))
    · rw [ite_eq_right bracket] at absent
      simp at absent

theorem slotIndex_found {slots : List (LowNormAssignment.Slot width)}
    (increase : sourcesIncrease slots.toArray = true) {k : ℕ} (inside : k < slots.length) {column : Fin width}
    (reads : slots[k].source = column) : SlotLayout.slotIndex slots column = some ⟨k, inside⟩ := by
  unfold SlotLayout.slotIndex
  obtain ⟨j, found⟩ := Option.isSome_iff_exists.mp
    ((List.find?_isSome (p := fun index : Fin slots.length => decide ((slots.get index).source = column))).mpr
      ⟨⟨k, inside⟩, List.mem_finRange _, by simpa using reads⟩)
  have hit : slots[j.val].source = column := by simpa using List.find?_some found
  rw [found]
  congr
  apply Fin.ext
  by_contra other
  have sorted := fun {i j : ℕ} (lt : i < j) (inside : j < slots.length) =>
    sources_lt (slots := slots.toArray) increase lt (by simpa using inside)
  simp only [List.getElem_toArray] at sorted
  rcases Nat.lt_or_gt_of_ne other with lt | gt
  · exact absurd (congrArg Fin.val (hit.trans reads.symm)) (Nat.ne_of_lt (sorted lt inside))
  · exact absurd (congrArg Fin.val (reads.trans hit.symm)) (Nat.ne_of_lt (sorted gt j.isLt))

theorem slotIndex_absent {slots : List (LowNormAssignment.Slot width)} {column : Fin width}
    (absent : ∀ k (inside : k < slots.length), slots[k].source ≠ column) :
    SlotLayout.slotIndex slots column = none := by
  unfold SlotLayout.slotIndex
  exact List.find?_eq_none.mpr fun index _ => by simpa using absent index.val index.isLt

theorem offsets_getElem? (slots : List (LowNormAssignment.Slot width)) {k : ℕ} (inside : k < slots.length) :
    (tableOf slots).offsets[k]? = some (LowNormAssignment.prefixWidth slots ⟨k, inside⟩) := by
  simp only [tableOf, List.getElem?_toArray, List.getElem?_scanl, LowNormAssignment.prefixWidth,
    LowNormAssignment.logicalWidth]
  rw [ite_eq_left (by simp; omega), List.map_take, List.sum_eq_foldl]

/-! ### Forms -/

theorem fastForm_eq {p : StepPackage.Package}
    (fits : ProductionAssignment.logicalWidth (slotsOf p) ≤ StepRelation.keyWidth)
    (increase : sourcesIncrease (tableOf (slotsOf p)).slots = true) {c : ℕ} (inside : c < p.width)
    {f : SparseForm StepRelation.keyWidth} (computed : fastForm (tableOf (slotsOf p)) c = some f) :
    f = (keyLayout p fits).map.form ⟨c, inside⟩ := by
  simp only [keyLayout, widen, SlotLayout.layout]
  unfold fastForm at computed
  split_ifs at computed with inPublic
  · obtain rfl := Option.some.inj computed.symm
    rw [dite_eq_left inPublic]
    rfl
  · rw [dite_eq_right inPublic]
    split at computed
    · rename_i k found
      obtain ⟨atK, reads⟩ := slotAt_found found
      split at computed
      · rename_i slot offset hslot hoffset
        split_ifs at computed with room
        obtain rfl := Option.some.inj computed.symm
        have inK : k < (slotsOf p).length := by simpa [tableOf] using atK
        obtain rfl : slot = (slotsOf p)[k] := by simpa [tableOf, List.getElem?_eq_getElem inK] using hslot.symm
        rw [offsets_getElem? _ inK, Option.some.injEq] at hoffset
        subst hoffset
        rw [slotIndex_found increase inK (column := ⟨c, inside⟩) (Fin.ext (by simpa [tableOf] using reads))]
        simp only [slotForm, RetainedSlot.form, widenForm_recompose, RetainedSlot.coordinateForms, List.map_ofFn]
        congr 1
        apply congrArg List.ofFn
        funext coordinate
        show SparseForm.singleton _ 1 = widenForm fits (SparseForm.singleton _ 1)
        simp only [widenForm, SparseForm.singleton, List.map_cons, List.map_nil]
        congr 3
        apply Fin.ext
        simp only [Fin.val_castLE, ProductionAssignment.privateColumn, LowNormAssignment.coordinateIndex]
        unfold LogicalPlan.publicCount ProductionAssignment.publicWidth
        omega
      · simp at computed
    · obtain rfl := Option.some.inj computed.symm
      rw [slotIndex_absent fun k inK same => slotAt_absent increase (by assumption) k (by simpa [tableOf] using inK)
        (by simpa [tableOf] using congrArg Fin.val same)]
      rfl
    · simp at computed

theorem compileTerms_congr {L : ℕ} {left right : SourceCompiler.SourceMap width L} :
    ∀ (terms : List (ℕ × F)) (leftBounded rightBounded),
      (∀ term ∈ terms, ∀ inside : term.1 < width, left.form ⟨term.1, inside⟩ = right.form ⟨term.1, inside⟩) →
      SourceCompiler.compileTerms left terms leftBounded = SourceCompiler.compileTerms right terms rightBounded
  | [], _, _, _ => rfl
  | term :: rest, leftBounded, rightBounded, same => by
      simp only [SourceCompiler.compileTerms]
      rw [same term (by simp) _, compileTerms_congr rest _ _ fun t member => same t (by simp [member])]

end CompressionLean.Checks.BoundedDepth
