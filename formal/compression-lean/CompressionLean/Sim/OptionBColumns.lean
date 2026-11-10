import CompressionLean.Spec.OptionBRuns

/-! The finish columns of option B's final program (`final_program.rs::Finish::new`,
`FinalProgram::run`). A column is live exactly when it is below the S-box columns and the program
reads it (`live_iff`, `Spec.OptionB.readColumns`). The form of a live column is a cell of the hint,
and the form of each other column is the constant 0 (`columnForms_spec`). -/

namespace CompressionLean.Sim.OptionB

open CompressionLean.Circuit CompressionLean.OptionB
open CompressionLean.Replay (Word)
open CompressionLean.Spec.OptionB
open NightstreamFPrime.Export.OptionB.StepPackage (Package PackagePiece)

/-! ### Live columns -/

theorem mark_fst {β : Type} (live : Array Bool) (terms : List (ℕ × β)) :
    mark Prod.fst live terms = mark id live (terms.map Prod.fst) := by
  simp only [mark, List.foldl_map, id]

theorem mark_append (live : Array Bool) (xs ys : List ℕ) : mark id (mark id live xs) ys = mark id live (xs ++ ys) := by
  simp only [mark, List.foldl_append]

theorem foldl_mark {δ : Type} (g : δ → List ℕ) (xs : List δ) (live : Array Bool) :
    xs.foldl (fun live x => mark id live (g x)) live = mark id live (xs.flatMap g) := by
  simp only [mark, List.foldl_flatMap]

theorem mark_live (columns : List ℕ) (live : Array Bool) (k : ℕ) :
    (mark id live columns).getD k false = true ↔ live.getD k false = true ∨ (k ∈ columns ∧ k < live.size) := by
  induction columns generalizing live with
  | nil => simp [mark]
  | cons c rest ih =>
      rw [show mark id live (c :: rest) = mark id (live.set! c true) rest from rfl, ih]
      simp only [Array.set!_eq_setIfInBounds, Array.size_setIfInBounds, List.mem_cons,
        Array.getD_eq_getD_getElem?, Array.getElem?_setIfInBounds]
      by_cases same : c = k
      · subst same
        by_cases inside : c < live.size <;> simp [inside]
      · simp [same, Ne.symm same]

theorem mark_size {α : Type} (column : α → ℕ) (terms : List α) (live : Array Bool) :
    (mark column live terms).size = live.size := by
  induction terms generalizing live with
  | nil => rfl
  | cons t rest ih =>
      rw [show mark column live (t :: rest) = mark column (live.set! (column t) true) rest from rfl, ih]
      simp

theorem replicate_dead (n k : ℕ) : (Array.replicate n false).getD k false = false := by
  rw [Array.getD_eq_getD_getElem?, Array.getElem?_replicate]
  split <;> rfl

theorem liveColumns_live (rows : List (Shrink.R1csRow Word)) (blocks : List NightstreamFPrime.Export.Generic.Frame.Block)
    (witnessWords sboxStart k : ℕ) :
    (liveColumns rows blocks witnessWords sboxStart).getD k false = true ↔ k < sboxStart ∧
      k ∈ rows.flatMap (fun row => (row.a ++ row.b ++ row.c).map Prod.fst) ++
        blocks.flatMap (fun block =>
          block.lanes.flatMap (fun lane => ((linearForm lane).getD (0, [])).2.map Prod.fst) ++
            List.range' (block.start + 1080) 16) ++ programReads witnessWords := by
  simp only [liveColumns, mark_fst, mark_append, foldl_mark, mark_live, Array.size_replicate, replicate_dead]
  simp [and_comm]

theorem live_iff (finish : Package) (k : ℕ) :
    (live finish).getD k false = true ↔ k < finish.sboxStart ∧ k ∈ readColumns finish :=
  liveColumns_live _ _ _ _ k

theorem live_size (finish : Package) : (live finish).size = finish.sboxStart := by
  simp only [live, liveColumns, mark_fst, mark_append, foldl_mark, mark_size, Array.size_replicate]

/-! ### Column forms -/

/-- The step of `columnForms`. -/
private def formStep (zeroForm : ℕ) (state : Array ℕ × ℕ) (flag : Bool) : Array ℕ × ℕ :=
  if flag then (state.1.push state.2, state.2 + 1) else (state.1.push zeroForm, state.2)

private theorem getD_push_lt (forms : Array ℕ) (form zeroForm : ℕ) {k : ℕ} (inside : k < forms.size) :
    (forms.push form).getD k zeroForm = forms.getD k zeroForm := by
  simp [Array.getD_eq_getD_getElem?, Array.getElem?_push, Nat.ne_of_lt inside]

private theorem getD_push_size (forms : Array ℕ) (form zeroForm : ℕ) :
    (forms.push form).getD forms.size zeroForm = form := by
  simp [Array.getD_eq_getD_getElem?]

private theorem formsFold (zeroForm : ℕ) : ∀ (flags : List Bool) (forms : Array ℕ) (next : ℕ),
    (flags.foldl (formStep zeroForm) (forms, next)).1.size = forms.size + flags.length ∧
    (∀ k < forms.size,
      (flags.foldl (formStep zeroForm) (forms, next)).1.getD k zeroForm = forms.getD k zeroForm) ∧
    ∀ j (inside : j < flags.length),
      (flags[j] = true →
        next ≤ (flags.foldl (formStep zeroForm) (forms, next)).1.getD (forms.size + j) zeroForm ∧
          (flags.foldl (formStep zeroForm) (forms, next)).1.getD (forms.size + j) zeroForm <
            next + flags.count true) ∧
      (flags[j] = false →
        (flags.foldl (formStep zeroForm) (forms, next)).1.getD (forms.size + j) zeroForm = zeroForm)
  | [], forms, next => ⟨rfl, fun _ _ => rfl, fun _ inside => absurd inside (Nat.not_lt_zero _)⟩
  | flag :: rest, forms, next => by
      have run : ∀ form next', formStep zeroForm (forms, next) flag = (forms.push form, next') →
          ((flag :: rest).foldl (formStep zeroForm) (forms, next)).1 =
            (rest.foldl (formStep zeroForm) (forms.push form, next')).1 := fun form next' step => by
        rw [List.foldl_cons, step]
      cases flag with
      | true =>
          obtain ⟨size, below, each⟩ := formsFold zeroForm rest (forms.push next) (next + 1)
          rw [run next (next + 1) rfl]
          have first := below forms.size (by simp)
          rw [getD_push_size] at first
          refine ⟨by rw [size, Array.size_push, List.length_cons]; omega, fun k inside => ?_, fun j inside => ?_⟩
          · rw [below k (by rw [Array.size_push]; omega), getD_push_lt _ _ _ inside]
          · cases j with
            | zero =>
                refine ⟨fun _ => ?_, fun dead => absurd dead (by simp)⟩
                simp only [Nat.add_zero, first, List.count_cons_self]
                omega
            | succ j =>
                have step := each j (by simpa using inside)
                rw [Array.size_push] at step
                rw [show forms.size + (j + 1) = forms.size + 1 + j by omega, List.getElem_cons_succ,
                  List.count_cons_self]
                exact ⟨fun flag => by have := step.1 flag; omega, step.2⟩
      | false =>
          obtain ⟨size, below, each⟩ := formsFold zeroForm rest (forms.push zeroForm) next
          rw [run zeroForm next rfl]
          have first := below forms.size (by simp)
          rw [getD_push_size] at first
          refine ⟨by rw [size, Array.size_push, List.length_cons]; omega, fun k inside => ?_, fun j inside => ?_⟩
          · rw [below k (by rw [Array.size_push]; omega), getD_push_lt _ _ _ inside]
          · cases j with
            | zero => exact ⟨fun live => absurd live (by simp), fun _ => by rw [Nat.add_zero, first]⟩
            | succ j =>
                have step := each j (by simpa using inside)
                rw [Array.size_push] at step
                rw [show forms.size + (j + 1) = forms.size + 1 + j by omega, List.getElem_cons_succ,
                  List.count_cons_of_ne (by simp)]
                exact step

/-- The form of a live column is a cell of the hint, and the form of each other column is
`zeroForm`. -/
theorem columnForms_spec (live : Array Bool) (first zeroForm : ℕ) :
    (columnForms live first zeroForm).size = live.size ∧ ∀ k < live.size,
      (live.getD k false = true → first ≤ (columnForms live first zeroForm).getD k zeroForm ∧
        (columnForms live first zeroForm).getD k zeroForm < first + live.count true) ∧
      (live.getD k false = false → (columnForms live first zeroForm).getD k zeroForm = zeroForm) := by
  obtain ⟨size, -, each⟩ := formsFold zeroForm live.toList #[] first
  have run : columnForms live first zeroForm = (live.toList.foldl (formStep zeroForm) (#[], first)).1 := by
    rw [Array.foldl_toList]
    rfl
  rw [run]
  refine ⟨by simpa using size, fun k inside => ?_⟩
  have step := each k (by simpa using inside)
  have flag : live.getD k false = live.toList[k]'(by simpa using inside) := by
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem inside, Option.getD_some, Array.getElem_toList]
  rw [Array.size_empty, Nat.zero_add, Array.count_toList] at step
  rw [flag]
  exact step

end CompressionLean.Sim.OptionB
