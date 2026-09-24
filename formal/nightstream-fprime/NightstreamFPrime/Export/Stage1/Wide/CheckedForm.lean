import NightstreamFPrime.Export.Stage1.Wide.FormSupport

/-! Exact sparse-form laws for the checked coordinate map. -/

namespace NightstreamFPrime.Export.Stage1.Wide.FormSupport

open NightstreamFPrime.Spec NightstreamFPrime.Layout
open ProductionRelation
open RetainedLayout (renameForm)

private abbrev Live (program : RetainedLayout.Program)
    (column : Fin (PerApplicationFixedPoint.logicalWidth program)) := RetainedLayout.Live program column.val

theorem rename_add (program : RetainedLayout.Program)
    (left right : SparseForm (PerApplicationFixedPoint.logicalWidth program))
    (hl : Supported (Live program) left) (hr : Supported (Live program) right)
    (supported : Supported (Live program) (SparseForm.add left right)) :
    renameForm program (SparseForm.add left right) supported =
      SparseForm.add (renameForm program left hl) (renameForm program right hr) := by
  apply congrArg SparseForm.mk
  exact List.pmap_append supported

theorem rename_scale (program : RetainedLayout.Program)
    (coefficient : F) (form : SparseForm (PerApplicationFixedPoint.logicalWidth program))
    (supported : Supported (Live program) form)
    (scaled : Supported (Live program) (SparseForm.scale coefficient form)) :
    renameForm program (SparseForm.scale coefficient form) scaled =
      SparseForm.scale coefficient (renameForm program form supported) := by
  apply congrArg SparseForm.mk
  simp only [renameForm, SparseForm.mapColumnsChecked, SparseForm.scale, List.pmap_map, List.map_pmap]

theorem rename_singleton (program : RetainedLayout.Program)
    (source : Fin (PerApplicationFixedPoint.logicalWidth program)) (coefficient : F)
    (live : RetainedLayout.Live program source.val)
    (supported : Supported (Live program) (SparseForm.singleton source coefficient)) :
    renameForm program (SparseForm.singleton source coefficient) supported =
      SparseForm.singleton (RetainedLayout.column program source live) coefficient := rfl

theorem rename_recompose (program : RetainedLayout.Program)
    (forms : List (SparseForm (PerApplicationFixedPoint.logicalWidth program)))
    (supported : ∀ form ∈ forms, Supported (Live program) form) :
    renameForm program (RetainedSlot.recomposeForms forms) (recompose forms supported) =
      RetainedSlot.recomposeForms (forms.pmap (fun form live => renameForm program form live) supported) := by
  induction forms with
  | nil => rfl
  | cons head tail ih =>
    let headLive := supported head List.mem_cons_self
    let tailLive := fun form member => supported form (List.mem_cons_of_mem head member)
    change renameForm program (SparseForm.add head (SparseForm.scale _ (RetainedSlot.recomposeForms tail))) _ = _
    rw [rename_add program head _ headLive (scale _ (recompose tail tailLive)),
      rename_scale program _ _ (recompose tail tailLive), ih tailLive]
    rfl

theorem rename_recompose_ofFn (program : RetainedLayout.Program) {count : Nat}
    (forms : Fin count → SparseForm (PerApplicationFixedPoint.logicalWidth program))
    (supported : ∀ index, Supported (Live program) (forms index))
    (complete : Supported (Live program) (RetainedSlot.recomposeForms (List.ofFn forms))) :
    renameForm program (RetainedSlot.recomposeForms (List.ofFn forms)) complete =
      RetainedSlot.recomposeForms (List.ofFn fun index => renameForm program (forms index) (supported index)) := by
  have each : ∀ form ∈ List.ofFn forms, Supported (Live program) form := by
    intro form member
    obtain ⟨index, rfl⟩ := List.mem_ofFn.mp member
    exact supported index
  rw [rename_recompose program _ each]
  apply congrArg RetainedSlot.recomposeForms
  apply List.ext_getElem
  · simp only [List.length_pmap, List.length_ofFn]
  · intro index hl hr
    simp only [List.getElem_pmap, List.getElem_ofFn]

end NightstreamFPrime.Export.Stage1.Wide.FormSupport
