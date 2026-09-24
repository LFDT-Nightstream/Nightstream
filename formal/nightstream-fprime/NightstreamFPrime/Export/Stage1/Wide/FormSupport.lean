import NightstreamFPrime.Export.Stage1.Wide.RetainedLayout

/-! Source support for the retained-coordinate move. These structural lemmas
inspect sparse constructors, not the full row or coordinate domains. -/

namespace NightstreamFPrime.Export.Stage1.Wide.FormSupport

open NightstreamFPrime.Spec NightstreamFPrime.Layout
open ProductionRelation

def Supported {columns : Nat} (predicate : Fin columns → Prop) (form : SparseForm columns) : Prop :=
  ∀ entry ∈ form.entries, predicate entry.column

theorem empty {columns : Nat} (predicate : Fin columns → Prop) : Supported predicate .empty := by
  simp [Supported, SparseForm.empty]

theorem add {columns : Nat} {predicate : Fin columns → Prop} {left right : SparseForm columns}
    (hl : Supported predicate left) (hr : Supported predicate right) :
    Supported predicate (SparseForm.add left right) := by
  intro entry member
  rcases List.mem_append.mp member with h | h
  · exact hl entry h
  · exact hr entry h

theorem scale {columns : Nat} {predicate : Fin columns → Prop} {form : SparseForm columns}
    (coefficient : F) (supported : Supported predicate form) :
    Supported predicate (SparseForm.scale coefficient form) := by
  intro entry member
  change entry ∈ form.entries.map (fun source => ⟨source.column, coefficient * source.coefficient⟩) at member
  obtain ⟨source, sourceMember, rfl⟩ := List.mem_map.mp member
  exact supported source sourceMember

theorem singleton {columns : Nat} {predicate : Fin columns → Prop} (column : Fin columns)
    (coefficient : F) (valid : predicate column) :
    Supported predicate (SparseForm.singleton column coefficient) := by
  intro entry member
  have eq : entry = ⟨column, coefficient⟩ := List.mem_singleton.mp member
  simpa only [eq] using valid

theorem recompose {columns : Nat} {predicate : Fin columns → Prop}
    (forms : List (SparseForm columns)) (supported : ∀ form ∈ forms, Supported predicate form) :
    Supported predicate (RetainedSlot.recomposeForms forms) := by
  induction forms with
  | nil => exact empty predicate
  | cons head tail ih =>
    exact add (supported head (by simp)) (scale _ (ih (fun form member =>
      supported form (List.mem_cons_of_mem head member))))

theorem block {sourceWidth columns : Nat} {predicate : Fin columns → Prop}
    (block : LowNormBlock.Block sourceWidth) (start : Nat)
    (fits : start + block.coordinateCount ≤ columns) (slot : Fin block.slotCount)
    (valid : ∀ column : Fin columns,
      start ≤ column.val → column.val < start + block.coordinateCount → predicate column) :
    Supported predicate (block.form start fits slot) := by
  apply recompose
  intro form member
  obtain ⟨coordinate, rfl⟩ := List.mem_ofFn.mp member
  apply singleton
  apply valid
  · change start ≤ start + _
    omega
  · change start + (block.coordinateOffset slot coordinate).val < start + block.coordinateCount
    have bound := (block.coordinateOffset slot coordinate).isLt
    omega

theorem get {columns : Nat} {predicate : Fin columns → Prop} (state : SparseLayer.State columns)
    (supported : ∀ lane, Supported predicate (state lane)) (index : Nat) :
    Supported predicate (SparseLayer.get state index) := by
  unfold SparseLayer.get
  split
  · exact supported _
  · exact empty _

theorem mat4 {columns : Nat} {predicate : Fin columns → Prop} (state : SparseLayer.State columns)
    (supported : ∀ lane, Supported predicate (state lane)) (base lane : Nat) :
    Supported predicate (SparseLayer.mat4 state base lane) := by
  rcases lane with _ | _ | _ | lane <;>
    simp only [SparseLayer.mat4, SparseLayer.add, SparseLayer.scale] <;>
    repeat' first | apply add | apply scale | exact get state supported _

theorem external {columns : Nat} {predicate : Fin columns → Prop} (state : SparseLayer.State columns)
    (supported : ∀ lane, Supported predicate (state lane)) (lane : Fin 8) :
    Supported predicate (SparseLayer.external state lane) := by
  unfold SparseLayer.external SparseLayer.add SparseLayer.block
  exact add (add (mat4 state supported _ _) (mat4 state supported _ _)) (mat4 state supported _ _)

def Common (program : RetainedLayout.Program) (column : Fin (PerApplicationFixedPoint.logicalWidth program)) : Prop :=
  column.val < RetainedLayout.hashEnd program ∨
    RetainedLayout.sharedStart program ≤ column.val ∧ column.val < RetainedLayout.sharedEnd program

theorem common_live (program : RetainedLayout.Program)
    (column : Fin (PerApplicationFixedPoint.logicalWidth program)) (common : Common program column) :
    RetainedLayout.Live program column.val := by
  rcases common with hash | shared
  · exact Or.inl hash
  · exact Or.inr (Or.inl shared)

theorem common_supported (program : RetainedLayout.Program)
    (form : SparseForm (PerApplicationFixedPoint.logicalWidth program))
    (supported : Supported (Common program) form) :
    ∀ entry ∈ form.entries, RetainedLayout.Live program entry.column.val :=
  fun entry member => common_live program entry.column (supported entry member)

theorem common_before (program : RetainedLayout.Program)
    (column : Fin (PerApplicationFixedPoint.logicalWidth program)) (common : Common program column) :
    (RetainedLayout.column program column (common_live program column common)).val < RetainedLayout.commonCount program := by
  obtain ⟨hash, start, stop, _⟩ := RetainedLayout.boundaries program
  rw [RetainedLayout.commonCount_eq]
  rcases common with before | ⟨after, before⟩
  · have mapped : RetainedLayout.column? program column.val = some column.val := by
      simp only [RetainedLayout.column?, if_pos before]
    rw [RetainedLayout.column_of_some _ _ _ _ mapped]
    rw [hash] at before
    omega
  · have afterHash : ¬column.val < RetainedLayout.hashEnd program := by
      rw [hash, start] at *
      omega
    have mapped : RetainedLayout.column? program column.val =
        some (RetainedLayout.hashEnd program + (column.val - RetainedLayout.sharedStart program)) := by
      simp only [RetainedLayout.column?, if_neg afterHash, if_pos (And.intro after before)]
    rw [RetainedLayout.column_of_some _ _ _ _ mapped, hash, start]
    rw [stop] at before
    omega

theorem renamed_before (program : RetainedLayout.Program)
    (form : SparseForm (PerApplicationFixedPoint.logicalWidth program))
    (supported : Supported (Common program) form) :
    Supported (fun column => column.val < RetainedLayout.commonCount program)
      (RetainedLayout.renameForm program form (common_supported program form supported)) := by
  intro entry member
  obtain ⟨source, sourceMember, same⟩ := List.mem_pmap.mp member
  subst entry
  exact common_before program source.column (supported source sourceMember)

end NightstreamFPrime.Export.Stage1.Wide.FormSupport
