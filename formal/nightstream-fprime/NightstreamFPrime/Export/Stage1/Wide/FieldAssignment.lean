import NightstreamFPrime.Layout.LowNormBlock
import NightstreamFPrime.Layout.ProductionRelation.SparseEvaluation

/-! Writes the candidate's retained output and quotient field block. The
outside-read theorem also preserves caller-owned transcript and input forms. -/

namespace NightstreamFPrime.Export.Stage1.Wide.FieldAssignment

open NightstreamFPrime.Spec NightstreamFPrime.Layout
open ProductionRelation
open Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

def block (count : Nat) : LowNormBlock.Block count where
  kind := .field
  slotCount := count
  source := id

def write {columns count : Nat} (start : Nat) (base : Assignment F columns)
    (values : Fin count → F) : Assignment F columns :=
  fun column => if owned : start ≤ column.val ∧ column.val < start + count * 41 then
    LowNormSlot.coordinate .field
      (values ⟨(column.val - start) / 41, by omega⟩)
      ⟨(column.val - start) % 41, Nat.mod_lt _ (by decide)⟩
  else base column

theorem outside {columns count : Nat} (start : Nat) (base : Assignment F columns)
    (values : Fin count → F) (column : Fin columns)
    (outside : column.val < start ∨ start + count * 41 ≤ column.val) :
    write start base values column = base column := by
  rw [write, dif_neg (by omega)]

theorem encodes {columns count : Nat} (start : Nat) (fits : start + (block count).coordinateCount ≤ columns)
    (base : Assignment F columns) (values : Fin count → F) :
    (block count).EncodesAt start fits (write start base values) values := by
  intro slot digit
  have slotBound : slot.val < count := slot.isLt
  have digitBound : digit.val < 41 := digit.isLt
  change write start base values ⟨start + (slot.val * 41 + digit.val), _⟩ =
    LowNormSlot.coordinate .field (values slot) digit
  rw [write, dif_pos (by dsimp only; constructor <;> omega)]
  apply congrArg₂ (LowNormSlot.coordinate .field)
  · apply congrArg values
    apply Fin.ext
    change (start + (slot.val * 41 + digit.val) - start) / 41 = slot.val
    omega
  · apply Fin.ext
    change (start + (slot.val * 41 + digit.val) - start) % 41 = digit.val
    omega

theorem form_eval_eq {columns : Nat} (left right : Assignment F columns) (form : SparseForm columns)
    (agrees : ∀ entry ∈ form.entries, left entry.column = right entry.column) :
    form.eval left = form.eval right := by
  rw [← SparseForm.evalSparse_eq_eval, ← SparseForm.evalSparse_eq_eval]
  unfold SparseForm.evalSparse
  have same : ∀ entries : List (SparseEntry columns),
      (∀ entry ∈ entries, left entry.column = right entry.column) → ∀ initial,
      entries.foldl (fun total entry => total + entry.coefficient * left entry.column) initial =
        entries.foldl (fun total entry => total + entry.coefficient * right entry.column) initial := by
    intro entries
    induction entries with
    | nil => intro _ _; rfl
    | cons entry rest ih =>
        intro reads initial
        simp only [List.foldl_cons]
        rw [reads entry (by simp)]
        exact ih (fun item member => reads item (by simp [member])) _
  exact same form.entries agrees 0

theorem owned_norm {columns count : Nat} (start : Nat) (base : Assignment F columns)
    (values : Fin count → F) (column : Fin columns)
    (owned : start ≤ column.val ∧ column.val < start + count * 41) :
    centeredMagnitude (write start base values column) < 2 := by
  rw [write, dif_pos owned]
  apply LowNormSlot.encode_norm .field _ trivial
  rw [← LowNormSlot.coordinateList_eq_encode]
  exact List.mem_ofFn.mpr ⟨_, rfl⟩

end NightstreamFPrime.Export.Stage1.Wide.FieldAssignment
