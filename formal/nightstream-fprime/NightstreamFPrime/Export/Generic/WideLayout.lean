import NightstreamFPrime.Export.Generic.LogicalPlan

/-! A layout over more logical columns (CW-10.2). A circuit key whose logical width exceeds its
circuit's reads the circuit's layout with the same forms in its first columns (`widen`). An
assignment of the circuit extends to the key with zeros past the circuit's columns (`extend`),
and the widened layout reads the extension as the layout reads the assignment
(`sourceEnv_widen`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Generic.WideLayout

open NightstreamFPrime.Spec
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

variable {narrow wide : Nat}

/-- A form over more logical columns, with the same entries. -/
def widenForm (le : narrow ≤ wide) (form : SparseForm narrow) : SparseForm wide :=
  ⟨form.entries.map fun entry => ⟨Fin.castLE le entry.column, entry.coefficient⟩⟩

/-- A layout over more logical columns. -/
def widen {width : Nat} (layout : LogicalPlan.Layout width narrow) (le : narrow ≤ wide) :
    LogicalPlan.Layout width wide where
  map := ⟨fun column => widenForm le (layout.map.form column)⟩
  publicFits := le_trans layout.publicFits le
  publicForm := fun column bound => by
    show widenForm le (layout.map.form column) = _
    rw [layout.publicForm column bound]
    rfl

/-- An assignment over more logical columns: the same values, then zeros. -/
def extend (assignment : Assignment F narrow) : Assignment F wide :=
  fun column => if inside : column.val < narrow then assignment ⟨column.val, inside⟩ else 0

theorem extend_castLE (le : narrow ≤ wide) (assignment : Assignment F narrow)
    (column : Fin narrow) :
    extend (wide := wide) assignment (Fin.castLE le column) = assignment column := by
  simp [extend, column.isLt]

theorem widenForm_eval (le : narrow ≤ wide) (assignment : Assignment F narrow) :
    ∀ entries : List (SparseEntry narrow),
      (widenForm le ⟨entries⟩).eval (extend assignment) = (⟨entries⟩ : SparseForm narrow).eval assignment
  | [] => by
      change (SparseForm.empty : SparseForm wide).eval _ = (SparseForm.empty : SparseForm narrow).eval _
      simp only [SparseForm.empty_eval]
  | entry :: rest => by
      have narrowSplit : (⟨entry :: rest⟩ : SparseForm narrow) =
          SparseForm.add (SparseForm.singleton entry.column entry.coefficient) ⟨rest⟩ := rfl
      have wideSplit : widenForm le ⟨entry :: rest⟩ =
          SparseForm.add (SparseForm.singleton (Fin.castLE le entry.column) entry.coefficient)
            (widenForm le ⟨rest⟩) := rfl
      rw [wideSplit, narrowSplit, SparseForm.add_eval, SparseForm.add_eval, SparseForm.singleton_eval,
        SparseForm.singleton_eval, extend_castLE, widenForm_eval le assignment rest]

theorem sourceEnv_widen {width : Nat} (layout : LogicalPlan.Layout width narrow) (le : narrow ≤ wide)
    (assignment : Assignment F narrow) :
    LogicalPlan.sourceEnv (widen layout le) (extend assignment) =
      LogicalPlan.sourceEnv layout assignment := by
  funext column
  unfold LogicalPlan.sourceEnv
  split
  · exact widenForm_eval le assignment _
  · rfl

theorem widenForm_eval_restrict (le : narrow ≤ wide) (assignment : Assignment F wide) :
    ∀ entries : List (SparseEntry narrow),
      (widenForm le ⟨entries⟩).eval assignment =
        (⟨entries⟩ : SparseForm narrow).eval (fun column => assignment (Fin.castLE le column))
  | [] => by
      change (SparseForm.empty : SparseForm wide).eval _ = (SparseForm.empty : SparseForm narrow).eval _
      simp only [SparseForm.empty_eval]
  | entry :: rest => by
      have narrowSplit : (⟨entry :: rest⟩ : SparseForm narrow) =
          SparseForm.add (SparseForm.singleton entry.column entry.coefficient) ⟨rest⟩ := rfl
      have wideSplit : widenForm le ⟨entry :: rest⟩ =
          SparseForm.add (SparseForm.singleton (Fin.castLE le entry.column) entry.coefficient)
            (widenForm le ⟨rest⟩) := rfl
      rw [wideSplit, narrowSplit, SparseForm.add_eval, SparseForm.add_eval,
        SparseForm.singleton_eval, SparseForm.singleton_eval, widenForm_eval_restrict le assignment rest]

theorem sourceEnv_widen_restrict {width : Nat} (layout : LogicalPlan.Layout width narrow)
    (le : narrow ≤ wide) (assignment : Assignment F wide) :
    LogicalPlan.sourceEnv (widen layout le) assignment =
      LogicalPlan.sourceEnv layout (fun column => assignment (Fin.castLE le column)) := by
  funext column
  unfold LogicalPlan.sourceEnv
  split
  · exact widenForm_eval_restrict le assignment _
  · rfl

theorem one_widen {width : Nat} (layout : LogicalPlan.Layout width narrow) (le : narrow ≤ wide)
    (assignment : Assignment F narrow) :
    extend (wide := wide) assignment (widen layout le).one = assignment layout.one :=
  extend_castLE le assignment layout.one

theorem extend_norm (assignment : Assignment F narrow)
    (bounded : ∀ column, centeredMagnitude (assignment column) < 2) (column : Fin wide) :
    centeredMagnitude (extend (wide := wide) assignment column) < 2 := by
  unfold extend
  split
  · exact bounded _
  · decide

end NightstreamFPrime.Export.Generic.WideLayout
