import Nightstream.Implementation.NebulaV2.ConditionalEqualityOneRows
import Nightstream.Implementation.NebulaV2.ConditionalEqualityRows

/-!
Contract: universal selector gating for an existing list of R1CS rows.

Assurance tier: implementation model.

Owns a three-row lowering for each source row. The first auxiliary stores the
source row's left product, the second stores its right linear combination, and
the third compares the auxiliaries only in the selected branch. Soundness and
honest completeness do not require a hidden normalization property of the
source row coefficients.

Does not own the Boolean constraint or protocol meaning of the selector wire,
absolute generated columns, or allocation of the two auxiliary columns.

Emits constraints: yes.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.SelectorGatedRows

open Nightstream.Implementation.R1CS

inductive SelectWhen where
  | zero
  | one
deriving DecidableEq, Repr

/-- Two fresh auxiliary columns for every source row. -/
structure Layout (source : List Row) where
  selectorColumn : Nat
  productColumn : Nat → Nat
  outputColumn : Nat → Nat

def productRow {source : List Row} (layout : Layout source)
    (index : Fin source.length) : Row :=
  let row := source.get index
  ⟨row.a, row.b, [(layout.productColumn index, 1)]⟩

def outputRow {source : List Row} (layout : Layout source)
    (index : Fin source.length) : Row :=
  let row := source.get index
  ⟨row.c, [(0, 1)], [(layout.outputColumn index, 1)]⟩

def selectorRow {source : List Row} (when : SelectWhen)
    (layout : Layout source) (index : Fin source.length) : Row :=
  let pair := (layout.productColumn index, layout.outputColumn index)
  match when with
  | .zero => ConditionalEqualityRows.row layout.selectorColumn pair
  | .one => ConditionalEqualityOneRows.row layout.selectorColumn pair

def blockRows {source : List Row} (when : SelectWhen)
    (layout : Layout source) (index : Fin source.length) : List Row :=
  [productRow layout index, outputRow layout index,
    selectorRow when layout index]

def rows {source : List Row} (when : SelectWhen)
    (layout : Layout source) : List Row :=
  (List.ofFn fun index : Fin source.length =>
    blockRows when layout index).flatten

theorem blockRows_length {source : List Row} (when : SelectWhen)
    (layout : Layout source) (index : Fin source.length) :
    (blockRows when layout index).length = 3 := by
  simp [blockRows]

theorem rows_length {source : List Row} (when : SelectWhen)
    (layout : Layout source) :
    (rows when layout).length = 3 * source.length := by
  simp [rows, blockRows, List.length_flatten, List.sum_ofFn, Nat.mul_comm]

/-- Exact auxiliary values used by the compiler witness. -/
structure AuxiliariesPlaced {source : List Row}
    (layout : Layout source) (assignment : Nat → Nat) : Prop where
  product : ∀ index : Fin source.length,
    assignment (layout.productColumn index.val) =
      lcEval assignment (source.get index).a *
        lcEval assignment (source.get index).b % goldilocksP
  output : ∀ index : Fin source.length,
    assignment (layout.outputColumn index.val) =
      lcEval assignment (source.get index).c

private theorem block_satisfied
    {source : List Row} {when : SelectWhen} {layout : Layout source}
    {assignment : Nat → Nat}
    (holds : Satisfies (rows when layout) assignment)
    (index : Fin source.length) :
    Satisfies (blockRows when layout index) assignment := by
  intro row member
  exact holds row (List.mem_flatten.mpr
    ⟨blockRows when layout index,
      List.mem_ofFn.mpr ⟨index, rfl⟩, member⟩)

private theorem product_value
    {source : List Row} {when : SelectWhen} {layout : Layout source}
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (holds : Satisfies (rows when layout) assignment)
    (index : Fin source.length) :
    lcEval assignment (source.get index).a *
          lcEval assignment (source.get index).b % goldilocksP =
        assignment (layout.productColumn index) := by
  have rowHolds := block_satisfied holds index
    (productRow layout index) (by simp [blockRows])
  simpa [productRow, RowHolds, lcEval,
    Nat.mod_eq_of_lt (canonical (layout.productColumn index))] using rowHolds

private theorem output_value
    {source : List Row} {when : SelectWhen} {layout : Layout source}
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows when layout) assignment)
    (index : Fin source.length) :
    lcEval assignment (source.get index).c =
      assignment (layout.outputColumn index) := by
  have rowHolds := block_satisfied holds index
    (outputRow layout index) (by simp [blockRows])
  simpa [outputRow, RowHolds, lcEval, one,
    Nat.mod_eq_of_lt (canonical (layout.outputColumn index))] using rowHolds

private theorem selected_auxiliaries_equal
    {source : List Row} {when : SelectWhen} {layout : Layout source}
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (selected : match when with
      | .zero => assignment layout.selectorColumn = 0
      | .one => assignment layout.selectorColumn = 1)
    (holds : Satisfies (rows when layout) assignment)
    (index : Fin source.length) :
    assignment (layout.productColumn index) =
      assignment (layout.outputColumn index) := by
  have gateHolds := block_satisfied holds index
    (selectorRow when layout index) (by simp [blockRows])
  cases when with
  | zero =>
      have gated : Satisfies
          (ConditionalEqualityRows.rows layout.selectorColumn
            [(layout.productColumn index, layout.outputColumn index)])
          assignment := by
        intro row member
        have rowExact : row = ConditionalEqualityRows.row
            layout.selectorColumn
            (layout.productColumn index, layout.outputColumn index) := by
          simpa [ConditionalEqualityRows.rows] using member
        subst row
        simpa [selectorRow] using gateHolds
      exact ConditionalEqualityRows.rows_sound_closed canonical one selected
        gated (layout.productColumn index, layout.outputColumn index) (by simp)
  | one =>
      have gated : Satisfies
          (ConditionalEqualityOneRows.rows layout.selectorColumn
            [(layout.productColumn index, layout.outputColumn index)])
          assignment := by
        intro row member
        have rowExact : row = ConditionalEqualityOneRows.row
            layout.selectorColumn
            (layout.productColumn index, layout.outputColumn index) := by
          simpa [ConditionalEqualityOneRows.rows] using member
        subst row
        simpa [selectorRow] using gateHolds
      exact ConditionalEqualityOneRows.rows_sound_one canonical one selected
        gated (layout.productColumn index, layout.outputColumn index) (by simp)

/-- In the selected branch, satisfaction of the lowered rows implies
satisfaction of every original source row. -/
theorem rows_sound_selected
    {source : List Row} {when : SelectWhen} {layout : Layout source}
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (selected : match when with
      | .zero => assignment layout.selectorColumn = 0
      | .one => assignment layout.selectorColumn = 1)
    (holds : Satisfies (rows when layout) assignment) :
    Satisfies source assignment := by
  intro sourceRow member
  rcases List.mem_iff_getElem.mp member with ⟨position, bound, rowAt⟩
  let index : Fin source.length := ⟨position, bound⟩
  rw [← rowAt]
  unfold RowHolds
  calc
    lcEval assignment (source.get index).a *
          lcEval assignment (source.get index).b % goldilocksP =
        assignment (layout.productColumn index) :=
      product_value canonical holds index
    _ = assignment (layout.outputColumn index) :=
      selected_auxiliaries_equal canonical one selected holds index
    _ = lcEval assignment (source.get index).c :=
      (output_value canonical one holds index).symm

private theorem productRow_complete
    {source : List Row} {layout : Layout source}
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (placed : AuxiliariesPlaced layout assignment)
    (index : Fin source.length) :
    RowHolds assignment (productRow layout index) := by
  simpa [productRow, RowHolds, lcEval,
    Nat.mod_eq_of_lt (canonical (layout.productColumn index))] using
      (placed.product index).symm

private theorem outputRow_complete
    {source : List Row} {layout : Layout source}
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : AuxiliariesPlaced layout assignment)
    (index : Fin source.length) :
    RowHolds assignment (outputRow layout index) := by
  simpa [outputRow, RowHolds, lcEval, one,
    Nat.mod_eq_of_lt (canonical (layout.outputColumn index))] using
      (placed.output index).symm

private theorem selectorRow_complete_selected
    {source : List Row} {when : SelectWhen} {layout : Layout source}
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (selected : match when with
      | .zero => assignment layout.selectorColumn = 0
      | .one => assignment layout.selectorColumn = 1)
    (index : Fin source.length)
    (equal : assignment (layout.productColumn index) =
      assignment (layout.outputColumn index))
    :
    RowHolds assignment (selectorRow when layout index) := by
  cases when with
  | zero =>
      have all := ConditionalEqualityRows.rows_complete_closed canonical one
        selected (pairs :=
          [(layout.productColumn index, layout.outputColumn index)]) (by
            intro pair member
            have pairExact : pair =
                (layout.productColumn index, layout.outputColumn index) := by
              simpa using member
            subst pair
            exact equal)
      exact all _ (by simp [ConditionalEqualityRows.rows, selectorRow])
  | one =>
      have all := ConditionalEqualityOneRows.rows_complete_one canonical one
        selected (pairs :=
          [(layout.productColumn index, layout.outputColumn index)]) (by
            intro pair member
            have pairExact : pair =
                (layout.productColumn index, layout.outputColumn index) := by
              simpa using member
            subst pair
            exact equal)
      exact all _ (by simp [ConditionalEqualityOneRows.rows, selectorRow])

private theorem selectorRow_complete_unselected
    {source : List Row} {when : SelectWhen} {layout : Layout source}
    {assignment : Nat → Nat}
    (one : assignment 0 = 1)
    (unselected : match when with
      | .zero => assignment layout.selectorColumn = 1
      | .one => assignment layout.selectorColumn = 0)
    (index : Fin source.length) :
    RowHolds assignment (selectorRow when layout index) := by
  cases when with
  | zero =>
      have all := ConditionalEqualityRows.rows_complete_active one unselected
        (pairs := [(layout.productColumn index, layout.outputColumn index)])
      exact all _ (by simp [ConditionalEqualityRows.rows, selectorRow])
  | one =>
      have all := ConditionalEqualityOneRows.rows_complete_zero unselected
        (pairs := [(layout.productColumn index, layout.outputColumn index)])
      exact all _ (by simp [ConditionalEqualityOneRows.rows, selectorRow])

private theorem block_complete_selected
    {source : List Row} {when : SelectWhen} {layout : Layout source}
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (selected : match when with
      | .zero => assignment layout.selectorColumn = 0
      | .one => assignment layout.selectorColumn = 1)
    (sourceHolds : Satisfies source assignment)
    (placed : AuxiliariesPlaced layout assignment)
    (index : Fin source.length) :
    Satisfies (blockRows when layout index) assignment := by
  intro row member
  simp only [blockRows, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl
  · exact productRow_complete canonical placed index
  · exact outputRow_complete canonical one placed index
  · apply selectorRow_complete_selected canonical one selected index
    calc
      assignment (layout.productColumn index) =
          lcEval assignment (source.get index).a *
            lcEval assignment (source.get index).b % goldilocksP :=
        placed.product index
      _ = lcEval assignment (source.get index).c :=
        sourceHolds _ (List.get_mem source index)
      _ = assignment (layout.outputColumn index) :=
        (placed.output index).symm

private theorem block_complete_unselected
    {source : List Row} {when : SelectWhen} {layout : Layout source}
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (unselected : match when with
      | .zero => assignment layout.selectorColumn = 1
      | .one => assignment layout.selectorColumn = 0)
    (placed : AuxiliariesPlaced layout assignment)
    (index : Fin source.length) :
    Satisfies (blockRows when layout index) assignment := by
  intro row member
  simp only [blockRows, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl
  · exact productRow_complete canonical placed index
  · exact outputRow_complete canonical one placed index
  · exact selectorRow_complete_unselected one unselected index

/-- An honest selected source relation satisfies the complete lowering. -/
theorem rows_complete_selected
    {source : List Row} {when : SelectWhen} {layout : Layout source}
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (selected : match when with
      | .zero => assignment layout.selectorColumn = 0
      | .one => assignment layout.selectorColumn = 1)
    (sourceHolds : Satisfies source assignment)
    (placed : AuxiliariesPlaced layout assignment) :
    Satisfies (rows when layout) assignment := by
  intro row member
  rcases List.mem_flatten.mp member with ⟨block, blockMember, rowMember⟩
  rcases List.mem_ofFn.mp blockMember with ⟨index, rfl⟩
  exact block_complete_selected canonical one selected sourceHolds placed index
    row rowMember

/-- The unselected branch constrains only the two deterministic auxiliary
values. It does not require the source relation to hold. -/
theorem rows_complete_unselected
    {source : List Row} {when : SelectWhen} {layout : Layout source}
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (unselected : match when with
      | .zero => assignment layout.selectorColumn = 1
      | .one => assignment layout.selectorColumn = 0)
    (placed : AuxiliariesPlaced layout assignment) :
    Satisfies (rows when layout) assignment := by
  intro row member
  rcases List.mem_flatten.mp member with ⟨block, blockMember, rowMember⟩
  rcases List.mem_ofFn.mp blockMember with ⟨index, rfl⟩
  exact block_complete_unselected canonical one unselected placed index row
    rowMember

end Nightstream.Implementation.NebulaV2.SelectorGatedRows
