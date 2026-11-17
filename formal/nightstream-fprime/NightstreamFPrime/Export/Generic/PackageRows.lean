import NightstreamFPrime.Export.Generic.PlanPush
import NightstreamFPrime.Export.Generic.SlotLayout
import NightstreamFPrime.Export.Generic.WideLayout
import NightstreamFPrime.Spec.Compression.Claim.Slot

/-! The dense rows of a step package (Rust `FactoredRows`) represent its relation
(`rows_represents`, the premise `represents` of `Export.BoundedDepth.FinalVerifier.Premises`).

`rows` takes each row of a relation over the source columns, with its entries
`(source column, coefficient)`, and the span of each source column: its first logical column and
its count of logical columns (`spanStart`, `spanWidth`). The span of a public column `c` is
`(c, 1)`. The span of a column with a slot is the slot's coordinates, and a coordinate `k` has
the weight `3^k` (`form_coefficient`, `RetainedSlot.recomposeForms`). Every other column has an
empty span. With the relation through `PlanPush.identity`, the dense matrices `L · G` of these
rows are the matrices of the relation through the slot layout, widened to the key's logical
width (`PlanPush.plan_coefficient`). Their columns have the blocks of the package's logical
width (`rows_support`, the premise `support`). -/

namespace NightstreamFPrime.Export.Generic.PackageRows

open NightstreamFPrime.Spec
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Spec.Folding.PiCCS
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.RingRelation.PiDECAlgebra.Radix (fieldOfNat)
open NightstreamFPrime.Export.Generic.LogicalPlan (Layout publicCount)

variable {width : Nat}

/-- The first logical column and the count of logical columns of source column `column` (Rust
`spans`): `(c, 1)` for a public column `c`, the coordinates of the column's slot after the
public columns, and no column otherwise. -/
def span (slots : List (LowNormAssignment.Slot width)) (column : Nat) : Nat × Nat :=
  if inside : column < width then
    if column < publicCount then (column, 1)
    else
      match SlotLayout.slotIndex slots ⟨column, inside⟩ with
      | some slot =>
          (ProductionAssignment.publicWidth + LowNormAssignment.prefixWidth slots slot,
            (slots.get slot).width)
      | none => (0, 0)
  else (0, 0)

/-- The first logical column of the span of a source column. -/
def spanStart (slots : List (LowNormAssignment.Slot width)) (column : Nat) : Nat :=
  (span slots column).1

/-- The count of logical columns of the span of a source column. -/
def spanWidth (slots : List (LowNormAssignment.Slot width)) (column : Nat) : Nat :=
  (span slots column).2

/-! ### The forms of the slot layout -/

private theorem width_pos {sourceWidth : Nat} (slot : LowNormAssignment.Slot sourceWidth) :
    0 < slot.width := by
  unfold LowNormAssignment.Slot.width
  cases slot.kind <;> decide

/-- A span ends at or before the package's logical width. -/
private theorem span_le (slots : List (LowNormAssignment.Slot width)) (column : Nat) :
    spanStart slots column + spanWidth slots column ≤
      ProductionAssignment.logicalWidth slots := by
  have publicEq : publicCount = 320 := rfl
  simp only [ProductionAssignment.logicalWidth, ProductionAssignment.publicWidth_eq]
  unfold spanStart spanWidth span
  by_cases inside : column < width
  · rw [dite_eq_left inside]
    by_cases isPublic : column < publicCount
    · rw [ite_eq_left isPublic]
      omega
    · rw [ite_eq_right isPublic]
      cases SlotLayout.slotIndex slots ⟨column, inside⟩ with
      | none => exact Nat.zero_le _
      | some slot =>
          have last : LowNormAssignment.prefixWidth slots slot + ((slots.get slot).width - 1) <
              LowNormAssignment.logicalWidth slots :=
            LowNormAssignment.prefixWidth_add_lt slots slot
              ⟨(slots.get slot).width - 1, Nat.sub_lt (width_pos _) Nat.one_pos⟩
          simp only [ProductionAssignment.publicWidth_eq]
          have := width_pos (slots.get slot)
          omega
  · rw [dite_eq_right inside]
    exact Nat.zero_le _

private theorem widenForm_coefficient {narrow wide : Nat} (le : narrow ≤ wide)
    (form : SparseForm narrow) (x : Fin wide) :
    (WideLayout.widenForm le form).coefficient x =
      if inside : x.val < narrow then form.coefficient ⟨x.val, inside⟩ else 0 := by
  obtain ⟨entries⟩ := form
  induction entries with
  | nil => split <;> rfl
  | cons entry rest inductionHypothesis =>
      have narrowSplit : (⟨entry :: rest⟩ : SparseForm narrow) =
          SparseForm.add (SparseForm.singleton entry.column entry.coefficient) ⟨rest⟩ := rfl
      have wideSplit : WideLayout.widenForm le ⟨entry :: rest⟩ =
          SparseForm.add (SparseForm.singleton (Fin.castLE le entry.column) entry.coefficient)
            (WideLayout.widenForm le ⟨rest⟩) := rfl
      rw [wideSplit, SparseForm.add_coefficient, inductionHypothesis, narrowSplit,
        SparseForm.singleton_coefficient]
      by_cases inside : x.val < narrow
      · simp only [inside, dite_true, SparseForm.add_coefficient, SparseForm.singleton_coefficient,
          Fin.ext_iff, Fin.val_castLE]
      · have different : Fin.castLE le entry.column ≠ x := fun same => inside (by
          rw [← same, Fin.val_castLE]
          exact entry.column.isLt)
        simp only [inside, dite_false, different, ite_false, add_zero]

/-- The widened layout reads the layout's forms on the first columns and nothing after them. -/
private theorem widen_coefficient {narrow wide : Nat} (layout : Layout width narrow)
    (le : narrow ≤ wide) (column : Nat) (x : Fin wide) :
    ((WideLayout.widen layout le).form column).coefficient x =
      if inside : x.val < narrow then (layout.form column).coefficient ⟨x.val, inside⟩
      else 0 := by
  unfold LogicalPlan.Layout.form
  by_cases bounded : column < width
  · rw [dite_eq_left bounded, dite_eq_left bounded]
    exact widenForm_coefficient le _ x
  · rw [dite_eq_right bounded, dite_eq_right bounded]
    split <;> rfl

/-- Coordinate `k` of a recomposed run of columns has the weight `3^k`. -/
private theorem recompose_coefficient {columns : Nat} :
    ∀ (count start : Nat) (column : Fin count → Fin columns),
      (∀ coordinate, (column coordinate).val = start + coordinate.val) → ∀ x : Fin columns,
        (RetainedSlot.recomposeForms (List.ofFn fun coordinate =>
            SparseForm.singleton (column coordinate) 1)).coefficient x =
          if start ≤ x.val ∧ x.val < start + count then fieldOfNat 3 ^ (x.val - start) else 0
  | 0, start, _, _, x => by
      rw [List.ofFn_zero, ite_eq_right (by omega)]
      rfl
  | count + 1, start, column, value, x => by
      rw [List.ofFn_succ]
      simp only [RetainedSlot.recomposeForms, SparseForm.add_coefficient,
        SparseForm.scale_coefficient, SparseForm.singleton_coefficient]
      rw [recompose_coefficient count (start + 1) (fun coordinate => column coordinate.succ)
        (fun coordinate => by rw [value, Fin.val_succ]; omega) x]
      have head : column 0 = x ↔ x.val = start := by
        rw [Fin.ext_iff, value]
        simp only [Fin.val_zero, add_zero, eq_comm]
      by_cases first : x.val = start
      · rw [ite_eq_left (head.mpr first), ite_eq_right (by omega), ite_eq_left (by omega), first,
          Nat.sub_self, pow_zero, mul_zero, add_zero]
      · rw [ite_eq_right (fun same => first (head.mp same)), zero_add]
        by_cases inside : start + 1 ≤ x.val ∧ x.val < start + 1 + count
        · rw [ite_eq_left inside, ite_eq_left (by omega),
            show x.val - start = x.val - (start + 1) + 1 by omega, pow_succ']
        · rw [ite_eq_right inside, ite_eq_right (by omega), mul_zero]

/-- The slot layout reads source column `column` with the weight `3^k` on cell `k` of its
span. -/
private theorem slot_coefficient (slots : List (LowNormAssignment.Slot width)) (column : Nat)
    (x : Fin (ProductionAssignment.logicalWidth slots)) :
    ((SlotLayout.layout slots).form column).coefficient x =
      if spanStart slots column ≤ x.val ∧
          x.val < spanStart slots column + spanWidth slots column then
        fieldOfNat 3 ^ (x.val - spanStart slots column)
      else 0 := by
  unfold LogicalPlan.Layout.form spanStart spanWidth span
  by_cases inside : column < width
  · rw [dite_eq_left inside, dite_eq_left inside]
    by_cases isPublic : column < publicCount
    · simp only [SlotLayout.layout, isPublic, dite_true, ite_true,
        SparseForm.singleton_coefficient, Fin.ext_iff]
      by_cases same : column = x.val
      · rw [ite_eq_left same, ite_eq_left (by omega), ← same, Nat.sub_self, pow_zero]
      · rw [ite_eq_right same, ite_eq_right (by omega)]
    · simp only [SlotLayout.layout, isPublic, dite_false, ite_false]
      cases SlotLayout.slotIndex slots ⟨column, inside⟩ with
      | none =>
          dsimp only
          rw [SparseForm.empty_coefficient, ite_eq_right (by omega)]
      | some slot =>
          exact recompose_coefficient _ _ _ (fun coordinate => by
            simp only [ProductionAssignment.privateColumn, LowNormAssignment.coordinateIndex]
            omega) x
  · rw [dite_eq_right inside, dite_eq_right inside, SparseForm.empty_coefficient,
      ite_eq_right (by omega)]

/-- The widened slot layout reads source column `column` with the weight `3^k` on cell `k` of its
span (Rust `spans` and `ratio = 3`). -/
theorem form_coefficient (slots : List (LowNormAssignment.Slot width)) {wide : Nat}
    (le : ProductionAssignment.logicalWidth slots ≤ wide) (column : Nat) (x : Fin wide) :
    ((WideLayout.widen (SlotLayout.layout slots) le).form column).coefficient x =
      if spanStart slots column ≤ x.val ∧
          x.val < spanStart slots column + spanWidth slots column then
        fieldOfNat 3 ^ (x.val - spanStart slots column)
      else 0 := by
  rw [widen_coefficient]
  by_cases inside : x.val < ProductionAssignment.logicalWidth slots
  · rw [dite_eq_left inside, slot_coefficient]
  · have := span_le slots column
    rw [dite_eq_right inside, ite_eq_right (by omega)]

/-! ### The dense rows -/

/-- The dense rows of a plan over the source columns (Rust `FactoredRows`): row `row` of matrix
`matrix` holds the entries `(source column, coefficient)` of port `matrix` of the plan's row, the
spans are those of the slots, and the ratio is 3. -/
def rows (slots : List (LowNormAssignment.Slot width)) (plan : Plan width) (columns : Nat) :
    Compression.Verifier.Factored F where
  rows := plan.rowCount
  columns := columns
  entries matrix row :=
    if inside : matrix < Spec.ProductionRelation.matrixCount ∧ row < plan.rowCount then
      (plan.portForm ⟨row, inside.2⟩ ⟨matrix, inside.1⟩).entries.map fun entry =>
        (entry.column.val, entry.coefficient)
    else []
  start := spanStart slots
  width := spanWidth slots
  ratio := fieldOfNat 3

open scoped NightstreamFPrime.Spec.Compression.Production

/-- The columns of the dense rows have the blocks of the package's logical width. -/
theorem rows_support (slots : List (LowNormAssignment.Slot width)) (plan : Plan width) :
    (rows slots plan (ringDegree * ConcreteRingColumnLayout.blockCount
        (ProductionAssignment.logicalWidth slots))).matrices.columns / ringDegree ≤
      ConcreteRingColumnLayout.blockCount (ProductionAssignment.logicalWidth slots) :=
  le_of_eq (Nat.mul_div_cancel_left _ (by decide))

/-- Entry `(row, x)` of matrix `matrix` of the dense rows, as the rows of `Represents` read it. -/
private theorem rows_entry (slots : List (LowNormAssignment.Slot width)) (plan : Plan width)
    (columns : Nat) (matrix : Fin Spec.ProductionRelation.matrixCount) (row x : Nat) :
    (if row < (rows slots plan columns).matrices.rows then
        (rows slots plan columns).matrices.entry matrix row x else 0) =
      if live : row < plan.rowCount then
        ((plan.portForm ⟨row, live⟩ matrix).entries.map fun entry =>
          if spanStart slots entry.column.val ≤ x ∧
              x < spanStart slots entry.column.val + spanWidth slots entry.column.val then
            entry.coefficient * fieldOfNat 3 ^ (x - spanStart slots entry.column.val)
          else 0).sum
      else 0 := by
  change (if row < plan.rowCount then (((rows slots plan columns).entries matrix row).map
    fun entry => (rows slots plan columns).run entry x).sum else 0) = _
  by_cases live : row < plan.rowCount
  · have entries : (rows slots plan columns).entries matrix row =
        (plan.portForm ⟨row, live⟩ matrix).entries.map fun entry =>
          (entry.column.val, entry.coefficient) :=
      dite_eq_left ⟨matrix.isLt, live⟩
    rw [ite_eq_left live, dite_eq_left live, entries, List.map_map]
    rfl
  · rw [ite_eq_right live, dite_eq_right live]

/-- At vertex `row` of the cube, matrix `port` of a plan is port `port` of row `row` below the
row count, and 0 from the row count. -/
private theorem matrix_vertex {columns : Nat} (plan : Plan columns)
    (port : Fin Spec.ProductionRelation.matrixCount) (row : Nat)
    (below : row < 2 ^ Lifecycle.cubeVariables) (x : Fin columns) :
    plan.matrix port (NumericBooleanDomain.vertex Lifecycle.cubeVariables ⟨row, below⟩) x =
      if live : row < plan.rowCount then (plan.portForm ⟨row, live⟩ port).coefficient x
      else 0 := by
  by_cases live : row < plan.rowCount
  · have vertex : NumericBooleanDomain.vertex Lifecycle.cubeVariables ⟨row, below⟩ =
        plan.rowLayout.toVertex ⟨row, live⟩ := rfl
    rw [dite_eq_left live, vertex]
    unfold Plan.matrix
    rw [plan.rowLayout.toColumn_toVertex]
  · have none : plan.rowLayout.toColumn?
        (NumericBooleanDomain.vertex Lifecycle.cubeVariables ⟨row, below⟩) = none :=
      (CanonicalRowLayout.toColumn?_eq_none_iff _ _ _ _).mpr (by
        rw [NumericBooleanDomain.index_vertex]
        exact Nat.le_of_not_lt live)
    rw [dite_eq_right live]
    unfold Plan.matrix
    rw [none]

/-- The dense rows of a step package, built from its relation over the source columns and the
spans of its slots, represent its relation through the slot layout, widened to a logical width
that holds the package's logical width. -/
theorem rows_represents (publicColumns : Nat) (pieces : List Pieces.Piece) (next width : Nat)
    (fits : CircuitRelation.Fits publicColumns pieces next width) (inside : publicCount ≤ width)
    (slots : List (LowNormAssignment.Slot width)) {logicalWidth : Nat}
    (le : ProductionAssignment.logicalWidth slots ≤ logicalWidth) :
    Compression.Claim.Represents
      (rows slots
        (CircuitRelation.plan publicColumns pieces next width fits (PlanPush.identity width inside))
        (ringDegree * ConcreteRingColumnLayout.blockCount
          (ProductionAssignment.logicalWidth slots))).matrices
      (CircuitRelation.plan publicColumns pieces next width fits
        (WideLayout.widen (SlotLayout.layout slots) le)).matrix := by
  have count := PlanPush.plan_rowCount publicColumns pieces next width fits
    (WideLayout.widen (SlotLayout.layout slots) le) (PlanPush.identity width inside)
  refine ⟨?_, ?_, ?_⟩
  · simp only [Compression.Verifier.Factored.matrices, rows, ConcreteRingColumnLayout.blockCount,
      RingCarrierLayout.carrierWidth, ringDegree]
    omega
  · exact (CircuitRelation.plan publicColumns pieces next width fits
      (PlanPush.identity width inside)).rowCount_le
  · intro matrix row x below _
    rw [rows_entry]
    by_cases live : row < (CircuitRelation.plan publicColumns pieces next width fits
        (PlanPush.identity width inside)).rowCount
    · rw [dite_eq_left live]
      by_cases within : x < logicalWidth
      · rw [dite_eq_left within, matrix_vertex, dite_eq_left (count ▸ live),
          PlanPush.plan_coefficient publicColumns pieces next width fits inside,
          PlanPush.push_coefficient]
        refine congrArg List.sum (List.map_congr_left fun entry _ => ?_)
        rw [form_coefficient]
        split_ifs
        · rfl
        · exact (mul_zero _).symm
      · rw [dite_eq_right within]
        refine List.sum_eq_zero fun value member => ?_
        obtain ⟨entry, _, rfl⟩ := List.mem_map.mp member
        have := span_le slots entry.column.val
        rw [ite_eq_right (by omega)]
    · rw [dite_eq_right live]
      split
      · rw [matrix_vertex, dite_eq_right (by omega)]
      · rfl

end NightstreamFPrime.Export.Generic.PackageRows
