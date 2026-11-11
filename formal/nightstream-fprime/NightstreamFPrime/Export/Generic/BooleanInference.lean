import NightstreamFPrime.Export.BoundedDepth.SlotCompleteness

set_option autoImplicit false

namespace NightstreamFPrime.Export.Generic.BooleanInference

open NightstreamFPrime.Spec NightstreamFPrime.Circuit NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Export.BoundedDepth.SlotCompleteness

structure Form where
  constant : F
  left : F
  right : F

def Form.eval (form : Form) (left right : F) : F :=
  form.constant + form.left * left + form.right * right

structure Gate where
  left : Nat
  right : Nat
  output : Nat
  a : Form
  b : Form
  rest : Form
  coefficient : F

def Gate.Holds (gate : Gate) (env : Env) : Prop :=
  gate.a.eval (env gate.left) (env gate.right) *
      gate.b.eval (env gate.left) (env gate.right) =
    gate.rest.eval (env gate.left) (env gate.right) + gate.coefficient * env gate.output

abbrev Gate.Truth (gate : Gate) (left right : F) : Prop :=
  let product := gate.a.eval left right * gate.b.eval left right
  let rest := gate.rest.eval left right
  product = rest ∨ product = rest + gate.coefficient

def Gate.checked (gate : Gate) : Bool :=
  decide (gate.coefficient ≠ 0 ∧ gate.Truth 0 0 ∧ gate.Truth 0 1 ∧
    gate.Truth 1 0 ∧ gate.Truth 1 1)

def Gate.Matches (gate : Gate) (row : R1CS.Row) : Prop :=
  ∀ env, env 0 = 1 → (row.Holds env ↔ gate.Holds env)

theorem boolean_of_truth {product rest coefficient output : F}
    (nonzero : coefficient ≠ 0)
    (holds : product = rest + coefficient * output)
    (truth : product = rest ∨ product = rest + coefficient) :
    output = 0 ∨ output = 1 := by
  rcases truth with truth | truth
  · left
    have zero : coefficient * output = 0 := by
      have := holds.symm.trans truth
      exact add_left_cancel (this.trans (add_zero rest).symm)
    exact (GoldilocksPrime.baseFieldNoZeroDivisors _ _ zero).resolve_left nonzero
  · right
    have same : coefficient * output = coefficient := add_left_cancel (holds.symm.trans truth)
    have zero : coefficient * (output - 1) = 0 := by
      rw [mul_sub, mul_one, same, sub_self]
    exact sub_eq_zero.mp ((GoldilocksPrime.baseFieldNoZeroDivisors _ _ zero).resolve_left nonzero)

theorem checked_truth {gate : Gate} (checked : gate.checked = true)
    {left right : F} (leftBit : LowNormSlot.Valid .bit left)
    (rightBit : LowNormSlot.Valid .bit right) : gate.Truth left right := by
  have check : gate.coefficient ≠ 0 ∧ gate.Truth 0 0 ∧ gate.Truth 0 1 ∧
      gate.Truth 1 0 ∧ gate.Truth 1 1 := of_decide_eq_true checked
  rcases leftBit with rfl | rfl <;> rcases rightBit with rfl | rfl
  · exact check.2.1
  · exact check.2.2.1
  · exact check.2.2.2.1
  · exact check.2.2.2.2

theorem gate_valid {gate : Gate} (checked : gate.checked = true) {env : Env}
    (holds : gate.Holds env)
    (leftBit : LowNormSlot.Valid .bit (env gate.left))
    (rightBit : LowNormSlot.Valid .bit (env gate.right)) :
    LowNormSlot.Valid .bit (env gate.output) := by
  have check : gate.coefficient ≠ 0 ∧ gate.Truth 0 0 ∧ gate.Truth 0 1 ∧
      gate.Truth 1 0 ∧ gate.Truth 1 1 := of_decide_eq_true checked
  exact boolean_of_truth check.1 holds (checked_truth checked leftBit rightBit)

inductive Derived (rows : List R1CS.Row) : Nat → Prop where
  | one : Derived rows 0
  | forced {row : R1CS.Row} {column : Nat} (member : row ∈ rows)
      (found : SlotLayout.forcedColumn? row = some column) : Derived rows column
  | gate {row : R1CS.Row} (gate : Gate) (member : row ∈ rows)
      (matched : gate.Matches row) (checked : gate.checked = true)
      (left : Derived rows gate.left) (right : Derived rows gate.right) :
      Derived rows gate.output

theorem derived_valid {rows : List R1CS.Row} {column : Nat} (derived : Derived rows column)
    {env : Env} (one : env 0 = 1) (holds : R1CS.RowsHold env rows) :
    LowNormSlot.Valid .bit (env column) := by
  induction derived with
  | one => exact Or.inr one
  | forced member found => exact NativeBlock.forced_valid found one (holds _ member)
  | gate gate member matched checked _ _ left right =>
      exact gate_valid checked ((matched env one).mp (holds _ member)) left right

def booleanRow (column : Nat) : R1CS.Row :=
  ⟨R1CS.LinearCombination.ofVar column, R1CS.LinearCombination.ofVar column,
    R1CS.LinearCombination.ofVar column⟩

theorem booleanRow_holds (column : Nat) (env : Env) :
    (booleanRow column).Holds env ↔ LowNormSlot.Valid .bit (env column) := by
  simp only [booleanRow, R1CS.Row.Holds, R1CS.LinearCombination.ofVar,
    R1CS.LinearCombination.eval, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil,
    one_mul, add_zero, zero_add]
  constructor
  · intro holds
    exact NativeBlock.bit_valid_of_boolean holds
  · rintro (value | value) <;> simp [value]

theorem redundant_rows {rows : List R1CS.Row} (columns : List Nat)
    (derived : ∀ column ∈ columns, Derived rows column) {env : Env} (one : env 0 = 1) :
    R1CS.RowsHold env (rows ++ columns.map booleanRow) ↔ R1CS.RowsHold env rows := by
  rw [R1CS.rowsHold_append]
  constructor
  · exact And.left
  · intro holds
    refine ⟨holds, ?_⟩
    intro row member
    obtain ⟨column, columnMember, rowEq⟩ := List.mem_map.mp member
    rw [← rowEq]
    exact (booleanRow_holds column env).mpr (derived_valid (derived column columnMember) one holds)

theorem slots_valid {width : Nat} {rows : List R1CS.Row}
    (slots : List (LowNormAssignment.Slot width)) (env : Env)
    (one : env 0 = 1) (holds : R1CS.RowsHold env rows)
    (kinds : ∀ slot ∈ slots, slot.kind = .field ∨
      slot.kind = .bit ∧ Derived rows slot.source.val) :
    ∀ slot ∈ slots, slot.Valid (fun column => env column.val) := by
  intro slot member
  rcases kinds slot member with field | ⟨bit, derived⟩
  · simp only [LowNormAssignment.Slot.Valid, field, LowNormSlot.Valid]
  · simpa only [LowNormAssignment.Slot.Valid, bit] using derived_valid derived one holds

theorem honest_layout {width : Nat} {rows : List R1CS.Row}
    (slots : List (LowNormAssignment.Slot width)) (env : Env)
    (one : env 0 = 1) (holds : R1CS.RowsHold env rows)
    (kinds : ∀ slot ∈ slots, slot.kind = .field ∨
      slot.kind = .bit ∧ Derived rows slot.source.val)
    (publicNorm : ∀ column, column < LogicalPlan.publicCount → centeredMagnitude (env column) < 2)
    (fits : LogicalPlan.Fits rows width)
    (covered : ∀ row ∈ rows, row.VarsSatisfy (fun column => column < width ∧
      (column < LogicalPlan.publicCount ∨ ∃ slot ∈ slots, slot.source.val = column))) :
    (LogicalPlan.plan rows width fits (SlotLayout.layout slots)).RowsZero
        (honestAssignment slots env) ∧
      ∀ column, centeredMagnitude (honestAssignment slots env column) < 2 := by
  have oneAt : honestAssignment slots env (SlotLayout.layout slots).one = 1 :=
    (ProductionAssignment.logicalAssignment_publicColumn _ slots _ ⟨0, by decide⟩).trans one
  constructor
  · apply (LogicalPlan.rowsZero_iff_rowsHold _ _ _ _ _ oneAt).mpr
    refine R1CS.rowsHold_of_agree rows _ env _ covered (fun column coverage => ?_) holds
    exact sourceEnv_honest slots env coverage.1 coverage.2
  · apply ProductionAssignment.logicalAssignment_norm
    · intro column
      exact publicNorm column.val column.isLt
    · exact slots_valid slots env one holds kinds

end NightstreamFPrime.Export.Generic.BooleanInference
