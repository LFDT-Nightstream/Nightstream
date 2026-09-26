import NightstreamFPrime.Layout.PiRlcWideSampler.Encoding
import NightstreamFPrime.Layout.ProductionRelation.SparseEvaluation

/-! Constructive completeness of the compact sampler matrix plan. The
caller owns the initial transcript and constant-one column below the sampler
region. The explicit assignment fills that region and preserves the caller. -/

namespace NightstreamFPrime.Layout.PiRlcWideSampler.Completeness

open NightstreamFPrime.Spec NightstreamFPrime.Circuit
open ProductionRelation BatchPlan BatchSemantics Witness Encoding
open Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

structure InputsBefore {columns : Nat} (interface : Interface columns) : Prop where
  one : interface.oneColumn.val < interface.start
  input : ∀ lane entry, entry ∈ (interface.initialState lane).entries → entry.column.val < interface.start

private theorem form_preserved {columns : Nat} (interface : Interface columns) (base : Assignment F columns)
    (initial : FieldState) (form : SparseForm columns)
    (before : ∀ entry ∈ form.entries, entry.column.val < interface.start) :
    form.eval (assignment interface base initial) = form.eval base := by
  rw [← SparseForm.evalSparse_eq_eval, ← SparseForm.evalSparse_eq_eval]
  unfold SparseForm.evalSparse
  have same : ∀ entries : List (SparseEntry columns),
      (∀ entry ∈ entries, entry.column.val < interface.start) → ∀ accumulator,
      entries.foldl (fun total entry => total + entry.coefficient * assignment interface base initial entry.column) accumulator =
        entries.foldl (fun total entry => total + entry.coefficient * base entry.column) accumulator := by
    intro entries
    induction entries with
    | nil => intro _ _; rfl
    | cons entry rest ih =>
        intro bounded accumulator
        simp only [List.foldl_cons]
        rw [assignment_outside interface base initial entry.column (Or.inl (bounded entry (by simp)))]
        exact ih (fun item member => bounded item (by simp [member])) _
  exact same form.entries before 0

theorem initial_preserved {columns : Nat} (interface : Interface columns) (base : Assignment F columns)
    (initial : FieldState) (before : InputsBefore interface) :
    SparseLayer.evalState (assignment interface base initial) interface.initialState =
      SparseLayer.evalState base interface.initialState := by
  funext lane
  exact form_preserved interface base initial _ (before.input lane)

theorem one_preserved {columns : Nat} (interface : Interface columns) (base : Assignment F columns)
    (initial : FieldState) (before : InputsBefore interface) (one : base interface.oneColumn = 1) :
    assignment interface base initial interface.oneColumn = 1 := by
  rw [assignment_outside interface base initial _ (Or.inl before.one), one]

theorem sbox_preserved {columns : Nat} (interface : Interface columns) (base : Assignment F columns)
    (initial : FieldState) (invocation : Fin 34) (row : Fin PoseidonRetainedSlots.rows.length) :
    (sbox interface invocation row).eval (assignment interface base initial) =
      PoseidonCompactWitness.retained (inputAt initial invocation.val) row := by
  unfold sbox
  rw [LowNormBlock.Block.form_eval _ _ _ _ _ (poseidon_encodes interface base initial _)]
  change PoseidonCompactWitness.retained (inputAt initial ((invocation.val * 86 + row.val) / 86))
    ⟨(invocation.val * 86 + row.val) % 86, _⟩ = _
  have rowBound : row.val < 86 := by simpa only [PoseidonRetainedSlots.rows_length] using row.isLt
  have quotient : (invocation.val * 86 + row.val) / 86 = invocation.val := by omega
  have remainder : (invocation.val * 86 + row.val) % 86 = row.val := by omega
  simp only [quotient, remainder]

private theorem child_complete {columns : Nat} (interface : Interface columns) (base : Assignment F columns)
    (initial : FieldState) (invocation : Fin 34)
    (one : assignment interface base initial interface.oneColumn = 1)
    (prior : SparseLayer.evalState (assignment interface base initial) (priorState interface invocation) =
      stateAt initial invocation.val) :
    PoseidonSboxPlan.RowsZero (PoseidonSboxFamilyPlan.invocationInterface (poseidonInterface interface) invocation)
        (assignment interface base initial) ∧
      SparseLayer.evalState (assignment interface base initial) (outputState interface invocation) =
        stateAt initial (invocation.val + 1) := by
  have inputs : SparseLayer.evalState (assignment interface base initial)
      ((poseidonInterface interface).input invocation) = inputAt initial invocation.val := by
    funext lane
    simp only [SparseLayer.evalState, poseidonInterface, inputAt, domainInput]
    split_ifs with entry
    · rw [SparseLayer.addConstant, SparseLayer.add, SparseForm.add_eval,
        SparseLayer.constant, SparseForm.singleton_eval, one, mul_one]
      exact congrArg (fun value => value + entryWord (invocation.val / 2) lane) (congrFun prior lane)
    · exact congrFun prior lane
  exact PoseidonCompactWitness.family_member (poseidonInterface interface) invocation
    (assignment interface base initial) (inputAt initial invocation.val) one inputs
    (sbox_preserved interface base initial invocation)

theorem children_complete {columns : Nat} (interface : Interface columns) (base : Assignment F columns)
    (before : InputsBefore interface) (one : base interface.oneColumn = 1) (invocation : Fin 34) :
    let initial := SparseLayer.evalState base interface.initialState
    PoseidonSboxPlan.RowsZero (PoseidonSboxFamilyPlan.invocationInterface (poseidonInterface interface) invocation)
        (assignment interface base initial) ∧
      SparseLayer.evalState (assignment interface base initial) (outputState interface invocation) =
        stateAt initial (invocation.val + 1) := by
  let initial := SparseLayer.evalState base interface.initialState
  have unit := one_preserved interface base initial before one
  have all : ∀ index, ∀ bounded : index < 34,
      PoseidonSboxPlan.RowsZero (PoseidonSboxFamilyPlan.invocationInterface
        (poseidonInterface interface) ⟨index, bounded⟩) (assignment interface base initial) ∧
      SparseLayer.evalState (assignment interface base initial) (outputState interface ⟨index, bounded⟩) =
        stateAt initial (index + 1) := by
    intro index
    induction index with
    | zero =>
        intro bounded
        apply child_complete interface base initial ⟨0, bounded⟩ unit
        simpa only [priorState, dif_pos rfl, stateAt] using! initial_preserved interface base initial before
    | succ index ih =>
        intro bounded
        apply child_complete interface base initial ⟨index + 1, bounded⟩ unit
        have previous := (ih (by omega)).2
        simpa only [priorState, Nat.add_one_ne_zero, ↓reduceDIte, Nat.add_sub_cancel_right] using previous
  exact all invocation.val invocation.isLt

theorem complete {columns : Nat} (compiled : RangePlan.Compiled) (interface : Interface columns)
    (base : Assignment F columns) (before : InputsBefore interface) (one : base interface.oneColumn = 1) :
    (plan compiled interface).RowsZero
      (assignment interface base (SparseLayer.evalState base interface.initialState)) := by
  let initial := SparseLayer.evalState base interface.initialState
  let completed := assignment interface base initial
  have unit := one_preserved interface base initial before one
  have children := children_complete interface base before one
  apply (rowsZero_iff compiled interface completed).mpr
  constructor
  · apply (PoseidonSboxFamilyPlan.planRowsZero_iff _ _ completed).mpr
    intro invocation
    exact (children invocation).1
  · apply (rangeFamily_zero_iff compiled interface completed).mpr
    intro source
    have inputs : ∀ lane : Fin 4,
        (outputState interface ⟨source.val * 2, by omega⟩ ⟨lane.val, by omega⟩).eval completed =
          rangeSources initial source lane.val := by
      intro lane
      have endpoint := congrFun (children ⟨source.val * 2, by omega⟩).2 ⟨lane.val, by omega⟩
      calc
        (outputState interface _ _).eval completed = stateAt initial (source.val * 2 + 1) _ := endpoint
        _ = rangeSources initial source lane.val := (rangeValues_input (drawAt initial source) lane).symm
    have preserves : (rangeSource interface source).Preserves completed (rangeSources initial source) :=
      Retained.sourceMap_preserves (rangeStart interface source) (rangeFits interface source)
        (fun lane => outputState interface ⟨source.val * 2, by omega⟩ ⟨lane.val, by omega⟩)
        completed (rangeSources initial source) inputs (rangeValues_helper _)
        (range_encodes interface base initial source)
    exact (compiled.plan_rows_iff (rangeInputs compiled interface source) completed
      (rangeSources initial source) unit (fun _ => preserves)).mpr (rangeValues_rows _)

end NightstreamFPrime.Layout.PiRlcWideSampler.Completeness
