import NightstreamFPrime.Export.Stage1.PiDECPoseidonNumericStep

/-!
Evaluate the existing Poseidon invocation rows as stored field values. The
schedule and retained coordinates remain owned by PoseidonSboxPlan. Numeric
state is stored between steps; no expanded sparse trace is executed.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECPoseidonNumericRows

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.ProductionRelation
open NightstreamFPrime.Spec.ProductionRelation.RowSemantics (PortValues)
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Layout.ProductionRelation
open PiDECPoseidonNumericStep

private def nextIndex (next : Nat) : Permutation.Step → Nat
  | .initialLayer => next
  | .initialFullRound _ => next + 8
  | .partialRound _ => next + 1
  | .terminalFullRound _ => next + 8

private theorem nextIndex_value {columns : Nat}
    (interface : PoseidonSboxPlan.Interface columns) (next : Nat)
    (state : PoseidonSboxPlan.State columns) (step : Permutation.Step) :
    nextIndex next step =
      (PoseidonSboxPlan.compileStep interface next state step).nextSbox := by
  cases step <;> rfl

private theorem stateStep_eq {columns : Nat} (read : Fin columns → F)
    (interface : PoseidonSboxPlan.Interface columns) (next : Nat)
    (state : PoseidonSboxPlan.State columns) (step : Permutation.Step) :
    stateStep read interface next (stateValues read state) step =
      stateValues read (PoseidonSboxPlan.compileStep interface next state step).state := by
  apply Vector.ext
  intro lane bounded
  exact congrFun ((stateStep_value read interface next state step).trans
    (stateValues_value read _).symm) ⟨lane, bounded⟩

/-- Preserve the existing schedule and lane order while carrying stored
numeric state. Every emitted value comes from the original sparse reads. -/
def rowsFrom {columns : Nat} (read : Fin columns → F)
    (interface : PoseidonSboxPlan.Interface columns) :
    Nat → Vector F 8 → List Permutation.Step → List PortValues
  | _, _, [] => []
  | next, state, step :: rest =>
      rowsStep read interface next state step ++
        rowsFrom read interface (nextIndex next step)
          (stateStep read interface next state step) rest

/-- The numeric schedule returns exactly the evaluated canonical S-box rows,
without an assignment-validity or constant-column premise. -/
theorem rowsFrom_value {columns : Nat} (read : Fin columns → F)
    (interface : PoseidonSboxPlan.Interface columns) (next : Nat)
    (state : PoseidonSboxPlan.State columns) (steps : List Permutation.Step) :
    rowsFrom read interface next (stateValues read state) steps =
      (PoseidonSboxPlan.compile interface next state steps).rows.map
        (rowValues read) := by
  induction steps generalizing next state with
  | nil => rfl
  | cons step rest inductionHypothesis =>
      rw [rowsFrom, rowsStep_value, stateStep_eq,
        nextIndex_value interface next state step, inductionHypothesis]
      simp only [PoseidonSboxPlan.compile, List.map_append]

/-- Carry the final stored state with the row list. The paired step shares
its retained reads between row construction and the next linear state. -/
def rowsWithState {columns : Nat} (read : Fin columns → F)
    (interface : PoseidonSboxPlan.Interface columns) :
    Nat → Vector F 8 → List Permutation.Step → List PortValues × Vector F 8
  | _, state, [] => ([], state)
  | next, state, step :: rest =>
      let current := stepValues read interface next state step
      let remaining := rowsWithState read interface (nextIndex next step) current.2 rest
      (current.1 ++ remaining.1, remaining.2)

private theorem rowsWithState_rows {columns : Nat} (read : Fin columns → F)
    (interface : PoseidonSboxPlan.Interface columns) (next : Nat)
    (state : Vector F 8) (steps : List Permutation.Step) :
    (rowsWithState read interface next state steps).1 =
      rowsFrom read interface next state steps := by
  induction steps generalizing next state with
  | nil => rfl
  | cons step rest inductionHypothesis =>
      rw [rowsWithState, stepValues_value]
      dsimp only
      rw [inductionHypothesis, rowsFrom]

private theorem rowsWithState_state {columns : Nat} (read : Fin columns → F)
    (interface : PoseidonSboxPlan.Interface columns) (next : Nat)
    (state : PoseidonSboxPlan.State columns) (steps : List Permutation.Step) :
    (rowsWithState read interface next (stateValues read state) steps).2 =
      stateValues read (PoseidonSboxPlan.compile interface next state steps).state := by
  induction steps generalizing next state with
  | nil => rfl
  | cons step rest inductionHypothesis =>
      rw [rowsWithState, stepValues_value]
      dsimp only
      rw [stateStep_eq, nextIndex_value, inductionHypothesis]
      rfl

private def referenceRowValues {columns : Nat} (read : Fin columns → F) :
    PoseidonSboxPlan.Row columns → PortValues
  | .sbox forms => rowValues read forms
  | .pin forms => RowSemantics.pin (forms.selector.evalSparse read)
      (forms.value.evalSparse read)

private theorem referenceRowValues_get {columns : Nat} (read : Fin columns → F)
    (row : PoseidonSboxPlan.Row columns) (port : Fin matrixCount) :
    (referenceRowValues read row).get port = (row.portForm port).evalSparse read := by
  cases row with
  | sbox forms => exact rowValues_get read forms port
  | pin forms =>
      simp only [referenceRowValues, SparseForm.evalSparse_eq_eval]
      fin_cases port <;>
        simp [PoseidonSboxPlan.Row.portForm, PoseidonSboxPlan.Row.meaningfulForm,
          PinRow.Forms.meaningfulForm, meaningfulPort?, RowSemantics.pin,
          RowSemantics.multiplication, RowSemantics.general, PortValues.get]

private def pinValues {columns : Nat} (read : Fin columns → F)
    (interface : PoseidonSboxPlan.Interface columns) : List PortValues :=
  let output := stateValues read (PoseidonSboxPlan.directOutput interface)
  List.ofFn fun lane : Fin 8 => RowSemantics.pin (read interface.oneColumn)
    ((interface.output lane).evalSparse read - output.get lane)

private theorem pinValues_value {columns : Nat} (read : Fin columns → F)
    (interface : PoseidonSboxPlan.Interface columns) :
    pinValues read interface =
      (PoseidonSboxPlan.outputRows interface).map
        (fun row => referenceRowValues read (.pin row)) := by
  unfold pinValues PoseidonSboxPlan.outputRows
  rw [List.map_ofFn]
  apply congrArg List.ofFn
  funext lane
  simp only [Function.comp_apply, referenceRowValues, SparseForm.evalSparse_eq_eval,
    PoseidonSboxPlan.selector, SparseForm.singleton_eval, one_mul,
    PoseidonSboxPlan.outputDifference, SparseForm.add_eval, SparseForm.scale_eval,
    PoseidonSboxPlan.trace_state_eq_directOutput, stateValues_value,
    SparseLayer.evalState, sub_eq_add_neg, neg_one_mul]

/-- Compute all 94 existing port-value records. The final eight pins reuse
the stored final state, so every retained S-box output is evaluated once. -/
def values {columns : Nat} (read : Fin columns → F)
    (interface : PoseidonSboxPlan.Interface columns) : List PortValues :=
  let produced := rowsWithState read interface 0
    (stateValues read interface.input) Permutation.schedule
  let selector := read interface.oneColumn
  produced.1 ++ List.ofFn fun lane : Fin 8 =>
    RowSemantics.pin selector
      ((interface.output lane).evalSparse read - produced.2.get lane)

private theorem values_eq_rows {columns : Nat} (read : Fin columns → F)
    (interface : PoseidonSboxPlan.Interface columns) :
    values read interface = (PoseidonSboxPlan.rows interface).map
      (referenceRowValues read) := by
  have finalState :
      (rowsWithState read interface 0 (stateValues read interface.input)
        Permutation.schedule).2 =
      stateValues read (PoseidonSboxPlan.directOutput interface) := by
    rw [rowsWithState_state]
    simpa only [PoseidonSboxPlan.trace] using
      congrArg (stateValues read) (PoseidonSboxPlan.trace_state_eq_directOutput interface)
  unfold values
  dsimp only
  rw [rowsWithState_rows, finalState]
  change rowsFrom read interface 0 (stateValues read interface.input)
    Permutation.schedule ++ pinValues read interface = _
  rw [rowsFrom_value, pinValues_value]
  simp only [PoseidonSboxPlan.rows, PoseidonSboxPlan.trace, List.map_append,
    List.map_map, Function.comp_def, referenceRowValues]

/-- The numeric result has exactly the row count of the canonical template. -/
theorem values_length {columns : Nat} (read : Fin columns → F)
    (interface : PoseidonSboxPlan.Interface columns) :
    (values read interface).length = 94 := by
  rw [values_eq_rows, List.length_map, PoseidonSboxPlan.rows_length]

/-- Store the complete invocation result for constant-time indexed reads. -/
def stored {columns : Nat} (read : Fin columns → F)
    (interface : PoseidonSboxPlan.Interface columns) : Vector PortValues 94 :=
  ⟨(values read interface).toArray, by simp only [List.size_toArray, values_length]⟩

/-- Every computed port equals the original row's sparse evaluation. This
includes the empty ports and requires no valid-row or selector assumption. -/
theorem stored_value {columns : Nat} (read : Fin columns → F)
    (interface : PoseidonSboxPlan.Interface columns) (row : Fin 94)
    (port : Fin matrixCount) :
    ((stored read interface).get row).get port =
      (((PoseidonSboxPlan.rows interface).get
        ⟨row.val, by rw [PoseidonSboxPlan.rows_length]; exact row.isLt⟩).portForm port).evalSparse read := by
  change ((values read interface).toArray[row.val]).get port = _
  rw [List.getElem_toArray]
  simp only [values_eq_rows, List.getElem_map]
  exact referenceRowValues_get read _ port

end NightstreamFPrime.Export.Stage1.PiDECPoseidonNumericRows
