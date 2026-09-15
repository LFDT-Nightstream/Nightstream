import NightstreamFPrime.Layout.ProductionRelation.PoseidonSboxPlan
import NightstreamFPrime.Layout.ProductionRelation.SparseEvaluation

/-!
Numeric evaluation of one existing Poseidon sparse compiler step under an
arbitrary assignment. Retained S-box outputs are read, and every state
transition is materialized in the existing eight-field vector.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECPoseidonNumericStep

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.ProductionRelation
open NightstreamFPrime.Spec.ProductionRelation.RowSemantics (PortValues)
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Layout.ProductionRelation

private theorem materialize_get (state : Layer.FState) :
    (Vector.ofFn state).get = state := by
  funext lane
  change (Vector.ofFn state)[lane.val] = state lane
  rw [Vector.getElem_ofFn]

/-- Materialize the eight input values without constructing new row forms. -/
def stateValues {logicalWidth : Nat} (read : Fin logicalWidth → F)
    (state : PoseidonSboxPlan.State logicalWidth) : Vector F 8 :=
  Vector.ofFn fun lane => (state lane).evalSparse read

private theorem stateValues_get {logicalWidth : Nat} (read : Fin logicalWidth → F)
    (state : PoseidonSboxPlan.State logicalWidth) (lane : Fin 8) :
    (stateValues read state).get lane = (state lane).evalSparse read := by
  change (Vector.ofFn (fun selected : Fin 8 => (state selected).evalSparse read))[lane.val] = _
  rw [Vector.getElem_ofFn]

theorem stateValues_value {logicalWidth : Nat} (read : Fin logicalWidth → F)
    (state : PoseidonSboxPlan.State logicalWidth) :
    (stateValues read state).get = SparseLayer.evalState read state := by
  funext lane
  rw [stateValues_get, SparseForm.evalSparse_eq_eval]
  rfl

/-- Read the same retained form as the symbolic compiler, including its
existing zero result outside the retained S-box domain. -/
def retainedValue {logicalWidth : Nat} (read : Fin logicalWidth → F)
    (interface : PoseidonSboxPlan.Interface logicalWidth) (index : Nat) : F :=
  (PoseidonSboxPlan.sboxOutputAt interface index).evalSparse read

private def fullState {logicalWidth : Nat} (read : Fin logicalWidth → F)
    (interface : PoseidonSboxPlan.Interface logicalWidth) (nextSbox : Nat) : Vector F 8 :=
  let outputs := Vector.ofFn fun lane : Fin 8 =>
    retainedValue read interface (nextSbox + lane.val)
  Vector.ofFn (Layer.externalF outputs.get)

private theorem fullState_value {logicalWidth : Nat} (read : Fin logicalWidth → F)
    (interface : PoseidonSboxPlan.Interface logicalWidth) (nextSbox : Nat) :
    (fullState read interface nextSbox).get =
      SparseLayer.evalState read
        (SparseLayer.external (PoseidonSboxPlan.fullOutput interface nextSbox)) := by
  simp only [fullState, materialize_get]
  funext lane
  change Layer.externalF (fun selected =>
      retainedValue read interface (nextSbox + selected.val)) lane =
    (SparseLayer.external (PoseidonSboxPlan.fullOutput interface nextSbox) lane).eval read
  rw [SparseLayer.eval_external]
  apply congrArg (fun state : Layer.FState => Layer.externalF state lane)
  funext selected
  simp only [retainedValue, SparseForm.evalSparse_eq_eval,
    SparseLayer.evalState, PoseidonSboxPlan.fullOutput]

private def partialState {logicalWidth : Nat} (read : Fin logicalWidth → F)
    (interface : PoseidonSboxPlan.Interface logicalWidth) (nextSbox : Nat)
    (state : Vector F 8) : Vector F 8 :=
  let output := retainedValue read interface nextSbox
  let replaced := Vector.ofFn fun lane : Fin 8 =>
    if lane.val = 0 then output else state.get lane
  Vector.ofFn (Layer.internalF replaced.get)

private theorem partialState_value {logicalWidth : Nat} (read : Fin logicalWidth → F)
    (interface : PoseidonSboxPlan.Interface logicalWidth) (nextSbox : Nat)
    (state : PoseidonSboxPlan.State logicalWidth) :
    (partialState read interface nextSbox (stateValues read state)).get =
      SparseLayer.evalState read
        (SparseLayer.internal (PoseidonSboxPlan.partialState interface nextSbox state)) := by
  simp only [partialState, materialize_get]
  funext lane
  change Layer.internalF (fun selected =>
      if selected.val = 0 then retainedValue read interface nextSbox
      else (stateValues read state).get selected) lane =
    (SparseLayer.internal (PoseidonSboxPlan.partialState interface nextSbox state) lane).eval read
  rw [SparseLayer.eval_internal]
  apply congrArg (fun numeric : Layer.FState => Layer.internalF numeric lane)
  funext selected
  by_cases zero : selected.val = 0 <;>
    simp only [PoseidonSboxPlan.partialState, PoseidonSboxPlan.partialOutput,
      SparseLayer.evalState, zero, if_true, if_false, retainedValue,
      stateValues_get, SparseForm.evalSparse_eq_eval]

/-- Apply only the existing linear state transition and retained-output
substitution, then store all eight fields before another step can read them. -/
def stateStep {logicalWidth : Nat} (read : Fin logicalWidth → F)
    (interface : PoseidonSboxPlan.Interface logicalWidth) (nextSbox : Nat)
    (state : Vector F 8) : Permutation.Step → Vector F 8
  | .initialLayer => Vector.ofFn (Layer.externalF state.get)
  | .initialFullRound _ => fullState read interface nextSbox
  | .partialRound _ => partialState read interface nextSbox state
  | .terminalFullRound _ => fullState read interface nextSbox

/-- The stored numeric state is the exact evaluation of compileStep.state.
The read function is arbitrary: no one-column, S-box, row-satisfaction or
parent-opening premise is required. -/
theorem stateStep_value {logicalWidth : Nat} (read : Fin logicalWidth → F)
    (interface : PoseidonSboxPlan.Interface logicalWidth) (nextSbox : Nat)
    (state : PoseidonSboxPlan.State logicalWidth) (step : Permutation.Step) :
    (stateStep read interface nextSbox (stateValues read state) step).get =
      SparseLayer.evalState read (PoseidonSboxPlan.compileStep interface nextSbox state step).state := by
  cases step with
  | initialLayer =>
      simp only [stateStep, materialize_get, stateValues_value, PoseidonSboxPlan.compileStep]
      funext lane
      exact (SparseLayer.eval_external read state lane).symm
  | initialFullRound round => exact fullState_value read interface nextSbox
  | partialRound round => exact partialState_value read interface nextSbox state
  | terminalFullRound round => exact fullState_value read interface nextSbox

/-- Store one existing S-box row under the same arbitrary sparse read. -/
def rowValues {logicalWidth : Nat} (read : Fin logicalWidth → F)
    (forms : SboxRow.Forms logicalWidth) : PortValues :=
  RowSemantics.sbox (forms.selector.evalSparse read)
    (forms.input.evalSparse read) (forms.output.evalSparse read)

/-- Every stored port equals the evaluation of its existing sparse form,
including the empty ports and matrix slot 13. -/
theorem rowValues_get {logicalWidth : Nat} (read : Fin logicalWidth → F)
    (forms : SboxRow.Forms logicalWidth) (port : Fin matrixCount) :
    (rowValues read forms).get port = (forms.portForm port).evalSparse read := by
  simp only [rowValues, SparseForm.evalSparse_eq_eval]
  fin_cases port <;>
    simp [SboxRow.Forms.portForm, SboxRow.Forms.meaningfulForm,
      meaningfulPort?, RowSemantics.sbox, RowSemantics.general,
      RowSemantics.PortValues.get]

private theorem selector_value {logicalWidth : Nat} (read : Fin logicalWidth → F)
    (interface : PoseidonSboxPlan.Interface logicalWidth) :
    (PoseidonSboxPlan.selector interface).evalSparse read =
      read interface.oneColumn := by
  simp only [PoseidonSboxPlan.selector, SparseForm.evalSparse_eq_eval,
    SparseForm.singleton_eval, one_mul]

private theorem addConstant_value {logicalWidth : Nat}
    (read : Fin logicalWidth → F) (oneColumn : Fin logicalWidth)
    (form : SparseForm logicalWidth) (constant : F) :
    (SparseLayer.addConstant oneColumn form constant).evalSparse read =
      form.evalSparse read + constant * read oneColumn := by
  simp only [SparseForm.evalSparse_eq_eval, SparseLayer.addConstant,
    SparseLayer.add, SparseLayer.constant, SparseForm.add_eval,
    SparseForm.singleton_eval]

private def fullRowValues {logicalWidth : Nat} (read : Fin logicalWidth → F)
    (interface : PoseidonSboxPlan.Interface logicalWidth)
    (constants : List (List Nat)) (round nextSbox : Nat)
    (state : Vector F 8) : List PortValues :=
  let selector := read interface.oneColumn
  List.ofFn fun lane : Fin 8 =>
    RowSemantics.sbox selector
      (state.get lane + Spec.Poseidon2.constantAt constants round lane.val * selector)
      (retainedValue read interface (nextSbox + lane.val))

private theorem fullRowValues_value {logicalWidth : Nat}
    (read : Fin logicalWidth → F) (interface : PoseidonSboxPlan.Interface logicalWidth)
    (constants : List (List Nat)) (round nextSbox : Nat)
    (state : PoseidonSboxPlan.State logicalWidth) :
    fullRowValues read interface constants round nextSbox (stateValues read state) =
      (PoseidonSboxPlan.fullRows interface constants round nextSbox state).map
        (rowValues read) := by
  unfold fullRowValues PoseidonSboxPlan.fullRows
  rw [List.map_ofFn]
  apply congrArg List.ofFn
  funext lane
  simp only [Function.comp_apply, rowValues, selector_value,
    PoseidonSboxPlan.fullInput, addConstant_value, PoseidonSboxPlan.fullOutput,
    retainedValue, stateValues_get]

/-- Compute the existing step rows from the stored state and retained reads.
Round constants multiply the selected one-column read; it need not equal one. -/
def rowsStep {logicalWidth : Nat} (read : Fin logicalWidth → F)
    (interface : PoseidonSboxPlan.Interface logicalWidth) (nextSbox : Nat)
    (state : Vector F 8) : Permutation.Step → List PortValues
  | .initialLayer => []
  | .initialFullRound round =>
      fullRowValues read interface Spec.Poseidon2.initialConstants round nextSbox state
  | .partialRound round =>
      let selector := read interface.oneColumn
      [RowSemantics.sbox selector
        (state.get 0 + Spec.Poseidon2.ofNat
          (Spec.Poseidon2.internalConstants.getD round 0) * selector)
        (retainedValue read interface nextSbox)]
  | .terminalFullRound round =>
      fullRowValues read interface Spec.Poseidon2.terminalConstants round nextSbox state

/-- Numeric row generation equals evaluation of the existing compileStep rows
in their exact order. The assignment and retained values are arbitrary. -/
theorem rowsStep_value {logicalWidth : Nat} (read : Fin logicalWidth → F)
    (interface : PoseidonSboxPlan.Interface logicalWidth) (nextSbox : Nat)
    (state : PoseidonSboxPlan.State logicalWidth) (step : Permutation.Step) :
    rowsStep read interface nextSbox (stateValues read state) step =
      (PoseidonSboxPlan.compileStep interface nextSbox state step).rows.map
        (rowValues read) := by
  cases step with
  | initialLayer => rfl
  | initialFullRound round =>
      exact fullRowValues_value read interface Spec.Poseidon2.initialConstants
        round nextSbox state
  | partialRound round =>
      simp only [rowsStep, PoseidonSboxPlan.compileStep, PoseidonSboxPlan.partialRows,
        List.map_cons, List.map_nil, rowValues, selector_value,
        PoseidonSboxPlan.partialInput, addConstant_value, PoseidonSboxPlan.partialOutput,
        retainedValue, stateValues_get]
  | terminalFullRound round =>
      exact fullRowValues_value read interface Spec.Poseidon2.terminalConstants
        round nextSbox state

private def fullStepValues {logicalWidth : Nat} (read : Fin logicalWidth → F)
    (interface : PoseidonSboxPlan.Interface logicalWidth)
    (constants : List (List Nat)) (round nextSbox : Nat)
    (state : Vector F 8) : List PortValues × Vector F 8 :=
  let selector := read interface.oneColumn
  let outputs := Vector.ofFn fun lane : Fin 8 =>
    retainedValue read interface (nextSbox + lane.val)
  (List.ofFn fun lane : Fin 8 =>
      RowSemantics.sbox selector
        (state.get lane + Spec.Poseidon2.constantAt constants round lane.val * selector)
        (outputs.get lane),
    Vector.ofFn (Layer.externalF outputs.get))

private theorem fullStepValues_value {logicalWidth : Nat}
    (read : Fin logicalWidth → F) (interface : PoseidonSboxPlan.Interface logicalWidth)
    (constants : List (List Nat)) (round nextSbox : Nat) (state : Vector F 8) :
    fullStepValues read interface constants round nextSbox state =
      (fullRowValues read interface constants round nextSbox state,
        fullState read interface nextSbox) := by
  apply Prod.ext
  · simp only [fullStepValues, fullRowValues, materialize_get]
  · rfl

/-- Compute each retained output once, then share it between the existing
row values and the stored next state. Missing retained indices still read zero. -/
def stepValues {logicalWidth : Nat} (read : Fin logicalWidth → F)
    (interface : PoseidonSboxPlan.Interface logicalWidth) (nextSbox : Nat)
    (state : Vector F 8) : Permutation.Step → List PortValues × Vector F 8
  | .initialLayer => ([], Vector.ofFn (Layer.externalF state.get))
  | .initialFullRound round =>
      fullStepValues read interface Spec.Poseidon2.initialConstants round nextSbox state
  | .partialRound round =>
      let selector := read interface.oneColumn
      let output := retainedValue read interface nextSbox
      let replaced := Vector.ofFn fun lane : Fin 8 =>
        if lane.val = 0 then output else state.get lane
      ([RowSemantics.sbox selector
          (state.get 0 + Spec.Poseidon2.ofNat
            (Spec.Poseidon2.internalConstants.getD round 0) * selector)
          output],
        Vector.ofFn (Layer.internalF replaced.get))
  | .terminalFullRound round =>
      fullStepValues read interface Spec.Poseidon2.terminalConstants round nextSbox state

/-- Total equality for every numeric state, sparse read, round and retained
index. No selector or row-satisfaction assumption is required. -/
theorem stepValues_value {logicalWidth : Nat} (read : Fin logicalWidth → F)
    (interface : PoseidonSboxPlan.Interface logicalWidth) (nextSbox : Nat)
    (state : Vector F 8) (step : Permutation.Step) :
    stepValues read interface nextSbox state step =
      (rowsStep read interface nextSbox state step,
        stateStep read interface nextSbox state step) := by
  cases step with
  | initialLayer => rfl
  | initialFullRound round =>
      exact fullStepValues_value read interface Spec.Poseidon2.initialConstants
        round nextSbox state
  | partialRound round => rfl
  | terminalFullRound round =>
      exact fullStepValues_value read interface Spec.Poseidon2.terminalConstants
        round nextSbox state

end NightstreamFPrime.Export.Stage1.PiDECPoseidonNumericStep
