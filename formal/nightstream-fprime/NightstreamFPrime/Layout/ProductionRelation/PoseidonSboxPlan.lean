import NightstreamFPrime.Layout.ProductionRelation.PinRow
import NightstreamFPrime.Layout.ProductionRelation.PoseidonRetainedSlots
import NightstreamFPrime.Layout.ProductionRelation.SboxRow
import NightstreamFPrime.Layout.ProductionRelation.SparseLayer

/-!
Owns the compact direct selective plan for one Poseidon2 permutation. The
plan retains and constrains only the 86 S-box outputs. All linear layers are
computed as sparse forms. Eight final pin rows bind the computed output to
the caller-owned output forms.

This module is one fixed-size template. Invocation placement belongs to the
Stage 1 export layer.
-/

namespace NightstreamFPrime.Layout.ProductionRelation.PoseidonSboxPlan

open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CCSResidualTable
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

abbrev State (logicalWidth : Nat) := SparseLayer.State logicalWidth

/-- Final-assignment forms supplied by one invocation owner. -/
structure Interface (logicalWidth : Nat) where
  oneColumn : Fin logicalWidth
  input : State logicalWidth
  sboxOutput : Fin PoseidonRetainedSlots.rows.length → SparseForm logicalWidth
  output : State logicalWidth

def sboxOutputAt {logicalWidth : Nat} (interface : Interface logicalWidth)
    (index : Nat) : SparseForm logicalWidth :=
  if bounded : index < PoseidonRetainedSlots.rows.length then
    interface.sboxOutput ⟨index, bounded⟩
  else
    .empty

def selector {logicalWidth : Nat} (interface : Interface logicalWidth) :
    SparseForm logicalWidth :=
  SparseForm.singleton interface.oneColumn 1

def fullInput {logicalWidth : Nat} (interface : Interface logicalWidth)
    (constants : List (List Nat)) (round : Nat) (state : State logicalWidth)
    (lane : Fin 8) : SparseForm logicalWidth :=
  SparseLayer.addConstant interface.oneColumn (state lane)
    (Spec.Poseidon2.constantAt constants round lane.val)

def fullOutput {logicalWidth : Nat} (interface : Interface logicalWidth)
    (nextSbox : Nat) : State logicalWidth :=
  fun lane => sboxOutputAt interface (nextSbox + lane.val)

def fullRows {logicalWidth : Nat} (interface : Interface logicalWidth)
    (constants : List (List Nat)) (round nextSbox : Nat)
    (state : State logicalWidth) : List (SboxRow.Forms logicalWidth) :=
  List.ofFn fun lane =>
    { selector := selector interface
      input := fullInput interface constants round state lane
      output := fullOutput interface nextSbox lane }

def partialInput {logicalWidth : Nat} (interface : Interface logicalWidth)
    (round : Nat) (state : State logicalWidth) : SparseForm logicalWidth :=
  SparseLayer.addConstant interface.oneColumn (state 0)
    (Spec.Poseidon2.ofNat
      (Spec.Poseidon2.internalConstants.getD round 0))

def partialOutput {logicalWidth : Nat} (interface : Interface logicalWidth)
    (nextSbox : Nat) : SparseForm logicalWidth :=
  sboxOutputAt interface nextSbox

def partialRows {logicalWidth : Nat} (interface : Interface logicalWidth)
    (round nextSbox : Nat) (state : State logicalWidth) :
    List (SboxRow.Forms logicalWidth) :=
  [{ selector := selector interface
     input := partialInput interface round state
     output := partialOutput interface nextSbox }]

def partialState {logicalWidth : Nat} (interface : Interface logicalWidth)
    (nextSbox : Nat) (state : State logicalWidth) : State logicalWidth :=
  fun lane => if lane.val = 0 then partialOutput interface nextSbox else state lane

def SboxRowsZero {logicalWidth : Nat}
    (assignment : Assignment F logicalWidth)
    (rows : List (SboxRow.Forms logicalWidth)) : Prop :=
  ∀ row ∈ rows, row.residual assignment = 0

private theorem seventhPower_eq_sboxF (value : F) :
    Spec.ProductionRelation.RowSemantics.seventhPower value =
      Layer.sboxF value := by
  simp [Spec.ProductionRelation.RowSemantics.seventhPower,
    Spec.Folding.PiCCS.PaperJoint.CCSResidualTable.pow,
    Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier.baseOps,
    Layer.sboxF, Spec.Poseidon2.sbox, mul_assoc]

theorem fullRowsZero_lane {logicalWidth : Nat}
    (interface : Interface logicalWidth) (constants : List (List Nat))
    (round nextSbox : Nat) (state : State logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment interface.oneColumn = 1)
    (rowsZero : SboxRowsZero assignment
      (fullRows interface constants round nextSbox state))
    (lane : Fin 8) :
    (fullOutput interface nextSbox lane).eval assignment =
      Layer.sboxF ((fullInput interface constants round state lane).eval
        assignment) := by
  let forms : SboxRow.Forms logicalWidth :=
    { selector := selector interface
      input := fullInput interface constants round state lane
      output := fullOutput interface nextSbox lane }
  have member : forms ∈ fullRows interface constants round nextSbox state := by
    unfold fullRows forms
    exact List.mem_ofFn.mpr ⟨lane, rfl⟩
  have zero := rowsZero forms member
  have preserves : forms.Preserves assignment
      ((fullInput interface constants round state lane).eval assignment)
      ((fullOutput interface nextSbox lane).eval assignment) := by
    refine ⟨?_, rfl, rfl⟩
    simp [forms, selector, one]
  have semantic := (SboxRow.Forms.residual_zero_iff forms assignment _ _
    preserves).mp zero
  rw [seventhPower_eq_sboxF] at semantic
  exact semantic.symm

theorem partialRowsZero_output {logicalWidth : Nat}
    (interface : Interface logicalWidth) (round nextSbox : Nat)
    (state : State logicalWidth) (assignment : Assignment F logicalWidth)
    (one : assignment interface.oneColumn = 1)
    (rowsZero : SboxRowsZero assignment
      (partialRows interface round nextSbox state)) :
    (partialOutput interface nextSbox).eval assignment =
      Layer.sboxF ((partialInput interface round state).eval assignment) := by
  let forms : SboxRow.Forms logicalWidth :=
    { selector := selector interface
      input := partialInput interface round state
      output := partialOutput interface nextSbox }
  have member : forms ∈ partialRows interface round nextSbox state := by
    simp [partialRows, forms]
  have zero := rowsZero forms member
  have preserves : forms.Preserves assignment
      ((partialInput interface round state).eval assignment)
      ((partialOutput interface nextSbox).eval assignment) := by
    refine ⟨?_, rfl, rfl⟩
    simp [forms, selector, one]
  have semantic := (SboxRow.Forms.residual_zero_iff forms assignment _ _
    preserves).mp zero
  rw [seventhPower_eq_sboxF] at semantic
  exact semantic.symm

/-- One direct schedule step and its exact retained-row suffix. -/
structure StepResult (logicalWidth : Nat) where
  nextSbox : Nat
  state : State logicalWidth
  rows : List (SboxRow.Forms logicalWidth)

def compileStep {logicalWidth : Nat} (interface : Interface logicalWidth)
    (nextSbox : Nat) (state : State logicalWidth) :
    Permutation.Step → StepResult logicalWidth
  | .initialLayer =>
      { nextSbox := nextSbox
        state := SparseLayer.external state
        rows := [] }
  | .initialFullRound round =>
      let outputs := fullOutput interface nextSbox
      { nextSbox := nextSbox + 8
        state := SparseLayer.external outputs
        rows := fullRows interface Spec.Poseidon2.initialConstants round
          nextSbox state }
  | .partialRound round =>
      { nextSbox := nextSbox + 1
        state := SparseLayer.internal (partialState interface nextSbox state)
        rows := partialRows interface round nextSbox state }
  | .terminalFullRound round =>
      let outputs := fullOutput interface nextSbox
      { nextSbox := nextSbox + 8
        state := SparseLayer.external outputs
        rows := fullRows interface Spec.Poseidon2.terminalConstants round
          nextSbox state }

/-- Zero residuals of one compact step force the exact reference Poseidon2
step on the evaluated sparse-form state. -/
theorem compileStep_sound {logicalWidth : Nat}
    (interface : Interface logicalWidth) (nextSbox : Nat)
    (state : State logicalWidth) (step : Permutation.Step)
    (assignment : Assignment F logicalWidth)
    (one : assignment interface.oneColumn = 1)
    (rowsZero : SboxRowsZero assignment
      (compileStep interface nextSbox state step).rows) :
    SparseLayer.evalState assignment
        (compileStep interface nextSbox state step).state =
      Permutation.applyF step (SparseLayer.evalState assignment state) := by
  cases step with
  | initialLayer =>
      funext lane
      exact SparseLayer.eval_external assignment state lane
  | initialFullRound round =>
      let outputs := fullOutput interface nextSbox
      have outputsEqual :
          SparseLayer.evalState assignment outputs =
            fun lane => Layer.sboxF
              ((fullInput interface Spec.Poseidon2.initialConstants round
                state lane).eval assignment) := by
        funext lane
        exact fullRowsZero_lane interface Spec.Poseidon2.initialConstants round
          nextSbox state assignment one rowsZero lane
      funext lane
      change
        (SparseLayer.external outputs lane).eval assignment =
          Layer.fullF Spec.Poseidon2.initialConstants round
            (SparseLayer.evalState assignment state) lane
      rw [SparseLayer.eval_external]
      unfold Layer.fullF
      apply congrFun (congrArg Layer.externalF ?_) lane
      funext index
      rw [outputsEqual]
      simp [fullInput, SparseLayer.evalState, one]
  | partialRound round =>
      have outputEqual := partialRowsZero_output interface round nextSbox state
        assignment one rowsZero
      funext lane
      change
        (SparseLayer.internal (partialState interface nextSbox state) lane).eval
            assignment =
          Layer.partialF round (SparseLayer.evalState assignment state) lane
      rw [SparseLayer.eval_internal]
      unfold Layer.partialF
      apply congrFun (congrArg Layer.internalF ?_) lane
      funext index
      by_cases zero : index.val = 0
      · have indexEq : index = 0 := Fin.ext zero
        subst index
        simp [partialState, outputEqual, partialInput, SparseLayer.evalState,
          one]
      · simp [partialState, SparseLayer.evalState, zero]
  | terminalFullRound round =>
      let outputs := fullOutput interface nextSbox
      have outputsEqual :
          SparseLayer.evalState assignment outputs =
            fun lane => Layer.sboxF
              ((fullInput interface Spec.Poseidon2.terminalConstants round
                state lane).eval assignment) := by
        funext lane
        exact fullRowsZero_lane interface Spec.Poseidon2.terminalConstants round
          nextSbox state assignment one rowsZero lane
      funext lane
      change
        (SparseLayer.external outputs lane).eval assignment =
          Layer.fullF Spec.Poseidon2.terminalConstants round
            (SparseLayer.evalState assignment state) lane
      rw [SparseLayer.eval_external]
      unfold Layer.fullF
      apply congrFun (congrArg Layer.externalF ?_) lane
      funext index
      rw [outputsEqual]
      simp [fullInput, SparseLayer.evalState, one]

/-- Structural schedule compiler. Rows stay in exact schedule and lane order. -/
def compile {logicalWidth : Nat} (interface : Interface logicalWidth) :
    Nat → State logicalWidth → List Permutation.Step →
      StepResult logicalWidth
  | nextSbox, state, [] =>
      { nextSbox := nextSbox, state := state, rows := [] }
  | nextSbox, state, step :: rest =>
      let head := compileStep interface nextSbox state step
      let tail := compile interface head.nextSbox head.state rest
      { nextSbox := tail.nextSbox
        state := tail.state
        rows := head.rows ++ tail.rows }

private theorem sboxRowsZero_append {logicalWidth : Nat}
    (assignment : Assignment F logicalWidth)
    (left right : List (SboxRow.Forms logicalWidth)) :
    SboxRowsZero assignment (left ++ right) ↔
      SboxRowsZero assignment left ∧ SboxRowsZero assignment right := by
  simp [SboxRowsZero, or_imp, forall_and]

/-- Zero residuals of a compact step sequence force the exact reference
Poseidon2 step sequence. -/
theorem compile_sound {logicalWidth : Nat}
    (interface : Interface logicalWidth) (nextSbox : Nat)
    (state : State logicalWidth) (steps : List Permutation.Step)
    (assignment : Assignment F logicalWidth)
    (one : assignment interface.oneColumn = 1)
    (rowsZero : SboxRowsZero assignment
      (compile interface nextSbox state steps).rows) :
    SparseLayer.evalState assignment
        (compile interface nextSbox state steps).state =
      Permutation.runF steps (SparseLayer.evalState assignment state) := by
  induction steps generalizing nextSbox state with
  | nil => rfl
  | cons step rest inductionHypothesis =>
      let head := compileStep interface nextSbox state step
      let tail := compile interface head.nextSbox head.state rest
      have rowsEqual :
          (compile interface nextSbox state (step :: rest)).rows =
            head.rows ++ tail.rows := by
        rfl
      rw [rowsEqual, sboxRowsZero_append] at rowsZero
      have headSound := compileStep_sound interface nextSbox state step
        assignment one rowsZero.1
      have tailSound := inductionHypothesis head.nextSbox head.state rowsZero.2
      change SparseLayer.evalState assignment tail.state =
        Permutation.runF (step :: rest)
          (SparseLayer.evalState assignment state)
      rw [tailSound]
      change
        Permutation.runF rest
            (SparseLayer.evalState assignment head.state) =
          Permutation.runF rest
            (Permutation.applyF step
              (SparseLayer.evalState assignment state))
      rw [headSound]

def trace {logicalWidth : Nat} (interface : Interface logicalWidth) :
    StepResult logicalWidth :=
  compile interface 0 interface.input Permutation.schedule

/-- Constant-time final state of the fixed production trace. The last
terminal full round owns S-box outputs 78 through 85 and the final external
linear layer. -/
def directOutput {logicalWidth : Nat} (interface : Interface logicalWidth) :
    State logicalWidth :=
  SparseLayer.external (fullOutput interface 78)

/-- The closed-form output is definitionally the output of the canonical
fixed schedule. -/
@[simp] theorem trace_state_eq_directOutput {logicalWidth : Nat}
    (interface : Interface logicalWidth) :
    (trace interface).state = directOutput interface := by
  rfl

def outputDifference {logicalWidth : Nat}
    (interface : Interface logicalWidth) (lane : Fin 8) :
    SparseForm logicalWidth :=
  SparseForm.add (interface.output lane)
    (SparseForm.scale (-1) ((trace interface).state lane))

def outputRows {logicalWidth : Nat} (interface : Interface logicalWidth) :
    List (PinRow.Forms logicalWidth) :=
  List.ofFn fun lane =>
    { selector := selector interface
      value := outputDifference interface lane }

def PinRowsZero {logicalWidth : Nat}
    (assignment : Assignment F logicalWidth)
    (rows : List (PinRow.Forms logicalWidth)) : Prop :=
  ∀ row ∈ rows, row.residual assignment = 0

theorem outputRowsZero_lane {logicalWidth : Nat}
    (interface : Interface logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment interface.oneColumn = 1)
    (rowsZero : PinRowsZero assignment (outputRows interface))
    (lane : Fin 8) :
    (interface.output lane).eval assignment =
      ((trace interface).state lane).eval assignment := by
  let forms : PinRow.Forms logicalWidth :=
    { selector := selector interface
      value := outputDifference interface lane }
  have member : forms ∈ outputRows interface := by
    unfold outputRows forms
    exact List.mem_ofFn.mpr ⟨lane, rfl⟩
  have zero := rowsZero forms member
  have preserves : forms.Preserves assignment
      (forms.value.eval assignment) := by
    refine ⟨?_, rfl⟩
    simp [forms, selector, one]
  have differenceZero := (PinRow.Forms.residual_zero_iff forms assignment _
    preserves).mp zero
  have differenceEval :
      forms.value.eval assignment =
        (interface.output lane).eval assignment -
          ((trace interface).state lane).eval assignment := by
    simp [forms, outputDifference, sub_eq_add_neg]
  rw [differenceEval] at differenceZero
  exact Lean.Grind.AddCommGroup.sub_eq_zero_iff.mp differenceZero

inductive Row (logicalWidth : Nat) where
  | sbox : SboxRow.Forms logicalWidth → Row logicalWidth
  | pin : PinRow.Forms logicalWidth → Row logicalWidth
deriving Repr, DecidableEq

namespace Row

def meaningfulForm {logicalWidth : Nat} (row : Row logicalWidth)
    (port : Fin Spec.ProductionRelation.meaningfulPortCount) :
    SparseForm logicalWidth :=
  match row with
  | .sbox forms => forms.meaningfulForm port
  | .pin forms => forms.meaningfulForm port

def residual {logicalWidth : Nat} (row : Row logicalWidth)
    (assignment : Assignment F logicalWidth) : F :=
  match row with
  | .sbox forms => forms.residual assignment
  | .pin forms => forms.residual assignment

def portForm {logicalWidth : Nat} (row : Row logicalWidth)
    (port : Fin Spec.ProductionRelation.matrixCount) :
    SparseForm logicalWidth :=
  match ProductionRelation.meaningfulPort? port with
  | some meaningful => row.meaningfulForm meaningful
  | none => .empty

def portImages {logicalWidth : Nat} (row : Row logicalWidth)
    (assignment : Assignment F logicalWidth) :
    Fin Spec.ProductionRelation.matrixCount → F :=
  fun port => (row.portForm port).eval assignment

theorem polynomial_eq_residual {logicalWidth : Nat} (row : Row logicalWidth)
    (assignment : Assignment F logicalWidth) :
    evaluatePolynomial
        Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier.baseOps
        Spec.ProductionRelation.polynomial (row.portImages assignment) =
      row.residual assignment := by
  cases row <;> rfl

end Row

def rows {logicalWidth : Nat} (interface : Interface logicalWidth) :
    List (Row logicalWidth) :=
  (trace interface).rows.map Row.sbox ++ (outputRows interface).map Row.pin

def RowsZero {logicalWidth : Nat} (interface : Interface logicalWidth)
    (assignment : Assignment F logicalWidth) : Prop :=
  ∀ row ∈ rows interface, row.residual assignment = 0

/-- Exact S-box equations for every retained trace row. -/
def SboxEquations {logicalWidth : Nat} (interface : Interface logicalWidth)
    (assignment : Assignment F logicalWidth) : Prop :=
  ∀ forms ∈ (trace interface).rows,
    forms.output.eval assignment =
      Layer.sboxF (forms.input.eval assignment)

/-- Exact equality between caller-owned outputs and the compiled trace. -/
def OutputEquations {logicalWidth : Nat} (interface : Interface logicalWidth)
    (assignment : Assignment F logicalWidth) : Prop :=
  ∀ lane,
    (interface.output lane).eval assignment =
      ((trace interface).state lane).eval assignment

private theorem fullRows_selector {logicalWidth : Nat}
    (interface : Interface logicalWidth) (constants : List (List Nat))
    (round nextSbox : Nat) (state : State logicalWidth) :
    ∀ forms ∈ fullRows interface constants round nextSbox state,
      forms.selector = selector interface := by
  intro forms member
  unfold fullRows at member
  rcases List.mem_ofFn.mp member with ⟨lane, rfl⟩
  rfl

private theorem partialRows_selector {logicalWidth : Nat}
    (interface : Interface logicalWidth) (round nextSbox : Nat)
    (state : State logicalWidth) :
    ∀ forms ∈ partialRows interface round nextSbox state,
      forms.selector = selector interface := by
  simp [partialRows]

private theorem compileStep_rows_selector {logicalWidth : Nat}
    (interface : Interface logicalWidth) (nextSbox : Nat)
    (state : State logicalWidth) (step : Permutation.Step) :
    ∀ forms ∈ (compileStep interface nextSbox state step).rows,
      forms.selector = selector interface := by
  cases step with
  | initialLayer => simp [compileStep]
  | initialFullRound round =>
      exact fullRows_selector interface Spec.Poseidon2.initialConstants round
        nextSbox state
  | partialRound round =>
      exact partialRows_selector interface round nextSbox state
  | terminalFullRound round =>
      exact fullRows_selector interface Spec.Poseidon2.terminalConstants round
        nextSbox state

private theorem compile_rows_selector {logicalWidth : Nat}
    (interface : Interface logicalWidth) (nextSbox : Nat)
    (state : State logicalWidth) (steps : List Permutation.Step) :
    ∀ forms ∈ (compile interface nextSbox state steps).rows,
      forms.selector = selector interface := by
  induction steps generalizing nextSbox state with
  | nil => simp [compile]
  | cons step rest inductionHypothesis =>
      intro forms member
      simp only [compile, List.mem_append] at member
      rcases member with headMember | tailMember
      · exact compileStep_rows_selector interface nextSbox state step forms
          headMember
      · exact inductionHypothesis
          (compileStep interface nextSbox state step).nextSbox
          (compileStep interface nextSbox state step).state forms tailMember

private theorem trace_rows_selector {logicalWidth : Nat}
    (interface : Interface logicalWidth) :
    ∀ forms ∈ (trace interface).rows,
      forms.selector = selector interface := by
  exact compile_rows_selector interface 0 interface.input Permutation.schedule

theorem sboxRowsZero_of_equations {logicalWidth : Nat}
    (interface : Interface logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment interface.oneColumn = 1)
    (equations : SboxEquations interface assignment) :
    SboxRowsZero assignment (trace interface).rows := by
  intro forms member
  have preserves : forms.Preserves assignment
      (forms.input.eval assignment) (forms.output.eval assignment) := by
    refine ⟨?_, rfl, rfl⟩
    rw [trace_rows_selector interface forms member]
    simp [selector, one]
  apply (SboxRow.Forms.residual_zero_iff forms assignment _ _ preserves).mpr
  rw [seventhPower_eq_sboxF]
  exact (equations forms member).symm

theorem pinRowsZero_of_equations {logicalWidth : Nat}
    (interface : Interface logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment interface.oneColumn = 1)
    (equations : OutputEquations interface assignment) :
    PinRowsZero assignment (outputRows interface) := by
  intro forms member
  unfold outputRows at member
  rcases List.mem_ofFn.mp member with ⟨lane, rfl⟩
  let row : PinRow.Forms logicalWidth :=
    { selector := selector interface
      value := outputDifference interface lane }
  have preserves : row.Preserves assignment (row.value.eval assignment) := by
    refine ⟨?_, rfl⟩
    simp [row, selector, one]
  apply (PinRow.Forms.residual_zero_iff row assignment _ preserves).mpr
  have differenceEval :
      row.value.eval assignment =
        (interface.output lane).eval assignment -
          ((trace interface).state lane).eval assignment := by
    simp [row, outputDifference, sub_eq_add_neg]
  rw [differenceEval]
  exact Lean.Grind.AddCommGroup.sub_eq_zero_iff.mpr (equations lane)

theorem rowsZero_of_equations {logicalWidth : Nat}
    (interface : Interface logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment interface.oneColumn = 1)
    (sboxEquations : SboxEquations interface assignment)
    (outputEquations : OutputEquations interface assignment) :
    RowsZero interface assignment := by
  intro row member
  simp only [rows, List.mem_append, List.mem_map] at member
  rcases member with ⟨forms, formsMember, rfl⟩ |
      ⟨forms, formsMember, rfl⟩
  · exact sboxRowsZero_of_equations interface assignment one sboxEquations
      forms formsMember
  · exact pinRowsZero_of_equations interface assignment one outputEquations
      forms formsMember

theorem rowsZero_implies_sboxRowsZero {logicalWidth : Nat}
    (interface : Interface logicalWidth)
    (assignment : Assignment F logicalWidth)
    (rowsZero : RowsZero interface assignment) :
    SboxRowsZero assignment (trace interface).rows := by
  intro forms member
  have selected := rowsZero (Row.sbox forms) (by
    unfold rows
    exact List.mem_append_left _ (List.mem_map_of_mem member))
  exact selected

theorem rowsZero_implies_pinRowsZero {logicalWidth : Nat}
    (interface : Interface logicalWidth)
    (assignment : Assignment F logicalWidth)
    (rowsZero : RowsZero interface assignment) :
    PinRowsZero assignment (outputRows interface) := by
  intro forms member
  have selected := rowsZero (Row.pin forms) (by
    unfold rows
    exact List.mem_append_right _ (List.mem_map_of_mem member))
  exact selected

/-- The complete 94-row template forces the exact Poseidon2 permutation on
the caller-owned input and output forms. -/
theorem rowsZero_implies_permute {logicalWidth : Nat}
    (interface : Interface logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment interface.oneColumn = 1)
    (rowsZero : RowsZero interface assignment) :
    List.ofFn (SparseLayer.evalState assignment interface.output) =
      Spec.Poseidon2.permute
        (List.ofFn (SparseLayer.evalState assignment interface.input)) := by
  have traceSound := compile_sound interface 0 interface.input
    Permutation.schedule assignment one
      (rowsZero_implies_sboxRowsZero interface assignment rowsZero)
  have outputsEqual :
      SparseLayer.evalState assignment interface.output =
        SparseLayer.evalState assignment (trace interface).state := by
    funext lane
    exact outputRowsZero_lane interface assignment one
      (rowsZero_implies_pinRowsZero interface assignment rowsZero) lane
  calc
    List.ofFn (SparseLayer.evalState assignment interface.output) =
        List.ofFn (SparseLayer.evalState assignment (trace interface).state) :=
      congrArg List.ofFn outputsEqual
    _ = List.ofFn (Permutation.runF Permutation.schedule
          (SparseLayer.evalState assignment interface.input)) :=
      congrArg List.ofFn traceSound
    _ = Permutation.runReference Permutation.schedule
          (List.ofFn (SparseLayer.evalState assignment interface.input)) :=
      Permutation.runF_eq_reference _ _
    _ = Spec.Poseidon2.permute
          (List.ofFn (SparseLayer.evalState assignment interface.input)) :=
      Permutation.runReference_schedule _

@[simp] theorem trace_nextSbox {logicalWidth : Nat}
    (interface : Interface logicalWidth) :
    (trace interface).nextSbox = 86 := by
  rfl

@[simp] theorem trace_rows_length {logicalWidth : Nat}
    (interface : Interface logicalWidth) :
    (trace interface).rows.length = 86 := by
  rfl

@[simp] theorem outputRows_length {logicalWidth : Nat}
    (interface : Interface logicalWidth) :
    (outputRows interface).length = 8 := by
  simp [outputRows]

@[simp] theorem rows_length {logicalWidth : Nat}
    (interface : Interface logicalWidth) :
    (rows interface).length = 94 := by
  simp [rows]

/-- Exact 14-matrix template plan. Slot 13 remains zero through the common
`ProductionRelation.Plan` constructor. -/
def plan {logicalWidth : Nat} (interface : Interface logicalWidth) :
    ProductionRelation.Plan logicalWidth where
  rowCount := (rows interface).length
  rowCount_le := by
    rw [rows_length]
    norm_num [NightstreamFPrime.Lifecycle.cubeVariables]
  forms := fun row port => (rows interface).get row |>.meaningfulForm port

@[simp] theorem plan_rowCount {logicalWidth : Nat}
    (interface : Interface logicalWidth) :
    (plan interface).rowCount = 94 := by
  simp [plan]

theorem plan_rowImage_at {logicalWidth : Nat}
    (interface : Interface logicalWidth)
    (assignment : Assignment F logicalWidth)
    (row : Fin (rows interface).length) :
    (plan interface).rowImage assignment
        ((plan interface).rowLayout.toVertex row) =
      ((rows interface).get row).portImages assignment := by
  funext port
  unfold ProductionRelation.Plan.rowImage
  rw [(plan interface).rowLayout.toColumn_toVertex]
  rfl

theorem plan_residual_at {logicalWidth : Nat}
    (interface : Interface logicalWidth)
    (assignment : Assignment F logicalWidth)
    (row : Fin (rows interface).length) :
    evaluatePolynomial
        Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier.baseOps
        Spec.ProductionRelation.polynomial
        ((plan interface).rowImage assignment
          ((plan interface).rowLayout.toVertex row)) =
      ((rows interface).get row).residual assignment := by
  rw [plan_rowImage_at]
  exact Row.polynomial_eq_residual _ _

def PlanRowsZero {logicalWidth : Nat} (interface : Interface logicalWidth)
    (assignment : Assignment F logicalWidth) : Prop :=
  ∀ row : Fin (rows interface).length,
    evaluatePolynomial
        Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier.baseOps
        Spec.ProductionRelation.polynomial
        ((plan interface).rowImage assignment
          ((plan interface).rowLayout.toVertex row)) = 0

theorem planRowsZero_implies_rowsZero {logicalWidth : Nat}
    (interface : Interface logicalWidth)
    (assignment : Assignment F logicalWidth)
    (planRowsZero : PlanRowsZero interface assignment) :
    RowsZero interface assignment := by
  intro row member
  rcases List.mem_iff_get.mp member with ⟨index, rfl⟩
  rw [← plan_residual_at interface assignment index]
  exact planRowsZero index

theorem rowsZero_implies_planRowsZero {logicalWidth : Nat}
    (interface : Interface logicalWidth)
    (assignment : Assignment F logicalWidth)
    (rowsZero : RowsZero interface assignment) :
    PlanRowsZero interface assignment := by
  intro row
  rw [plan_residual_at interface assignment row]
  exact rowsZero ((rows interface).get row) (List.get_mem _ _)

/-- Exact Poseidon2 trace equations construct a satisfying assignment for
all live rows of the actual 14-matrix plan. -/
theorem planRowsZero_of_equations {logicalWidth : Nat}
    (interface : Interface logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment interface.oneColumn = 1)
    (sboxEquations : SboxEquations interface assignment)
    (outputEquations : OutputEquations interface assignment) :
    PlanRowsZero interface assignment :=
  rowsZero_implies_planRowsZero interface assignment
    (rowsZero_of_equations interface assignment one sboxEquations
      outputEquations)

/-- Live rows of the actual 14-matrix plan force the exact Poseidon2
permutation. Padding and matrix slot 13 are zero by the common plan type. -/
theorem planRowsZero_implies_permute {logicalWidth : Nat}
    (interface : Interface logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment interface.oneColumn = 1)
    (planRowsZero : PlanRowsZero interface assignment) :
    List.ofFn (SparseLayer.evalState assignment interface.output) =
      Spec.Poseidon2.permute
        (List.ofFn (SparseLayer.evalState assignment interface.input)) :=
  rowsZero_implies_permute interface assignment one
    (planRowsZero_implies_rowsZero interface assignment planRowsZero)

end NightstreamFPrime.Layout.ProductionRelation.PoseidonSboxPlan
