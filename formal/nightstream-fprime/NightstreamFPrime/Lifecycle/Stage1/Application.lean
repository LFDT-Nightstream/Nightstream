import NightstreamFPrime.Circuit.VariableSupport
import NightstreamFPrime.Lifecycle.Types

namespace NightstreamFPrime.Lifecycle.Stage1.Application

open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec

def stateWordCount : Nat := 4

abbrev StateIndex := Fin stateWordCount

structure Interface (witnessWordCount : Nat) where
  input : Nat → StateIndex → Expr
  witness : Nat → Fin witnessWordCount → Expr
  output : Nat → StateIndex → Expr

structure InputsBelow {witnessWordCount : Nat}
    (interface : Interface witnessWordCount) (offset : Nat) : Prop where
  input : ∀ index, (interface.input offset index).VarsBelow offset
  witness : ∀ index, (interface.witness offset index).VarsBelow offset
  output : ∀ index, (interface.output offset index).VarsBelow offset

structure InputsSupported {witnessWordCount : Nat}
    (interface : Interface witnessWordCount) (offset : Nat)
    (allowed : Nat → Prop) : Prop where
  input : ∀ index, (interface.input offset index).VarsSatisfy allowed
  witness : ∀ index, (interface.witness offset index).VarsSatisfy allowed
  output : ∀ index, (interface.output offset index).VarsSatisfy allowed

def inputState {witnessWordCount : Nat}
    (interface : Interface witnessWordCount) (offset : Nat) (env : Env) :
    AppState :=
  List.ofFn fun index => (interface.input offset index).eval env

def witnessValue {witnessWordCount : Nat}
    (interface : Interface witnessWordCount) (offset : Nat) (env : Env) :
    AppWitness :=
  List.ofFn fun index => (interface.witness offset index).eval env

def outputState {witnessWordCount : Nat}
    (interface : Interface witnessWordCount) (offset : Nat) (env : Env) :
    AppState :=
  List.ofFn fun index => (interface.output offset index).eval env

@[simp] theorem inputState_length {witnessWordCount : Nat}
    (interface : Interface witnessWordCount) (offset : Nat) (env : Env) :
    (inputState interface offset env).length = stateWordCount := by
  simp [inputState]

@[simp] theorem witnessValue_length {witnessWordCount : Nat}
    (interface : Interface witnessWordCount) (offset : Nat) (env : Env) :
    (witnessValue interface offset env).length = witnessWordCount := by
  simp [witnessValue]

@[simp] theorem outputState_length {witnessWordCount : Nat}
    (interface : Interface witnessWordCount) (offset : Nat) (env : Env) :
    (outputState interface offset env).length = stateWordCount := by
  simp [outputState]

def Holds (step : AppState → AppWitness → AppState)
    {witnessWordCount : Nat} (interface : Interface witnessWordCount)
    (offset : Nat) (env : Env) : Prop :=
  outputState interface offset env =
    step (inputState interface offset env) (witnessValue interface offset env)

structure Program where
  witnessWordCount : Nat
  step : AppState → AppWitness → AppState
  step_length : ∀ state witness, state.length = stateWordCount →
    (step state witness).length = stateWordCount
  circuit : Interface witnessWordCount → FormalCircuit
  spec_iff : ∀ interface offset env,
    (circuit interface).spec offset env ↔ Holds step interface offset env
  assumptions_of_inputsBelow : ∀ interface offset env,
    InputsBelow interface offset → (circuit interface).assumptions offset env
  constraintsSupported : ∀ interface offset env allowed,
    (circuit interface).assumptions offset env →
      InputsSupported interface offset allowed →
      (∀ index,
        offset ≤ index →
        index < offset +
          localLength (Circuit.ops (circuit interface).main offset) →
        allowed index) →
      ∀ expression ∈ flatConstraints (Circuit.ops (circuit interface).main offset),
        expression.VarsSatisfy allowed

namespace Program

theorem assumptions (program : Program)
    (interface : Interface program.witnessWordCount) (offset : Nat) (env : Env)
    (inputs : InputsBelow interface offset) :
    (program.circuit interface).assumptions offset env :=
  program.assumptions_of_inputsBelow interface offset env inputs

theorem support (program : Program)
    (interface : Interface program.witnessWordCount) (offset : Nat) (env : Env)
    (allowed : Nat → Prop)
    (assumptions : (program.circuit interface).assumptions offset env)
    (inputs : InputsSupported interface offset allowed)
    (localSupport : ∀ index,
      offset ≤ index →
      index < offset + localLength
        (Circuit.ops (program.circuit interface).main offset) →
      allowed index) :
    ∀ expression ∈ flatConstraints
        (Circuit.ops (program.circuit interface).main offset),
      expression.VarsSatisfy allowed :=
  program.constraintsSupported interface offset env allowed assumptions inputs
    localSupport

theorem scope (program : Program)
    (interface : Interface program.witnessWordCount) (offset : Nat) (env : Env)
    (inputs : InputsBelow interface offset)
    (assumptions : (program.circuit interface).assumptions offset env) :
    ∀ expression ∈ flatConstraints
        (Circuit.ops (program.circuit interface).main offset),
      expression.VarsBelow
        (offset + localLength
          (Circuit.ops (program.circuit interface).main offset)) := by
  let bound := offset + localLength
    (Circuit.ops (program.circuit interface).main offset)
  have supported := program.support interface offset env
    (fun index => index < bound) assumptions (by
      refine {
        input := fun index => ?_
        witness := fun index => ?_
        output := fun index => ?_ }
      · apply (Expr.varsSatisfy_lt_iff_varsBelow _ bound).2
        exact Expr.VarsBelow.mono _ (inputs.input index) (by
          unfold bound
          omega)
      · apply (Expr.varsSatisfy_lt_iff_varsBelow _ bound).2
        exact Expr.VarsBelow.mono _ (inputs.witness index) (by
          unfold bound
          omega)
      · apply (Expr.varsSatisfy_lt_iff_varsBelow _ bound).2
        exact Expr.VarsBelow.mono _ (inputs.output index) (by
          unfold bound
          omega)) (by
      intro index _ upper
      exact upper)
  intro expression member
  apply (Expr.varsSatisfy_lt_iff_varsBelow expression bound).1
  exact supported expression member

theorem soundness (program : Program)
    (interface : Interface program.witnessWordCount) (offset : Nat) (env : Env)
    (assumptions : (program.circuit interface).assumptions offset env)
    (rows : holds env (Circuit.ops (program.circuit interface).main offset)) :
    Holds program.step interface offset env := by
  apply (program.spec_iff interface offset env).mp
  exact (program.circuit interface).soundness env offset assumptions rows

theorem completeness (program : Program)
    (interface : Interface program.witnessWordCount) (offset : Nat) (env : Env)
    (assumptions : (program.circuit interface).assumptions offset env)
    (specification : Holds program.step interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (program.circuit interface).main offset)) ∧
      holdsFlat completed
        (Circuit.ops (program.circuit interface).main offset) := by
  apply (program.circuit interface).completeness env offset assumptions
  exact (program.spec_iff interface offset env).mpr specification

theorem holds_of_agree_below
    (program : Program)
    (interface : Interface program.witnessWordCount)
    (offset : Nat) (before after : Env)
    (inputs : InputsBelow interface offset)
    (agrees : ∀ index, index < offset → after index = before index)
    (holds : Holds program.step interface offset before) :
    Holds program.step interface offset after := by
  unfold Holds inputState witnessValue outputState at holds ⊢
  have inputEq :
      List.ofFn (fun index => (interface.input offset index).eval after) =
        List.ofFn (fun index => (interface.input offset index).eval before) := by
    apply congrArg List.ofFn
    funext index
    exact (interface.input offset index).eval_eq_of_agree_below offset
      after before (inputs.input index) agrees
  have witnessEq :
      List.ofFn (fun index => (interface.witness offset index).eval after) =
        List.ofFn (fun index => (interface.witness offset index).eval before) := by
    apply congrArg List.ofFn
    funext index
    exact (interface.witness offset index).eval_eq_of_agree_below offset
      after before (inputs.witness index) agrees
  have outputEq :
      List.ofFn (fun index => (interface.output offset index).eval after) =
        List.ofFn (fun index => (interface.output offset index).eval before) := by
    apply congrArg List.ofFn
    funext index
    exact (interface.output offset index).eval_eq_of_agree_below offset
      after before (inputs.output index) agrees
  rw [outputEq, inputEq, witnessEq]
  exact holds

theorem holds_of_values_eq
    (program : Program)
    (interface : Interface program.witnessWordCount)
    (offset : Nat) (before after : Env)
    (inputEq : inputState interface offset before =
      inputState interface offset after)
    (witnessEq : witnessValue interface offset before =
      witnessValue interface offset after)
    (outputEq : outputState interface offset before =
      outputState interface offset after)
    (holds : Holds program.step interface offset before) :
    Holds program.step interface offset after := by
  unfold Holds
  rw [← inputEq, ← witnessEq, ← outputEq]
  exact holds

theorem step_output_length (program : Program)
    (interface : Interface program.witnessWordCount) (offset : Nat) (env : Env)
    (specification : Holds program.step interface offset env) :
    (program.step (inputState interface offset env)
      (witnessValue interface offset env)).length = stateWordCount := by
  rw [← specification]
  exact outputState_length interface offset env

end Program

end NightstreamFPrime.Lifecycle.Stage1.Application
