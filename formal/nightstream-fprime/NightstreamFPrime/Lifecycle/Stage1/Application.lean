import NightstreamFPrime.Circuit.VariableSupport
import NightstreamFPrime.Lifecycle.Types

/-!
Owns the Lean interface for one verifier-selected Stage 1 application.

Each production application supplies one closed `Program`. Its circuit proves
the exact transition implemented by `step`. The final Stage 1 assembler fixes
that program before package construction; neither the prover nor the runtime
package loader supplies it.

This module does not select a production application, assign physical columns,
compute an application identity, or modify the canonical package.
-/

namespace NightstreamFPrime.Lifecycle.Stage1.Application

open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec

/-- Stage 1 carries exactly four application-state words in each state hash. -/
def stateWordCount : Nat := 4

abbrev StateIndex := Fin stateWordCount

/-- External wires of one concrete application circuit. The witness width is
fixed by the Lean-authored program. -/
structure Interface (witnessWordCount : Nat) where
  input : Nat → StateIndex → Expr
  witness : Nat → Fin witnessWordCount → Expr
  output : Nat → StateIndex → Expr

/-- Exact caller-owned application wires. A concrete layout proves that all
of them precede the application's local allocation. -/
structure InputsBelow {witnessWordCount : Nat}
    (interface : Interface witnessWordCount) (offset : Nat) : Prop where
  input : ∀ index, (interface.input offset index).VarsBelow offset
  witness : ∀ index, (interface.witness offset index).VarsBelow offset
  output : ∀ index, (interface.output offset index).VarsBelow offset

/-- Exact caller-selected support for all external application expressions. -/
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

/-- Exact semantic obligation of one application circuit. -/
def Holds (step : AppState → AppWitness → AppState)
    {witnessWordCount : Nat} (interface : Interface witnessWordCount)
    (offset : Nat) (env : Env) : Prop :=
  outputState interface offset env =
    step (inputState interface offset env) (witnessValue interface offset env)

/-- A Lean-authored application is one proved circuit for one exact step
function. The proof field is erased during execution. -/
structure Program where
  witnessWordCount : Nat
  step : AppState → AppWitness → AppState
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

/-- A layout-owned wire-range proof supplies every application-circuit
assumption. -/
theorem assumptions (program : Program)
    (interface : Interface program.witnessWordCount) (offset : Nat) (env : Env)
    (inputs : InputsBelow interface offset) :
    (program.circuit interface).assumptions offset env :=
  program.assumptions_of_inputsBelow interface offset env inputs

/-- The selected application preserves one caller-selected support set. -/
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

/-- The stronger support contract implies the original causal row scope. -/
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

/-- Arbitrary satisfying circuit witnesses implement the selected application
transition. -/
theorem soundness (program : Program)
    (interface : Interface program.witnessWordCount) (offset : Nat) (env : Env)
    (assumptions : (program.circuit interface).assumptions offset env)
    (rows : holds env (Circuit.ops (program.circuit interface).main offset)) :
    Holds program.step interface offset env := by
  apply (program.spec_iff interface offset env).mp
  exact (program.circuit interface).soundness env offset assumptions rows

/-- Every valid selected application transition has a witness for the exact
flattened rows emitted from its Lean circuit. -/
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

/-- The application relation depends only on its declared input, witness, and
output expressions below the child start. -/
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

/-- The selected application relation transports through equality of the
three values it reads. -/
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

/-- Acceptance also proves that the selected step returns the exact four-word
state required by the Stage 1 state-hash ABI. -/
theorem step_output_length (program : Program)
    (interface : Interface program.witnessWordCount) (offset : Nat) (env : Env)
    (specification : Holds program.step interface offset env) :
    (program.step (inputState interface offset env)
      (witnessValue interface offset env)).length = stateWordCount := by
  rw [← specification]
  exact outputState_length interface offset env

end Program

end NightstreamFPrime.Lifecycle.Stage1.Application
