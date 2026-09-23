import NightstreamFPrime.Circuit.VariableSupport
import NightstreamFPrime.Lifecycle.Types

/-!
Owns application input/output expressions and transition predicates.
Program selection and checked circuit specializations are in `Application`.
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

end NightstreamFPrime.Lifecycle.Stage1.Application
