import Nightstream.Implementation.Lowering.Goldilocks.Codec

/-!
Contract: Lean-owned application semantics for the repository's 42-times-6
WASM integration fixture.

Assurance tier: model-level.

Owns: the seven-coordinate carried state, the two-batch deterministic
transition, the verifier-owned initial memory word `42`, and the derived
terminal result `252`.

Does not own: general WASM semantics, WAT parsing, a production deployment,
physical rows, NIFS, Rust, or equality with the Rust benchmark.

The first transition represents the benchmark's first three instructions:
push address zero, load the word at that address, and push six. The second
transition represents multiplication and terminal output. The polynomial
transition is total on the field carrier; the named benchmark execution starts
at the canonical zero phase.

Emits constraints: none.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6

open Nightstream.Implementation.Lowering.Goldilocks

/-- Ordered state coordinates carried between three-instruction batches. -/
inductive StateCoordinate where
  | phase
  | memoryWord
  | leftOperand
  | rightOperand
  | output
  | halted
  | trapped
deriving DecidableEq, Repr

def StateCoordinate.index : StateCoordinate -> Fin 7
  | .phase => ⟨0, by decide⟩
  | .memoryWord => ⟨1, by decide⟩
  | .leftOperand => ⟨2, by decide⟩
  | .rightOperand => ⟨3, by decide⟩
  | .output => ⟨4, by decide⟩
  | .halted => ⟨5, by decide⟩
  | .trapped => ⟨6, by decide⟩

/-- Application state in the exact coordinate order above. -/
abbrev State := Fin 7 -> Field

theorem State.ext_coordinates
    {left right : State}
    (coordinates :
      ∀ coordinate : StateCoordinate,
        left coordinate.index = right coordinate.index) :
    left = right := by
  funext ⟨index, indexLt⟩
  have alternatives :
      index = 0 ∨ index = 1 ∨ index = 2 ∨ index = 3 ∨
        index = 4 ∨ index = 5 ∨ index = 6 := by
    omega
  rcases alternatives with
    rfl | rfl | rfl | rfl | rfl | rfl | rfl
  · exact coordinates .phase
  · exact coordinates .memoryWord
  · exact coordinates .leftOperand
  · exact coordinates .rightOperand
  · exact coordinates .output
  · exact coordinates .halted
  · exact coordinates .trapped

/-- The benchmark transition is deterministic and needs no private advice. -/
abbrev Witness := Fin 0 -> Field

def read (state : State) (coordinate : StateCoordinate) : Field :=
  state coordinate.index

def writeCoordinates
    (phase memoryWord leftOperand rightOperand output halted trapped : Field) :
    State
  | ⟨0, _⟩ => phase
  | ⟨1, _⟩ => memoryWord
  | ⟨2, _⟩ => leftOperand
  | ⟨3, _⟩ => rightOperand
  | ⟨4, _⟩ => output
  | ⟨5, _⟩ => halted
  | ⟨6, _⟩ => trapped

/-- Canonical benchmark entry state. The static data segment owns word `42`
at address zero; all execution coordinates start at zero. -/
def initial : State :=
  writeCoordinates 0 42 0 0 0 0 0

/-- One application transition.

At phase zero, it preserves the verifier-owned memory word and prepares
operands `memoryWord` and `6`. At phase one, it retains those operands,
multiplies them into the output, and marks the state halted. The formulas are
defined for every field phase, so the physical recipe need not assume a hidden
Boolean-domain premise. -/
def step (state : State) (_witness : Witness) : State :=
  let phase := read state .phase
  let memoryWord := read state .memoryWord
  let left := read state .leftOperand
  let right := read state .rightOperand
  let output := read state .output
  let trapped := read state .trapped
  writeCoordinates
    1
    memoryWord
    (memoryWord + phase * (left - memoryWord))
    (6 + phase * (right - 6))
    (output + phase * (left * right - output))
    phase
    trapped

def noWitness : Witness := Fin.elim0

def afterPreparation : State :=
  step initial noWitness

def final : State :=
  step afterPreparation noWitness

@[simp] theorem initial_phase :
    read initial .phase = 0 := rfl

@[simp] theorem initial_memoryWord :
    read initial .memoryWord = 42 := rfl

@[simp] theorem afterPreparation_phase :
    read afterPreparation .phase = 1 := by
  decide

@[simp] theorem afterPreparation_left :
    read afterPreparation .leftOperand = 42 := by
  decide

@[simp] theorem afterPreparation_right :
    read afterPreparation .rightOperand = 6 := by
  decide

@[simp] theorem afterPreparation_output :
    read afterPreparation .output = 0 := by
  decide

@[simp] theorem final_output :
    read final .output = 252 := by
  decide

@[simp] theorem final_halted :
    read final .halted = 1 := by
  decide

@[simp] theorem final_not_trapped :
    read final .trapped = 0 := by
  decide

/-- The named benchmark result is derived by two applications of the
transition; it is not a supplied acceptance value. -/
theorem benchmark_computes_252 :
    read (step (step initial noWitness) noWitness) .output = 252 :=
  final_output

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6
