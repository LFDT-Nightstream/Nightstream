import NightstreamFPrime.Export.OptionB.StepExtend
import NightstreamFPrime.Export.OptionB.IvcWalk

/-! Completeness of option-B IVC (CW-10.4): the terminal accepts the honest prover's proof of
every valid run. A statement is honestly accepted when the terminal accepts the bottom proof, or a
recursive proof whose slots are honest (`Accepts`). One valid step keeps this
(`accepts_step`): from the bottom proof by `StepExtend.base_extend`, and from a recursive proof by
`StepExtend.light_extend` when the selected slot is below the light cap and by
`StepExtend.refresh_extend` otherwise, as Rust picks the circuit from the slot depth. The terminal
does not read the circuit selector (`TerminalBridge.holdsFor_recursive_control`), so each step may
pick its own. Induction over the run gives `run_complete`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.OptionB.RunCompleteness

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.OptionB
open NightstreamFPrime.Export.OptionB.TerminalBridge
open NightstreamFPrime.Export.OptionB.StepExtend

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}
  {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
  {ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits)}
  {circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits}
  {step : AppState → AppWitness → AppState}
  {cubeFits : Phi81CarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}

variable (ajtai circuitKeys step) in
/-- The terminal accepts `statement` with the bottom proof, or with a recursive proof whose slots
are honest. -/
def Accepts (cap : Nat) (control : AppState → AppWitness → Fin 2)
    (statement : TerminalStatement AppState) : Prop :=
  Terminal.HoldsFor cap ajtai circuitKeys step control statement .bottom ∨
    ∃ payload, Terminal.HoldsFor cap ajtai circuitKeys step control statement (.recursive payload) ∧
      Honest (ajtai := ajtai) payload

/-- One valid step keeps the terminal's honest acceptance. -/
theorem accepts_step (exports : Exports relation ajtai circuitKeys step cubeFits)
    (complete : Complete exports) (keysShaped : ∀ key, (circuitKeys key).digest.length = 4)
    (cap : Nat) (control : AppState → AppWitness → Fin 2) (statement : TerminalStatement AppState)
    (accepts : Accepts ajtai circuitKeys step cap control statement)
    (witness : AppWitness) (valid : exports.program.valid statement.zi witness)
    (iterationNext : statement.iteration + 1 < goldilocksModulus)
    (witnessLength : witness.length = exports.program.witnessWordCount) :
    Accepts ajtai circuitKeys step cap control
      ⟨statement.iteration + 1, statement.z0, step statement.zi witness⟩ := by
  have nextLength : ∀ (current : AppState),
      current.length = Stage1.Application.stateWordCount →
        (step current witness).length = Stage1.Application.stateWordCount := by
    intro current currentLength
    rw [← exports.programStep]
    exact exports.program.step_length current witness currentLength
  have transport : ∀ (control' : AppState → AppWitness → Fin 2) payload,
      Terminal.HoldsFor cap ajtai circuitKeys step control'
          ⟨statement.iteration + 1, statement.z0, step statement.zi witness⟩ (.recursive payload) →
        Terminal.HoldsFor cap ajtai circuitKeys step control
          ⟨statement.iteration + 1, statement.z0, step statement.zi witness⟩ (.recursive payload) :=
    fun control' payload holds =>
      (holdsFor_recursive_control cap ajtai circuitKeys step control' control _ payload).mp holds
  rcases accepts with bottom | ⟨payload, holds, honest⟩
  · obtain ⟨statementValid, iterationZero, ziEq⟩ :=
      (Terminal.holdsFor_bottom_iff cap ajtai circuitKeys step control statement).mp bottom
    have bottom' := (Terminal.holdsFor_bottom_iff cap ajtai circuitKeys step (fun _ _ => 0)
      statement).mpr ⟨statementValid, iterationZero, ziEq⟩
    obtain ⟨next, nextHolds, nextHonest⟩ := base_extend exports complete keysShaped cap
      (fun _ _ => 0) statement bottom' witness valid rfl (nextLength _ statementValid.2.2)
      witnessLength
    exact Or.inr ⟨next, transport _ next nextHolds, nextHonest⟩
  · obtain ⟨statementValid, pcValid, _⟩ :=
      (Terminal.holdsFor_recursive_iff cap ajtai circuitKeys step control statement payload).mp
        holds
    have selectedEq : ∀ other : InRange 2 payload.pc, selectedIndex other = selectedIndex pcValid :=
      fun _ => Fin.ext rfl
    by_cases room : (payload.running (selectedIndex pcValid)).2 < Schedule.productionLightFoldCap
    · have holds' := (holdsFor_recursive_control cap ajtai circuitKeys step control (fun _ _ => 0)
        statement payload).mp holds
      obtain ⟨next, nextHolds, nextHonest⟩ := light_extend exports complete keysShaped cap
        (fun _ _ => 0) statement payload holds' honest witness valid rfl
        (fun other => by rw [selectedEq other]; exact room) iterationNext
        (nextLength _ statementValid.2.2) witnessLength
      exact Or.inr ⟨next, transport _ next nextHolds, nextHonest⟩
    · have holds' := (holdsFor_recursive_control cap ajtai circuitKeys step control (fun _ _ => 1)
        statement payload).mp holds
      obtain ⟨next, nextHolds, nextHonest⟩ := refresh_extend exports complete keysShaped cap
        (fun _ _ => 1) statement payload holds' honest witness valid rfl iterationNext
        (nextLength _ statementValid.2.2) witnessLength
      exact Or.inr ⟨next, transport _ next nextHolds, nextHonest⟩

/-- Completeness of option-B IVC: for every valid run of the application from `z0` whose
witnesses have the application's witness length, the terminal accepts the honest prover's proof
of the run's last state. -/
theorem run_complete (exports : Exports relation ajtai circuitKeys step cubeFits)
    (complete : Complete exports) (keysShaped : ∀ key, (circuitKeys key).digest.length = 4)
    (cap : Nat) (control : AppState → AppWitness → Fin 2) (z0 : AppState)
    (z0Length : z0.length = Stage1.Application.stateWordCount) :
    ∀ witnesses : List AppWitness, IvcWalk.ValidFrom exports.program z0 witnesses →
      (∀ witness ∈ witnesses, witness.length = exports.program.witnessWordCount) →
      witnesses.length < goldilocksModulus →
        Accepts ajtai circuitKeys step cap control ⟨witnesses.length, z0, witnesses.foldl step z0⟩ := by
  intro witnesses
  induction witnesses using List.reverseRecOn with
  | nil =>
      intro _ _ _
      left
      exact (Terminal.holdsFor_bottom_iff cap ajtai circuitKeys step control _).mpr
        ⟨⟨show (0 : Nat) < goldilocksModulus by decide, z0Length, z0Length⟩, rfl, rfl⟩
  | append_singleton earlier witness inductionHypothesis =>
      intro validRun lengths lengthBound
      rw [IvcWalk.validFrom_append] at validRun
      have earlierBound : earlier.length < goldilocksModulus := by
        simp only [List.length_append, List.length_singleton] at lengthBound
        omega
      have earlierAccepts := inductionHypothesis validRun.1
        (fun other inside => lengths other (List.mem_append_left _ inside)) earlierBound
      have foldEq : earlier.foldl exports.program.step z0 = earlier.foldl step z0 :=
        congrArg (fun next => earlier.foldl next z0) exports.programStep
      have lastValid : exports.program.valid (earlier.foldl step z0) witness := by
        rw [← foldEq]
        exact validRun.2.1
      have stepped := accepts_step exports complete keysShaped cap control _ earlierAccepts witness
        lastValid (by simpa using lengthBound)
        (lengths witness (List.mem_append_right _ (List.mem_singleton_self _)))
      simpa only [List.length_append, List.length_singleton, List.foldl_append, List.foldl_cons,
        List.foldl_nil] using stepped

end NightstreamFPrime.Export.OptionB.RunCompleteness
