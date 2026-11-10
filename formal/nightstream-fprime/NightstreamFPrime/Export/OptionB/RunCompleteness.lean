import NightstreamFPrime.Export.OptionB.StepExtend
import NightstreamFPrime.Export.OptionB.IvcWalk

/-! Completeness of option-B IVC (CW-10.4). The verifier of a proof with its witnesses
(`Verifies`, SPEC.md Algorithm 6) requires the terminal relation and, for a recursive proof, the
slot checks `StepExtend.Honest`: the depth of each slot is at most the light cap, and its witness
is within the accumulator bound of its depth (Cyclo, ePrint 2026/359, Theorem 3 item 1). The
terminal relation alone admits each slot witness below `B`. It is the relation that extraction
reaches (`FinishWalk.holdsFor_of_extracted`), and the verifier implies it (`holds_of_verifies`).

One valid step extends each proof that the verifier accepts (`verifies_step`, HyperNova
Definition 11 (i)): the bottom proof by `StepExtend.base_extend`, and a recursive proof by
`StepExtend.light_extend` when the selected slot is below the light cap and by
`StepExtend.refresh_extend` otherwise, as Rust picks the circuit from the slot depth. The terminal
does not read the circuit selector (`TerminalBridge.holdsFor_recursive_control`), so each step may
pick its own. A statement is honestly accepted when the verifier accepts some proof of it
(`Accepts`, `accepts_step`). Induction over the run gives `run_complete`. -/

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
and fresh instance are honest at the package widths `packageWidths`. -/
def Accepts (packageWidths : Fin 2 → Nat) (cap : Nat) (control : AppState → AppWitness → Fin 2)
    (statement : TerminalStatement AppState) : Prop :=
  Terminal.HoldsFor cap ajtai circuitKeys step control statement .bottom ∨
    ∃ payload, Terminal.HoldsFor cap ajtai circuitKeys step control statement (.recursive payload) ∧
      Honest (ajtai := ajtai) packageWidths payload

variable (ajtai circuitKeys step) in
/-- The verifier of a proof with its witnesses: the terminal relation accepts the proof, and the
slots and the fresh instance of a recursive proof pass the checks of `StepExtend.Honest`. -/
def Verifies (packageWidths : Fin 2 → Nat) (cap : Nat) (control : AppState → AppWitness → Fin 2)
    (statement : TerminalStatement AppState)
    (proof : Terminal.ProofEnvelope (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    Prop :=
  Terminal.HoldsFor cap ajtai circuitKeys step control statement proof ∧
    match proof with
    | .bottom => True
    | .recursive payload => Honest (ajtai := ajtai) packageWidths payload

/-- The verifier accepts only what the terminal relation accepts, so each bound on an acceptance
of the terminal relation is a bound on an acceptance of the verifier. -/
theorem holds_of_verifies {packageWidths : Fin 2 → Nat} {cap : Nat}
    {control : AppState → AppWitness → Fin 2} {statement : TerminalStatement AppState}
    {proof : Terminal.ProofEnvelope (logicalWidth := logicalWidth) (publicFits := publicFits)}
    (verifies : Verifies ajtai circuitKeys step packageWidths cap control statement proof) :
    Terminal.HoldsFor cap ajtai circuitKeys step control statement proof :=
  And.left verifies

theorem verifies_bottom_iff (packageWidths : Fin 2 → Nat) (cap : Nat)
    (control : AppState → AppWitness → Fin 2) (statement : TerminalStatement AppState) :
    Verifies ajtai circuitKeys step packageWidths cap control statement .bottom ↔
      Terminal.HoldsFor cap ajtai circuitKeys step control statement .bottom :=
  ⟨And.left, fun holds => And.intro holds True.intro⟩

theorem verifies_recursive_iff (packageWidths : Fin 2 → Nat) (cap : Nat)
    (control : AppState → AppWitness → Fin 2) (statement : TerminalStatement AppState)
    (payload : TerminalProof (SlotData logicalWidth publicFits)
      (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits))
      (StepFresh logicalWidth publicFits)
      (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits)) 2) :
    Verifies ajtai circuitKeys step packageWidths cap control statement (.recursive payload) ↔
      Terminal.HoldsFor cap ajtai circuitKeys step control statement (.recursive payload) ∧
        Honest (ajtai := ajtai) packageWidths payload :=
  Iff.rfl

/-- HyperNova Definition 11 (i) for option B: one valid step extends each proof that the verifier
accepts to a recursive proof that the verifier accepts at the next iteration. -/
theorem verifies_step (exports : Exports relation ajtai circuitKeys step cubeFits)
    (complete : Complete exports) (keysShaped : ∀ key, (circuitKeys key).digest.length = 4)
    (cap : Nat) (control : AppState → AppWitness → Fin 2) (statement : TerminalStatement AppState)
    (proof : Terminal.ProofEnvelope (logicalWidth := logicalWidth) (publicFits := publicFits))
    (verifies : Verifies ajtai circuitKeys step complete.packageWidth cap control statement proof)
    (witness : AppWitness) (valid : exports.program.valid statement.zi witness)
    (iterationNext : statement.iteration + 1 < goldilocksModulus)
    (witnessLength : witness.length = exports.program.witnessWordCount) :
    ∃ next, Verifies ajtai circuitKeys step complete.packageWidth cap control
      ⟨statement.iteration + 1, statement.z0, step statement.zi witness⟩ (.recursive next) := by
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
  cases proof with
  | bottom =>
    have bottom := (verifies_bottom_iff complete.packageWidth cap control statement).mp verifies
    obtain ⟨statementValid, iterationZero, ziEq⟩ :=
      (Terminal.holdsFor_bottom_iff cap ajtai circuitKeys step control statement).mp bottom
    have bottom' := (Terminal.holdsFor_bottom_iff cap ajtai circuitKeys step (fun _ _ => 0)
      statement).mpr ⟨statementValid, iterationZero, ziEq⟩
    obtain ⟨next, nextHolds, nextHonest⟩ := base_extend exports complete keysShaped cap
      (fun _ _ => 0) statement bottom' witness valid rfl (nextLength _ statementValid.2.2)
      witnessLength
    exact ⟨next, (verifies_recursive_iff complete.packageWidth cap control _ next).mpr
      ⟨transport _ next nextHolds, nextHonest⟩⟩
  | recursive payload =>
    obtain ⟨holds, honest⟩ :=
      (verifies_recursive_iff complete.packageWidth cap control statement payload).mp verifies
    obtain ⟨statementValid, pcValid, _⟩ :=
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
      exact ⟨next, (verifies_recursive_iff complete.packageWidth cap control _ next).mpr
        ⟨transport _ next nextHolds, nextHonest⟩⟩
    · have holds' := (holdsFor_recursive_control cap ajtai circuitKeys step control (fun _ _ => 1)
        statement payload).mp holds
      obtain ⟨next, nextHolds, nextHonest⟩ := refresh_extend exports complete keysShaped cap
        (fun _ _ => 1) statement payload holds' honest witness valid rfl iterationNext
        (nextLength _ statementValid.2.2) witnessLength
      exact ⟨next, (verifies_recursive_iff complete.packageWidth cap control _ next).mpr
        ⟨transport _ next nextHolds, nextHonest⟩⟩

/-- One valid step keeps the honest acceptance of a statement. -/
theorem accepts_step (exports : Exports relation ajtai circuitKeys step cubeFits)
    (complete : Complete exports) (keysShaped : ∀ key, (circuitKeys key).digest.length = 4)
    (cap : Nat) (control : AppState → AppWitness → Fin 2) (statement : TerminalStatement AppState)
    (accepts : Accepts ajtai circuitKeys step complete.packageWidth cap control statement)
    (witness : AppWitness) (valid : exports.program.valid statement.zi witness)
    (iterationNext : statement.iteration + 1 < goldilocksModulus)
    (witnessLength : witness.length = exports.program.witnessWordCount) :
    Accepts ajtai circuitKeys step complete.packageWidth cap control
      ⟨statement.iteration + 1, statement.z0, step statement.zi witness⟩ := by
  rcases accepts with bottom | ⟨payload, holds, honest⟩
  · obtain ⟨next, verified⟩ := verifies_step exports complete keysShaped cap control statement
      .bottom ((verifies_bottom_iff complete.packageWidth cap control statement).mpr bottom)
      witness valid iterationNext witnessLength
    exact Or.inr ⟨next, (verifies_recursive_iff complete.packageWidth cap control _ next).mp
      verified⟩
  · obtain ⟨next, verified⟩ := verifies_step exports complete keysShaped cap control statement
      (.recursive payload)
      ((verifies_recursive_iff complete.packageWidth cap control statement payload).mpr
        ⟨holds, honest⟩)
      witness valid iterationNext witnessLength
    exact Or.inr ⟨next, (verifies_recursive_iff complete.packageWidth cap control _ next).mp
      verified⟩

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
        Accepts ajtai circuitKeys step complete.packageWidth cap control
          ⟨witnesses.length, z0, witnesses.foldl step z0⟩ := by
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
