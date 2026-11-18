import NightstreamFPrime.Export.BoundedDepth.StepExtend
import NightstreamFPrime.Export.BoundedDepth.IvcWalk

/-! Completeness of bounded-depth IVC (CW-10.4). The verifier of a proof with its witnesses
(`Verifies`, SPEC.md Algorithm 6) requires the terminal relation and, for a recursive proof, the
slot checks `StepExtend.Honest`: the depth of each slot is at most the light cap, and its witness
is within the accumulator bound of its depth (Cyclo, ePrint 2026/359, Theorem 3 item 1). The
terminal relation alone admits each slot witness below `B`. The verifier implies it
(`holds_of_verifies`).

`Verifies` requires zeros only on the ring blocks of the fresh witness that start at or past its
package width (`BlockSupport.Supported`). Rust's direct verifier requires zeros on each cell from
that width (SPEC.md Algorithm 6, Rejects). The honest prover's proofs of `verifies_step` and
`run_complete` have these zeros (`FreshZeroTail`).

One valid step extends each proof that the verifier accepts (`verifies_step`, HyperNova
Definition 11 (i)) to a proof that the verifier accepts and whose fresh witness has the zero tail:
the bottom proof by `StepExtend.base_extend`, and a recursive proof by
`StepExtend.light_extend` when the selected slot is below the light cap and by
`StepExtend.refresh_extend` otherwise, as Rust picks the circuit from the slot depth. The terminal
does not read the circuit selector (`TerminalBridge.holdsFor_recursive_control`), so each step may
pick its own. Induction over the run gives `run_complete`. A run of at most `cap + 1` steps takes
the light fold at each step after the base step, so its proof keeps slot 1 at the default slot of
depth 0 and has the light circuit last (`light_run_complete`, the light mode of SPEC.md
Algorithm 9). -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.BoundedDepth.RunCompleteness

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.BoundedDepth
open NightstreamFPrime.Export.BoundedDepth.TerminalBridge
open NightstreamFPrime.Export.BoundedDepth.StepExtend

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    RingCarrierLayout.carrierWidth logicalWidth}
  {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
  {ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits)}
  {circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits}
  {step : AppState → AppWitness → AppState}
  {cubeFits : RingCarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}

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

/-- The fresh witness of a recursive proof is zero on every cell from the package width of the
circuit that ran last (`BlockSupport.ZeroTail`). This is the width at which `Verifies` checks its
ring blocks (`StepExtend.Honest.freshSupport`), and the zero tail that Rust's direct verifier
checks (SPEC.md Algorithm 6, Rejects). The bottom proof has no fresh witness. -/
def FreshZeroTail (packageWidths : Fin 2 → Nat)
    (proof : Terminal.ProofEnvelope (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    Prop :=
  match proof with
  | .bottom => True
  | .recursive payload => ∀ pcValid : InRange 2 payload.pc,
      BlockSupport.ZeroTail (packageWidths (selectedIndex pcValid)) payload.freshWitness

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

/-- HyperNova Definition 11 (i) for the bounded-depth scheme: one valid step extends each proof
that the verifier accepts to a recursive proof that the verifier accepts at the next iteration.
The fresh witness of the new proof is zero on every cell from its package width
(`FreshZeroTail`). -/
theorem verifies_step (exports : Exports relation ajtai circuitKeys step cubeFits)
    (complete : Complete exports) (keysShaped : ∀ key, (circuitKeys key).digest.length = 4)
    (cap : Nat) (control : AppState → AppWitness → Fin 2) (statement : TerminalStatement AppState)
    (proof : Terminal.ProofEnvelope (logicalWidth := logicalWidth) (publicFits := publicFits))
    (verifies : Verifies ajtai circuitKeys step complete.packageWidth cap control statement proof)
    (witness : AppWitness) (valid : exports.program.valid statement.zi witness)
    (iterationNext : statement.iteration + 1 < fieldModulus)
    (witnessLength : witness.length = exports.program.witnessWordCount) :
    ∃ next, Verifies ajtai circuitKeys step complete.packageWidth cap control
        ⟨statement.iteration + 1, statement.z0, step statement.zi witness⟩ (.recursive next) ∧
      FreshZeroTail complete.packageWidth (.recursive next) := by
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
    obtain ⟨next, nextHolds, nextHonest, -, -, nextTail⟩ := base_extend exports complete keysShaped
      cap
      (fun _ _ => 0) statement bottom' witness valid rfl (nextLength _ statementValid.2.2)
      witnessLength
    exact ⟨next, (verifies_recursive_iff complete.packageWidth cap control _ next).mpr
      ⟨transport _ next nextHolds, nextHonest⟩, nextTail⟩
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
      obtain ⟨next, nextHolds, nextHonest, -, -, nextTail⟩ := light_extend exports complete
        keysShaped cap (fun _ _ => 0) statement payload holds' honest witness valid rfl
        (fun other => by rw [selectedEq other]; exact room) iterationNext
        (nextLength _ statementValid.2.2) witnessLength
      exact ⟨next, (verifies_recursive_iff complete.packageWidth cap control _ next).mpr
        ⟨transport _ next nextHolds, nextHonest⟩, nextTail⟩
    · have holds' := (holdsFor_recursive_control cap ajtai circuitKeys step control (fun _ _ => 1)
        statement payload).mp holds
      obtain ⟨next, nextHolds, nextHonest, nextTail⟩ := refresh_extend exports complete
        keysShaped cap (fun _ _ => 1) statement payload holds' honest witness valid rfl
        iterationNext (nextLength _ statementValid.2.2) witnessLength
      exact ⟨next, (verifies_recursive_iff complete.packageWidth cap control _ next).mpr
        ⟨transport _ next nextHolds, nextHonest⟩, nextTail⟩

/-- Completeness of bounded-depth IVC: for every valid run of the application from `z0` whose
witnesses have the application's witness length, the verifier accepts the honest prover's proof
of the run's last state, and the proof's fresh witness is zero on every cell from its package
width (`FreshZeroTail`). -/
theorem run_complete (exports : Exports relation ajtai circuitKeys step cubeFits)
    (complete : Complete exports) (keysShaped : ∀ key, (circuitKeys key).digest.length = 4)
    (cap : Nat) (control : AppState → AppWitness → Fin 2) (z0 : AppState)
    (z0Length : z0.length = Stage1.Application.stateWordCount) :
    ∀ witnesses : List AppWitness, IvcWalk.ValidFrom exports.program z0 witnesses →
      (∀ witness ∈ witnesses, witness.length = exports.program.witnessWordCount) →
      witnesses.length < fieldModulus →
        ∃ proof, Verifies ajtai circuitKeys step complete.packageWidth cap control
            ⟨witnesses.length, z0, witnesses.foldl step z0⟩ proof ∧
          FreshZeroTail complete.packageWidth proof := by
  intro witnesses
  induction witnesses using List.reverseRecOn with
  | nil =>
      intro _ _ _
      refine ⟨.bottom, (verifies_bottom_iff complete.packageWidth cap control _).mpr ?_, trivial⟩
      exact (Terminal.holdsFor_bottom_iff cap ajtai circuitKeys step control _).mpr
        ⟨⟨show (0 : Nat) < fieldModulus by decide, z0Length, z0Length⟩, rfl, rfl⟩
  | append_singleton earlier witness inductionHypothesis =>
      intro validRun lengths lengthBound
      rw [IvcWalk.validFrom_append] at validRun
      have earlierBound : earlier.length < fieldModulus := by
        simp only [List.length_append, List.length_singleton] at lengthBound
        omega
      obtain ⟨proof, verifies, -⟩ := inductionHypothesis validRun.1
        (fun other inside => lengths other (List.mem_append_left _ inside)) earlierBound
      have foldEq : earlier.foldl exports.program.step z0 = earlier.foldl step z0 :=
        congrArg (fun next => earlier.foldl next z0) exports.programStep
      have lastValid : exports.program.valid (earlier.foldl step z0) witness := by
        rw [← foldEq]
        exact validRun.2.1
      obtain ⟨next, verified, tail⟩ := verifies_step exports complete keysShaped cap control _
        proof verifies witness lastValid (by simpa using lengthBound)
        (lengths witness (List.mem_append_right _ (List.mem_singleton_self _)))
      refine ⟨.recursive next, ?_, tail⟩
      simpa only [List.length_append, List.length_singleton, List.foldl_append, List.foldl_cons,
        List.foldl_nil] using verified

/-- Completeness of the light compression (SPEC.md, Algorithm 9): for every valid run of `n` steps
of the application from `z0` with `1 ≤ n ≤ cap + 1`, whose witnesses have the application's
witness length, the verifier accepts the honest prover's recursive proof of the run's last state,
whose last circuit is the light circuit, whose slot 1 is the default slot at depth 0, and whose
slot 0 has depth `n − 1`. Its fresh witness is zero on every cell from its package width
(`FreshZeroTail`). The first step is the base step (`StepExtend.base_extend`), and each
later step is a light fold into slot 0 below the cap (`StepExtend.light_extend`), so no step of
the run is a refresh fold. -/
theorem light_run_complete (exports : Exports relation ajtai circuitKeys step cubeFits)
    (complete : Complete exports) (keysShaped : ∀ key, (circuitKeys key).digest.length = 4)
    (cap : Nat) (control : AppState → AppWitness → Fin 2) (z0 : AppState)
    (z0Length : z0.length = Stage1.Application.stateWordCount) :
    ∀ witnesses : List AppWitness, IvcWalk.ValidFrom exports.program z0 witnesses →
      (∀ witness ∈ witnesses, witness.length = exports.program.witnessWordCount) →
      0 < witnesses.length → witnesses.length ≤ Schedule.productionLightFoldCap + 1 →
        ∃ payload, Verifies ajtai circuitKeys step complete.packageWidth cap control
            ⟨witnesses.length, z0, witnesses.foldl step z0⟩ (.recursive payload) ∧
          payload.pc = 1 ∧ payload.running 1 = (defaultSlot, 0) ∧
          (payload.running 0).2 = witnesses.length - 1 ∧
          FreshZeroTail complete.packageWidth (.recursive payload) := by
  have transport : ∀ (statement : TerminalStatement AppState)
      (control' : AppState → AppWitness → Fin 2) payload,
      Terminal.HoldsFor cap ajtai circuitKeys step control' statement (.recursive payload) →
        Terminal.HoldsFor cap ajtai circuitKeys step control statement (.recursive payload) :=
    fun statement control' payload holds =>
      (holdsFor_recursive_control cap ajtai circuitKeys step control' control _ payload).mp holds
  intro witnesses
  induction witnesses using List.reverseRecOn with
  | nil => exact fun _ _ positive => absurd positive (Nat.lt_irrefl 0)
  | append_singleton earlier witness inductionHypothesis =>
      intro validRun lengths _ lengthBound
      rw [IvcWalk.validFrom_append] at validRun
      have earlierLengths : ∀ other ∈ earlier, other.length = exports.program.witnessWordCount :=
        fun other inside => lengths other (List.mem_append_left _ inside)
      have witnessLength := lengths witness (List.mem_append_right _ (List.mem_singleton_self _))
      have foldEq : earlier.foldl exports.program.step z0 = earlier.foldl step z0 :=
        congrArg (fun next => earlier.foldl next z0) exports.programStep
      have lastValid : exports.program.valid (earlier.foldl step z0) witness := by
        rw [← foldEq]
        exact validRun.2.1
      have stepLength : ∀ current : AppState, current.length = Stage1.Application.stateWordCount →
          (step current witness).length = Stage1.Application.stateWordCount := by
        intro current currentLength
        rw [← exports.programStep]
        exact exports.program.step_length current witness currentLength
      simp only [List.length_append, List.length_singleton, List.foldl_append, List.foldl_cons,
        List.foldl_nil, Nat.add_sub_cancel] at lengthBound ⊢
      rcases Nat.eq_zero_or_pos earlier.length with empty | positive
      · rw [List.length_eq_zero_iff.mp empty]
        have bottom := (Terminal.holdsFor_bottom_iff cap ajtai circuitKeys step (fun _ _ => 0)
          ⟨0, z0, z0⟩).mpr ⟨⟨show (0 : Nat) < fieldModulus by decide, z0Length, z0Length⟩, rfl, rfl⟩
        rw [List.length_eq_zero_iff.mp empty] at lastValid
        obtain ⟨next, nextHolds, nextHonest, pcOne, defaults, tail⟩ := base_extend exports complete
          keysShaped cap (fun _ _ => 0) ⟨0, z0, z0⟩ bottom witness lastValid rfl
          (stepLength z0 z0Length) witnessLength
        exact ⟨next, (verifies_recursive_iff complete.packageWidth cap control _ next).mpr
          ⟨transport _ _ next nextHolds, nextHonest⟩, pcOne, defaults 1, by rw [defaults 0]; rfl,
          tail⟩
      · obtain ⟨payload, verifies, pcOne, slotOne, depthZero, -⟩ := inductionHypothesis validRun.1
          earlierLengths positive (by omega)
        obtain ⟨holds, honest⟩ := (verifies_recursive_iff complete.packageWidth cap control _
          payload).mp verifies
        have holds' := (holdsFor_recursive_control cap ajtai circuitKeys step control
          (fun _ _ => 0) _ payload).mp holds
        have currentLength := ((Terminal.holdsFor_recursive_iff cap ajtai circuitKeys step control _
          payload).mp holds).1.2.2
        have selectedZero : ∀ pcValid : InRange 2 payload.pc, selectedIndex pcValid = 0 :=
          fun _ => Fin.ext (by simp [selectedIndex, pcOne])
        obtain ⟨next, nextHolds, nextHonest, nextPc, nextRunning, tail⟩ := light_extend exports
          complete keysShaped cap (fun _ _ => 0) ⟨earlier.length, z0, earlier.foldl step z0⟩
          payload holds' honest witness lastValid rfl
          (fun pcValid => by rw [selectedZero pcValid, depthZero]; omega)
          (lt_of_le_of_lt lengthBound (by decide))
          (stepLength _ currentLength) witnessLength
        have pcValid : InRange 2 payload.pc := ⟨by rw [pcOne], by rw [pcOne]; decide⟩
        obtain ⟨nextDepth, nextOther⟩ := nextRunning pcValid
        rw [selectedZero pcValid] at nextDepth nextOther
        refine ⟨next, (verifies_recursive_iff complete.packageWidth cap control _ next).mpr
          ⟨transport _ _ next nextHolds, nextHonest⟩, nextPc, ?_, ?_, tail⟩
        · rw [nextOther 1 (by decide), slotOne]
        · rw [nextDepth, depthZero]
          omega

end NightstreamFPrime.Export.BoundedDepth.RunCompleteness
