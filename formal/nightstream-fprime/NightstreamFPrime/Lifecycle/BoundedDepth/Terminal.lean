import NightstreamFPrime.Lifecycle.BoundedDepth.Step
import NightstreamFPrime.Lifecycle.Relation

/-! The bounded-depth terminal (CW-4). It is HyperNova Construction 2's outer terminal with the bounded-depth scheme's
two slots: at iteration 0 the current state is the initial state; otherwise the fresh instance's
public input is the hash of the state with both slots, every slot's claim holds under its
circuit's relation as a combined claim (so its norm is below `B`, the decider check of CW-6), and
the fresh instance holds under the relation of the circuit that ran last. The terminal reads only
the setup's verifier keys, so it is the same for both circuits (`transition_forCircuit`). -/

namespace NightstreamFPrime.Lifecycle.BoundedDepth.Terminal

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper
open NightstreamFPrime.Lifecycle.PaperAlgebra

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ RingCarrierLayout.carrierWidth logicalWidth}

abbrev Witness := PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)

abbrev ProofEnvelope := OuterTerminalProof (SlotData logicalWidth publicFits)
  (Witness (logicalWidth := logicalWidth) (publicFits := publicFits))
  (StepFresh logicalWidth publicFits)
  (Witness (logicalWidth := logicalWidth) (publicFits := publicFits)) 2

/-- Each slot's claim holds under its circuit's relation; the fresh instance holds under the
relation of the circuit that ran last. -/
noncomputable def relations
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    TerminalRelations (CircuitKey logicalWidth publicFits) (SlotData logicalWidth publicFits)
      (Witness (logicalWidth := logicalWidth) (publicFits := publicFits))
      (StepFresh logicalWidth publicFits)
      (Witness (logicalWidth := logicalWidth) (publicFits := publicFits)) 2 where
  runningHolds := fun _ key slot witness =>
    CE.Holds (semantics ajtai) productionGlobalParams (toClaim (key.system ajtai) slot.1) witness
  freshHolds := fun _ key fresh witness =>
    CCS.Holds (semantics ajtai) productionGlobalParams (freshStatement key.relation fresh) witness

def StatementValid (statement : TerminalStatement AppState) : Prop :=
  statement.iteration < fieldModulus ∧
    statement.z0.length = Stage1.Application.stateWordCount ∧
    statement.zi.length = Stage1.Application.stateWordCount

/-- The bounded-depth terminal relation. -/
noncomputable def HoldsFor (cap : Nat)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
    (step : AppState → AppWitness → AppState) (control : AppState → AppWitness → Fin 2)
    (statement : TerminalStatement AppState)
    (proof : ProofEnvelope (logicalWidth := logicalWidth) (publicFits := publicFits)) : Prop :=
  StatementValid statement ∧
    OuterTerminalTransition ((setup cap ajtai circuitKeys).forCircuit 0) (machine step control)
      (relations ajtai) statement proof

/-- The terminal reads only the verifier keys, which both circuits share. -/
theorem transition_forCircuit (cap : Nat)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
    (step : AppState → AppWitness → AppState) (control : AppState → AppWitness → Fin 2)
    (circuit : Fin 2) (statement : TerminalStatement AppState)
    (proof : ProofEnvelope (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    OuterTerminalTransition ((setup cap ajtai circuitKeys).forCircuit circuit)
        (machine step control) (relations ajtai) statement proof ↔
      OuterTerminalTransition ((setup cap ajtai circuitKeys).forCircuit 0)
        (machine step control) (relations ajtai) statement proof := by
  cases proof <;> rfl

theorem holdsFor_bottom_iff (cap : Nat)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
    (step : AppState → AppWitness → AppState) (control : AppState → AppWitness → Fin 2)
    (statement : TerminalStatement AppState) :
    HoldsFor cap ajtai circuitKeys step control statement .bottom ↔
      StatementValid statement ∧ statement.iteration = 0 ∧ statement.zi = statement.z0 :=
  Iff.rfl

/-- A recursive terminal: the state hash binds both slots and the fresh instance, both slot claims
hold, and the fresh instance holds under the relation of the circuit that ran last. -/
theorem holdsFor_recursive_iff (cap : Nat)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
    (step : AppState → AppWitness → AppState) (control : AppState → AppWitness → Fin 2)
    (statement : TerminalStatement AppState)
    (payload : TerminalProof (SlotData logicalWidth publicFits)
      (Witness (logicalWidth := logicalWidth) (publicFits := publicFits))
      (StepFresh logicalWidth publicFits)
      (Witness (logicalWidth := logicalWidth) (publicFits := publicFits)) 2) :
    HoldsFor cap ajtai circuitKeys step control statement (.recursive payload) ↔
      StatementValid statement ∧ ∃ pcValid : InRange 2 payload.pc,
        0 < statement.iteration ∧
        payload.fresh.publicInputs ⟨0, by decide⟩ =
          encHash (publicFits := publicFits) (stateHash (toState {
            verifierKeys := circuitKeys
            iteration := statement.iteration
            z0 := statement.z0
            current := statement.zi
            running := payload.running
            pc := payload.pc })) ∧
        (∀ slot, CE.Holds (semantics ajtai) productionGlobalParams
          (toClaim ((circuitKeys slot).system ajtai) (payload.running slot).1)
          (payload.runningWitness slot)) ∧
        CCS.Holds (semantics ajtai) productionGlobalParams
          (freshStatement (circuitKeys (selectedIndex pcValid)).relation payload.fresh)
          payload.freshWitness :=
  Iff.rfl

theorem rejects_counter_shift (cap : Nat)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
    (step : AppState → AppWitness → AppState) (control : AppState → AppWitness → Fin 2)
    (statement : TerminalStatement AppState)
    (proof : ProofEnvelope (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    ¬ HoldsFor cap ajtai circuitKeys step control
      { statement with iteration := statement.iteration + fieldModulus } proof := by
  intro accepted
  have bound := accepted.1.1
  dsimp only at bound
  omega

end NightstreamFPrime.Lifecycle.BoundedDepth.Terminal
