import NightstreamFPrime.Lifecycle.OptionB.RefreshFold
import NightstreamFPrime.Lifecycle.OptionB.StateHashGadget
import NightstreamFPrime.Lifecycle.Stage1.Application

/-! The refresh step circuit's soundness (CW-3.3): the rows of the refresh circuit's gadgets, the
words that represent the step's input and output, the slot-select spec and the next slots'
digests give the refresh case
of `stepHolds_iff`. The gadgets are the light step's (`LightStep`) without the depth guard, with
the refresh fold (`RefreshFold`) in place of the light fold. A refresh may run at any depth, and
the selected slot takes depth 0 (`Schedule.nextDepth`). -/

namespace NightstreamFPrime.Lifecycle.OptionB.RefreshStep

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- The gadgets of the refresh step circuit and their offsets. The prior state hash reads two slot
gadgets; the next state hash reads two digests, and one slot gadget hashes the new slot. -/
structure Leaves (logicalWidth : Nat)
    (publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) (witnessWordCount : Nat) where
  application : Stage1.Application.Interface witnessWordCount
  applicationOffset : Nat
  prior : PriorStateHash.Interface
  priorOffset : Nat
  priorSlots : Fin 2 → OutputHash.Interface
  priorSlotOffsets : Fin 2 → Nat
  next : OutputHash.Interface
  nextOffset : Nat
  nextDigest : Fin 2 → Fin 4 → Expr
  newSlot : OutputHash.Interface
  newSlotOffset : Nat
  dec : Lifecycle.PiDEC.v1_2.Formal.Interface logicalWidth publicFits
  decOffset : Nat
  piccs : RefreshPiCCSPhase.Leaves logicalWidth 8 publicFits
  rlc : Lifecycle.PiRLC.v1_2.Formal.Interface logicalWidth publicFits
  rlcOffset : Nat

variable {witnessWordCount : Nat}

/-- Each gadget's rows hold. The fold gadgets are stated for `relation`, the relation of the
selected slot's circuit. -/
structure SpecHolds (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (step : AppState → AppWitness → AppState) (control : AppState → AppWitness → Fin 2)
    (circuit : Fin 2) (leaves : Leaves logicalWidth publicFits witnessWordCount) (env : Env) :
    Prop where
  application : Stage1.Application.Holds step leaves.application leaves.applicationOffset env
  dispatch : Stage1.Application.Valid (fun state witness => control state witness = circuit)
    leaves.application leaves.applicationOffset env
  prior : PriorStateHash.SpecHolds leaves.prior leaves.priorOffset env
  priorSlots : ∀ slot,
    OutputHash.SpecHolds (leaves.priorSlots slot) (leaves.priorSlotOffsets slot) env
  next : OutputHash.SpecHolds leaves.next leaves.nextOffset env
  newSlot : OutputHash.SpecHolds leaves.newSlot leaves.newSlotOffset env
  dec : Lifecycle.PiDEC.v1_2.Formal.SpecHolds relation leaves.dec leaves.decOffset env
  piccs : RefreshPiCCSPhase.SpecHolds relation leaves.piccs env
  rlc : Lifecycle.PiRLC.v1_2.Formal.SpecHolds relation leaves.rlc leaves.rlcOffset env

variable (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
  (step : AppState → AppWitness → AppState) (control : AppState → AppWitness → Fin 2)
  (circuit : Fin 2) (input : StepInput logicalWidth publicFits)
  (output : StepOutput logicalWidth publicFits)

/-- The circuit's words represent the step's input and output. `selected` is the circuit's
prior-circuit selector and `proof` the refresh proof that its words carry; the proof matters only
after the base case. -/
structure Represents (leaves : Leaves logicalWidth publicFits witnessWordCount) (selected : Fin 2)
    (proof : Spec.Folding.Refresh.Proof PaperAlgebra.Commitment productionShape 8) (env : Env) :
    Prop where
  applicationInput :
    Stage1.Application.inputState leaves.application leaves.applicationOffset env = input.zi
  applicationWitness :
    Stage1.Application.witnessValue leaves.application leaves.applicationOffset env =
      input.witness
  applicationOutput :
    Stage1.Application.outputState leaves.application leaves.applicationOffset env =
      output.zNext
  pcNext : output.pcNext = circuit
  priorSlotWords : ∀ slot, SlotRepresents (leaves.priorSlots slot) (leaves.priorSlotOffsets slot)
    env (input.running slot)
  priorStateWords : StateRepresents (leaves.prior.preimage leaves.priorOffset)
    (fun slot => gadgetDigest (leaves.priorSlots slot) (leaves.priorSlotOffsets slot) env) env
    (toState (Paper.priorHashPreimage
      ((setup Folding.Schedule.productionLightFoldCap ajtai circuitKeys).forCircuit circuit) input))
  priorPublicInput : ∀ column, (leaves.prior.publicInput leaves.priorOffset column).eval env =
    input.fresh.publicInputs ⟨0, by decide⟩ column
  nextStateWords : StateRepresents (leaves.next.preimage leaves.nextOffset)
    (fun slot => digestValue (leaves.nextDigest slot) env) env
    (toState (Paper.nextHashPreimage
      ((setup Folding.Schedule.productionLightFoldCap ajtai circuitKeys).forCircuit circuit) input
      output))
  nextDigest : output.x = gadgetDigest leaves.next leaves.nextOffset env
  decWired : RefreshFold.DecWired leaves.dec leaves.decOffset (input.running selected).1
    proof.split env
  piccsWired : RefreshPiCCSPhase.Wired (circuitKeys selected).relation ajtai
    (RefreshFold.children (circuitKeys selected).relation ajtai (input.running selected).1
      proof.split) input.fresh proof.nifs leaves.piccs env
  rlcWired : RefreshFold.RlcWired leaves.dec leaves.decOffset leaves.piccs leaves.rlc
    leaves.rlcOffset input.fresh proof.split proof.nifs.piCcsOutput env
  recursive : input.iteration ≠ 0 → input.nifsProof = .refresh proof

/-- Soundness of the refresh step circuit: its rows, its words and the slot select give the step
relation of a refresh circuit (`stepHolds_iff`) at the production cap. -/
theorem spec_implies_stepHolds (refresh : BoundedDepth.circuitStep circuit = .refresh)
    (leaves : Leaves logicalWidth publicFits witnessWordCount) (selected : Fin 2)
    (proof : Spec.Folding.Refresh.Proof PaperAlgebra.Commitment productionShape 8) (env : Env)
    (specification : SpecHolds (circuitKeys selected).relation step control circuit leaves env)
    (represents : Represents ajtai circuitKeys circuit input output leaves selected proof env)
    (select : SelectSpec input output selected
      (RefreshFold.nextSlot (circuitKeys selected).relation leaves.rlc leaves.rlcOffset env) 0)
    (nextDigests : ∀ slot,
      digestValue (leaves.nextDigest slot) env = slotDigest (output.runningNext slot)) :
    StepHolds Folding.Schedule.productionLightFoldCap ajtai circuitKeys step control circuit input
      output := by
  rw [stepHolds_iff]
  refine ⟨?_, represents.pcNext, ?_, ?_, ?_⟩
  · have valid := specification.dispatch
    unfold Stage1.Application.Valid at valid
    rwa [represents.applicationInput, represents.applicationWitness] at valid
  · have holds := specification.application
    unfold Stage1.Application.Holds at holds
    rwa [represents.applicationInput, represents.applicationWitness,
      represents.applicationOutput] at holds
  · exact builder_implies_output (cap := Folding.Schedule.productionLightFoldCap) (ajtai := ajtai)
      (circuitKeys := circuitKeys) (step := step) (control := control) (circuit := circuit)
      (input := input) (output := output) specification.next nextDigests
      represents.nextStateWords represents.nextDigest
  · by_cases zero : input.iteration = 0
    · exact Or.inl ⟨zero, select.base zero⟩
    · right
      obtain ⟨pcEq, claimEq, depthEq, others⟩ := select.recursive zero
      have proofEq := represents.recursive zero
      have priorPcValid : Paper.InRange 2 input.priorPc := by
        have bound := selected.isLt
        unfold Paper.InRange
        omega
      have selectedEq : Paper.selectedIndex priorPcValid = selected :=
        Fin.ext (by simp [Paper.selectedIndex, pcEq])
      refine ⟨priorPcValid, ?_⟩
      dsimp only
      rw [selectedEq, refresh]
      refine ⟨Nat.pos_of_ne_zero zero, ?_, (fun light => nomatch light), ?_, depthEq, others⟩
      · exact builder_implies_prior (cap := Folding.Schedule.productionLightFoldCap)
          (ajtai := ajtai)
          (circuitKeys := circuitKeys) (step := step) (control := control) (circuit := circuit)
          (input := input) specification.prior
          (fun slot => gadgetDigest_eq (specification.priorSlots slot)
            (represents.priorSlotWords slot))
          represents.priorStateWords represents.priorPublicInput
      · rw [claimEq]
        exact proofEq ▸ RefreshFold.spec_implies_fold (circuitKeys selected).relation ajtai
          (input.running selected).1 input.fresh proof leaves.dec leaves.decOffset leaves.piccs
          leaves.rlc leaves.rlcOffset env specification.dec represents.decWired specification.piccs
          represents.piccsWired specification.rlc represents.rlcWired

end NightstreamFPrime.Lifecycle.OptionB.RefreshStep
