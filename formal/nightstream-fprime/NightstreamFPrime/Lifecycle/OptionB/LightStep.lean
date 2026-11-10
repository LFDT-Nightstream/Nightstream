import NightstreamFPrime.Lifecycle.OptionB.LightFold
import NightstreamFPrime.Lifecycle.OptionB.StateHashGadget
import NightstreamFPrime.Lifecycle.OptionB.DepthGuard
import NightstreamFPrime.Lifecycle.Stage1.Application

/-! The light step circuit's soundness (CW-2.5c): the rows of the light circuit's gadgets, the
words that represent the step's input and output, and the slot-select spec give the light case
of `stepHolds_iff` (HyperNova Construction 2 with option B's two slots). The application gadget
is today's (`Stage1.Application`); its validity predicate is the dispatch `control zi w =
circuit`. The digest-select rows give `SelectSpec` with the incremented depth word as the next
depth, and the next slots' digests that the next state hash reads. The prior state hash reads the
prior digest words; after the base case each is the digest of its prior slot (`priorDigests`). -/

namespace NightstreamFPrime.Lifecycle.OptionB.LightStep

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Range
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- The gadgets of the light step circuit and their offsets. The prior and next state hashes read
two digests each, and one slot gadget hashes the new slot. -/
structure Leaves (logicalWidth : Nat)
    (publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) (witnessWordCount : Nat) where
  application : Stage1.Application.Interface witnessWordCount
  applicationOffset : Nat
  depth : CanonicalU64.Interface
  depthOffset : Nat
  prior : PriorStateHash.Interface
  priorOffset : Nat
  priorDigest : Fin 2 → Fin 4 → Expr
  next : OutputHash.Interface
  nextOffset : Nat
  nextDigest : Fin 2 → Fin 4 → Expr
  newSlot : OutputHash.Interface
  newSlotOffset : Nat
  piccs : LightPiCCSPhase.Leaves logicalWidth 8 publicFits
  rlc : LightFold.RlcLeaves logicalWidth publicFits

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
  depth : CanonicalU64.SpecHolds leaves.depth leaves.depthOffset env
  depthHigh : DepthGuard.HighBitsZero leaves.depthOffset env
  prior : PriorStateHash.SpecHolds leaves.prior leaves.priorOffset env
  next : OutputHash.SpecHolds leaves.next leaves.nextOffset env
  newSlot : OutputHash.SpecHolds leaves.newSlot leaves.newSlotOffset env
  piccs : LightPiCCSPhase.SpecHolds relation leaves.piccs env
  rlc : LightFold.RlcSpecHolds leaves.rlc env

variable (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
  (step : AppState → AppWitness → AppState) (control : AppState → AppWitness → Fin 2)
  (circuit : Fin 2) (input : StepInput logicalWidth publicFits)
  (output : StepOutput logicalWidth publicFits)

/-- The circuit's words represent the step's input and output. `selected` is the circuit's
prior-circuit selector and `proof` the light proof that its words carry; the proof and the
depth bound matter only after the base case. -/
structure Represents (leaves : Leaves logicalWidth publicFits witnessWordCount) (selected : Fin 2)
    (proof : Spec.Folding.Light.Proof LightShape 8) (env : Env) : Prop where
  applicationInput :
    Stage1.Application.inputState leaves.application leaves.applicationOffset env = input.zi
  applicationWitness :
    Stage1.Application.witnessValue leaves.application leaves.applicationOffset env =
      input.witness
  applicationOutput :
    Stage1.Application.outputState leaves.application leaves.applicationOffset env =
      output.zNext
  pcNext : output.pcNext = circuit
  priorStateWords : StateRepresents (leaves.prior.preimage leaves.priorOffset)
    (fun slot => digestValue (leaves.priorDigest slot) env) env
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
  piccsWired : LightPiCCSPhase.Wired (circuitKeys selected).relation ajtai
    (LightFold.lightRunning (input.running selected).1)
    (OptionBFolds.freshToLight input.fresh) proof leaves.piccs env
  rlcWired : LightFold.RlcWired leaves.piccs leaves.rlc (input.running selected).1 input.fresh
    proof.piCcsOutput env
  depthWord : (leaves.depth.source leaves.depthOffset).eval env =
    natWord (input.running selected).2
  recursive : input.iteration ≠ 0 →
    input.nifsProof = .light proof ∧ (input.running selected).2 < goldilocksModulus

/-- Soundness of the light step circuit: its rows, its words, the slot select and, after the base
case, prior digests that are the prior slots' digests give the step relation of a light circuit
(`stepHolds_iff`) at the production cap. -/
theorem spec_implies_stepHolds (light : BoundedDepth.circuitStep circuit = .light)
    (leaves : Leaves logicalWidth publicFits witnessWordCount) (selected : Fin 2)
    (proof : Spec.Folding.Light.Proof LightShape 8) (env : Env)
    (specification : SpecHolds (circuitKeys selected).relation step control circuit leaves env)
    (represents : Represents ajtai circuitKeys circuit input output leaves selected proof env)
    (select : SelectSpec input output selected (LightFold.nextSlot leaves.piccs leaves.rlc env)
      ((leaves.depth.source leaves.depthOffset).eval env + 1).val)
    (priorDigests : input.iteration ≠ 0 → ∀ slot,
      digestValue (leaves.priorDigest slot) env = slotDigest (input.running slot))
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
      obtain ⟨proofEq, depthBound⟩ := represents.recursive zero
      have priorPcValid : Paper.InRange 2 input.priorPc := by
        have bound := selected.isLt
        unfold Paper.InRange
        omega
      have selectedEq : Paper.selectedIndex priorPcValid = selected :=
        Fin.ext (by simp [Paper.selectedIndex, pcEq])
      have terms := DepthGuard.light_terms leaves.depth env leaves.depthOffset
        specification.depth specification.depthHigh depthBound represents.depthWord
      refine ⟨priorPcValid, ?_⟩
      dsimp only
      rw [selectedEq, light]
      refine ⟨Nat.pos_of_ne_zero zero, ?_, fun _ => terms.1, ?_, ?_, others⟩
      · exact builder_implies_prior (cap := Folding.Schedule.productionLightFoldCap)
          (ajtai := ajtai)
          (circuitKeys := circuitKeys) (step := step) (control := control) (circuit := circuit)
          (input := input) specification.prior (priorDigests zero) represents.priorStateWords represents.priorPublicInput
      · rw [claimEq]
        exact proofEq ▸ LightFold.spec_implies_fold (circuitKeys selected).relation ajtai
          (input.running selected).1 input.fresh proof leaves.piccs leaves.rlc env
          specification.piccs represents.piccsWired specification.rlc represents.rlcWired
      · rw [depthEq, ← terms.2]
        have below :
            Folding.Schedule.nextDepth (input.running selected).2 .light < goldilocksModulus := by
          have cap := terms.1
          simp only [Folding.Schedule.nextDepth, Folding.Schedule.productionLightFoldCap] at cap ⊢
          have modulus : 65 ≤ goldilocksModulus := by decide
          omega
        simp only [natWord, Poseidon2.ofNat]
        exact Nat.mod_eq_of_lt below

end NightstreamFPrime.Lifecycle.OptionB.LightStep
