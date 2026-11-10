import NightstreamFPrime.Lifecycle.BoundedDepth.Keys
import NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepth

/-! The bounded-depth scheme's step relation: per-circuit HyperNova Construction 2 (`BoundedDepth`) with the light
circuit at slot 0 and the refresh circuit at slot 1. Both circuits run the same application step,
so the Nebula phase is in both (gate G2). The application chooses the circuit (`control`); the
light verifier's depth guard keeps every schedule sound.

A slot carries `SlotData`, the claim and depth that the state hash binds. A fold reads the claim
with the slot circuit's constraint system at stage `combined`, runs the bounded-depth fold of the
circuit that runs now, and keeps the output's single evaluation family (`slotFolds`). Every fold
output has that constraint system and stage (`BoundedDepthKeys.folds_output`), so `slotFolds` accepts
exactly the folds that `BoundedDepthFolds.folds` accepts on the claims the slots carry
(`slotFolds_eq_some_iff`). `stepHolds_iff` states the step relation in the terms a step circuit
checks. -/

namespace NightstreamFPrime.Lifecycle.BoundedDepth

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2
open NightstreamFPrime.Spec.HyperNova.NonInteractiveMultiFold (Accepts)
open NightstreamFPrime.Lifecycle.PaperAlgebra

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

/-- One step circuit: the key digest that the state hash binds and the circuit's relation. Both
circuits share the Ajtai key, so the fold keys are `keys relation ajtai`. -/
structure CircuitKey (logicalWidth : Nat)
    (publicFits :
      ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth) where
  digest : KeyDigest
  relation : ProductionKey.LogicalRelation logicalWidth publicFits

abbrev StepClaim (logicalWidth : Nat)
    (publicFits :
      ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth) :=
  BoundedDepthFolds.Claim PaperAlgebra.Commitment
    (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
    productionShape (Phi81CarrierLayout.carrierWidth logicalWidth)
    (Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth logicalWidth))

/-- A fold proof of either circuit. Every production relation has degree bound 8
(`ProductionKey.degreeBound_eq`). -/
abbrev StepProof := BoundedDepthFolds.FoldProof PaperAlgebra.Commitment productionShape 8 8

abbrev StepFresh (logicalWidth : Nat)
    (publicFits :
      ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth) :=
  Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)

def toClaim (system : PaperAlgebra.Structure logicalWidth)
    (claim : SlotClaim logicalWidth publicFits) : StepClaim logicalWidth publicFits where
  constraintSystem := system
  commitment := claim.commitment
  publicInput := claim.publicInput
  point := claim.point
  evaluations := #[claim.evaluations]
  stage := .combined

def fromClaim (claim : StepClaim logicalWidth publicFits) :
    Option (SlotClaim logicalWidth publicFits) :=
  if size : claim.evaluations.size = 1 then
    some ⟨claim.commitment, claim.publicInput, claim.point, claim.evaluations[0]⟩
  else none

theorem fromClaim_toClaim (system : PaperAlgebra.Structure logicalWidth)
    (claim : SlotClaim logicalWidth publicFits) :
    fromClaim (toClaim system claim) = some claim := by
  simp [fromClaim, toClaim]

theorem toClaim_of_fromClaim {system : PaperAlgebra.Structure logicalWidth}
    {output : StepClaim logicalWidth publicFits} {claim : SlotClaim logicalWidth publicFits}
    (systemEq : output.constraintSystem = system) (stageEq : output.stage = .combined)
    (read : fromClaim output = some claim) : output = toClaim system claim := by
  obtain ⟨_, _, _, _, evaluations, _⟩ := output
  simp only at systemEq stageEq
  subst systemEq stageEq
  unfold fromClaim at read
  split at read
  · next size =>
      cases read
      simp only [toClaim, CE.Instance.mk.injEq, true_and, and_true]
      apply Array.ext
      · simpa using size
      · intro index _ inside
        have zero : index = 0 := by simpa using inside
        subst zero
        simp
  · cases read

/-- The constraint system that a circuit's slot claims are read with. -/
noncomputable def CircuitKey.system (ajtai : AjtaiKey (logicalWidth := logicalWidth)
      (publicFits := publicFits)) (key : CircuitKey logicalWidth publicFits) :
    PaperAlgebra.Structure logicalWidth :=
  (refreshKey key.relation ajtai).relationSource

/-- The bounded-depth folds on the claims that slots carry. -/
noncomputable def slotFolds
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    Spec.HyperNova.Construction2.BoundedDepth.Folds (CircuitKey logicalWidth publicFits) (SlotClaim logicalWidth publicFits)
      (StepFresh logicalWidth publicFits) StepProof :=
  fun step key claim fresh proof =>
    (BoundedDepthFolds.folds step (keys key.relation ajtai) (toClaim (key.system ajtai) claim) fresh
      proof).bind fromClaim

/-- `slotFolds` accepts exactly the folds that `BoundedDepthFolds.folds` accepts on carried claims. -/
theorem slotFolds_eq_some_iff
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (step : Schedule.Step) (key : CircuitKey logicalWidth publicFits)
    (claim : SlotClaim logicalWidth publicFits) (fresh : StepFresh logicalWidth publicFits)
    (proof : StepProof) (next : SlotClaim logicalWidth publicFits) :
    slotFolds ajtai step key claim fresh proof = some next ↔
      BoundedDepthFolds.folds step (keys key.relation ajtai) (toClaim (key.system ajtai) claim) fresh
        proof = some (toClaim (key.system ajtai) next) := by
  unfold slotFolds
  rw [Option.bind_eq_some_iff]
  constructor
  · rintro ⟨output, accepted, read⟩
    have shape := HyperNova.Construction2.BoundedDepthKeys.folds_output
      (refreshKey key.relation ajtai) rfl (lightParts key.relation ajtai) accepted
    rw [accepted, toClaim_of_fromClaim shape.1 shape.2 read]
    rfl
  · intro accepted
    exact ⟨_, accepted, fromClaim_toClaim _ _⟩

/-- HyperNova Definition 12's default running instance, as a slot claim. -/
def defaultSlot : SlotClaim logicalWidth publicFits where
  commitment := (defaultRunning (publicFits := publicFits)).commitments ⟨0, by decide⟩
  publicInput := (defaultRunning (publicFits := publicFits)).publicInputs ⟨0, by decide⟩
  point := (defaultRunning (publicFits := publicFits)).point
  evaluations := (defaultRunning (publicFits := publicFits)).evaluations ⟨0, by decide⟩

noncomputable def setup (cap : Nat)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits) :=
  Spec.HyperNova.Construction2.BoundedDepth.setup cap (slotFolds ajtai) circuitKeys defaultSlot

/-- The state that a Construction 2 hash preimage describes. -/
def toState
    (preimage : Paper.HashPreimage (CircuitKey logicalWidth publicFits) AppState
      (SlotData logicalWidth publicFits) 2) :
    State logicalWidth publicFits where
  verifierKeys := fun circuit => (preimage.verifierKeys circuit).digest
  iteration := preimage.iteration
  z0 := preimage.z0
  current := preimage.current
  pc := preimage.pc
  slots := preimage.running

def machine (step : AppState → AppWitness → AppState)
    (control : AppState → AppWitness → Fin 2) :
    Paper.Machine (CircuitKey logicalWidth publicFits) Digest AppState AppWitness
      (SlotData logicalWidth publicFits) (StepFresh logicalWidth publicFits)
      (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits)) 2 where
  control := control
  step := fun _ state witness => step state witness
  freshPublic := fun fresh => fresh.publicInputs ⟨0, by decide⟩
  encodeInstance := encHash (publicFits := publicFits)
  hash := fun preimage => stateHash (toState preimage)

/-- The machine hash binds the key digests, the counters, both states and both slots, or
Poseidon2 has a collision (`stateHash_collision`). -/
theorem hash_binds
    {left right : Paper.HashPreimage (CircuitKey logicalWidth publicFits) AppState
      (SlotData logicalWidth publicFits) 2}
    (leftWell : (toState left).WellFormed) (rightWell : (toState right).WellFormed)
    (same : stateHash (toState left) = stateHash (toState right)) :
    ((fun circuit => (left.verifierKeys circuit).digest) =
          (fun circuit => (right.verifierKeys circuit).digest) ∧
        left.iteration = right.iteration ∧ left.z0 = right.z0 ∧ left.current = right.current ∧
        left.pc = right.pc ∧ left.running = right.running) ∨
      StateCollision (toState left) (toState right) := by
  by_cases equal : toState left = toState right
  · exact Or.inl ⟨congrArg State.verifierKeys equal, congrArg State.iteration equal,
      congrArg State.z0 equal, congrArg State.current equal, congrArg State.pc equal,
      congrArg State.slots equal⟩
  · exact Or.inr (stateHash_collision leftWell rightWell equal same)

abbrev StepInput (logicalWidth : Nat)
    (publicFits :
      ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth) :=
  Paper.Input (CircuitKey logicalWidth publicFits) AppState AppWitness
    (SlotData logicalWidth publicFits) (StepFresh logicalWidth publicFits) StepProof 2

abbrev StepOutput (logicalWidth : Nat)
    (publicFits :
      ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth) :=
  Paper.Output Digest AppState (SlotData logicalWidth publicFits) 2

/-- The spec of the slot-select rows: at iteration 0 the initial state is the current state and
both slots are the default slot at depth 0; otherwise the prior circuit is `selected`, its slot
takes the fold's claim `next` at depth `nextDepth`, and the other slot is copied. The light step
writes the incremented depth word and the refresh step writes 0. -/
structure SelectSpec (input : StepInput logicalWidth publicFits)
    (output : StepOutput logicalWidth publicFits) (selected : Fin 2)
    (next : SlotClaim logicalWidth publicFits) (nextDepth : Nat) : Prop where
  base : input.iteration = 0 →
    input.z0 = input.zi ∧ output.runningNext = fun _ => (defaultSlot, 0)
  recursive : input.iteration ≠ 0 →
    input.priorPc = selected.val + 1 ∧ (output.runningNext selected).1 = next ∧
      (output.runningNext selected).2 = nextDepth ∧
      ∀ slot, slot ≠ selected → output.runningNext slot = input.running slot

/-- The bounded-depth scheme's step relation for circuit `circuit`. -/
noncomputable def StepHolds (cap : Nat)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
    (step : AppState → AppWitness → AppState) (control : AppState → AppWitness → Fin 2)
    (circuit : Fin 2) (input : StepInput logicalWidth publicFits)
    (output : StepOutput logicalWidth publicFits) : Prop :=
  PerCircuit.Transition (setup cap ajtai circuitKeys) (machine step control) circuit input output

/-- The slot verifier of the circuit that runs now: the depth guard, the bounded-depth fold of that
circuit with the selected slot's keys, and the depth update. -/
theorem accepts_iff (cap : Nat)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuit : Fin 2) (key : CircuitKey logicalWidth publicFits)
    (slot : SlotData logicalWidth publicFits) (fresh : StepFresh logicalWidth publicFits)
    (proof : StepProof) (next : SlotData logicalWidth publicFits) :
    Accepts (Spec.HyperNova.Construction2.BoundedDepth.verifier cap (slotFolds ajtai) circuit) key slot fresh proof next ↔
      (Spec.HyperNova.Construction2.BoundedDepth.circuitStep circuit = .light → slot.2 < cap) ∧
        BoundedDepthFolds.folds (Spec.HyperNova.Construction2.BoundedDepth.circuitStep circuit) (keys key.relation ajtai)
            (toClaim (key.system ajtai) slot.1) fresh proof =
          some (toClaim (key.system ajtai) next.1) ∧
        next.2 = Schedule.nextDepth slot.2 (Spec.HyperNova.Construction2.BoundedDepth.circuitStep circuit) := by
  rw [Spec.HyperNova.Construction2.BoundedDepth.accepts_iff, slotFolds_eq_some_iff]

/-- The step relation in the terms a step circuit checks: the dispatch, the application step,
the output state hash, and either the base case or, for the selected prior slot, the prior state
hash in the fresh instance, the depth guard, the fold and the depth update, with the other slot
unchanged. -/
theorem stepHolds_iff (cap : Nat)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
    (step : AppState → AppWitness → AppState) (control : AppState → AppWitness → Fin 2)
    (circuit : Fin 2) (input : StepInput logicalWidth publicFits)
    (output : StepOutput logicalWidth publicFits) :
    StepHolds cap ajtai circuitKeys step control circuit input output ↔
      control input.zi input.witness = circuit ∧
      output.pcNext = circuit ∧
      output.zNext = step input.zi input.witness ∧
      output.x = stateHash (toState
        (Paper.nextHashPreimage ((setup cap ajtai circuitKeys).forCircuit circuit) input
          output)) ∧
      ((input.iteration = 0 ∧ input.z0 = input.zi ∧
          output.runningNext = fun _ => (defaultSlot, 0)) ∨
        ∃ priorPcValid : Paper.InRange 2 input.priorPc,
          let selected := Paper.selectedIndex priorPcValid
          let key := circuitKeys selected
          0 < input.iteration ∧
          input.fresh.publicInputs ⟨0, by decide⟩ =
            encHash (publicFits := publicFits) (stateHash (toState
              (Paper.priorHashPreimage ((setup cap ajtai circuitKeys).forCircuit circuit)
                input))) ∧
          (Spec.HyperNova.Construction2.BoundedDepth.circuitStep circuit = .light → (input.running selected).2 < cap) ∧
          BoundedDepthFolds.folds (Spec.HyperNova.Construction2.BoundedDepth.circuitStep circuit) (keys key.relation ajtai)
              (toClaim (key.system ajtai) (input.running selected).1) input.fresh
              input.nifsProof =
            some (toClaim (key.system ajtai) (output.runningNext selected).1) ∧
          (output.runningNext selected).2 =
            Schedule.nextDepth (input.running selected).2 (Spec.HyperNova.Construction2.BoundedDepth.circuitStep circuit) ∧
          ∀ slot, slot ≠ selected → output.runningNext slot = input.running slot) := by
  unfold StepHolds PerCircuit.Transition Paper.Transition
  simp only [machine, setup, Spec.HyperNova.Construction2.BoundedDepth.setup, PerCircuit.Setup.forCircuit, accepts_iff,
    and_assoc]

end NightstreamFPrime.Lifecycle.BoundedDepth
