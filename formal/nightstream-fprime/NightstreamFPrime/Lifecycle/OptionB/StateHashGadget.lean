import NightstreamFPrime.Lifecycle.OptionB.Step
import NightstreamFPrime.Lifecycle.OutputHash
import NightstreamFPrime.Lifecycle.PriorStateHashCore

/-! Option B's state hash in a step circuit (CW-2.4): one `OutputHash` gadget per hashed slot and
one hash gadget for the 38-word state preimage, whose last eight words are two slot digests.
`builder_implies_output` and `builder_implies_prior` give the output state hash and the prior
state hash in the fresh instance of `stepHolds_iff` from the gadget specifications, the words the
circuit holds and the value of each slot digest. A step runs three slot digests (both prior slots
and the new slot; the other next digest is a prior digest, `DigestSelect`) and two state hashes:
3 × 190,704 + 2 × 5,480 = 583,072 hash recipes (`slotGadget_hashLength`,
`stateGadget_hashLength`), against 2 × 2,542,720 = 5,085,440 for today's two flat state hashes
(`PilotProduction.hashWitnessCount_eq`). -/

namespace NightstreamFPrime.Lifecycle.OptionB

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2
open NightstreamFPrime.Lifecycle.PaperAlgebra

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

/-- The value of four digest words. -/
def digestValue (digest : Fin 4 → Expr) (env : Env) : Digest :=
  List.ofFn fun lane => (digest lane).eval env

/-- The four digest words of a hash gadget's output lanes. -/
def gadgetDigest (gadget : OutputHash.Interface) (offset : Nat) (env : Env) : Digest :=
  digestValue (gadget.digest offset) env

/-- The circuit's words for a slot gadget's preimage are the slot domain and the slot's words. -/
def SlotRepresents (gadget : OutputHash.Interface) (offset : Nat) (env : Env)
    (slot : SlotData logicalWidth publicFits) : Prop :=
  Hash.evalList env (gadget.preimage offset) = slotDomainChunk ++ serializeSlot slot

/-- The circuit's words for the state preimage, with the slot digests `digests`. -/
def StateRepresents (preimage : List Expr) (digests : Fin 2 → Digest) (env : Env)
    (state : State logicalWidth publicFits) : Prop :=
  Hash.evalList env preimage =
    stateDomainChunk ++ state.verifierKeys 0 ++ state.verifierKeys 1 ++
      [natWord state.iteration] ++ state.z0 ++ state.current ++ [natWord state.pc] ++
      digests 0 ++ digests 1

theorem gadgetDigest_eq {gadget : OutputHash.Interface} {offset : Nat} {env : Env}
    {slot : SlotData logicalWidth publicFits} (spec : OutputHash.SpecHolds gadget offset env)
    (represents : SlotRepresents gadget offset env slot) :
    gadgetDigest gadget offset env = slotDigest slot := by
  unfold gadgetDigest digestValue slotDigest
  rw [← represents]
  exact spec

/-- With each slot digest the digest of the state's slot, the state words are the state
preimage. -/
theorem statePreimage_of_digests {preimage : List Expr} {digests : Fin 2 → Digest} {env : Env}
    {state : State logicalWidth publicFits}
    (digestsEq : ∀ slot, digests slot = slotDigest (state.slots slot))
    (stateWords : StateRepresents preimage digests env state) :
    Hash.evalList env preimage = statePreimage state := by
  rw [stateWords, digestsEq 0, digestsEq 1]
  rfl

theorem output_stateHash {output : OutputHash.Interface} {offset : Nat}
    {digests : Fin 2 → Digest} {env : Env} {state : State logicalWidth publicFits}
    (spec : OutputHash.SpecHolds output offset env)
    (digestsEq : ∀ slot, digests slot = slotDigest (state.slots slot))
    (stateWords : StateRepresents (output.preimage offset) digests env state) :
    gadgetDigest output offset env = stateHash state := by
  unfold gadgetDigest digestValue stateHash
  rw [← statePreimage_of_digests digestsEq stateWords]
  exact spec

theorem prior_publicInput {prior : PriorStateHash.Interface} {offset : Nat}
    {digests : Fin 2 → Digest} {env : Env} {state : State logicalWidth publicFits}
    (spec : PriorStateHash.SpecHolds prior offset env)
    (digestsEq : ∀ slot, digests slot = slotDigest (state.slots slot))
    (stateWords : StateRepresents (prior.preimage offset) digests env state) :
    (fun column => (prior.publicInput offset column).eval env) =
      encHash (publicFits := publicFits) (stateHash state) := by
  rw [spec, statePreimage_of_digests digestsEq stateWords]
  rfl

section Step

variable (cap : Nat) (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
  (step : AppState → AppWitness → AppState) (control : AppState → AppWitness → Fin 2)
  (circuit : Fin 2) (input : StepInput logicalWidth publicFits)
  (output : StepOutput logicalWidth publicFits)

/-- The output state hash of `stepHolds_iff` from the output gadget and the next slot digests. -/
theorem builder_implies_output {outputGadget : OutputHash.Interface} {offset : Nat}
    {digests : Fin 2 → Digest} {env : Env}
    (spec : OutputHash.SpecHolds outputGadget offset env)
    (digestsEq : ∀ slot, digests slot = slotDigest (output.runningNext slot))
    (stateWords : StateRepresents (outputGadget.preimage offset) digests env
      (toState (Paper.nextHashPreimage ((setup cap ajtai circuitKeys).forCircuit circuit) input
        output)))
    (digest : output.x = gadgetDigest outputGadget offset env) :
    Paper.OutputHolds ((setup cap ajtai circuitKeys).forCircuit circuit) (machine step control)
      input output := by
  rw [Paper.OutputHolds, digest]
  exact output_stateHash
    (state := toState (Paper.nextHashPreimage ((setup cap ajtai circuitKeys).forCircuit circuit)
      input output)) spec digestsEq stateWords

/-- The prior state hash in the fresh instance, as `stepHolds_iff` requires it. -/
theorem builder_implies_prior {prior : PriorStateHash.Interface} {offset : Nat}
    {digests : Fin 2 → Digest} {env : Env}
    (spec : PriorStateHash.SpecHolds prior offset env)
    (digestsEq : ∀ slot, digests slot = slotDigest (input.running slot))
    (stateWords : StateRepresents (prior.preimage offset) digests env
      (toState (Paper.priorHashPreimage ((setup cap ajtai circuitKeys).forCircuit circuit)
        input)))
    (publicInput : ∀ column,
      (prior.publicInput offset column).eval env =
        input.fresh.publicInputs ⟨0, by decide⟩ column) :
    (machine step control).freshPublic input.fresh =
      (machine step control).encodeInstance ((machine step control).hash
        (Paper.priorHashPreimage ((setup cap ajtai circuitKeys).forCircuit circuit) input)) := by
  have hashed := prior_publicInput
    (state := toState (Paper.priorHashPreimage ((setup cap ajtai circuitKeys).forCircuit circuit)
      input)) spec digestsEq stateWords
  funext column
  exact (publicInput column).symm.trans (congrFun hashed column)

end Step

/-- A slot gadget hashes 2,067 words: 173 absorptions and the padding permutation. -/
theorem slotGadget_hashLength {gadget : OutputHash.Interface} {offset : Nat} {env : Env}
    {slot : SlotData logicalWidth publicFits} (represents : SlotRepresents gadget offset env slot) :
    OutputHash.hashLength gadget offset = 190704 := by
  have length := congrArg List.length represents
  simp only [Hash.evalList, List.length_map, List.length_append, serializeSlot_length_eq] at length
  simp [OutputHash.hashLength, Hash.inputChunks, length, slotDomainChunk, domainChunk,
    Poseidon2.rate]

/-- A state gadget hashes 38 words: four absorptions and the padding permutation. -/
theorem stateGadget_hashLength {gadget : OutputHash.Interface} {offset : Nat}
    {digests : Fin 2 → Digest} {env : Env} {state : State logicalWidth publicFits}
    (well : state.WellFormed) (digestLength : ∀ slot, (digests slot).length = 4)
    (represents : StateRepresents (gadget.preimage offset) digests env state) :
    OutputHash.hashLength gadget offset = 5480 := by
  have length := congrArg List.length represents
  simp only [Hash.evalList, List.length_map, List.length_append, List.length_singleton,
    digestLength, well.verifierKeys, well.z0, well.current] at length
  simp [OutputHash.hashLength, Hash.inputChunks, length, stateDomainChunk, domainChunk,
    Poseidon2.rate]

end NightstreamFPrime.Lifecycle.OptionB
