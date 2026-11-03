import NightstreamFPrime.Lifecycle.XOut
import NightstreamFPrime.Lifecycle.Transcript

/-! Option B's state hash with slot digests (owner decision 2026-10-31). Each slot's claim and
depth are hashed to a four-word digest under their own domain. The state hash binds the two
circuits' verifier-key digests, the step counter, the initial and current states, the program
counter and the two slot digests. HyperNova Construction 2 step 5 hashes the running instances
directly; the extra level is the recorded difference. `stateHash_collision`: two different
well-formed states with one state hash give a Poseidon2 collision on an input pair that the two
states fix (`StateCollision`). A collision of some pair always exists, so only a pair that the
states name is an event that can have a small probability. -/

namespace NightstreamFPrime.Lifecycle.OptionB

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction (EvaluationFamily)
open NightstreamFPrime.Lifecycle.PaperAlgebra

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

/-- A slot's accumulator claim. The slot's key fixes the constraint system and the norm stage,
so the claim does not carry them. -/
structure SlotClaim (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth)
    where
  commitment : PaperAlgebra.Commitment
  publicInput : PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits)
  point : CubePoint K cubeVariables
  evaluations : EvaluationFamily K productionShape

/-- A slot: its claim and its depth, the light folds since its last refresh
(`BoundedDepth.Running`). -/
abbrev SlotData (logicalWidth : Nat)
    (publicFits :
      ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth) :=
  SlotClaim logicalWidth publicFits × Nat

/-- Twelve words: the ASCII bytes of `label`, packed eight to a word (SPEC.md 4.2). -/
def domainChunk (label : String) : List F :=
  (List.range Poseidon2.rate).map fun word =>
    packBytes (((label.toList.map Char.toNat).drop (8 * word)).take 8)

def slotDomainChunk : List F := domainChunk "HyperNova/NIVC/slot/v3"

def stateDomainChunk : List F := domainChunk "HyperNova/NIVC/state/v3"

def serializeSlot (slot : SlotData logicalWidth publicFits) : List F :=
  serializeCommitment slot.1.commitment ++ serializeEvaluations slot.1.evaluations ++
    serializePoint slot.1.point ++
    serializePublicInput (publicFits := publicFits) slot.1.publicInput ++ [natWord slot.2]

def slotDigest (slot : SlotData logicalWidth publicFits) : Digest :=
  Poseidon2.hash (slotDomainChunk ++ serializeSlot slot)

/-- What the state hash binds. -/
structure State (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth)
    where
  verifierKeys : Fin 2 → KeyDigest
  iteration : Nat
  z0 : AppState
  current : AppState
  pc : Nat
  slots : Fin 2 → SlotData logicalWidth publicFits

def statePreimage (state : State logicalWidth publicFits) : List F :=
  stateDomainChunk ++ state.verifierKeys 0 ++ state.verifierKeys 1 ++ [natWord state.iteration] ++
    state.z0 ++ state.current ++ [natWord state.pc] ++ slotDigest (state.slots 0) ++
    slotDigest (state.slots 1)

def stateHash (state : State logicalWidth publicFits) : Digest :=
  Poseidon2.hash (statePreimage state)

/-- Key digests and states have four words, and every counted word is below `q`. -/
structure State.WellFormed (state : State logicalWidth publicFits) : Prop where
  verifierKeys : ∀ circuit, (state.verifierKeys circuit).length = 4
  z0 : state.z0.length = 4
  current : state.current.length = 4
  iteration : state.iteration < goldilocksModulus
  pc : state.pc < goldilocksModulus
  depth : ∀ slot, (state.slots slot).2 < goldilocksModulus

theorem domainChunks_distinct :
    slotDomainChunk ≠ stateDomainChunk ∧ slotDomainChunk ≠ Lifecycle.stateDomainChunk ∧
      stateDomainChunk ≠ Lifecycle.stateDomainChunk ∧
      slotDomainChunk ≠ Transcript.foldDomainChunk ∧
      stateDomainChunk ≠ Transcript.foldDomainChunk := by
  decide

theorem natWord_injective {left right : Nat} (leftBound : left < goldilocksModulus)
    (rightBound : right < goldilocksModulus) (same : natWord left = natWord right) :
    left = right := by
  have values := congrArg Fin.val same
  simpa [natWord, Poseidon2.ofNat, Nat.mod_eq_of_lt leftBound, Nat.mod_eq_of_lt rightBound]
    using values

theorem serializePoint_injective : Function.Injective serializePoint := by
  intro left right same
  have coordinates : left.coordinates = right.coordinates := serializeKs_injective same
  cases left
  cases right
  simp only at coordinates
  subst coordinates
  rfl

theorem serializeSlot_length (slot : SlotData logicalWidth publicFits) :
    (serializeSlot slot).length =
      productionProfile.commitmentWidth * ringDegree +
        (productionShape.matrixCount + 1) * productionShape.coefficientCount * 2 +
        cubeVariables * 2 + (FullShape logicalWidth publicFits).publicWidth + 1 := by
  simp [serializeSlot, Nat.add_assoc]

/-- A slot is 2,055 words: κd = 1,188, `Eval_K` and `Eval_A` 540, the point 56, the public input
270 and the depth 1 (the CW-0 census number). -/
theorem serializeSlot_length_eq (slot : SlotData logicalWidth publicFits) :
    (serializeSlot slot).length = 2055 := by
  rw [serializeSlot_length, publicWidth_eq]
  rfl

/-- A well-formed state preimage is 38 words. -/
theorem statePreimage_length {state : State logicalWidth publicFits} (well : state.WellFormed) :
    (statePreimage state).length = 38 := by
  simp [statePreimage, slotDigest, Poseidon2.hash_length, well.verifierKeys, well.z0, well.current,
    Poseidon2.digestLen]
  rfl

theorem serializeSlot_injective {left right : SlotData logicalWidth publicFits}
    (leftDepth : left.2 < goldilocksModulus) (rightDepth : right.2 < goldilocksModulus)
    (same : serializeSlot left = serializeSlot right) : left = right := by
  unfold serializeSlot at same
  obtain ⟨rest, depthWords⟩ := List.append_inj' same rfl
  obtain ⟨rest, publicWords⟩ := List.append_inj' rest (by simp)
  obtain ⟨rest, pointWords⟩ := List.append_inj' rest (by simp)
  obtain ⟨commitmentWords, evaluationWords⟩ := List.append_inj' rest (by simp)
  have depth := natWord_injective leftDepth rightDepth (List.cons.inj depthWords).1
  have publicInput := serializePublicInput_injective publicWords
  have point := serializePoint_injective pointWords
  have commitment := serializeCommitment_injective commitmentWords
  have evaluations := serializeEvaluations_injective evaluationWords
  obtain ⟨⟨_, _, _, _⟩, _⟩ := left
  obtain ⟨⟨_, _, _, _⟩, _⟩ := right
  simp only at depth publicInput point commitment evaluations
  subst depth publicInput point commitment evaluations
  rfl

theorem statePreimage_injective {left right : State logicalWidth publicFits}
    (leftWell : left.WellFormed) (rightWell : right.WellFormed)
    (same : statePreimage left = statePreimage right) :
    left.verifierKeys = right.verifierKeys ∧ left.iteration = right.iteration ∧
      left.z0 = right.z0 ∧ left.current = right.current ∧ left.pc = right.pc ∧
      ∀ slot, slotDigest (left.slots slot) = slotDigest (right.slots slot) := by
  unfold statePreimage at same
  obtain ⟨rest, digest1⟩ := List.append_inj' same (by simp [slotDigest, Poseidon2.hash_length])
  obtain ⟨rest, digest0⟩ := List.append_inj' rest (by simp [slotDigest, Poseidon2.hash_length])
  obtain ⟨rest, pcWords⟩ := List.append_inj' rest rfl
  obtain ⟨rest, current⟩ := List.append_inj' rest (by rw [leftWell.current, rightWell.current])
  obtain ⟨rest, z0⟩ := List.append_inj' rest (by rw [leftWell.z0, rightWell.z0])
  obtain ⟨rest, iterationWords⟩ := List.append_inj' rest rfl
  obtain ⟨rest, key1⟩ := List.append_inj' rest
    (by rw [leftWell.verifierKeys 1, rightWell.verifierKeys 1])
  obtain ⟨_, key0⟩ := List.append_inj' rest
    (by rw [leftWell.verifierKeys 0, rightWell.verifierKeys 0])
  refine ⟨?_, natWord_injective leftWell.iteration rightWell.iteration
      (List.cons.inj iterationWords).1, z0, current,
    natWord_injective leftWell.pc rightWell.pc (List.cons.inj pcWords).1, ?_⟩
  · funext circuit
    match circuit with
    | ⟨0, _⟩ => exact key0
    | ⟨1, _⟩ => exact key1
  · intro slot
    match slot with
    | ⟨0, _⟩ => exact digest0
    | ⟨1, _⟩ => exact digest1

/-- A Poseidon2 collision on an input pair that two states fix: their state preimages, or the
slot preimages of one slot. -/
def StateCollision (left right : State logicalWidth publicFits) : Prop :=
  (statePreimage left ≠ statePreimage right ∧ stateHash left = stateHash right) ∨
    ∃ slot, serializeSlot (left.slots slot) ≠ serializeSlot (right.slots slot) ∧
      slotDigest (left.slots slot) = slotDigest (right.slots slot)

/-- Two different well-formed states with one state hash give a collision on their pair: the
state preimages differ, or a slot differs and its slot preimages differ. -/
theorem stateHash_collision {left right : State logicalWidth publicFits}
    (leftWell : left.WellFormed) (rightWell : right.WellFormed) (different : left ≠ right)
    (same : stateHash left = stateHash right) : StateCollision left right := by
  by_cases preimages : statePreimage left = statePreimage right
  · obtain ⟨keys, iteration, z0, current, pc, digests⟩ :=
      statePreimage_injective leftWell rightWell preimages
    have slotDiffers : ∃ slot, left.slots slot ≠ right.slots slot := by
      refine Classical.byContradiction fun none => different ?_
      have slots : left.slots = right.slots := funext fun slot =>
        Classical.byContradiction fun differs => none ⟨slot, differs⟩
      cases left
      cases right
      simp only at keys iteration z0 current pc slots
      subst keys iteration z0 current pc slots
      rfl
    obtain ⟨slot, differs⟩ := slotDiffers
    refine Or.inr ⟨slot, fun inputs => differs ?_, digests slot⟩
    exact serializeSlot_injective (leftWell.depth slot) (rightWell.depth slot) inputs
  · exact Or.inl ⟨preimages, same⟩

end NightstreamFPrime.Lifecycle.OptionB
