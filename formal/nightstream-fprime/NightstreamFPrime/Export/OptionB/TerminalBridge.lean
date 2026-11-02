import NightstreamFPrime.Export.OptionB.StepRelation
import NightstreamFPrime.Lifecycle.OptionB.Terminal

/-! The bridge from the exported step relations to the option-B terminal (CW-5.10). The setup's
key of each circuit carries that circuit's exported relation. If the terminal accepts a recursive
proof, the fresh instance holds under the relation of the circuit that ran last, so that circuit's
step relation holds in some environment (`light_freshHolds`, `refresh_freshHolds`), and the step's
output hash is the terminal's state hash. The state hash binds the decoded key digests to the
setup's keys, and the counters, states and slot claims to the terminal's, or Poseidon2 has a
collision (`stateHash_binds_words`). So the last step holds under the setup's keys and ends in the
terminal's statement (`bridge`).

The state hash reads every counter as one word (`natWord`). The decoded next iteration is the
input iteration plus one, which can be `q`, whose word is 0; the terminal requires
`0 < iteration`, so the iterations agree (`iteration_of_words`). A terminal slot's depth is any
`Nat`, so the bridge gives the depth words, not the depths. -/

namespace NightstreamFPrime.Export.OptionB.TerminalBridge

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.OptionB
open NightstreamFPrime.Layout.OptionB
open NightstreamFPrime.Export.Generic
open NightstreamFPrime.Export.OptionB.StepRelation

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- Two different Poseidon2 inputs with one hash. -/
def Collision : Prop :=
  ∃ input input', input ≠ input' ∧ Poseidon2.hash input = Poseidon2.hash input'

/-- The key digests and both application states have four words; the counters are any `Nat`. -/
structure Shaped (state : State logicalWidth publicFits) : Prop where
  verifierKeys : ∀ circuit, (state.verifierKeys circuit).length = 4
  z0 : state.z0.length = 4
  current : state.current.length = 4

/-- What one state hash fixes when every counter is read as one word. -/
structure SameWords (left right : State logicalWidth publicFits) : Prop where
  verifierKeys : left.verifierKeys = right.verifierKeys
  iteration : natWord left.iteration = natWord right.iteration
  z0 : left.z0 = right.z0
  current : left.current = right.current
  pc : natWord left.pc = natWord right.pc
  claims : ∀ slot, (left.slots slot).1 = (right.slots slot).1
  depths : ∀ slot, natWord (left.slots slot).2 = natWord (right.slots slot).2

theorem serializeSlot_words {left right : SlotData logicalWidth publicFits}
    (same : serializeSlot left = serializeSlot right) :
    left.1 = right.1 ∧ natWord left.2 = natWord right.2 := by
  unfold serializeSlot at same
  obtain ⟨rest, depthWords⟩ := List.append_inj' same rfl
  obtain ⟨rest, publicWords⟩ := List.append_inj' rest (by simp)
  obtain ⟨rest, pointWords⟩ := List.append_inj' rest (by simp)
  obtain ⟨commitmentWords, evaluationWords⟩ := List.append_inj' rest (by simp)
  have publicInput := serializePublicInput_injective publicWords
  have point := serializePoint_injective pointWords
  have commitment := serializeCommitment_injective commitmentWords
  have evaluations := serializeEvaluations_injective evaluationWords
  refine ⟨?_, (List.cons.inj depthWords).1⟩
  obtain ⟨⟨_, _, _, _⟩, _⟩ := left
  obtain ⟨⟨_, _, _, _⟩, _⟩ := right
  simp only at publicInput point commitment evaluations
  subst publicInput point commitment evaluations
  rfl

/-- Two shaped states with one state hash have the same words, or Poseidon2 has a collision. -/
theorem stateHash_binds_words {left right : State logicalWidth publicFits}
    (leftShaped : Shaped left) (rightShaped : Shaped right)
    (same : stateHash left = stateHash right) : SameWords left right ∨ Collision := by
  by_cases preimages : statePreimage left = statePreimage right
  · unfold statePreimage at preimages
    obtain ⟨rest, digest1⟩ :=
      List.append_inj' preimages (by simp [slotDigest, Poseidon2.hash_length])
    obtain ⟨rest, digest0⟩ := List.append_inj' rest (by simp [slotDigest, Poseidon2.hash_length])
    obtain ⟨rest, pcWords⟩ := List.append_inj' rest rfl
    obtain ⟨rest, current⟩ :=
      List.append_inj' rest (by rw [leftShaped.current, rightShaped.current])
    obtain ⟨rest, z0⟩ := List.append_inj' rest (by rw [leftShaped.z0, rightShaped.z0])
    obtain ⟨rest, iterationWords⟩ := List.append_inj' rest rfl
    obtain ⟨rest, key1⟩ := List.append_inj' rest
      (by rw [leftShaped.verifierKeys 1, rightShaped.verifierKeys 1])
    obtain ⟨_, key0⟩ := List.append_inj' rest
      (by rw [leftShaped.verifierKeys 0, rightShaped.verifierKeys 0])
    have digests : ∀ slot, slotDigest (left.slots slot) = slotDigest (right.slots slot) := by
      intro slot
      match slot with
      | ⟨0, _⟩ => exact digest0
      | ⟨1, _⟩ => exact digest1
    by_cases slots : ∀ slot, serializeSlot (left.slots slot) = serializeSlot (right.slots slot)
    · refine Or.inl ⟨?_, (List.cons.inj iterationWords).1, z0, current,
        (List.cons.inj pcWords).1, fun slot => (serializeSlot_words (slots slot)).1,
        fun slot => (serializeSlot_words (slots slot)).2⟩
      funext circuit
      match circuit with
      | ⟨0, _⟩ => exact key0
      | ⟨1, _⟩ => exact key1
    · obtain ⟨slot, differs⟩ := Classical.not_forall.mp slots
      exact Or.inr ⟨slotDomainChunk ++ serializeSlot (left.slots slot),
        slotDomainChunk ++ serializeSlot (right.slots slot),
        fun inputs => differs (List.append_cancel_left inputs), digests slot⟩
  · exact Or.inr ⟨_, _, preimages, same⟩

/-- The next iteration's word is the terminal's, and the terminal's iteration is positive and
below `q`, so the next iteration does not wrap. -/
theorem iteration_of_words {inputIteration iteration : Nat}
    (inputBound : inputIteration < goldilocksModulus) (bound : iteration < goldilocksModulus)
    (positive : 0 < iteration) (words : natWord (inputIteration + 1) = natWord iteration) :
    inputIteration + 1 = iteration := by
  have values := congrArg Fin.val words
  simp only [natWord, Poseidon2.ofNat, Nat.mod_eq_of_lt bound] at values
  by_cases wraps : inputIteration + 1 = goldilocksModulus
  · rw [wraps, Nat.mod_self] at values
    omega
  · rwa [Nat.mod_eq_of_lt (by omega)] at values

theorem ofFn_getD {digest : Digest} (length : digest.length = 4) :
    (List.ofFn fun word : Fin 4 => digest.getD word.val 0) = digest := by
  match digest, length with
  | [_, _, _, _], _ => rfl

/-- The last step ends in the terminal's statement: the next iteration, both states, the program
counter, the slot claims and the slot depth words. -/
structure EndsIn (statement : TerminalStatement AppState)
    (running : Fin 2 → SlotData logicalWidth publicFits) (pc : Nat) (circuit : Fin 2)
    (input : StepInput logicalWidth publicFits) (output : StepOutput logicalWidth publicFits) :
    Prop where
  iteration : input.iteration + 1 = statement.iteration
  z0 : input.z0 = statement.z0
  zNext : output.zNext = statement.zi
  pc : oneBased circuit = pc
  claims : ∀ slot, (output.runningNext slot).1 = (running slot).1
  depths : ∀ slot, natWord (output.runningNext slot).2 = natWord (running slot).2

/-- The state a recursive terminal hashes. -/
def terminalState (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
    (statement : TerminalStatement AppState) (running : Fin 2 → SlotData logicalWidth publicFits)
    (pc : Nat) : State logicalWidth publicFits :=
  toState
    { verifierKeys := circuitKeys
      iteration := statement.iteration
      z0 := statement.z0
      current := statement.zi
      running := running
      pc := pc }

theorem decodedKeys_eq {circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits}
    {verifierKeys : Fin 2 → Fin 4 → Expr} {env : Env}
    (digests : ∀ key, (decodedKeys circuitKeys verifierKeys env key).digest =
      (circuitKeys key).digest) :
    decodedKeys circuitKeys verifierKeys env = circuitKeys := by
  funext key
  calc decodedKeys circuitKeys verifierKeys env key
      = ⟨(decodedKeys circuitKeys verifierKeys env key).digest, (circuitKeys key).relation⟩ :=
        rfl
    _ = ⟨(circuitKeys key).digest, (circuitKeys key).relation⟩ := by rw [digests key]
    _ = circuitKeys key := rfl

/-- A step that holds under decoded keys, whose output hash is a recursive terminal's state
hash, holds under the setup's keys and ends in the terminal's statement, or Poseidon2 has a
collision. -/
theorem bind_step (cap : Nat)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
    (keysShaped : ∀ key, (circuitKeys key).digest.length = 4)
    (step : AppState → AppWitness → AppState) (control : AppState → AppWitness → Fin 2)
    (circuit : Fin 2) (verifierKeys : Fin 2 → Fin 4 → Expr) (env : Env)
    (input : StepInput logicalWidth publicFits) (output : StepOutput logicalWidth publicFits)
    (holds : StepHolds cap ajtai (decodedKeys circuitKeys verifierKeys env) step control circuit
      input output)
    (iterationBound : input.iteration < goldilocksModulus) (z0Length : input.z0.length = 4)
    (zNextLength : output.zNext.length = 4)
    (statement : TerminalStatement AppState) (valid : Terminal.StatementValid statement)
    (positive : 0 < statement.iteration) (running : Fin 2 → SlotData logicalWidth publicFits)
    (pc : Nat) (pcValid : InRange 2 pc)
    (outputHash : output.x = stateHash (terminalState circuitKeys statement running pc)) :
    (StepHolds cap ajtai circuitKeys step control circuit input output ∧
      EndsIn statement running pc circuit input output) ∨ Collision := by
  obtain ⟨_, pcNext, _, nextHash, _⟩ :=
    (stepHolds_iff cap ajtai _ step control circuit input output).1 holds
  rw [outputHash] at nextHash
  have nextShaped : Shaped (toState (nextHashPreimage
      ((setup cap ajtai (decodedKeys circuitKeys verifierKeys env)).forCircuit circuit) input
      output)) :=
    ⟨fun _ => List.length_ofFn, z0Length, zNextLength⟩
  have terminalShaped : Shaped (terminalState circuitKeys statement running pc) :=
    ⟨keysShaped, valid.2.1, valid.2.2⟩
  rcases stateHash_binds_words nextShaped terminalShaped nextHash.symm with words | collision
  · have keys := decodedKeys_eq fun key => congrFun words.verifierKeys key
    rw [keys] at holds
    have small : 2 < goldilocksModulus := by decide
    refine Or.inl ⟨holds, iteration_of_words iterationBound valid.1 positive words.iteration,
      words.z0, words.current, ?_, words.claims, words.depths⟩
    have same := natWord_injective
      (Nat.lt_of_le_of_lt (show oneBased output.pcNext ≤ 2 by unfold oneBased; omega) small)
      (Nat.lt_of_le_of_lt pcValid.2 small) words.pc
    rw [← pcNext]
    exact same
  · exact Or.inr collision

section Circuits

variable (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
  (step : AppState → AppWitness → AppState) (control : AppState → AppWitness → Fin 2)
  (cubeFits : Phi81CarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables)

/-- A recursive terminal whose last circuit is the light circuit, with the light circuit's
exported relation in the setup, gives the light step under the setup's keys, ending in the
terminal's statement, or a Poseidon2 collision. -/
theorem light_bridge (keysShaped : ∀ key, (circuitKeys key).digest.length = 4) (width : Nat)
    (publicInside : ringDegree * PaperAlgebra.publicRingColumns ≤ width)
    (program : Stage1.Application.Program) (stepEq : program.step = step)
    (validControl : ∀ state witness, program.valid state witness → control state witness = 0)
    (firstFresh : Nat)
    (fits : CircuitRelation.Fits CircuitRelation.publicColumns (CircuitRelation.pieces
      (LightAssemblerCircuit.stepCircuit program base relation 0 ajtai)
      (LightAssembler.sharedEnd base program.witnessWordCount)
      (lightDigestColumn program)) firstFresh width)
    (layout : LogicalPlan.Layout width logicalWidth)
    (lightKey : (circuitKeys 0).relation =
      (CircuitRelation.plan _ _ firstFresh width fits layout).logicalRelation
        (publicFits := publicFits) cubeFits)
    (cap : Nat) (statement : TerminalStatement AppState)
    (payload : TerminalProof (SlotData logicalWidth publicFits)
      (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits))
      (StepFresh logicalWidth publicFits)
      (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits)) 2)
    (accepted : Terminal.HoldsFor cap ajtai circuitKeys step control statement
      (.recursive payload))
    (ranLight : payload.pc = 1) :
    (∃ input output, StepHolds Folding.Schedule.productionLightFoldCap ajtai circuitKeys step
        control 0 input output ∧ EndsIn statement payload.running payload.pc 0 input output) ∨
      Collision := by
  obtain ⟨valid, pcValid, positive, freshHash, _, freshHolds⟩ :=
    (Terminal.holdsFor_recursive_iff cap ajtai circuitKeys step control statement payload).1
      accepted
  have selected : selectedIndex pcValid = 0 := Fin.ext (by simp [selectedIndex, ranLight])
  rw [selected, lightKey] at freshHolds
  obtain ⟨env, holds, words⟩ := light_freshHolds program relation ajtai circuitKeys step control
    stepEq validControl firstFresh width fits layout cubeFits publicInside payload.fresh
    payload.freshWitness _ freshHash freshHolds
  rcases bind_step _ ajtai circuitKeys keysShaped step control 0 _ env _ _ holds (Fin.isLt _)
      List.length_ofFn List.length_ofFn statement valid positive payload.running payload.pc
      pcValid (words.trans (ofFn_getD (Poseidon2.hash_length _))) with
    ⟨holds, ends⟩ | collision
  · exact Or.inl ⟨_, _, holds, ends⟩
  · exact Or.inr collision

/-- The refresh case of `light_bridge`. -/
theorem refresh_bridge (keysShaped : ∀ key, (circuitKeys key).digest.length = 4) (width : Nat)
    (publicInside : ringDegree * PaperAlgebra.publicRingColumns ≤ width)
    (program : Stage1.Application.Program) (stepEq : program.step = step)
    (validControl : ∀ state witness, program.valid state witness → control state witness = 1)
    (firstFresh : Nat)
    (fits : CircuitRelation.Fits CircuitRelation.publicColumns (CircuitRelation.pieces
      (RefreshAssemblerCircuit.stepCircuit program base relation 1 ajtai)
      (RefreshAssembler.sharedEnd base program.witnessWordCount)
      (refreshDigestColumn program)) firstFresh width)
    (layout : LogicalPlan.Layout width logicalWidth)
    (refreshKey : (circuitKeys 1).relation =
      (CircuitRelation.plan _ _ firstFresh width fits layout).logicalRelation
        (publicFits := publicFits) cubeFits)
    (cap : Nat) (statement : TerminalStatement AppState)
    (payload : TerminalProof (SlotData logicalWidth publicFits)
      (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits))
      (StepFresh logicalWidth publicFits)
      (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits)) 2)
    (accepted : Terminal.HoldsFor cap ajtai circuitKeys step control statement
      (.recursive payload))
    (ranRefresh : payload.pc = 2) :
    (∃ input output, StepHolds Folding.Schedule.productionLightFoldCap ajtai circuitKeys step
        control 1 input output ∧ EndsIn statement payload.running payload.pc 1 input output) ∨
      Collision := by
  obtain ⟨valid, pcValid, positive, freshHash, _, freshHolds⟩ :=
    (Terminal.holdsFor_recursive_iff cap ajtai circuitKeys step control statement payload).1
      accepted
  have selected : selectedIndex pcValid = 1 := Fin.ext (by simp [selectedIndex, ranRefresh])
  rw [selected, refreshKey] at freshHolds
  obtain ⟨env, holds, words⟩ := refresh_freshHolds program relation ajtai circuitKeys step
    control stepEq validControl firstFresh width fits layout cubeFits publicInside payload.fresh
    payload.freshWitness _ freshHash freshHolds
  rcases bind_step _ ajtai circuitKeys keysShaped step control 1 _ env _ _ holds (Fin.isLt _)
      List.length_ofFn List.length_ofFn statement valid positive payload.running payload.pc
      pcValid (words.trans (ofFn_getD (Poseidon2.hash_length _))) with
    ⟨holds, ends⟩ | collision
  · exact Or.inl ⟨_, _, holds, ends⟩
  · exact Or.inr collision

/-- Option B's export-to-terminal bridge. The setup's keys carry the two exported relations, and
the terminal accepts a recursive proof. Then the circuit that ran last holds its step relation
under the setup's keys and ends in the terminal's statement, or Poseidon2 has a collision. -/
theorem bridge (keysShaped : ∀ key, (circuitKeys key).digest.length = 4)
    (lightProgram refreshProgram : Stage1.Application.Program)
    (lightStep : lightProgram.step = step) (refreshStep : refreshProgram.step = step)
    (lightControl : ∀ state witness, lightProgram.valid state witness →
      control state witness = 0)
    (refreshControl : ∀ state witness, refreshProgram.valid state witness →
      control state witness = 1)
    (lightFresh refreshFresh lightWidth refreshWidth : Nat)
    (lightFits : CircuitRelation.Fits CircuitRelation.publicColumns (CircuitRelation.pieces
      (LightAssemblerCircuit.stepCircuit lightProgram base relation 0 ajtai)
      (LightAssembler.sharedEnd base lightProgram.witnessWordCount)
      (lightDigestColumn lightProgram)) lightFresh lightWidth)
    (refreshFits : CircuitRelation.Fits CircuitRelation.publicColumns (CircuitRelation.pieces
      (RefreshAssemblerCircuit.stepCircuit refreshProgram base relation 1 ajtai)
      (RefreshAssembler.sharedEnd base refreshProgram.witnessWordCount)
      (refreshDigestColumn refreshProgram)) refreshFresh refreshWidth)
    (lightLayout : LogicalPlan.Layout lightWidth logicalWidth)
    (refreshLayout : LogicalPlan.Layout refreshWidth logicalWidth)
    (lightInside : ringDegree * PaperAlgebra.publicRingColumns ≤ lightWidth)
    (refreshInside : ringDegree * PaperAlgebra.publicRingColumns ≤ refreshWidth)
    (lightKey : (circuitKeys 0).relation =
      (CircuitRelation.plan _ _ lightFresh lightWidth lightFits lightLayout).logicalRelation
        (publicFits := publicFits) cubeFits)
    (refreshKey : (circuitKeys 1).relation =
      (CircuitRelation.plan _ _ refreshFresh refreshWidth refreshFits refreshLayout).logicalRelation
        (publicFits := publicFits) cubeFits)
    (cap : Nat) (statement : TerminalStatement AppState)
    (payload : TerminalProof (SlotData logicalWidth publicFits)
      (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits))
      (StepFresh logicalWidth publicFits)
      (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits)) 2)
    (accepted : Terminal.HoldsFor cap ajtai circuitKeys step control statement
      (.recursive payload)) :
    (∃ circuit input output, StepHolds Folding.Schedule.productionLightFoldCap ajtai circuitKeys
        step control circuit input output ∧
        EndsIn statement payload.running payload.pc circuit input output) ∨ Collision := by
  obtain ⟨_, pcValid, _⟩ :=
    (Terminal.holdsFor_recursive_iff cap ajtai circuitKeys step control statement payload).1
      accepted
  have cases : payload.pc = 1 ∨ payload.pc = 2 := by
    obtain ⟨lower, upper⟩ := pcValid
    omega
  rcases cases with ranLight | ranRefresh
  · rcases light_bridge relation ajtai circuitKeys step control cubeFits keysShaped lightWidth
        lightInside lightProgram lightStep lightControl lightFresh lightFits lightLayout lightKey cap
        statement payload accepted ranLight with ⟨input, output, last⟩ | collision
    · exact Or.inl ⟨0, input, output, last⟩
    · exact Or.inr collision
  · rcases refresh_bridge relation ajtai circuitKeys step control cubeFits keysShaped refreshWidth
        refreshInside refreshProgram refreshStep refreshControl refreshFresh refreshFits
        refreshLayout refreshKey cap statement payload accepted ranRefresh with
      ⟨input, output, last⟩ | collision
    · exact Or.inl ⟨1, input, output, last⟩
    · exact Or.inr collision

end Circuits

end NightstreamFPrime.Export.OptionB.TerminalBridge
