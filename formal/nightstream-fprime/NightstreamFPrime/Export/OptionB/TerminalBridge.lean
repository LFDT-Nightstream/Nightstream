import NightstreamFPrime.Export.OptionB.StepRelation
import NightstreamFPrime.Lifecycle.OptionB.Terminal

/-! The bridge from the exported step relations to the option-B terminal (CW-5.10). The setup's
key of each circuit carries that circuit's exported relation. If the terminal accepts a recursive
proof, the fresh instance holds under the relation of the circuit that ran last, so that circuit's
step relation holds in the environment that the fresh witness decodes to (`Exports.decode`), and
the step's output hash is the terminal's state hash. The state hash binds the decoded key digests
to the setup's keys, and the counters, states and slot claims to the terminal's, or Poseidon2 has
a collision on the decoded next state and the terminal's state (`stateHash_binds_words`,
`Collision`). So the last step holds under the setup's keys and ends in the terminal's statement
(`bridge`). The decode is a function of the proof, so the collision names two states that a
reduction computes from the proof.

The circuit selector of Construction 2 (`Paper.Machine.control`) reads only the application state
and witness. Option B's scheduler picks the circuit from the slot depth, which the application
state does not hold: a light fold while the slot is below the cap, a refresh fold at the cap.
Both circuits run the same application step, so the selector does not change the computed state,
and the light depth guard bounds every schedule (`Folding.Schedule`). The bridge therefore takes
the selector that returns the circuit that ran (`chosen`). The terminal does not read the
selector (`holdsFor_recursive_control`), so a terminal under any selector gives the step.

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

/-- The selector that returns `circuit`: the circuit that the prover runs. -/
def chosen (circuit : Fin 2) : AppState → AppWitness → Fin 2 := fun _ _ => circuit

/-- The terminal does not read the circuit selector. -/
theorem holdsFor_recursive_control (cap : Nat)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
    (step : AppState → AppWitness → AppState) (control control' : AppState → AppWitness → Fin 2)
    (statement : TerminalStatement AppState)
    (payload : TerminalProof (SlotData logicalWidth publicFits)
      (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits))
      (StepFresh logicalWidth publicFits)
      (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits)) 2) :
    Terminal.HoldsFor cap ajtai circuitKeys step control statement (.recursive payload) ↔
      Terminal.HoldsFor cap ajtai circuitKeys step control' statement (.recursive payload) :=
  (Terminal.holdsFor_recursive_iff cap ajtai circuitKeys step control statement payload).trans
    (Terminal.holdsFor_recursive_iff cap ajtai circuitKeys step control' statement payload).symm

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

/-- Two shaped states with one state hash have the same words, or they collide. -/
theorem stateHash_binds_words {left right : State logicalWidth publicFits}
    (leftShaped : Shaped left) (rightShaped : Shaped right)
    (same : stateHash left = stateHash right) :
    SameWords left right ∨ StateCollision left right := by
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
      exact Or.inr (Or.inr ⟨slot, differs, digests slot⟩)
  · exact Or.inr (Or.inl ⟨preimages, same⟩)

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
hash, holds under the setup's keys and ends in the terminal's statement, or the step's next state
and the terminal's state collide. -/
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
      EndsIn statement running pc circuit input output) ∨
      StateCollision (toState (nextHashPreimage
          ((setup cap ajtai (decodedKeys circuitKeys verifierKeys env)).forCircuit circuit) input
          output))
        (terminalState circuitKeys statement running pc) := by
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

/-- The circuit that a program counter names: the refresh circuit at 2, the light circuit
otherwise. -/
def circuitOf (pc : Nat) : Fin 2 := if pc = 2 then 1 else 0

/-- A decoded step: its input, its output and the key digests that its shared words carry. -/
structure Decoded (logicalWidth : Nat)
    (publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) where
  input : StepInput logicalWidth publicFits
  output : StepOutput logicalWidth publicFits
  keys : Fin 2 → CircuitKey logicalWidth publicFits

section Circuits

variable (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
  (step : AppState → AppWitness → AppState)
  (cubeFits : Phi81CarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables)

/-- The two exported step circuits whose relations the setup's keys carry. Both run the
application step `step`. -/
structure Exports where
  lightProgram : Stage1.Application.Program
  refreshProgram : Stage1.Application.Program
  lightStep : lightProgram.step = step
  refreshStep : refreshProgram.step = step
  lightFresh : Nat
  refreshFresh : Nat
  lightWidth : Nat
  refreshWidth : Nat
  lightFits : CircuitRelation.Fits CircuitRelation.publicColumns (CircuitRelation.pieces
    (LightAssemblerCircuit.stepCircuit lightProgram base relation 0 ajtai)
    (LightAssembler.sharedEnd base lightProgram.witnessWordCount)
    (lightDigestColumn lightProgram)) lightFresh lightWidth
  refreshFits : CircuitRelation.Fits CircuitRelation.publicColumns (CircuitRelation.pieces
    (RefreshAssemblerCircuit.stepCircuit refreshProgram base relation 1 ajtai)
    (RefreshAssembler.sharedEnd base refreshProgram.witnessWordCount)
    (refreshDigestColumn refreshProgram)) refreshFresh refreshWidth
  lightLayout : LogicalPlan.Layout lightWidth logicalWidth
  refreshLayout : LogicalPlan.Layout refreshWidth logicalWidth
  lightInside : ringDegree * PaperAlgebra.publicRingColumns ≤ lightWidth
  refreshInside : ringDegree * PaperAlgebra.publicRingColumns ≤ refreshWidth
  lightKey : (circuitKeys 0).relation =
    (CircuitRelation.plan _ _ lightFresh lightWidth lightFits lightLayout).logicalRelation
      (publicFits := publicFits) cubeFits
  refreshKey : (circuitKeys 1).relation =
    (CircuitRelation.plan _ _ refreshFresh refreshWidth refreshFits refreshLayout).logicalRelation
      (publicFits := publicFits) cubeFits

variable {relation ajtai circuitKeys step cubeFits}

/-- The step that a fresh witness of `circuit` decodes to. -/
def Exports.decode (exports : Exports relation ajtai circuitKeys step cubeFits) (circuit : Fin 2)
    (witness : Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    Decoded logicalWidth publicFits :=
  let assignment := Layout.ProductionRelation.Plan.logicalAssignment witness
  if circuit = 0 then
    let env := lightEnv exports.lightProgram relation ajtai exports.lightLayout assignment
    ⟨lightInput exports.lightProgram env, lightOutput exports.lightProgram 0 env,
      decodedKeys circuitKeys (lightInterface (logicalWidth := logicalWidth)
        (publicFits := publicFits) exports.lightProgram).verifierKeys env⟩
  else
    let env := refreshEnv exports.refreshProgram relation ajtai exports.refreshLayout assignment
    ⟨refreshInput exports.refreshProgram env, refreshOutput exports.refreshProgram 1 env,
      decodedKeys circuitKeys (refreshInterface (logicalWidth := logicalWidth)
        (publicFits := publicFits) exports.refreshProgram).verifierKeys env⟩

/-- The step that a recursive terminal's fresh witness decodes to. -/
def Exports.last (exports : Exports relation ajtai circuitKeys step cubeFits)
    (payload : TerminalProof (SlotData logicalWidth publicFits)
      (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits))
      (StepFresh logicalWidth publicFits)
      (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits)) 2) :
    Decoded logicalWidth publicFits :=
  exports.decode (circuitOf payload.pc) payload.freshWitness

/-- The collision of the bridge: the next state of the decoded last step and the terminal's state
collide. -/
def Collision (exports : Exports relation ajtai circuitKeys step cubeFits)
    (statement : TerminalStatement AppState)
    (payload : TerminalProof (SlotData logicalWidth publicFits)
      (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits))
      (StepFresh logicalWidth publicFits)
      (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits)) 2) : Prop :=
  StateCollision
    (toState (nextHashPreimage
      ((setup Folding.Schedule.productionLightFoldCap ajtai (exports.last payload).keys).forCircuit
        (circuitOf payload.pc)) (exports.last payload).input (exports.last payload).output))
    (terminalState circuitKeys statement payload.running payload.pc)

/-- What the bridge reads from a recursive terminal: a valid statement, a program counter that
names a circuit, a positive iteration, a fresh public input that encodes the terminal's state
hash, and a fresh instance that holds under the relation of the circuit that ran last. The slot
witnesses are not read. -/
structure Linked (statement : TerminalStatement AppState)
    (payload : TerminalProof (SlotData logicalWidth publicFits)
      (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits))
      (StepFresh logicalWidth publicFits)
      (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits)) 2) : Prop where
  valid : Terminal.StatementValid statement
  pcValid : InRange 2 payload.pc
  positive : 0 < statement.iteration
  freshHash : payload.fresh.publicInputs ⟨0, by decide⟩ = encHash (publicFits := publicFits)
    (stateHash (terminalState circuitKeys statement payload.running payload.pc))
  freshHolds : CCS.Holds (PaperAlgebra.semantics ajtai) productionGlobalParams
    (freshStatement (circuitKeys (circuitOf payload.pc)).relation payload.fresh)
    payload.freshWitness

/-- A recursive terminal that the terminal relation accepts is linked. -/
theorem linked_of_holdsFor (control : AppState → AppWitness → Fin 2) (cap : Nat)
    (statement : TerminalStatement AppState)
    (payload : TerminalProof (SlotData logicalWidth publicFits)
      (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits))
      (StepFresh logicalWidth publicFits)
      (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits)) 2)
    (accepted : Terminal.HoldsFor cap ajtai circuitKeys step control statement
      (.recursive payload)) :
    Linked (ajtai := ajtai) (circuitKeys := circuitKeys) statement payload := by
  obtain ⟨valid, pcValid, positive, freshHash, _, freshHolds⟩ :=
    (Terminal.holdsFor_recursive_iff cap ajtai circuitKeys step control statement payload).1
      accepted
  have selected : selectedIndex pcValid = circuitOf payload.pc := by
    obtain ⟨lower, upper⟩ := pcValid
    apply Fin.ext
    simp only [selectedIndex, circuitOf]
    split <;> simp <;> omega
  rw [selected] at freshHolds
  exact ⟨valid, pcValid, positive, freshHash, freshHolds⟩

/-- Option B's export-to-terminal bridge. The setup's keys carry the two exported relations, and
a recursive terminal is linked. Then the circuit that ran last holds its step relation on the
decoded step, with that circuit as the selector (`chosen`), under the setup's keys and ends in
the terminal's statement, or the decoded next state and the terminal's state collide. -/
theorem bridge_linked (exports : Exports relation ajtai circuitKeys step cubeFits)
    (keysShaped : ∀ key, (circuitKeys key).digest.length = 4)
    (statement : TerminalStatement AppState)
    (payload : TerminalProof (SlotData logicalWidth publicFits)
      (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits))
      (StepFresh logicalWidth publicFits)
      (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits)) 2)
    (linked : Linked (ajtai := ajtai) (circuitKeys := circuitKeys) statement payload) :
    (StepHolds Folding.Schedule.productionLightFoldCap ajtai circuitKeys step
        (chosen (circuitOf payload.pc)) (circuitOf payload.pc) (exports.last payload).input
        (exports.last payload).output ∧
      EndsIn statement payload.running payload.pc (circuitOf payload.pc)
        (exports.last payload).input (exports.last payload).output) ∨
      Collision exports statement payload := by
  obtain ⟨valid, pcValid, positive, freshHash, freshHolds⟩ := linked
  have cases : payload.pc = 1 ∨ payload.pc = 2 := by
    obtain ⟨lower, upper⟩ := pcValid
    omega
  have digestEq := ofFn_getD (Poseidon2.hash_length (statePreimage
    (terminalState circuitKeys statement payload.running payload.pc)))
  rcases cases with ranLight | ranRefresh
  · have circuitEq : circuitOf payload.pc = 0 := by simp [circuitOf, ranLight]
    rw [circuitEq, exports.lightKey] at freshHolds
    obtain ⟨holds, words⟩ := light_freshHolds exports.lightProgram relation ajtai circuitKeys
      step (chosen 0) exports.lightStep (fun _ _ _ => rfl) exports.lightFresh exports.lightWidth
      exports.lightFits exports.lightLayout cubeFits exports.lightInside payload.fresh
      payload.freshWitness _ freshHash freshHolds
    simp only [Collision, Exports.last, circuitEq, Exports.decode, ↓reduceIte]
    exact bind_step _ ajtai circuitKeys keysShaped step (chosen 0) 0 _ _ _ _ holds (Fin.isLt _)
      List.length_ofFn List.length_ofFn statement valid positive payload.running payload.pc
      pcValid (words.trans digestEq)
  · have circuitEq : circuitOf payload.pc = 1 := by simp [circuitOf, ranRefresh]
    rw [circuitEq, exports.refreshKey] at freshHolds
    obtain ⟨holds, words⟩ := refresh_freshHolds exports.refreshProgram relation ajtai circuitKeys
      step (chosen 1) exports.refreshStep (fun _ _ _ => rfl) exports.refreshFresh
      exports.refreshWidth exports.refreshFits exports.refreshLayout cubeFits
      exports.refreshInside payload.fresh payload.freshWitness _ freshHash freshHolds
    simp only [Collision, Exports.last, circuitEq, Exports.decode, Fin.one_eq_zero_iff,
      OfNat.ofNat_ne_one, ↓reduceIte]
    exact bind_step _ ajtai circuitKeys keysShaped step (chosen 1) 1 _ _ _ _ holds (Fin.isLt _)
      List.length_ofFn List.length_ofFn statement valid positive payload.running payload.pc
      pcValid (words.trans digestEq)

/-- `bridge_linked` for a recursive terminal that the terminal relation accepts under any
selector. -/
theorem bridge (exports : Exports relation ajtai circuitKeys step cubeFits)
    (keysShaped : ∀ key, (circuitKeys key).digest.length = 4)
    (control : AppState → AppWitness → Fin 2) (cap : Nat) (statement : TerminalStatement AppState)
    (payload : TerminalProof (SlotData logicalWidth publicFits)
      (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits))
      (StepFresh logicalWidth publicFits)
      (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits)) 2)
    (accepted : Terminal.HoldsFor cap ajtai circuitKeys step control statement
      (.recursive payload)) :
    (StepHolds Folding.Schedule.productionLightFoldCap ajtai circuitKeys step
        (chosen (circuitOf payload.pc)) (circuitOf payload.pc) (exports.last payload).input
        (exports.last payload).output ∧
      EndsIn statement payload.running payload.pc (circuitOf payload.pc)
        (exports.last payload).input (exports.last payload).output) ∨
      Collision exports statement payload :=
  bridge_linked exports keysShaped statement payload
    (linked_of_holdsFor control cap statement payload accepted)

end Circuits

end NightstreamFPrime.Export.OptionB.TerminalBridge
