import NightstreamFPrime.Export.BoundedDepth.StepRelation
import NightstreamFPrime.Lifecycle.BoundedDepth.Terminal

/-! The bridge from the exported step relations to the bounded-depth terminal (CW-5.10). The setup's
key of each circuit carries that circuit's exported relation. If the terminal accepts a recursive
proof, the fresh instance holds under the relation of the circuit that ran last, so that circuit's
step relation holds in the environment that the fresh witness decodes to (`Exports.decode`), and
the step's output hash is the terminal's state hash. The state hash binds the decoded key digests
to the setup's keys, and the counters, states and slot claims to the terminal's, or Poseidon2 has
a collision on the decoded next state and the terminal's state (`stateHash_binds_words`,
`Collision`). So the last step holds under the setup's keys and ends in the terminal's statement
(`bridge`). The decode is a function of the proof, so the collision names two inputs that a
reduction computes from the proof.

The light circuit reads only the active slot's words (`StepRelation.lightInput`), so the decode
takes the other slot from the terminal's slots. Its step relation needs the other slot's prior
digest word to be that slot's digest. After the base case the light rows copy that word into the
next state's words, which the output hash hashes; so either the next state's words are the
terminal's state preimage and the premise holds, or the two lists collide (`WordCollision`).

The circuit selector of Construction 2 (`Paper.Machine.control`) reads only the application state
and witness. The bounded-depth scheme's scheduler picks the circuit from the slot depth, which the application
state does not hold: a light fold while the slot is below the cap, a refresh fold at the cap.
Both circuits run the same application step, so the selector does not change the computed state,
and the light depth guard bounds every schedule (`Folding.Schedule`). The bridge therefore takes
the selector that returns the circuit that ran on a valid step (`chosen`), so the step's witness
is valid (`valid_of_chosen`). The terminal does not read the selector
(`holdsFor_recursive_control`), so a terminal under any selector gives the step.

The state hash reads every counter as one word (`natWord`). The decoded next iteration is the
input iteration plus one, which can be `q`, whose word is 0; the terminal requires
`0 < iteration`, so the iterations agree (`iteration_of_words`). A terminal slot's depth is any
`Nat`, so the bridge gives the depth words, not the depths. -/

namespace NightstreamFPrime.Export.BoundedDepth.TerminalBridge

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.BoundedDepth
open NightstreamFPrime.Layout.BoundedDepth
open NightstreamFPrime.Export.Generic
open NightstreamFPrime.Export.BoundedDepth.StepRelation

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

attribute [local instance] Classical.propDecidable in
/-- The selector that returns `circuit`, the circuit that the prover runs, on a step that
`program` admits, and the other circuit otherwise. -/
noncomputable def chosen (program : Stage1.Application.Program) (circuit : Fin 2) :
    AppState → AppWitness → Fin 2 :=
  fun state witness => if program.valid state witness then circuit else circuit + 1

/-- A step on which `chosen` returns its circuit is valid. -/
theorem valid_of_chosen {program : Stage1.Application.Program} {circuit : Fin 2}
    {state : AppState} {witness : AppWitness} (picked : chosen program circuit state witness = circuit) :
    program.valid state witness := by
  refine Classical.byContradiction fun invalid => ?_
  simp only [chosen, invalid, ↓reduceIte] at picked
  revert picked
  fin_cases circuit <;> decide

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

/-- A shaped state's preimage has 38 words. -/
theorem statePreimage_length {state : State logicalWidth publicFits} (shaped : Shaped state) :
    (statePreimage state).length = 38 := by
  simp [statePreimage, slotDigest, Poseidon2.hash_length, shaped.verifierKeys, shaped.z0,
    shaped.current, Poseidon2.digestLen]
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
  · exact Or.inr (Or.inl ⟨preimages, by
      rw [statePreimage_length leftShaped, statePreimage_length rightShaped], same⟩)

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
application `program`, whose step is `step`. -/
structure Exports where
  program : Stage1.Application.Program
  programStep : program.step = step
  lightFresh : Nat
  refreshFresh : Nat
  lightWidth : Nat
  refreshWidth : Nat
  lightFits : CircuitRelation.Fits CircuitRelation.publicColumns (CircuitRelation.pieces
    (LightAssemblerCircuit.stepCircuit program base relation 0 ajtai)
    (LightAssembler.sharedEnd base program.witnessWordCount)
    (lightDigestColumn program)) lightFresh lightWidth
  refreshFits : CircuitRelation.Fits CircuitRelation.publicColumns (CircuitRelation.pieces
    (RefreshAssemblerCircuit.stepCircuit program base relation 1 ajtai)
    (RefreshAssembler.sharedEnd base program.witnessWordCount)
    (refreshDigestColumn program)) refreshFresh refreshWidth
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

/-- The step that a fresh witness of `circuit` decodes to. A light step takes its other slot from
`running`. -/
def Exports.decode (exports : Exports relation ajtai circuitKeys step cubeFits) (circuit : Fin 2)
    (witness : Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits))
    (running : Fin 2 → SlotData logicalWidth publicFits) :
    Decoded logicalWidth publicFits :=
  let assignment := Layout.ProductionRelation.Plan.logicalAssignment witness
  if circuit = 0 then
    let env := lightEnv exports.program relation ajtai exports.lightFresh exports.lightLayout assignment
    ⟨lightInput exports.program running env, lightOutput exports.program relation 0 running env,
      decodedKeys circuitKeys (lightInterface (logicalWidth := logicalWidth)
        (publicFits := publicFits) exports.program).verifierKeys env⟩
  else
    let env := refreshEnv exports.program relation ajtai exports.refreshFresh exports.refreshLayout assignment
    ⟨refreshInput exports.program env, refreshOutput exports.program relation 1 env,
      decodedKeys circuitKeys (refreshInterface (logicalWidth := logicalWidth)
        (publicFits := publicFits) exports.program).verifierKeys env⟩

/-- The step that a recursive terminal's fresh witness decodes to, with the terminal's slots. -/
def Exports.last (exports : Exports relation ajtai circuitKeys step cubeFits)
    (payload : TerminalProof (SlotData logicalWidth publicFits)
      (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits))
      (StepFresh logicalWidth publicFits)
      (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits)) 2) :
    Decoded logicalWidth publicFits :=
  exports.decode (circuitOf payload.pc) payload.freshWitness payload.running

/-- A decoded step's witness has the program's word count. -/
theorem Exports.last_witness_length (exports : Exports relation ajtai circuitKeys step cubeFits)
    (payload : TerminalProof (SlotData logicalWidth publicFits)
      (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits))
      (StepFresh logicalWidth publicFits)
      (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits)) 2) :
    (exports.last payload).input.witness.length = exports.program.witnessWordCount := by
  unfold Exports.last Exports.decode
  split <;> simp [lightInput, refreshInput]

/-- A Poseidon2 collision on two inputs of one length. -/
def WordCollision (left right : List F) : Prop :=
  left ≠ right ∧ left.length = right.length ∧ Poseidon2.hash left = Poseidon2.hash right

/-- The next state's words of the light step that a recursive terminal's fresh witness decodes
to. -/
def Exports.lightNextWords (exports : Exports relation ajtai circuitKeys step cubeFits)
    (payload : TerminalProof (SlotData logicalWidth publicFits)
      (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits))
      (StepFresh logicalWidth publicFits)
      (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits)) 2) : List F :=
  StepRelation.lightNextWords (logicalWidth := logicalWidth) (publicFits := publicFits)
    exports.program 0
    (lightEnv exports.program relation ajtai exports.lightFresh exports.lightLayout
      (Layout.ProductionRelation.Plan.logicalAssignment payload.freshWitness))

/-- The collision of the bridge: the next state of the decoded last step and the terminal's state
collide, or the last step is light and its next state's words and the terminal's state preimage
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
      (terminalState circuitKeys statement payload.running payload.pc) ∨
    (circuitOf payload.pc = 0 ∧ WordCollision (exports.lightNextWords payload)
      (statePreimage (terminalState circuitKeys statement payload.running payload.pc)))

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

/-- The bounded-depth scheme's export-to-terminal bridge. The setup's keys carry the two exported relations, and
a recursive terminal is linked. Then the circuit that ran last holds its step relation on the
decoded step, with the selector that returns that circuit on a valid step (`chosen`), under the
setup's keys and ends in the terminal's statement, or the decoded next state and the terminal's
state collide. -/
theorem bridge_linked (exports : Exports relation ajtai circuitKeys step cubeFits)
    (keysShaped : ∀ key, (circuitKeys key).digest.length = 4)
    (statement : TerminalStatement AppState)
    (payload : TerminalProof (SlotData logicalWidth publicFits)
      (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits))
      (StepFresh logicalWidth publicFits)
      (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits)) 2)
    (linked : Linked (ajtai := ajtai) (circuitKeys := circuitKeys) statement payload) :
    (StepHolds Folding.Schedule.productionLightFoldCap ajtai circuitKeys step
        (chosen exports.program (circuitOf payload.pc)) (circuitOf payload.pc) (exports.last payload).input
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
    obtain ⟨hashEq, nextOther⟩ := light_freshWords exports.program relation ajtai step
      (chosen exports.program 0) exports.programStep (fun _ _ valid => ite_eq_left valid)
      exports.lightFresh exports.lightWidth exports.lightFits exports.lightLayout cubeFits
      exports.lightInside payload.fresh payload.freshWitness _ freshHash freshHolds
    simp only [Collision, Exports.last, Exports.lightNextWords, circuitEq, Exports.decode,
      ↓reduceIte]
    by_cases same : StepRelation.lightNextWords (logicalWidth := logicalWidth)
        (publicFits := publicFits) exports.program 0 (lightEnv exports.program relation ajtai exports.lightFresh exports.lightLayout
          (Layout.ProductionRelation.Plan.logicalAssignment payload.freshWitness)) =
        statePreimage (terminalState circuitKeys statement payload.running payload.pc)
    · have others : OtherDigest exports.program payload.running
          (lightEnv exports.program relation ajtai exports.lightFresh exports.lightLayout
            (Layout.ProductionRelation.Plan.logicalAssignment payload.freshWitness)) := by
        intro nonzero slot other
        rw [← nextOther nonzero slot other]
        exact nextDigests_of_words exports.program same slot
      obtain ⟨holds, words⟩ := light_freshHolds exports.program relation ajtai circuitKeys
        step (chosen exports.program 0) exports.programStep (fun _ _ valid => ite_eq_left valid)
        exports.lightFresh exports.lightWidth
        exports.lightFits exports.lightLayout cubeFits exports.lightInside payload.fresh
        payload.freshWitness _ freshHash freshHolds payload.running others
      exact (bind_step _ ajtai circuitKeys keysShaped step (chosen exports.program 0) 0 _ _ _ _
        holds (Fin.isLt _)
        List.length_ofFn List.length_ofFn statement valid positive payload.running payload.pc
        pcValid (words.trans digestEq)).imp_right Or.inl
    · exact Or.inr (Or.inr ⟨trivial, same,
        (lightNextWords_length exports.program 0 _).trans
          (statePreimage_length (state := terminalState circuitKeys statement payload.running
            payload.pc) ⟨keysShaped, valid.2.1, valid.2.2⟩).symm,
        hashEq.trans digestEq⟩)
  · have circuitEq : circuitOf payload.pc = 1 := by simp [circuitOf, ranRefresh]
    rw [circuitEq, exports.refreshKey] at freshHolds
    obtain ⟨holds, words⟩ := refresh_freshHolds exports.program relation ajtai circuitKeys
      step (chosen exports.program 1) exports.programStep (fun _ _ valid => ite_eq_left valid)
      exports.refreshFresh
      exports.refreshWidth exports.refreshFits exports.refreshLayout cubeFits
      exports.refreshInside payload.fresh payload.freshWitness _ freshHash freshHolds
    simp only [Collision, Exports.last, circuitEq, Exports.decode, Fin.one_eq_zero_iff,
      OfNat.ofNat_ne_one, ↓reduceIte]
    exact (bind_step _ ajtai circuitKeys keysShaped step (chosen exports.program 1) 1 _ _ _ _
      holds (Fin.isLt _)
      List.length_ofFn List.length_ofFn statement valid positive payload.running payload.pc
      pcValid (words.trans digestEq)).imp_right Or.inl

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
        (chosen exports.program (circuitOf payload.pc)) (circuitOf payload.pc) (exports.last payload).input
        (exports.last payload).output ∧
      EndsIn statement payload.running payload.pc (circuitOf payload.pc)
        (exports.last payload).input (exports.last payload).output) ∨
      Collision exports statement payload :=
  bridge_linked exports keysShaped statement payload
    (linked_of_holdsFor control cap statement payload accepted)

end Circuits

end NightstreamFPrime.Export.BoundedDepth.TerminalBridge
