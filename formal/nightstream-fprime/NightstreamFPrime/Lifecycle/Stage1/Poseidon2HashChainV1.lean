import NightstreamFPrime.Gadgets.Poseidon2.Support
import NightstreamFPrime.Lifecycle.Stage1.Application

/-!
Owns the first verifier-selected Stage 1 application. The application hashes
one exact 40-byte domain tag, the four-word prior state, and the four-word
message. This module does not select a package, key, or verifier policy.
-/

namespace NightstreamFPrime.Lifecycle.Stage1.Poseidon2HashChainV1

open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Spec

def messageWordCount : Nat := 4

abbrev MessageIndex := Fin messageWordCount

/-- ASCII bytes of `Nightstream/Stage1/Poseidon2HashChain/v1`, in order. -/
def domainTagBytes : List Nat :=
  [78, 105, 103, 104, 116, 115, 116, 114, 101, 97,
   109, 47, 83, 116, 97, 103, 101, 49, 47, 80,
   111, 115, 101, 105, 100, 111, 110, 50, 72, 97,
   115, 104, 67, 104, 97, 105, 110, 47, 118, 49]

/-- Each approved ASCII byte is mapped directly into Goldilocks. -/
def domainTag : List F := domainTagBytes.map Poseidon2.ofNat

@[simp] theorem domainTagBytes_length : domainTagBytes.length = 40 := by
  rfl

@[simp] theorem domainTag_length : domainTag.length = 40 := by
  simp [domainTag]

def preimage (priorState message : List F) : List F :=
  domainTag ++ priorState ++ message

theorem preimage_length (priorState message : List F)
    (priorLength : priorState.length = Application.stateWordCount)
    (messageLength : message.length = messageWordCount) :
    (preimage priorState message).length = 48 := by
  simp [preimage, priorLength, messageLength, Application.stateWordCount,
    messageWordCount]

/-- Exact application semantics approved by the owner. -/
def step (priorState message : List F) : List F :=
  Poseidon2.hash (preimage priorState message)

private theorem externalLayer_length (state : Poseidon2.State) :
    (Poseidon2.externalLayer state).length = Poseidon2.width := by
  simp [Poseidon2.externalLayer]

private theorem internalLayer_length (state : Poseidon2.State) :
    (Poseidon2.internalLayer state).length = Poseidon2.width := by
  simp [Poseidon2.internalLayer]

private theorem fullRound_length (rows : List (List Nat)) (round : Nat)
    (state : Poseidon2.State) :
    (Poseidon2.fullRound rows round state).length = Poseidon2.width := by
  exact externalLayer_length _

private theorem partialRound_length (round : Nat) (state : Poseidon2.State) :
    (Poseidon2.partialRound round state).length = Poseidon2.width := by
  exact internalLayer_length _

private theorem rounds_length (roundStep : Nat → Poseidon2.State →
    Poseidon2.State)
    (stepLength : ∀ round state,
      (roundStep round state).length = Poseidon2.width)
    (rounds : List Nat) (state : Poseidon2.State)
    (stateLength : state.length = Poseidon2.width) :
    (rounds.foldl (fun current round => roundStep round current) state).length =
      Poseidon2.width := by
  induction rounds generalizing state with
  | nil => exact stateLength
  | cons round rest inductionHypothesis =>
      exact inductionHypothesis _ (stepLength round state)

private theorem permute_length (state : Poseidon2.State) :
    (Poseidon2.permute state).length = Poseidon2.width := by
  unfold Poseidon2.permute Poseidon2.rounds
  apply rounds_length
  · exact fullRound_length Poseidon2.terminalConstants
  · apply rounds_length
    · exact partialRound_length
    · apply rounds_length
      · exact fullRound_length Poseidon2.initialConstants
      · exact externalLayer_length state

theorem step_output_length (priorState message : List F) :
    (step priorState message).length = Application.stateWordCount := by
  unfold step Poseidon2.hash
  dsimp only
  rw [List.length_take, permute_length]
  norm_num [Poseidon2.digestLen, Poseidon2.width,
    Application.stateWordCount]

def inputExpressions (interface : Application.Interface messageWordCount)
    (offset : Nat) : List Expr :=
  domainTag.map Expr.const ++
    List.ofFn (interface.input offset) ++
    List.ofFn (interface.witness offset)

def hashInterface (interface : Application.Interface messageWordCount) :
    Formal.Interface where
  input := inputExpressions interface
  expected := interface.output

def circuit (interface : Application.Interface messageWordCount) :
    FormalCircuit :=
  Formal.circuit (hashInterface interface)

@[simp] theorem inputExpressions_length
    (interface : Application.Interface messageWordCount) (offset : Nat) :
    (inputExpressions interface offset).length = 48 := by
  simp [inputExpressions, Application.stateWordCount, messageWordCount]

theorem inputChunks_length
    (interface : Application.Interface messageWordCount) (offset : Nat) :
    (Hash.inputChunks (inputExpressions interface offset)).length = 12 := by
  unfold Hash.inputChunks
  rw [List.length_map, List.length_range, inputExpressions_length]
  norm_num [Poseidon2.rate]

theorem compile_recipes_length
    (interface : Application.Interface messageWordCount) (offset : Nat) :
    (Hash.compile offset (inputExpressions interface offset)).recipes.length =
      7696 := by
  rw [Hash.compile_recipes_length, inputChunks_length]

theorem circuit_localLength
    (interface : Application.Interface messageWordCount) (offset : Nat) :
    localLength (Circuit.ops (circuit interface).main offset) = 7696 := by
  change localLength (Circuit.ops (Formal.main (hashInterface interface)) offset) =
    7696
  rw [Formal.main_ops, Formal.opsAt_localLength]
  exact compile_recipes_length interface offset

theorem circuit_rowCount
    (interface : Application.Interface messageWordCount) (offset : Nat) :
    (flatConstraints (Circuit.ops (circuit interface).main offset)).length =
      7700 := by
  change (flatConstraints (Circuit.ops
    (Formal.main (hashInterface interface)) offset)).length = 7700
  rw [Formal.flatConstraints_length_eq]
  change (Hash.compile offset
    (inputExpressions interface offset)).recipes.length + 4 = 7700
  rw [compile_recipes_length]

private theorem eval_inputExpressions
    (interface : Application.Interface messageWordCount)
    (offset : Nat) (env : Env) :
    Hash.evalList env (inputExpressions interface offset) =
      preimage (Application.inputState interface offset env)
        (Application.witnessValue interface offset env) := by
  simp [Hash.evalList, inputExpressions, preimage, domainTag,
    Application.inputState, Application.witnessValue, List.map_map,
    Function.comp_def]

theorem spec_iff (interface : Application.Interface messageWordCount)
    (offset : Nat) (env : Env) :
    (circuit interface).spec offset env ↔
      Application.Holds step interface offset env := by
  change Formal.SpecHolds (hashInterface interface) offset env ↔
    Application.Holds step interface offset env
  unfold Formal.SpecHolds Application.Holds step
  simp only [hashInterface, Application.outputState]
  rw [eval_inputExpressions]
  rfl

private theorem inputExpressions_varsBelow
    (interface : Application.Interface messageWordCount) (offset : Nat)
    (inputs : Application.InputsBelow interface offset) :
    ∀ expression ∈ inputExpressions interface offset,
      expression.VarsBelow offset := by
  intro expression member
  simp only [inputExpressions, List.mem_append] at member
  rcases member with (tagMember | inputMember) | witnessMember
  · rcases List.mem_map.mp tagMember with ⟨value, _, rfl⟩
    trivial
  · rw [List.mem_ofFn'] at inputMember
    rcases inputMember with ⟨index, rfl⟩
    exact inputs.input index
  · rw [List.mem_ofFn'] at witnessMember
    rcases witnessMember with ⟨index, rfl⟩
    exact inputs.witness index

theorem assumptions_of_inputsBelow
    (interface : Application.Interface messageWordCount)
    (offset : Nat) (env : Env)
    (inputs : Application.InputsBelow interface offset) :
    (circuit interface).assumptions offset env := by
  change Formal.Assumptions (hashInterface interface) offset env
  exact ⟨inputExpressions_varsBelow interface offset inputs,
    inputs.output⟩

private theorem inputExpressions_varsSatisfy
    (interface : Application.Interface messageWordCount) (offset : Nat)
    (allowed : Nat → Prop)
    (inputs : Application.InputsSupported interface offset allowed) :
    ∀ expression ∈ inputExpressions interface offset,
      expression.VarsSatisfy allowed := by
  intro expression member
  simp only [inputExpressions, List.mem_append] at member
  rcases member with (tagMember | inputMember) | witnessMember
  · rcases List.mem_map.mp tagMember with ⟨value, _, rfl⟩
    trivial
  · rw [List.mem_ofFn'] at inputMember
    rcases inputMember with ⟨index, rfl⟩
    exact inputs.input index
  · rw [List.mem_ofFn'] at witnessMember
    rcases witnessMember with ⟨index, rfl⟩
    exact inputs.witness index

theorem constraintsSupported
    (interface : Application.Interface messageWordCount)
    (offset : Nat) (env : Env) (allowed : Nat → Prop)
    (_assumptions : (circuit interface).assumptions offset env)
    (inputs : Application.InputsSupported interface offset allowed)
    (localSupport : ∀ index,
      offset ≤ index →
      index < offset + localLength
        (Circuit.ops (circuit interface).main offset) →
      allowed index) :
    ∀ expression ∈
        flatConstraints (Circuit.ops (circuit interface).main offset),
      expression.VarsSatisfy allowed := by
  have supported := Support.formalFlatConstraints_supported
    (hashInterface interface) offset allowed
    (inputExpressions_varsSatisfy interface offset allowed inputs)
    inputs.output (by
      intro index lower upper
      apply localSupport index lower
      simpa [circuit] using upper)
  simpa [circuit] using supported

/-- The closed verifier-owned application program. -/
def program : Application.Program where
  witnessWordCount := messageWordCount
  step := step
  circuit := circuit
  spec_iff := spec_iff
  assumptions_of_inputsBelow := assumptions_of_inputsBelow
  constraintsSupported := constraintsSupported

@[simp] theorem program_witnessWordCount : program.witnessWordCount = 4 := by
  rfl

@[simp] theorem program_step : program.step = step := by
  rfl

theorem program_localLength
    (interface : Application.Interface program.witnessWordCount)
    (offset : Nat) :
    localLength (Circuit.ops (program.circuit interface).main offset) =
      7696 := by
  exact circuit_localLength interface offset

theorem program_rowCount
    (interface : Application.Interface program.witnessWordCount)
    (offset : Nat) :
    (flatConstraints (Circuit.ops (program.circuit interface).main offset)).length =
      7700 := by
  exact circuit_rowCount interface offset

end NightstreamFPrime.Lifecycle.Stage1.Poseidon2HashChainV1
