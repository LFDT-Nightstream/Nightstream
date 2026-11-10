import NightstreamFPrime.Export.Generic.CircuitRelation
import NightstreamFPrime.Layout.OptionB.StepCircuits
import NightstreamFPrime.Layout.ProductionRelation.AcceptedOpening

/-! The option-B step relations (CW-5.3). Each step circuit is exported with its shared words at
`base = 270`, after the public input `x` (`LogicalPlan`). From a satisfying assignment of a
circuit's logical relation, the decoders read the step's input, output, key digests and fold proof
from the shared words, with the next slots that the digest select fixes; `lightRepresents` and
`refreshRepresents` hold by construction. The light circuit reads only the active slot's words, so
its decoder takes the other slot from outside (`lightInput`): the terminal bridge gives the
terminal's slot. The light step relation then needs the other slot's prior digest word to be that
slot's digest (`OtherDigest`); `light_freshWords` gives, from the rows alone, the words that the
bridge reads to show it. The finish circuit's decoder (`finishStepOutput`) reads the next slots'
words instead. With
`CircuitRelation.rowsZero_implies_spec` this gives each circuit's case of the option-B step
relation (`light_stepHolds`, `refresh_stepHolds`) and the output digest that `x` encodes, in the
environment that the assignment decodes to (`lightEnv`, `refreshEnv`): the assignment with the
dead Poseidon2 locals completed. The decoded key digests are what the shared words carry; the
state hash in `x` binds them to the setup's keys (`State.stateHash_collision`), which the terminal
uses. `light_freshHolds` and
`refresh_freshHolds` start from the CCS statement that the folding scheme extracts for a fresh
instance (`Plan.freshHolds_implies_rowsAndPublic`). -/

namespace NightstreamFPrime.Export.OptionB.StepRelation

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Spec.HyperNova.Construction2
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PiCCS.v1_2
open NightstreamFPrime.Lifecycle.OptionB
open NightstreamFPrime.Lifecycle.OptionB.SlotWords
open NightstreamFPrime.Layout.OptionB
open NightstreamFPrime.Export.Generic

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- The public input occupies source columns 0 to 269; the shared words start after it. -/
def base : Nat := ringDegree * PaperAlgebra.publicRingColumns

/-- The logical width that both circuit keys share. It is at least each step circuit's logical
width, so a key with fewer columns commits a witness that is zero past them. -/
def keyWidth : Nat := 2 ^ 26

theorem keyPublicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth keyWidth := by
  decide

theorem keyCubeFits : Phi81CarrierLayout.carrierWidth keyWidth ≤ 2 ^ cubeVariables := by
  decide

/-- The relation argument of the step circuits at `keyWidth`. The emitter builds the circuits with
it; their rows do not read its matrices. -/
def keyRelation : ProductionKey.LogicalRelation keyWidth keyPublicFits :=
  ⟨fun _ _ _ => 0, keyCubeFits⟩

theorem natWord_val (value : F) : natWord value.val = value := by
  apply Fin.ext
  simp [natWord, Poseidon2.ofNat, Nat.mod_eq_of_lt value.isLt]

/-- The key digests that the shared words carry, with the setup's relations. -/
def decodedKeys (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
    (verifierKeys : Fin 2 → Fin 4 → Expr) (env : Env) :
    Fin 2 → CircuitKey logicalWidth publicFits :=
  fun key =>
    ⟨List.ofFn fun lane => (verifierKeys key lane).eval env, (circuitKeys key).relation⟩

section Light

variable (program : Stage1.Application.Program)

abbrev lightInterface :=
  LightAssembler.interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
    program.witnessWordCount

def lightProof (env : Env) : Spec.Folding.Light.Proof LightShape 8 :=
  ⟨fun roundIndex =>
      ((lightInterface (logicalWidth := logicalWidth) (publicFits := publicFits) program).round
        roundIndex).semanticPolynomial env,
    ⟨fun source coefficient =>
        ((lightInterface (logicalWidth := logicalWidth) (publicFits := publicFits)
          program).output.padCoordinate source coefficient).eval env,
      fun source matrix coefficient =>
        ((lightInterface (logicalWidth := logicalWidth) (publicFits := publicFits)
          program).output.matrixCoordinate source matrix coefficient).eval env⟩⟩

/-- A slot whose depth is read as one word: the claim, and the depth modulo `q`. It has the slot's
words and digest, because `serializeSlot` reads the depth as one word. -/
def wordDepth (slot : SlotData logicalWidth publicFits) : SlotData logicalWidth publicFits :=
  (slot.1, (natWord slot.2).val)

theorem serializeSlot_wordDepth (slot : SlotData logicalWidth publicFits) :
    serializeSlot (wordDepth slot) = serializeSlot slot := by
  simp only [serializeSlot, wordDepth, natWord_val]

theorem slotDigest_wordDepth (slot : SlotData logicalWidth publicFits) :
    slotDigest (wordDepth slot) = slotDigest slot := by
  rw [slotDigest, slotDigest, serializeSlot_wordDepth]

theorem wordDepth_lt (slot : SlotData logicalWidth publicFits) :
    (wordDepth slot).2 < goldilocksModulus :=
  Fin.isLt _

theorem wordDepth_of_lt {slot : SlotData logicalWidth publicFits}
    (bound : slot.2 < goldilocksModulus) : wordDepth slot = slot := by
  obtain ⟨claim, depth⟩ := slot
  show (claim, (natWord depth).val) = (claim, depth)
  simp only [natWord, Poseidon2.ofNat, Nat.mod_eq_of_lt bound]

/-- The light step's input. The light circuit reads only the active slot's words: the selected
slot is the active slot, and the other slot is `running`'s, with its depth read as one word. -/
def lightInput (running : Fin 2 → SlotData logicalWidth publicFits) (env : Env) :
    StepInput logicalWidth publicFits :=
  let interface := lightInterface (logicalWidth := logicalWidth) (publicFits := publicFits) program
  { iteration := (interface.iteration.eval env).val
    z0 := List.ofFn fun lane => (interface.initialState lane).eval env
    zi := List.ofFn fun lane => (interface.currentState lane).eval env
    running := fun slot =>
      if slot = LightStepLayout.selectedSlot program interface env then interface.active.value env
      else wordDepth (running slot)
    fresh := StatementAbsorption.evalFresh interface.fresh env
    priorPc := (interface.priorPc.eval env).val
    witness := List.ofFn fun index => (interface.witness index).eval env
    nifsProof := .light (lightProof (logicalWidth := logicalWidth) (publicFits := publicFits)
      program env) }

/-- Every decoded slot's depth is a field value. -/
theorem lightInput_depth (running : Fin 2 → SlotData logicalWidth publicFits) (env : Env)
    (slot : Fin 2) :
    ((lightInput (logicalWidth := logicalWidth) (publicFits := publicFits) program running
      env).running slot).2 < goldilocksModulus := by
  simp only [lightInput]
  split
  · exact Fin.isLt _
  · exact wordDepth_lt _

/-- After the base case, the other slot's prior digest word is the digest of `running`'s other
slot. The light circuit does not read that slot's words. -/
def OtherDigest (running : Fin 2 → SlotData logicalWidth publicFits) (env : Env) : Prop :=
  let interface := lightInterface (logicalWidth := logicalWidth) (publicFits := publicFits) program
  (interface.iteration.eval env).val ≠ 0 → ∀ slot,
    slot ≠ LightStepLayout.selectedSlot program interface env →
      digestValue (interface.priorDigest slot) env = slotDigest (running slot)

/-- The step's output. Its next slots are the digest select's: the new slot (the light fold's
claim at the next depth) in the selected slot after the base case (`DigestSelect.nextSlots`). -/
def lightOutput (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (circuit : Fin 2) (running : Fin 2 → SlotData logicalWidth publicFits) (env : Env) :
    StepOutput logicalWidth publicFits :=
  let interface := lightInterface (logicalWidth := logicalWidth) (publicFits := publicFits) program
  let input := lightInput (logicalWidth := logicalWidth) (publicFits := publicFits) program running
    env
  { zNext := List.ofFn fun lane => (interface.nextState lane).eval env
    runningNext := DigestSelect.nextSlots input.iteration input.running
      (LightStepLayout.selectedSlot program interface env)
      ((LightStepLayout.newSlot program relation circuit interface
        (LightAssembler.sharedEnd base program.witnessWordCount)).value env)
    pcNext := circuit
    x := List.ofFn fun lane => (interface.outputDigest lane).eval env }

/-- The output's next slots, for an environment that is not unfolded. -/
theorem lightOutput_runningNext (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (circuit : Fin 2) (running : Fin 2 → SlotData logicalWidth publicFits) (env : Env) :
    (lightOutput program relation circuit running env).runningNext =
      DigestSelect.nextSlots (lightInput program running env).iteration
        (lightInput program running env).running
        (LightStepLayout.selectedSlot program
          (lightInterface (logicalWidth := logicalWidth) (publicFits := publicFits) program) env)
        ((LightStepLayout.newSlot program relation circuit
          (lightInterface (logicalWidth := logicalWidth) (publicFits := publicFits) program)
          (LightAssembler.sharedEnd base program.witnessWordCount)).value env) :=
  rfl

theorem lightRepresents (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits) (circuit : Fin 2)
    (running : Fin 2 → SlotData logicalWidth publicFits) (env : Env)
    (others : OtherDigest program running env) :
    LightStepSoundness.StepRepresents program
      (lightInterface (logicalWidth := logicalWidth) (publicFits := publicFits) program) ajtai
      (decodedKeys circuitKeys (lightInterface (logicalWidth := logicalWidth)
        (publicFits := publicFits) program).verifierKeys env)
      circuit (lightInput program running env) (lightOutput program relation circuit running env)
      (lightProof (logicalWidth := logicalWidth) (publicFits := publicFits) program env) env where
  iteration := (natWord_val _).symm
  iterationBound := Fin.isLt _
  priorPc := (natWord_val _).symm
  priorPcBound := Fin.isLt _
  verifierKeys := fun _ => rfl
  initialState := rfl
  currentState := rfl
  nextState := rfl
  witness := rfl
  active := by simp [lightInput]
  otherDigests := fun nonzero slot other => by
    simp only [lightInput, other, ↓reduceIte, slotDigest_wordDepth]
    exact others nonzero slot other
  pcNext := rfl
  outputDigest := rfl
  fresh := rfl
  rounds := fun _ => rfl
  outputPad := fun _ _ => rfl
  outputMatrix := fun _ _ _ => rfl
  proof := fun _ => rfl

/-- The source column of output digest word `word`. -/
def lightDigestColumn (word : Fin 4) : Nat :=
  LightAssembler.digestsStart base program.witnessWordCount + 16 + word.val

/-- The environment that an assignment of the light circuit decodes to
(`CircuitRelation.decodeEnv`). -/
def lightEnv (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    {width : Nat} (layout : LogicalPlan.Layout width logicalWidth)
    (assignment : Assignment F logicalWidth) : Env :=
  CircuitRelation.decodeEnv CircuitRelation.publicColumns (CircuitRelation.pieces
    (LightAssemblerCircuit.stepCircuit program base relation 0 ajtai)
    (LightAssembler.sharedEnd base program.witnessWordCount) (lightDigestColumn program)) layout
    assignment

/-- A satisfying assignment of the light circuit's logical relation whose public input encodes
`digest` gives the light case of the option-B step relation, with output hash `digest`, on the
step with the other slot of `running` when that slot's digest is its prior digest word. -/
theorem light_stepHolds (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
    (step : AppState → AppWitness → AppState) (control : AppState → AppWitness → Fin 2)
    (stepEq : program.step = step)
    (validControl : ∀ state witness, program.valid state witness → control state witness = 0)
    (firstFresh width : Nat)
    (fits : CircuitRelation.Fits CircuitRelation.publicColumns (CircuitRelation.pieces
      (LightAssemblerCircuit.stepCircuit program base relation 0 ajtai)
      (LightAssembler.sharedEnd base program.witnessWordCount)
      (lightDigestColumn program)) firstFresh width)
    (layout : LogicalPlan.Layout width logicalWidth)
    (assignment : Assignment F logicalWidth) (digest : Digest)
    (encoded : LogicalPlan.publicInput (publicFits := publicFits)
      (LogicalPlan.sourceEnv layout assignment) = encHash digest)
    (rowsZero : (CircuitRelation.plan _ _ firstFresh width fits layout).RowsZero assignment)
    (running : Fin 2 → SlotData logicalWidth publicFits)
    (others : OtherDigest program running (lightEnv program relation ajtai layout assignment)) :
    let env := lightEnv program relation ajtai layout assignment
    StepHolds Folding.Schedule.productionLightFoldCap ajtai
        (decodedKeys circuitKeys (lightInterface (logicalWidth := logicalWidth)
          (publicFits := publicFits) program).verifierKeys env)
        step control 0 (lightInput program running env)
        (lightOutput program relation 0 running env) ∧
      (lightOutput (logicalWidth := logicalWidth) (publicFits := publicFits) program relation
        0 running env).x =
        List.ofFn fun word : Fin 4 => digest.getD word.val 0 := by
  obtain ⟨rows, words⟩ := CircuitRelation.rowsZero_implies_spec
    (LightAssemblerCircuit.stepCircuit program base relation 0 ajtai)
    (LightAssembler.sharedEnd base program.witnessWordCount) firstFresh
    (lightDigestColumn program) fits layout assignment digest encoded rowsZero (fun _ => rfl)
  exact ⟨LightAssembler.rows_implies_stepHolds program relation base ajtai _ step control 0 rfl
    stepEq validControl _ rows _ _ _
    (lightRepresents program relation ajtai circuitKeys 0 running _ others)
    (lightOutput_runningNext program relation 0 running _),
    congrArg List.ofFn (funext words)⟩

/-- The words of the light step's next state, which its output hash hashes. -/
def lightNextWords (circuit : Fin 2) (env : Env) : List F :=
  let interface := lightInterface (logicalWidth := logicalWidth) (publicFits := publicFits) program
  Hash.evalList env (LightStepLayout.statePreimageExpr program interface (interface.iteration + 1)
    interface.nextState (Expr.const (natWord (circuit.val + 1))) interface.nextDigest)

/-- The rows alone, with a public input that encodes `digest`: `digest` is the hash of the next
state's words, and after the base case the other slot's next digest word is its prior digest
word. -/
theorem light_rowsWords (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (step : AppState → AppWitness → AppState) (control : AppState → AppWitness → Fin 2)
    (stepEq : program.step = step)
    (validControl : ∀ state witness, program.valid state witness → control state witness = 0)
    (firstFresh width : Nat)
    (fits : CircuitRelation.Fits CircuitRelation.publicColumns (CircuitRelation.pieces
      (LightAssemblerCircuit.stepCircuit program base relation 0 ajtai)
      (LightAssembler.sharedEnd base program.witnessWordCount)
      (lightDigestColumn program)) firstFresh width)
    (layout : LogicalPlan.Layout width logicalWidth)
    (assignment : Assignment F logicalWidth) (digest : Digest)
    (encoded : LogicalPlan.publicInput (publicFits := publicFits)
      (LogicalPlan.sourceEnv layout assignment) = encHash digest)
    (rowsZero : (CircuitRelation.plan _ _ firstFresh width fits layout).RowsZero assignment) :
    let env := lightEnv program relation ajtai layout assignment
    let interface := lightInterface (logicalWidth := logicalWidth) (publicFits := publicFits)
      program
    Poseidon2.hash (lightNextWords (logicalWidth := logicalWidth) (publicFits := publicFits) program 0 env) =
        List.ofFn (fun word : Fin 4 => digest.getD word.val 0) ∧
      ((interface.iteration.eval env).val ≠ 0 → ∀ slot,
        slot ≠ LightStepLayout.selectedSlot program interface env →
          digestValue (interface.nextDigest slot) env =
            digestValue (interface.priorDigest slot) env) := by
  obtain ⟨rows, words⟩ := CircuitRelation.rowsZero_implies_spec
    (LightAssemblerCircuit.stepCircuit program base relation 0 ajtai)
    (LightAssembler.sharedEnd base program.witnessWordCount) firstFresh
    (lightDigestColumn program) fits layout assignment digest encoded rowsZero (fun _ => rfl)
  obtain ⟨hashEq, other⟩ := LightAssembler.rows_nextWords program relation base step control 0
    stepEq validControl _ rows
  exact ⟨hashEq.symm.trans (congrArg List.ofFn (funext words)),
    fun nonzero slot different => other (fun zero => nonzero ((congrArg Fin.val zero).trans rfl))
      slot different⟩

/-- Next-state words that are a state's preimage give the state's slot digests as the next
digest words. -/
theorem nextDigests_of_words {circuit : Fin 2} {env : Env} {state : State logicalWidth publicFits}
    (same : lightNextWords (logicalWidth := logicalWidth) (publicFits := publicFits) program circuit env = statePreimage state)
    (slot : Fin 2) :
    digestValue ((lightInterface (logicalWidth := logicalWidth) (publicFits := publicFits)
      program).nextDigest slot) env = slotDigest (state.slots slot) := by
  unfold lightNextWords at same
  rw [LightStepSoundness.statePreimageExpr_eval] at same
  unfold statePreimage at same
  obtain ⟨rest, digest1⟩ :=
    List.append_inj' same (by simp [slotDigest, Poseidon2.hash_length, Poseidon2.digestLen])
  obtain ⟨_, digest0⟩ :=
    List.append_inj' rest (by simp [slotDigest, Poseidon2.hash_length, Poseidon2.digestLen])
  match slot with
  | ⟨0, _⟩ => exact digest0
  | ⟨1, _⟩ => exact digest1

theorem lightNextWords_length (circuit : Fin 2) (env : Env) :
    (lightNextWords (logicalWidth := logicalWidth) (publicFits := publicFits) program circuit env).length = 38 := by
  unfold lightNextWords
  rw [LightStepSoundness.statePreimageExpr_eval]
  simp only [List.length_append, List.length_ofFn, List.length_singleton]
  rfl

end Light

section Refresh

variable (program : Stage1.Application.Program)

abbrev refreshInterface :=
  RefreshAssembler.interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
    program.witnessWordCount

def refreshProof (env : Env) :
    Spec.Folding.Refresh.Proof PaperAlgebra.Commitment productionShape 8 :=
  let interface :=
    refreshInterface (logicalWidth := logicalWidth) (publicFits := publicFits) program
  let commitments := fun child row lane => (interface.childCommitment child row lane).eval env
  let evaluations := fun child =>
    StatementAbsorption.evalEvaluation (interface.childEvaluation child) env
  ⟨⟨commitments, evaluations⟩,
    ⟨fun roundIndex => (interface.round roundIndex).semanticPolynomial env,
      ⟨fun source coefficient => (interface.output.padCoordinate source coefficient).eval env,
        fun source matrix coefficient =>
          (interface.output.matrixCoordinate source matrix coefficient).eval env⟩,
      commitments, evaluations⟩⟩

def refreshInput (env : Env) : StepInput logicalWidth publicFits :=
  let interface :=
    refreshInterface (logicalWidth := logicalWidth) (publicFits := publicFits) program
  { iteration := (interface.iteration.eval env).val
    z0 := List.ofFn fun lane => (interface.initialState lane).eval env
    zi := List.ofFn fun lane => (interface.currentState lane).eval env
    running := fun slot => (interface.prior slot).value env
    fresh := StatementAbsorption.evalFresh interface.fresh env
    priorPc := (interface.priorPc.eval env).val
    witness := List.ofFn fun index => (interface.witness index).eval env
    nifsProof := .refresh (refreshProof (logicalWidth := logicalWidth)
      (publicFits := publicFits) program env) }

/-- The step's output. Its next slots are the digest select's: the new slot (the refresh fold's
claim at depth 0) in the selected slot after the base case (`DigestSelect.nextSlots`). -/
def refreshOutput (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (circuit : Fin 2) (env : Env) : StepOutput logicalWidth publicFits :=
  let interface :=
    refreshInterface (logicalWidth := logicalWidth) (publicFits := publicFits) program
  let input := refreshInput (logicalWidth := logicalWidth) (publicFits := publicFits) program env
  { zNext := List.ofFn fun lane => (interface.nextState lane).eval env
    runningNext := DigestSelect.nextSlots input.iteration input.running
      (DigestSelect.selected (RefreshStepLayout.selectInterface program interface)
        (RefreshStepLayout.selectOffset program relation circuit interface
          (RefreshAssembler.sharedEnd base program.witnessWordCount)) env)
      ((RefreshStepLayout.newSlot program relation circuit interface
        (RefreshAssembler.sharedEnd base program.witnessWordCount)).value env)
    pcNext := circuit
    x := List.ofFn fun lane => (interface.outputDigest lane).eval env }

/-- The output's next slots, for an environment that is not unfolded. -/
theorem refreshOutput_runningNext
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) (circuit : Fin 2)
    (env : Env) :
    (refreshOutput program relation circuit env).runningNext =
      DigestSelect.nextSlots
        (refreshInput (logicalWidth := logicalWidth) (publicFits := publicFits) program
          env).iteration
        (refreshInput (logicalWidth := logicalWidth) (publicFits := publicFits) program
          env).running
        (DigestSelect.selected (RefreshStepLayout.selectInterface program
            (refreshInterface (logicalWidth := logicalWidth) (publicFits := publicFits) program))
          (RefreshStepLayout.selectOffset program relation circuit
            (refreshInterface (logicalWidth := logicalWidth) (publicFits := publicFits) program)
            (RefreshAssembler.sharedEnd base program.witnessWordCount)) env)
        ((RefreshStepLayout.newSlot program relation circuit
          (refreshInterface (logicalWidth := logicalWidth) (publicFits := publicFits) program)
          (RefreshAssembler.sharedEnd base program.witnessWordCount)).value env) :=
  rfl

/-- The finish circuit's output: its next slots are the next slots' words, which the finish
circuit's slot select fixes and its final program reads. -/
def finishStepOutput (circuit : Fin 2) (env : Env) : StepOutput logicalWidth publicFits :=
  let interface :=
    refreshInterface (logicalWidth := logicalWidth) (publicFits := publicFits) program
  { zNext := List.ofFn fun lane => (interface.nextState lane).eval env
    runningNext := fun slot => (interface.next slot).value env
    pcNext := circuit
    x := List.ofFn fun lane => (interface.outputDigest lane).eval env }

theorem refreshRepresents (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits) (circuit : Fin 2) (env : Env) :
    RefreshStepSoundness.StepRepresents program
      (refreshInterface (logicalWidth := logicalWidth) (publicFits := publicFits) program) ajtai
      (decodedKeys circuitKeys (refreshInterface (logicalWidth := logicalWidth)
        (publicFits := publicFits) program).verifierKeys env)
      circuit (refreshInput program env) (refreshOutput program relation circuit env)
      (refreshProof (logicalWidth := logicalWidth) (publicFits := publicFits) program env)
      env where
  iteration := (natWord_val _).symm
  iterationBound := Fin.isLt _
  priorPc := (natWord_val _).symm
  priorPcBound := Fin.isLt _
  verifierKeys := fun _ => rfl
  initialState := rfl
  currentState := rfl
  nextState := rfl
  witness := rfl
  priorSlots := fun _ => SlotExpr.words_eval _ env
  priorDepth := fun _ => Fin.isLt _
  pcNext := rfl
  outputDigest := rfl
  fresh := rfl
  childCommitments := fun _ _ _ => rfl
  childEvaluations := fun _ => rfl
  rounds := fun _ => rfl
  outputPad := fun _ _ => rfl
  outputMatrix := fun _ _ _ => rfl
  proof := fun _ => rfl

def refreshDigestColumn (word : Fin 4) : Nat :=
  LightAssembler.digestsStart base program.witnessWordCount + 16 + word.val

/-- The environment that an assignment of the refresh circuit decodes to
(`CircuitRelation.decodeEnv`). -/
def refreshEnv (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    {width : Nat} (layout : LogicalPlan.Layout width logicalWidth)
    (assignment : Assignment F logicalWidth) : Env :=
  CircuitRelation.decodeEnv CircuitRelation.publicColumns (CircuitRelation.pieces
    (RefreshAssemblerCircuit.stepCircuit program base relation 1 ajtai)
    (RefreshAssembler.sharedEnd base program.witnessWordCount) (refreshDigestColumn program)) layout
    assignment

/-- A satisfying assignment of the refresh circuit's logical relation whose public input
encodes `digest` gives the refresh case of the option-B step relation, with output hash
`digest`. -/
theorem refresh_stepHolds (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
    (step : AppState → AppWitness → AppState) (control : AppState → AppWitness → Fin 2)
    (stepEq : program.step = step)
    (validControl : ∀ state witness, program.valid state witness → control state witness = 1)
    (firstFresh width : Nat)
    (fits : CircuitRelation.Fits CircuitRelation.publicColumns (CircuitRelation.pieces
      (RefreshAssemblerCircuit.stepCircuit program base relation 1 ajtai)
      (RefreshAssembler.sharedEnd base program.witnessWordCount)
      (refreshDigestColumn program)) firstFresh width)
    (layout : LogicalPlan.Layout width logicalWidth)
    (assignment : Assignment F logicalWidth) (digest : Digest)
    (encoded : LogicalPlan.publicInput (publicFits := publicFits)
      (LogicalPlan.sourceEnv layout assignment) = encHash digest)
    (rowsZero : (CircuitRelation.plan _ _ firstFresh width fits layout).RowsZero assignment) :
    let env := refreshEnv program relation ajtai layout assignment
    StepHolds Folding.Schedule.productionLightFoldCap ajtai
        (decodedKeys circuitKeys (refreshInterface (logicalWidth := logicalWidth)
          (publicFits := publicFits) program).verifierKeys env)
        step control 1 (refreshInput program env) (refreshOutput program relation 1 env) ∧
      (refreshOutput (logicalWidth := logicalWidth) (publicFits := publicFits) program relation
        1 env).x =
        List.ofFn fun word : Fin 4 => digest.getD word.val 0 := by
  obtain ⟨rows, words⟩ := CircuitRelation.rowsZero_implies_spec
    (RefreshAssemblerCircuit.stepCircuit program base relation 1 ajtai)
    (RefreshAssembler.sharedEnd base program.witnessWordCount) firstFresh
    (refreshDigestColumn program) fits layout assignment digest encoded rowsZero (fun _ => rfl)
  exact ⟨RefreshAssembler.rows_implies_stepHolds program relation base ajtai _ step control 1 rfl
    stepEq validControl _ rows _ _ _ (refreshRepresents program relation ajtai circuitKeys 1 _)
    (refreshOutput_runningNext program relation 1 _),
    congrArg List.ofFn (funext words)⟩

end Refresh

/-! ### From the fresh statement the fold extracts -/

/-- The first logical columns are the public input of `projectPublicInput`. -/
theorem publicInput_eq_project {width : Nat} (layout : LogicalPlan.Layout width logicalWidth)
    (assignment : Assignment F logicalWidth)
    (publicInside : ringDegree * PaperAlgebra.publicRingColumns ≤ width) :
    LogicalPlan.publicInput (publicFits := publicFits) (LogicalPlan.sourceEnv layout assignment) =
      Phi81Relation.projectPublicInput
        (shape := PaperAlgebra.FullShape logicalWidth publicFits)
        (Phi81CarrierLayout.extendAssignment 0 assignment) := by
  funext column
  have inside : column.val < logicalWidth :=
    Nat.lt_of_lt_of_le column.isLt layout.publicFits
  have same : (PaperAlgebra.FullShape logicalWidth publicFits).publicColumn column =
      Phi81CarrierLayout.embedLogical (⟨column.val, inside⟩ : Fin logicalWidth) := by
    apply Fin.ext
    rfl
  change LogicalPlan.sourceEnv layout assignment column.val = _
  rw [Phi81Relation.projectPublicInput, same, Phi81CarrierLayout.extendAssignment_embedLogical,
    LogicalPlan.sourceEnv_public layout assignment column.val column.isLt
      (Nat.lt_of_lt_of_le column.isLt publicInside)]

/-- A fresh instance that holds under the light circuit's relation, with public input
`encHash digest`, gives the light case of the option-B step relation with output hash `digest`. -/
theorem light_freshHolds (program : Stage1.Application.Program)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
    (step : AppState → AppWitness → AppState) (control : AppState → AppWitness → Fin 2)
    (stepEq : program.step = step)
    (validControl : ∀ state witness, program.valid state witness → control state witness = 0)
    (firstFresh width : Nat)
    (fits : CircuitRelation.Fits CircuitRelation.publicColumns (CircuitRelation.pieces
      (LightAssemblerCircuit.stepCircuit program base relation 0 ajtai)
      (LightAssembler.sharedEnd base program.witnessWordCount)
      (lightDigestColumn program)) firstFresh width)
    (layout : LogicalPlan.Layout width logicalWidth)
    (cubeFits : Phi81CarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables)
    (publicInside : ringDegree * PaperAlgebra.publicRingColumns ≤ width)
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (assignment : Assignment F (Phi81CarrierLayout.carrierWidth logicalWidth))
    (digest : Digest) (encoded : fresh.publicInputs ⟨0, by decide⟩ = encHash digest)
    (accepted : CCS.Holds (PaperAlgebra.semantics ajtai) productionGlobalParams
      (freshStatement ((CircuitRelation.plan _ _ firstFresh width fits layout).logicalRelation
        (publicFits := publicFits) cubeFits) fresh) assignment)
    (running : Fin 2 → SlotData logicalWidth publicFits)
    (others : OtherDigest program running (lightEnv program relation ajtai layout
      (Layout.ProductionRelation.Plan.logicalAssignment assignment))) :
    let env := lightEnv program relation ajtai layout
      (Layout.ProductionRelation.Plan.logicalAssignment assignment)
    StepHolds Folding.Schedule.productionLightFoldCap ajtai
        (decodedKeys circuitKeys (lightInterface (logicalWidth := logicalWidth)
          (publicFits := publicFits) program).verifierKeys env)
        step control 0 (lightInput program running env)
        (lightOutput program relation 0 running env) ∧
      (lightOutput (logicalWidth := logicalWidth) (publicFits := publicFits) program relation
        0 running env).x =
        List.ofFn fun word : Fin 4 => digest.getD word.val 0 := by
  obtain ⟨rowsZero, publicEq⟩ :=
    Layout.ProductionRelation.Plan.freshHolds_implies_rowsAndPublic _ cubeFits ajtai fresh
      assignment layout.publicFits accepted
  exact light_stepHolds program relation ajtai circuitKeys step control stepEq validControl
    firstFresh width fits layout _ digest
    ((publicInput_eq_project layout _ publicInside).trans (publicEq.trans encoded)) rowsZero
    running others

/-- A fresh instance that holds under the light circuit's relation, with public input
`encHash digest`: `digest` is the hash of the light step's next state words, and after the base
case the other slot's next digest word is its prior digest word (`light_rowsWords`). -/
theorem light_freshWords (program : Stage1.Application.Program)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (step : AppState → AppWitness → AppState) (control : AppState → AppWitness → Fin 2)
    (stepEq : program.step = step)
    (validControl : ∀ state witness, program.valid state witness → control state witness = 0)
    (firstFresh width : Nat)
    (fits : CircuitRelation.Fits CircuitRelation.publicColumns (CircuitRelation.pieces
      (LightAssemblerCircuit.stepCircuit program base relation 0 ajtai)
      (LightAssembler.sharedEnd base program.witnessWordCount)
      (lightDigestColumn program)) firstFresh width)
    (layout : LogicalPlan.Layout width logicalWidth)
    (cubeFits : Phi81CarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables)
    (publicInside : ringDegree * PaperAlgebra.publicRingColumns ≤ width)
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (assignment : Assignment F (Phi81CarrierLayout.carrierWidth logicalWidth))
    (digest : Digest) (encoded : fresh.publicInputs ⟨0, by decide⟩ = encHash digest)
    (accepted : CCS.Holds (PaperAlgebra.semantics ajtai) productionGlobalParams
      (freshStatement ((CircuitRelation.plan _ _ firstFresh width fits layout).logicalRelation
        (publicFits := publicFits) cubeFits) fresh) assignment) :
    let env := lightEnv program relation ajtai layout
      (Layout.ProductionRelation.Plan.logicalAssignment assignment)
    let interface := lightInterface (logicalWidth := logicalWidth) (publicFits := publicFits)
      program
    Poseidon2.hash (lightNextWords (logicalWidth := logicalWidth) (publicFits := publicFits)
        program 0 env) = List.ofFn (fun word : Fin 4 => digest.getD word.val 0) ∧
      ((interface.iteration.eval env).val ≠ 0 → ∀ slot,
        slot ≠ LightStepLayout.selectedSlot program interface env →
          digestValue (interface.nextDigest slot) env =
            digestValue (interface.priorDigest slot) env) := by
  obtain ⟨rowsZero, publicEq⟩ :=
    Layout.ProductionRelation.Plan.freshHolds_implies_rowsAndPublic _ cubeFits ajtai fresh
      assignment layout.publicFits accepted
  exact light_rowsWords program relation ajtai step control stepEq validControl firstFresh width
    fits layout _ digest
    ((publicInput_eq_project layout _ publicInside).trans (publicEq.trans encoded)) rowsZero

/-- A fresh instance that holds under the refresh circuit's relation, with public input
`encHash digest`, gives the refresh case of the option-B step relation with output hash
`digest`. -/
theorem refresh_freshHolds (program : Stage1.Application.Program)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
    (step : AppState → AppWitness → AppState) (control : AppState → AppWitness → Fin 2)
    (stepEq : program.step = step)
    (validControl : ∀ state witness, program.valid state witness → control state witness = 1)
    (firstFresh width : Nat)
    (fits : CircuitRelation.Fits CircuitRelation.publicColumns (CircuitRelation.pieces
      (RefreshAssemblerCircuit.stepCircuit program base relation 1 ajtai)
      (RefreshAssembler.sharedEnd base program.witnessWordCount)
      (refreshDigestColumn program)) firstFresh width)
    (layout : LogicalPlan.Layout width logicalWidth)
    (cubeFits : Phi81CarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables)
    (publicInside : ringDegree * PaperAlgebra.publicRingColumns ≤ width)
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (assignment : Assignment F (Phi81CarrierLayout.carrierWidth logicalWidth))
    (digest : Digest) (encoded : fresh.publicInputs ⟨0, by decide⟩ = encHash digest)
    (accepted : CCS.Holds (PaperAlgebra.semantics ajtai) productionGlobalParams
      (freshStatement ((CircuitRelation.plan _ _ firstFresh width fits layout).logicalRelation
        (publicFits := publicFits) cubeFits) fresh) assignment) :
    let env := refreshEnv program relation ajtai layout
      (Layout.ProductionRelation.Plan.logicalAssignment assignment)
    StepHolds Folding.Schedule.productionLightFoldCap ajtai
        (decodedKeys circuitKeys (refreshInterface (logicalWidth := logicalWidth)
          (publicFits := publicFits) program).verifierKeys env)
        step control 1 (refreshInput program env) (refreshOutput program relation 1 env) ∧
      (refreshOutput (logicalWidth := logicalWidth) (publicFits := publicFits) program relation
        1 env).x =
        List.ofFn fun word : Fin 4 => digest.getD word.val 0 := by
  obtain ⟨rowsZero, publicEq⟩ :=
    Layout.ProductionRelation.Plan.freshHolds_implies_rowsAndPublic _ cubeFits ajtai fresh
      assignment layout.publicFits accepted
  exact refresh_stepHolds program relation ajtai circuitKeys step control stepEq validControl
    firstFresh width fits layout _ digest
    ((publicInput_eq_project layout _ publicInside).trans (publicEq.trans encoded)) rowsZero

end NightstreamFPrime.Export.OptionB.StepRelation
