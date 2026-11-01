import NightstreamFPrime.Lifecycle.OptionB.LightStepLayout

/-! The light step circuit's soundness (CW-2.5d, d5b). `StepRepresents` states what the step's
shared words carry: the counters, key digests, states and witness of the step's input and output,
both prior and both next slots, the fresh instance, the light proof and the output hash. With it,
the rows of `LightStepLayout.main` give the light case of `stepHolds_iff` at the production cap
(`rows_implies_stepHolds`): the slot-select rows pick the active slot, the two fold layouts fold it
with the fresh instance, and `LightStep.spec_implies_stepHolds` assembles the step. -/

namespace NightstreamFPrime.Lifecycle.OptionB.LightStepSoundness

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Spec.HyperNova.Construction2
open NightstreamFPrime.Lifecycle.PiCCS.v1_2
open NightstreamFPrime.Lifecycle.PiRLC.v1_2
open NightstreamFPrime.Lifecycle.OptionB.SlotWords
open NightstreamFPrime.Lifecycle.OptionB.LightStepLayout

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}
  (program : Stage1.Application.Program)
  (interface : LightStepLayout.Interface logicalWidth publicFits program.witnessWordCount)

theorem statePreimageExpr_eval (iteration : Expr) (current : Fin 4 → Expr) (pc : Expr)
    (digests : Fin 2 → Fin 4 → Expr) (env : Env) :
    Hash.evalList env (statePreimageExpr program interface iteration current pc digests) =
      stateDomainChunk ++ List.ofFn (fun lane => (interface.verifierKeys 0 lane).eval env) ++
        List.ofFn (fun lane => (interface.verifierKeys 1 lane).eval env) ++ [iteration.eval env] ++
        List.ofFn (fun lane => (interface.initialState lane).eval env) ++
        List.ofFn (fun lane => (current lane).eval env) ++ [pc.eval env] ++
        List.ofFn (fun lane => (digests 0 lane).eval env) ++
        List.ofFn (fun lane => (digests 1 lane).eval env) := by
  simp [statePreimageExpr, Hash.evalList, Function.comp_def, List.map_map]

/-- The new slot's claim is the light fold's next slot claim (`LightFold.nextSlot`). -/
theorem newSlot_value (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (circuit : Fin 2) (offset : Nat) (env : Env) :
    ((newSlot program relation circuit interface offset).value env).1 =
      LightFold.nextSlot (piccsLeaves program relation circuit interface offset)
        (rlcLeaves program relation circuit interface offset) env := by
  simp only [SlotExpr.value, LightFold.nextSlot]
  congr 1

/-- What the step's shared words carry. -/
structure StepRepresents (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth)
      (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits) (circuit : Fin 2)
    (input : StepInput logicalWidth publicFits) (output : StepOutput logicalWidth publicFits)
    (proof : Spec.Folding.Light.Proof LightShape 8) (env : Env) : Prop where
  iteration : interface.iteration.eval env = natWord input.iteration
  iterationBound : input.iteration < goldilocksModulus
  priorPc : interface.priorPc.eval env = natWord input.priorPc
  priorPcBound : input.priorPc < goldilocksModulus
  verifierKeys : ∀ key,
    List.ofFn (fun lane => (interface.verifierKeys key lane).eval env) = (circuitKeys key).digest
  initialState : List.ofFn (fun lane => (interface.initialState lane).eval env) = input.z0
  currentState : List.ofFn (fun lane => (interface.currentState lane).eval env) = input.zi
  nextState : List.ofFn (fun lane => (interface.nextState lane).eval env) = output.zNext
  witness : List.ofFn (fun index => (interface.witness index).eval env) = input.witness
  priorSlots : ∀ slot,
    Hash.evalList env (interface.prior slot).words = serializeSlot (input.running slot)
  priorDepth : ∀ slot, (input.running slot).2 < goldilocksModulus
  nextSlots : ∀ slot,
    Hash.evalList env (interface.next slot).words = serializeSlot (output.runningNext slot)
  nextDepth : ∀ slot, (output.runningNext slot).2 < goldilocksModulus
  pcNext : output.pcNext = circuit
  outputDigest : output.x = List.ofFn (fun lane => (interface.outputDigest lane).eval env)
  fresh : StatementAbsorption.evalFresh interface.fresh env = input.fresh
  rounds : ∀ roundIndex,
    (interface.round roundIndex).semanticPolynomial env = proof.piCcsRounds roundIndex
  outputPad : ∀ source coefficient,
    (interface.output.padCoordinate source coefficient).eval env =
      proof.piCcsOutput.padCoordinate source coefficient
  outputMatrix : ∀ source matrix coefficient,
    (interface.output.matrixCoordinate source matrix coefficient).eval env =
      proof.piCcsOutput.matrixCoordinate source matrix coefficient
  proof : input.iteration ≠ 0 → input.nifsProof = .light proof

private theorem slotPreimage_eval (words : List Expr) (env : Env) :
    Hash.evalList env (slotDomainChunk.map Expr.const ++ words) =
      slotDomainChunk ++ Hash.evalList env words := by
  simp [Hash.evalList, List.map_map, Function.comp_def]

private theorem natWord_succ (count : Nat) : natWord (count + 1) = natWord count + 1 := by
  apply Fin.ext
  simp [natWord, Poseidon2.ofNat, Fin.val_add, Nat.add_mod]

private theorem eval_one (env : Env) : (1 : Expr).eval env = 1 := rfl

/-- The step's words carry what the light Π_CCS layout reads: the active slot's claim as the
running claim, the fresh instance and the light proof. -/
theorem piccsRepresents
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits) (circuit : Fin 2)
    (input : StepInput logicalWidth publicFits) (output : StepOutput logicalWidth publicFits)
    (proof : Spec.Folding.Light.Proof LightShape 8) (env : Env)
    (represents : StepRepresents program interface ajtai circuitKeys circuit input output proof
      env)
    (selected : Fin 2) (activeValue : interface.active.value env = input.running selected)
    (offset : Nat) :
    LightPiCCSLayout.Represents (LightFold.lightRunning (input.running selected).1)
      (OptionBFolds.freshToLight input.fresh) proof (piccsInterface program interface) offset
      env where
  fresh := represents.fresh
  runningPoint := by
    rw [← activeValue]
    simp [canonicalFinIndices, List.map_ofFn, LightFold.lightRunning, SlotExpr.value,
      StatementAbsorption.evalPoint]
    rfl
  runningPad := fun coefficient => by
    rw [← activeValue]
    rfl
  runningMatrix := fun matrix coefficient => by
    rw [← activeValue]
    rfl
  rounds := represents.rounds
  outputPad := represents.outputPad
  outputMatrix := represents.outputMatrix

/-- The step's words carry what the light Π_RLC layout reads: the active slot's commitment and
public input, the fresh instance and the light proof's Π_CCS output. -/
theorem rlcRepresents (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (circuit : Fin 2) (offset : Nat)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
    (input : StepInput logicalWidth publicFits) (output : StepOutput logicalWidth publicFits)
    (proof : Spec.Folding.Light.Proof LightShape 8) (env : Env)
    (represents : StepRepresents program interface ajtai circuitKeys circuit input output proof
      env)
    (selected : Fin 2) (activeValue : interface.active.value env = input.running selected) :
    LightRlcLayout.Represents (piccsLeaves program relation circuit interface offset)
      (rlcInterface program interface) (rlcOffset program relation circuit interface offset)
      (input.running selected).1 input.fresh proof.piCcsOutput env where
  runningCommitment := by
    rw [← activeValue]
    rfl
  runningPublicInput := by
    rw [← activeValue]
    rfl
  freshWords := represents.fresh
  outputPad := represents.outputPad
  outputMatrix := represents.outputMatrix

/-- Soundness of the light step circuit: its rows and the values of its shared words give the
step relation of a light circuit (`stepHolds_iff`) at the production cap. The application program
computes `step`, and its validity predicate implies the dispatch. -/
theorem rows_implies_stepHolds
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
    (step : AppState → AppWitness → AppState) (control : AppState → AppWitness → Fin 2)
    (circuit : Fin 2) (light : BoundedDepth.circuitStep circuit = .light)
    (stepEq : program.step = step)
    (validControl : ∀ state witness, program.valid state witness →
      control state witness = circuit)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) (offset : Nat) (env : Env)
    (assumptions : LightStepLayout.Assumptions program relation circuit interface offset env)
    (rows : holds env (Circuit.ops (LightStepLayout.main program relation circuit interface)
      offset))
    (input : StepInput logicalWidth publicFits) (output : StepOutput logicalWidth publicFits)
    (proof : Spec.Folding.Light.Proof LightShape 8)
    (represents : StepRepresents program interface ajtai circuitKeys circuit input output proof
      env) :
    StepHolds Folding.Schedule.productionLightFoldCap ajtai circuitKeys step control circuit input
      output := by
  obtain ⟨specification, selectRows⟩ := LightStepLayout.soundness program relation circuit
    interface offset step control stepEq validControl env assumptions rows
    (circuitKeys (SlotSelect.selected
      (LightStepLayout.selectInterface program relation circuit interface offset)
      (LightStepLayout.selectOffset program relation circuit interface offset) env)).relation
  obtain ⟨select, activeValue⟩ := SlotSelect.rows_implies_select
    (LightStepLayout.selectInterface program relation circuit interface offset)
    (LightStepLayout.selectOffset program relation circuit interface offset) env input output
    selectRows represents.iteration represents.iterationBound represents.priorPc
    represents.priorPcBound represents.initialState represents.currentState represents.priorSlots
    represents.priorDepth represents.nextSlots represents.nextDepth
  have nextEq : (((LightStepLayout.selectInterface program relation circuit interface offset).new
      (LightStepLayout.selectOffset program relation circuit interface offset)).value env).1 =
      LightFold.nextSlot (piccsLeaves program relation circuit interface offset)
        (rlcLeaves program relation circuit interface offset) env :=
    newSlot_value program interface relation circuit offset env
  rw [nextEq] at select
  generalize SlotSelect.selected
    (LightStepLayout.selectInterface program relation circuit interface offset)
    (LightStepLayout.selectOffset program relation circuit interface offset) env = selected
    at specification select activeValue
  refine LightStep.spec_implies_stepHolds ajtai circuitKeys step control circuit input output
    light (LightStepLayout.leaves program relation circuit interface offset) selected proof env
    specification ⟨represents.currentState, represents.witness, represents.nextState,
      represents.pcNext, fun slot => ?_, ?_, fun column => ?_, fun slot => ?_, ?_,
      represents.outputDigest,
      LightPiCCSLayout.wired relation (piccsOffset program circuit interface offset)
        (circuitKeys selected).relation ajtai (LightFold.lightRunning (input.running selected).1)
        (OptionBFolds.freshToLight input.fresh) proof (piccsInterface program interface) env ?_,
      LightRlcLayout.wired (piccsLeaves program relation circuit interface offset)
        (rlcInterface program interface) (rlcOffset program relation circuit interface offset)
        (input.running selected).1 input.fresh proof.piCcsOutput env ?_, ?_,
      fun nonzero => ⟨represents.proof nonzero, represents.priorDepth selected⟩⟩ select
  · exact (slotPreimage_eval _ env).trans
      (congrArg (slotDomainChunk ++ ·) (represents.priorSlots slot))
  · unfold StateRepresents
    change Hash.evalList env (statePreimageExpr program interface interface.iteration
      interface.currentState interface.priorPc interface.priorDigest) = _
    rw [statePreimageExpr_eval, represents.iteration, represents.initialState,
      represents.currentState, represents.priorPc, represents.verifierKeys 0,
      represents.verifierKeys 1]
    rfl
  · rw [← represents.fresh]
    rfl
  · exact (slotPreimage_eval _ env).trans
      (congrArg (slotDomainChunk ++ ·) (represents.nextSlots slot))
  · unfold StateRepresents
    change Hash.evalList env (statePreimageExpr program interface (interface.iteration + 1)
      interface.nextState (Expr.const (natWord (circuit.val + 1))) interface.nextDigest) = _
    rw [statePreimageExpr_eval, Expr.eval_hadd, eval_one, represents.iteration, ← natWord_succ,
      represents.initialState, represents.nextState, represents.verifierKeys 0,
      represents.verifierKeys 1, Expr.eval_const, ← represents.pcNext]
    rfl
  · exact piccsRepresents program interface ajtai circuitKeys circuit input output proof env
      represents selected activeValue _
  · exact rlcRepresents program interface relation circuit offset ajtai circuitKeys input output
      proof env represents selected activeValue
  · rw [← activeValue]
    exact (SlotExpr.natWord_val _).symm

end NightstreamFPrime.Lifecycle.OptionB.LightStepSoundness
