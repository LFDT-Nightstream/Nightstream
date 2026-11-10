import NightstreamFPrime.Lifecycle.BoundedDepth.RefreshStepLayout

/-! The refresh step circuit's soundness (CW-3.4c). `StepRepresents` states what the step's shared
words carry: the counters, key digests, states and witness of the step's input and output, both
prior slots, the fresh instance, the refresh proof's children, rounds and Π_CCS output, and the
output hash. With it and next slots that `DigestSelect.nextSlots` gives, the rows of
`RefreshStepLayout.main` give the refresh case of `stepHolds_iff` (`rows_implies_stepHolds`): the
digest-select rows pick the active slot and the next slots' digests, Π_DEC splits the active
slot, the refresh Π_CCS and Π_RLC fold the children with the fresh instance, and
`RefreshStep.spec_implies_stepHolds` assembles the step. -/

namespace NightstreamFPrime.Lifecycle.BoundedDepth.RefreshStepSoundness

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Spec.HyperNova.Construction2
open NightstreamFPrime.Lifecycle.PiCCS.v1_2
open NightstreamFPrime.Lifecycle.BoundedDepth.SlotWords
open NightstreamFPrime.Lifecycle.BoundedDepth.RefreshStepLayout

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}
  (program : Stage1.Application.Program)
  (interface : RefreshStepLayout.Interface logicalWidth publicFits program.witnessWordCount)

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

/-- The new slot is the refresh fold's next slot claim (`RefreshFold.nextSlot`) at depth 0. -/
theorem newSlot_value (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (circuit : Fin 2) (offset : Nat) (env : Env) :
    (newSlot program relation circuit interface offset).value env =
      (RefreshFold.nextSlot relation (rlcInterface program relation circuit interface offset)
        (rlcOffset program relation circuit interface offset) env, 0) :=
  rfl

/-- What the step's shared words carry. -/
structure StepRepresents (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth)
      (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits) (circuit : Fin 2)
    (input : StepInput logicalWidth publicFits) (output : StepOutput logicalWidth publicFits)
    (proof : Spec.Folding.Refresh.Proof PaperAlgebra.Commitment productionShape 8) (env : Env) :
    Prop where
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
  pcNext : output.pcNext = circuit
  outputDigest : output.x = List.ofFn (fun lane => (interface.outputDigest lane).eval env)
  fresh : StatementAbsorption.evalFresh interface.fresh env = input.fresh
  childCommitments : ∀ child row lane,
    (interface.childCommitment child row lane).eval env = proof.split.commitments child row lane
  childEvaluations : ∀ child,
    StatementAbsorption.evalEvaluation (interface.childEvaluation child) env =
      proof.split.evaluations child
  rounds : ∀ roundIndex,
    (interface.round roundIndex).semanticPolynomial env = proof.nifs.piCcsRounds roundIndex
  outputPad : ∀ source coefficient,
    (interface.output.padCoordinate source coefficient).eval env =
      proof.nifs.piCcsOutput.padCoordinate source coefficient
  outputMatrix : ∀ source matrix coefficient,
    (interface.output.matrixCoordinate source matrix coefficient).eval env =
      proof.nifs.piCcsOutput.matrixCoordinate source matrix coefficient
  proof : input.iteration ≠ 0 → input.nifsProof = .refresh proof

private theorem slotPreimage_eval (words : List Expr) (env : Env) :
    Hash.evalList env (slotDomainChunk.map Expr.const ++ words) =
      slotDomainChunk ++ Hash.evalList env words := by
  simp [Hash.evalList, List.map_map, Function.comp_def]

private theorem natWord_succ (count : Nat) : natWord (count + 1) = natWord count + 1 := by
  apply Fin.ext
  simp [natWord, Poseidon2.ofNat, Fin.val_add, Nat.add_mod]

private theorem eval_one (env : Env) : (1 : Expr).eval env = 1 := rfl

section Wires

variable (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits) (circuit : Fin 2)
  (input : StepInput logicalWidth publicFits) (output : StepOutput logicalWidth publicFits)
  (proof : Spec.Folding.Refresh.Proof PaperAlgebra.Commitment productionShape 8) (env : Env)
  (represents : StepRepresents program interface ajtai circuitKeys circuit input output proof env)
  (selected : Fin 2) (activeValue : interface.active.value env = input.running selected)

include represents activeValue

/-- The step's words carry what today's Π_DEC reads: the active slot's claim as the parent and
the proof's children as the messages. -/
theorem decWired (offset : Nat) :
    RefreshFold.DecWired (decInterface program interface)
      (decOffset program circuit interface offset) (input.running selected).1 proof.split env where
  parentCommitment := fun row lane => by
    rw [← activeValue]
    rfl
  parentPublicInput := fun column => by
    rw [← activeValue]
    rfl
  parentPoint := by
    rw [← activeValue]
    rfl
  parentEvaluation := by
    rw [← activeValue]
    rfl
  messageCommitment := represents.childCommitments
  messageEvaluation := represents.childEvaluations

/-- The step's words carry what the refresh Π_CCS layout reads: the children of the active slot,
the fresh instance and the refresh proof. -/
theorem piccsRepresents (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (offset : Nat) :
    RefreshPiCCSLayout.Represents
      (RefreshFold.children relation ajtai (input.running selected).1 proof.split) input.fresh
      proof.nifs (piccsInterface program interface) offset env where
  fresh := represents.fresh
  point := by
    rw [← activeValue]
    simp [canonicalFinIndices, List.map_ofFn, RefreshFold.children,
      Spec.Folding.Refresh.children, toClaim, SlotExpr.value, StatementAbsorption.evalPoint]
    rfl
  commitments := represents.childCommitments
  evaluations := represents.childEvaluations
  rounds := represents.rounds
  outputPad := represents.outputPad
  outputMatrix := represents.outputMatrix

omit activeValue in
/-- The step's words carry what today's Π_RLC reads: the fresh instance and the children as
sources, the Π_DEC digit words as the children's public inputs, and the Π_CCS output families. -/
theorem rlcWired (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (offset : Nat) :
    RefreshFold.RlcWired (decInterface program interface)
      (decOffset program circuit interface offset)
      (piccsLeaves program relation circuit interface offset)
      (rlcInterface program relation circuit interface offset)
      (rlcOffset program relation circuit interface offset) input.fresh proof.split
      proof.nifs.piCcsOutput env where
  samplerStart := by
    simp only [rlcInterface, Lifecycle.PiRLC.v1_2.Sampler.evalState]
  point := by
    simp only [rlcInterface, Lifecycle.PiRLC.v1_2.InputBinding.evalPoint,
      StatementAbsorption.evalPoint, RoundTranscript.evalRoundPoint, canonicalFinIndices,
      List.map_ofFn]
    rfl
  commitment := fun source row lane => by
    change ((Fin.addCases interface.fresh.commitment interface.childCommitment source :
        Fin productionProfile.commitmentWidth → Fin ringDegree → Expr) row lane).eval env =
      (Fin.addCases input.fresh.commitments proof.split.commitments source :
        PaperAlgebra.Commitment) row lane
    refine Fin.addCases (fun index => ?_) (fun child => ?_) source
    · simp only [Fin.addCases_left]
      rw [← represents.fresh]
      rfl
    · simp only [Fin.addCases_right]
      exact represents.childCommitments child row lane
  publicInput := fun source column => by
    change ((Fin.addCases interface.fresh.publicInput interface.digit source :
        Fin (PaperAlgebra.FullShape logicalWidth publicFits).publicWidth → Expr) column).eval
          env =
      (Fin.addCases input.fresh.publicInputs
        (fun child column =>
          ((decInterface program interface).digit (decOffset program circuit interface offset)
            (RefreshFold.outputIndex child) column).eval env) source :
          PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
        column
    refine Fin.addCases (fun index => ?_) (fun child => ?_) source
    · simp only [Fin.addCases_left]
      rw [← represents.fresh]
      rfl
    · simp only [Fin.addCases_right]
      rfl
  evaluation := fun source => by
    change StatementAbsorption.evalEvaluation
        ⟨interface.output.padCoordinate source, interface.output.matrixCoordinate source⟩ env =
      _
    simp only [StatementAbsorption.evalEvaluation, represents.outputPad, represents.outputMatrix]

end Wires

/-- Soundness of the refresh step circuit: its rows, the values of its shared words and the next
slots of `DigestSelect.nextSlots` give the step relation of a refresh circuit (`stepHolds_iff`)
at the production cap. The application program computes `step`, and its validity predicate
implies the dispatch. -/
theorem rows_implies_stepHolds
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
    (step : AppState → AppWitness → AppState) (control : AppState → AppWitness → Fin 2)
    (circuit : Fin 2) (refresh : Spec.HyperNova.Construction2.BoundedDepth.circuitStep circuit = .refresh)
    (stepEq : program.step = step)
    (validControl : ∀ state witness, program.valid state witness →
      control state witness = circuit)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) (offset : Nat) (env : Env)
    (assumptions : RefreshStepLayout.Assumptions program relation circuit interface offset env)
    (rows : holds env (Circuit.ops (RefreshStepLayout.main program relation circuit interface)
      offset))
    (input : StepInput logicalWidth publicFits) (output : StepOutput logicalWidth publicFits)
    (proof : Spec.Folding.Refresh.Proof PaperAlgebra.Commitment productionShape 8)
    (represents : StepRepresents program interface ajtai circuitKeys circuit input output proof
      env)
    (next : output.runningNext = DigestSelect.nextSlots input.iteration input.running
      (DigestSelect.selected (RefreshStepLayout.selectInterface program interface)
        (RefreshStepLayout.selectOffset program relation circuit interface offset) env)
      ((newSlot program relation circuit interface offset).value env)) :
    StepHolds Folding.Schedule.productionLightFoldCap ajtai circuitKeys step control circuit input
      output := by
  obtain ⟨specification, selectRows⟩ := RefreshStepLayout.soundness program relation circuit
    interface offset step control stepEq validControl env assumptions rows
    (circuitKeys (DigestSelect.selected (RefreshStepLayout.selectInterface program interface)
      (RefreshStepLayout.selectOffset program relation circuit interface offset) env)).relation
  have priorDigests := fun slot => gadgetDigest_eq (specification.priorSlots slot)
    ((slotPreimage_eval _ env).trans (congrArg (slotDomainChunk ++ ·) (represents.priorSlots slot)))
  obtain ⟨select, activeValue, nextDigests⟩ := DigestSelect.rows_implies_select
    (RefreshStepLayout.selectInterface program interface)
    (RefreshStepLayout.selectOffset program relation circuit interface offset) env input output
    ((newSlot program relation circuit interface offset).value env)
    selectRows represents.iteration represents.iterationBound represents.priorPc
    represents.priorPcBound represents.initialState represents.currentState represents.priorSlots
    represents.priorDepth priorDigests
    (LightStepLayout.slotHash_digest _ _ _ env specification.newSlot) next
  rw [newSlot_value program interface relation circuit offset env] at select
  generalize DigestSelect.selected (RefreshStepLayout.selectInterface program interface)
    (RefreshStepLayout.selectOffset program relation circuit interface offset) env = selected
    at specification select activeValue
  refine RefreshStep.spec_implies_stepHolds ajtai circuitKeys step control circuit input output
    refresh (RefreshStepLayout.leaves program relation circuit interface offset) selected proof
    env specification ⟨represents.currentState, represents.witness, represents.nextState,
      represents.pcNext, fun slot => ?_, ?_, fun column => ?_, ?_, represents.outputDigest,
      decWired program interface ajtai circuitKeys circuit input output proof env represents
        selected activeValue offset,
      RefreshPiCCSLayout.wired relation (piccsOffset program circuit interface offset)
        (circuitKeys selected).relation ajtai
        (RefreshFold.children (circuitKeys selected).relation ajtai (input.running selected).1
          proof.split) input.fresh proof.nifs (piccsInterface program interface) env
        (piccsRepresents program interface ajtai circuitKeys circuit input output proof env
          represents selected activeValue (circuitKeys selected).relation
          (piccsOffset program circuit interface offset)),
      rlcWired program interface ajtai circuitKeys circuit input output proof env represents
        relation offset,
      represents.proof⟩ select nextDigests
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
  · unfold StateRepresents
    change Hash.evalList env (statePreimageExpr program interface (interface.iteration + 1)
      interface.nextState (Expr.const (natWord (circuit.val + 1))) interface.nextDigest) = _
    rw [statePreimageExpr_eval, Expr.eval_hadd, eval_one, represents.iteration, ← natWord_succ,
      represents.initialState, represents.nextState, represents.verifierKeys 0,
      represents.verifierKeys 1, Expr.eval_const, ← represents.pcNext]
    rfl

end NightstreamFPrime.Lifecycle.BoundedDepth.RefreshStepSoundness
