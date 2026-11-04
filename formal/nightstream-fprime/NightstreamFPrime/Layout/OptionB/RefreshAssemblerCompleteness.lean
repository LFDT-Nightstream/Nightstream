import NightstreamFPrime.Layout.OptionB.RefreshAssemblerSource
import NightstreamFPrime.Layout.OptionB.RefreshStepCompleteness
import NightstreamFPrime.Layout.OptionB.RefreshPiCCSCompleteness
import NightstreamFPrime.Layout.OptionB.SharedWordsAbove
import NightstreamFPrime.Layout.OptionB.StepFootprint

/-! Completeness of the refresh step circuit (CW-3.8). An honest refresh step (`StepHolds` with
the application's validity, the prior-state binding of the fresh instance, and a refresh proof
that the refresh fold accepts for the selected slot) has an assignment of the circuit at the
source layout: the honest source values (`RefreshAssemblerSource.honestValues`), then the seven
step gadgets (`completeGadgetsFrom`), today's Π_DEC (`completeDecFrom`, whose phase the accepted
fold and the honest digit words give), the refresh Π_CCS (`RefreshPiCCSCompleteness.completeFrom`),
today's Π_RLC (`completeRlcPhaseFrom`), and the slot-select rows (`SlotSelect.rows_of_select`).
Every row holds, and the words carry the step (`complete`). -/

namespace NightstreamFPrime.Layout.OptionB.RefreshAssemblerCompleteness

open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.OptionB

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2
open NightstreamFPrime.Lifecycle.PiCCS.v1_2
open NightstreamFPrime.Lifecycle.OptionB.SlotWords
open NightstreamFPrime.Lifecycle.OptionB.RefreshStepLayout
open NightstreamFPrime.Layout.OptionB.LightAssembler (slotAt slotWordCount slotsStart digestsStart)
open NightstreamFPrime.Layout.OptionB.RefreshAssembler
open NightstreamFPrime.Layout.OptionB.RefreshAssemblerSource

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

private theorem natWord_succ (count : Nat) : natWord (count + 1) = natWord count + 1 := by
  apply Fin.ext
  simp [natWord, Poseidon2.ofNat, Fin.val_add, Nat.add_mod]

private theorem natWord_ne_zero {count : Nat} (bound : count < goldilocksModulus)
    (nonzero : count ≠ 0) : natWord count ≠ 0 := fun zero =>
  nonzero (natWord_injective bound (by decide) (zero.trans rfl))

private theorem slotPreimage_eval (words : List Expr) (env : Env) :
    Hash.evalList env (slotDomainChunk.map Expr.const ++ words) =
      slotDomainChunk ++ Hash.evalList env words := by
  simp [Hash.evalList, List.map_map, Function.comp_def]

private theorem defaultWords_eval (env : Env) :
    Hash.evalList env (SlotSelect.defaultWords (logicalWidth := logicalWidth)
        (publicFits := publicFits)) =
      serializeSlot ((defaultSlot, 0) : SlotData logicalWidth publicFits) := by
  simp [SlotSelect.defaultWords, Hash.evalList, List.map_map, Function.comp_def]

private theorem eval_one (env : Env) : (1 : Expr).eval env = 1 := rfl

private theorem output_ext
    {left right : Lifecycle.PiDEC.v1_2.OutputBinding.Output logicalWidth publicFits}
    (constraintSystem : left.constraintSystem = right.constraintSystem)
    (commitment : left.commitment = right.commitment)
    (publicInput : left.publicInput = right.publicInput)
    (point : left.point = right.point)
    (evaluations : left.evaluations = right.evaluations)
    (stage : left.stage = right.stage) : left = right := by
  cases left
  cases right
  simp_all

private theorem flatConstraints_append (left right : List Op) :
    flatConstraints (left ++ right) = flatConstraints left ++ flatConstraints right := by
  simp [flatConstraints, List.flatMap_append]

section Groups

variable (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
  (input : StepInput logicalWidth publicFits) (output : StepOutput logicalWidth publicFits)
  (proof : Spec.Folding.Refresh.Proof PaperAlgebra.Commitment productionShape 8)
  (selected : Fin 2) (folded : SlotData logicalWidth publicFits)

theorem active_group :
    (honestValues ajtai circuitKeys input output proof selected folded).slots ⟨2, by decide⟩ =
      input.running selected := by
  simp only [honestValues, show ¬(2 : Nat) < 2 by decide, ↓reduceIte]

theorem inner_group (slot : Fin 2) :
    (honestValues ajtai circuitKeys input output proof selected folded).slots
        ⟨3 + slot.val, by omega⟩ =
      if slot = selected then folded else input.running slot := by
  have bound := slot.isLt
  simp only [honestValues, show ¬3 + slot.val < 2 by omega, show ¬3 + slot.val = 2 by omega,
    show 3 + slot.val < 5 by omega, ↓reduceIte, Nat.add_sub_cancel_left]
  by_cases same : slot = selected
  · subst same
    simp
  · have notVal : slot.val ≠ selected.val := fun equal => same (Fin.ext equal)
    simp only [notVal, same, ↓reduceIte]
    exact congrArg input.running (LightAssemblerSource.finAt_eq _ slot _ (Nat.mod_eq_of_lt bound))

theorem priorDigest_group (slot : Fin 2) :
    (honestValues ajtai circuitKeys input output proof selected folded).digests
        ⟨slot.val, by omega⟩ = slotDigest (input.running slot) := by
  have bound := slot.isLt
  simp only [honestValues, show slot.val < 2 from bound, ↓reduceIte]
  exact congrArg (fun index => slotDigest (input.running index))
    (LightAssemblerSource.finAt_eq _ slot _ (Nat.mod_eq_of_lt bound))

theorem nextDigest_group (slot : Fin 2) :
    (honestValues ajtai circuitKeys input output proof selected folded).digests
        ⟨2 + slot.val, by omega⟩ = slotDigest (output.runningNext slot) := by
  have bound := slot.isLt
  simp only [honestValues, show ¬2 + slot.val < 2 by omega, show 2 + slot.val < 4 by omega,
    ↓reduceIte, Nat.add_sub_cancel_left]
  exact congrArg (fun index => slotDigest (output.runningNext index))
    (LightAssemblerSource.finAt_eq _ slot _ (Nat.mod_eq_of_lt bound))

end Groups

/-- `complete` with any values `front` below `base`: the assignment carries `front` there, since
the shared words lie at or above `base` (`SharedWordsAbove`). The exported relation puts its
public input below `base`. Every constraint of the circuit reads below its last local
(`StepFootprint.refresh_assembler_select_below` for the slot-select rows). -/
theorem completeWith (program : Stage1.Application.Program)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) (base : Nat)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
    (step : AppState → AppWitness → AppState) (control : AppState → AppWitness → Fin 2)
    (circuit : Fin 2) (refresh : BoundedDepth.circuitStep circuit = .refresh)
    (stepEq : program.step = step)
    (input : StepInput logicalWidth publicFits) (output : StepOutput logicalWidth publicFits)
    (proof : Spec.Folding.Refresh.Proof PaperAlgebra.Commitment productionShape 8)
    (selected : Fin 2) (folded : SlotClaim logicalWidth publicFits)
    (holds : StepHolds Folding.Schedule.productionLightFoldCap ajtai circuitKeys step control
      circuit input output)
    (valid : program.valid input.zi input.witness)
    (selectedPc : input.iteration ≠ 0 → input.priorPc = selected.val + 1)
    (freshPublic : input.fresh.publicInputs ⟨0, by decide⟩ =
      encHash (publicFits := publicFits) (stateHash (toState (Paper.priorHashPreimage
        ((setup Folding.Schedule.productionLightFoldCap ajtai circuitKeys).forCircuit circuit)
        input))))
    (proofEq : input.iteration ≠ 0 → input.nifsProof = .refresh proof)
    (accepted : OptionBFolds.folds .refresh (keys (circuitKeys selected).relation ajtai)
      (toClaim ((circuitKeys selected).system ajtai) (input.running selected).1) input.fresh
      (.refresh proof) = some (toClaim ((circuitKeys selected).system ajtai) folded))
    (iterationBound : input.iteration < goldilocksModulus)
    (priorPcBound : input.priorPc < goldilocksModulus)
    (priorDepth : ∀ slot, (input.running slot).2 < goldilocksModulus)
    (nextDepth : ∀ slot, (output.runningNext slot).2 < goldilocksModulus)
    (keyLength : ∀ key, (circuitKeys key).digest.length = 4)
    (z0Length : input.z0.length = 4) (ziLength : input.zi.length = 4)
    (zNextLength : output.zNext.length = 4)
    (witnessLength : input.witness.length = program.witnessWordCount) (front : Env) :
    ∃ env, (∀ column, column < base → env column = front column) ∧
      (∀ expression ∈ flatConstraints (Circuit.ops (RefreshStepLayout.main program relation circuit
        (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)) (sharedEnd base program.witnessWordCount)),
        expression.VarsBelow (sharedEnd base program.witnessWordCount + localLength (Circuit.ops
          (RefreshStepLayout.main program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)) (sharedEnd base program.witnessWordCount)))) ∧
      holdsFlat env (Circuit.ops (RefreshStepLayout.main program relation circuit
        (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)) (sharedEnd base program.witnessWordCount)) ∧
      RefreshStepSoundness.StepRepresents program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) ajtai circuitKeys circuit input output
        proof env := by
  obtain ⟨_, pcNext, zNextEq, xEq, cases⟩ := (stepHolds_iff _ _ _ _ _ _ _ _).mp holds
  have xLength : output.x.length = 4 := by
    rw [xEq]
    exact Spec.Poseidon2.hash_length _
  obtain ⟨foldedSlot, foldedSlotEq⟩ : ∃ slot : SlotData logicalWidth publicFits,
      slot = (folded, 0) := ⟨_, rfl⟩
  have foldedDepth : foldedSlot.2 < goldilocksModulus := by
    rw [foldedSlotEq]
    exact (by decide : (0 : Nat) < goldilocksModulus)
  have starts : base ≤ slotsStart base program.witnessWordCount ∧
      base ≤ digestsStart base program.witnessWordCount ∧
      base ≤ digitsStart base program.witnessWordCount ∧
      base ≤ sharedEnd base program.witnessWordCount := by
    simp only [sharedEnd, outputStart, roundsStart, digitsStart, childEvaluationsStart,
      childCommitmentsStart, LightAssembler.freshStart, digestsStart, slotsStart,
      LightAssembler.witnessStart]
    omega
  obtain ⟨slotsAbove, digestsAbove, digitsAbove, endAbove⟩ := starts
  obtain ⟨source, sourceEq⟩ : ∃ env : Env, env = SharedWordsAbove.withFront front
      (honestEnv ajtai circuitKeys input output proof selected foldedSlot base
        program.witnessWordCount) base := ⟨_, rfl⟩
  have starAbove : ∀ (target : Env) column, base ≤ column →
      SharedWordsAbove.withFront (honestEnv ajtai circuitKeys input output proof selected
        foldedSlot base program.witnessWordCount) target base column = target column :=
    fun _ _ above => SharedWordsAbove.withFront_above above
  have sourceAgree : SharedWordsAbove.Shared (honestEnv ajtai circuitKeys input output proof
      selected foldedSlot base program.witnessWordCount) source base
      (sharedEnd base program.witnessWordCount) := by
    intro column above _
    rw [sourceEq]
    exact SharedWordsAbove.withFront_above above
  have agreeOf : ∀ completed : Sequence.Prefix source (sharedEnd base program.witnessWordCount),
      SharedWordsAbove.Shared (honestEnv ajtai circuitKeys input output proof selected foldedSlot
        base program.witnessWordCount) completed.current base
        (sharedEnd base program.witnessWordCount) :=
    fun completed column above below => (completed.agrees column (Or.inl below)).trans
      (sourceAgree column above below)
  have repAt := fun (target : Env) (agree : SharedWordsAbove.Shared (honestEnv ajtai circuitKeys input
      output proof selected foldedSlot base program.witnessWordCount) target base
      (sharedEnd base program.witnessWordCount)) =>
    SharedWordsAbove.refresh_stepRepresents (starAbove target) program ajtai circuitKeys circuit
      input output proof
      (stepRepresents ajtai circuitKeys input output proof selected foldedSlot base _ program
        circuit (SharedWordsAbove.withFront_agree agree) iterationBound priorPcBound priorDepth
        nextDepth keyLength z0Length ziLength zNextLength witnessLength xLength pcNext proofEq)
  have activeAt : ∀ (target : Env) (agree : SharedWordsAbove.Shared (honestEnv ajtai circuitKeys input
      output proof selected foldedSlot base program.witnessWordCount) target base
      (sharedEnd base program.witnessWordCount)),
      (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).active.value target = input.running selected :=
    fun target agree => (SharedWordsAbove.slotAt_value_eq (starAbove target) (by omega)).trans
      ((target_slot_value ajtai circuitKeys input output proof selected
        foldedSlot base program.witnessWordCount _ (SharedWordsAbove.withFront_agree agree)
        ⟨2, by decide⟩ (by
          rw [active_group]
          exact priorDepth selected)).trans
        (active_group ajtai circuitKeys input output proof selected foldedSlot))
  have digestLength : ∀ block, ((honestValues ajtai circuitKeys input output proof selected
      foldedSlot).digests block).length = 4 := by
    intro block
    simp only [honestValues]
    split_ifs
    · exact Spec.Poseidon2.hash_length _
    · exact Spec.Poseidon2.hash_length _
    · exact xLength
  have digestLanes := fun (target : Env) (agree : SharedWordsAbove.Shared (honestEnv ajtai circuitKeys input
      output proof selected foldedSlot base program.witnessWordCount) target base
      (sharedEnd base program.witnessWordCount)) (block : Fin 5) (start : Nat)
      (startEq : start = digestsStart base program.witnessWordCount + 4 * block.val) =>
    show List.ofFn (fun lane : Fin 4 => target (start + lane.val)) =
        (honestValues ajtai circuitKeys input output proof selected foldedSlot).digests block by
      rw [← LightAssemblerSource.ofFn_getD 0 _ (digestLength block)]
      apply congrArg List.ofFn
      funext lane
      have blockBound := block.isLt
      have laneBound := lane.isLt
      rw [startEq, Nat.add_assoc]
      exact (starAbove target _ (by omega)).symm.trans
        ((target_digest ajtai circuitKeys input output proof selected foldedSlot base
          program.witnessWordCount _ (SharedWordsAbove.withFront_agree agree)
          (4 * block.val + lane.val) (by omega)).trans
          (LightAssemblerSource.quadColumn_eq _ _ block _ laneBound))
  have priorDigestWords : ∀ (target : Env) (agree : SharedWordsAbove.Shared (honestEnv ajtai circuitKeys input
      output proof selected foldedSlot base program.witnessWordCount) target base
      (sharedEnd base program.witnessWordCount)) (slot : Fin 2),
      List.ofFn (fun lane => ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).priorDigest slot lane).eval target) =
          slotDigest (input.running slot) := by
    intro target agree slot
    have lanes := digestLanes target agree ⟨slot.val, by omega⟩ _ rfl
    rw [priorDigest_group] at lanes
    exact lanes
  have nextDigestWords : ∀ (target : Env) (agree : SharedWordsAbove.Shared (honestEnv ajtai circuitKeys input
      output proof selected foldedSlot base program.witnessWordCount) target base
      (sharedEnd base program.witnessWordCount)) (slot : Fin 2),
      List.ofFn (fun lane => ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).nextDigest slot lane).eval target) =
          slotDigest (output.runningNext slot) := by
    intro target agree slot
    have lanes := digestLanes target agree ⟨2 + slot.val, by omega⟩
      (digestsStart base program.witnessWordCount + 8 + 4 * slot.val) (by simp only; omega)
    rw [nextDigest_group] at lanes
    exact lanes
  have rep0 := repAt source sourceAgree
  -- The step gadgets on the source values.
  have application : (applicationCircuit program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)).spec (sharedEnd base program.witnessWordCount) source := by
    apply (program.spec_iff _ _ _).mpr
    constructor
    · change List.ofFn (fun lane => ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).nextState lane).eval source) =
        program.step (List.ofFn fun lane => ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).currentState lane).eval source)
          (List.ofFn fun index => ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).witness index).eval source)
      rw [rep0.nextState, rep0.currentState, rep0.witness, stepEq]
      exact zNextEq
    · change program.valid (List.ofFn fun lane => ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).currentState lane).eval source)
          (List.ofFn fun index => ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).witness index).eval source)
      rw [rep0.currentState, rep0.witness]
      exact valid
  have priorSlotSpec : ∀ slot, OutputHash.SpecHolds (priorSlotInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) slot)
      (sharedEnd base program.witnessWordCount) source := by
    intro slot
    change List.ofFn (fun lane => ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).priorDigest slot lane).eval source) =
      Spec.Poseidon2.hash (Hash.evalList source (slotDomainChunk.map Expr.const ++
        ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).prior slot).words))
    rw [slotPreimage_eval, rep0.priorSlots slot]
    exact priorDigestWords source sourceAgree slot
  have nextSlotSpec : ∀ slot, OutputHash.SpecHolds (nextSlotInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) slot)
      (sharedEnd base program.witnessWordCount) source := by
    intro slot
    change List.ofFn (fun lane => ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).nextDigest slot lane).eval source) =
      Spec.Poseidon2.hash (Hash.evalList source (slotDomainChunk.map Expr.const ++
        ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).next slot).words))
    rw [slotPreimage_eval, rep0.nextSlots slot]
    exact nextDigestWords source sourceAgree slot
  have priorStateSpec : PriorStateHash.SpecHolds (priorStateInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)) (sharedEnd base program.witnessWordCount)
      source := by
    have publicEq := congrArg (fun fresh => fresh.publicInputs ⟨0, by decide⟩) rep0.fresh
    change (fun column => ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).fresh.publicInput ⟨0, by decide⟩ column).eval source) =
      PriorStateHash.encodedHash (Spec.Poseidon2.hash (Hash.evalList source
        (statePreimageExpr program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).iteration (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).currentState (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).priorPc
          (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).priorDigest)))
    refine publicEq.trans (freshPublic.trans ?_)
    rw [RefreshStepSoundness.statePreimageExpr_eval, rep0.iteration, rep0.initialState,
      rep0.currentState, rep0.priorPc, rep0.verifierKeys 0, rep0.verifierKeys 1,
      priorDigestWords source sourceAgree 0, priorDigestWords source sourceAgree 1]
    rfl
  have nextStateSpec : OutputHash.SpecHolds (nextStateInterface program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)) (sharedEnd base program.witnessWordCount)
      source := by
    change List.ofFn (fun lane => ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).outputDigest lane).eval source) =
      Spec.Poseidon2.hash (Hash.evalList source
        (statePreimageExpr program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).iteration + 1) (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).nextState
          (Expr.const (natWord (circuit.val + 1))) (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).nextDigest))
    rw [← rep0.outputDigest, xEq, RefreshStepSoundness.statePreimageExpr_eval, Expr.eval_hadd,
      eval_one, rep0.iteration, ← natWord_succ, rep0.initialState, rep0.nextState,
      rep0.verifierKeys 0, rep0.verifierKeys 1, Expr.eval_const, ← pcNext,
      nextDigestWords source sourceAgree 0, nextDigestWords source sourceAgree 1]
    rfl
  -- The accepted fold: Π_DEC on the slot, the refresh Π_CCS check, and the Π_RLC parent.
  have foldedEq := accepted
  obtain ⟨decAccepted, piccsAccepted, -⟩ :=
    (Spec.Folding.Refresh.verify_eq_some_iff _ _ _ _ _).mp accepted
  -- Fill the gadgets.
  have below := inputsBelow (logicalWidth := logicalWidth) (publicFits := publicFits) base
    program.witnessWordCount
  obtain ⟨p1, ops1, end1, preserves1⟩ := RefreshStepCompleteness.completeGadgetsFrom program
    circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount) relation (Sequence.empty source (sharedEnd base program.witnessWordCount)) rfl below application
    priorSlotSpec priorStateSpec nextSlotSpec nextStateSpec
  -- Today's Π_DEC: the accepted fold and the honest digit words give its phase.
  have rep1 := repAt p1.current (agreeOf p1)
  have active1 := activeAt p1.current (agreeOf p1)
  have attemptEq := RefreshFold.decAttempt_eq relation ajtai (input.running selected).1
    proof.split (decInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)) (decOffset program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) p1.current
    (RefreshStepSoundness.decWired program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) ajtai circuitKeys circuit input output proof
      p1.current rep1 selected active1 (sharedEnd base program.witnessWordCount))
  have digitWord : ∀ child column,
      ((decInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)).digit (decOffset program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) child
        column).eval p1.current =
        (PaperAlgebra.publicInputSplit ajtai).split (input.running selected).1.publicInput child
          column := by
    intro child column
    exact digit_eval p1.current _ (fun word bound =>
      (starAbove p1.current _ (by omega)).symm.trans (target_digit ajtai circuitKeys input output
        proof selected foldedSlot base program.witnessWordCount _
        (SharedWordsAbove.withFront_agree (agreeOf p1)) word bound))
      (childOf child) column
  have decPhase : Lifecycle.PiDEC.v1_2.Semantics.PhaseHolds relation ajtai
      (decInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)) (decOffset program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) p1.current := by
    have outputEq : Lifecycle.PiDEC.v1_2.Semantics.output relation (decInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount))
        (decOffset program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) p1.current =
        Spec.Folding.PiDEC.PaperVerifier.children (PaperAlgebra.publicInputSplit ajtai)
          (Lifecycle.PiDEC.v1_2.Semantics.inputAttempt relation (decInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount))
            (decOffset program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) p1.current) := by
      have parentPublic : (Lifecycle.PiDEC.v1_2.Semantics.inputAttempt relation
          (decInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)) (decOffset program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))
          p1.current).parent.publicInput = (input.running selected).1.publicInput :=
        funext (RefreshStepSoundness.decWired program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) ajtai circuitKeys circuit input output
          proof p1.current rep1 selected active1 (sharedEnd base program.witnessWordCount)).parentPublicInput
      funext child
      refine output_ext ?_ ?_ ?_ ?_ ?_ ?_
      · rfl
      · rfl
      · change (fun column => ((decInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)).digit
            (decOffset program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) child column).eval p1.current) =
          (PaperAlgebra.publicInputSplit ajtai).split
            (Lifecycle.PiDEC.v1_2.Semantics.inputAttempt relation (decInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount))
              (decOffset program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) p1.current).parent.publicInput child
        rw [parentPublic]
        funext column
        exact digitWord child column
      · rfl
      · rfl
      · rfl
    refine ⟨?_, ?_⟩
    · rw [outputEq]
      rfl
    · rw [outputEq]
      change Spec.Folding.PiDEC.PaperVerifier.Accepted _ _ _
        (Lifecycle.PiDEC.v1_2.Semantics.inputAttempt relation (decInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount))
          (decOffset program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) p1.current)
      rw [attemptEq]
      exact ⟨decAccepted.parentBounded, decAccepted.parentCombined,
        decAccepted.parentEvaluationSize, decAccepted.messageEvaluationSize,
        decAccepted.commitmentEquation, decAccepted.evaluationEquation⟩
  obtain ⟨p2, ops2, end2, preserves2⟩ := RefreshStepCompleteness.completeDecFrom program circuit
    (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount) relation ajtai p1 end1 below decPhase
  -- The refresh Π_CCS and today's Π_RLC.
  have piccsBelow := RefreshStepBounds.piccsBelow program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount) below
  obtain ⟨p3, ops3, end3, preserves3⟩ := RefreshPiCCSCompleteness.completeFrom relation
    (piccsInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)) (piccsOffset program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) ajtai
    (RefreshFold.children (circuitKeys selected).relation ajtai (input.running selected).1
      proof.split) input.fresh proof.nifs p2 end2 piccsBelow (circuitKeys selected).relation
    (RefreshStepSoundness.piccsRepresents program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) ajtai circuitKeys circuit input output
      proof p2.current (repAt p2.current (agreeOf p2)) selected (activeAt p2.current (agreeOf p2))
      (circuitKeys selected).relation _)
    piccsAccepted
  obtain ⟨p4, ops4, end4, preserves4, rlcPhase⟩ :=
    RefreshStepCompleteness.completeRlcPhaseFrom relation ajtai
      (rlcInterface program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))
      (rlcOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) p3 end3
      (RefreshStepBounds.rlcAssumptions program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount) below p3.current)
  -- The rows so far and the words on the filled values.
  have filledRows : holdsFlat p4.current
      (gadgetOps program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount) ++
        Circuit.ops (Lifecycle.PiDEC.v1_2.Formal.main relation (decInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)))
          (decOffset program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) ++
        RefreshPiCCSLayout.opsAt relation (piccsInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount))
          (piccsOffset program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) ++
        Circuit.ops (Lifecycle.PiRLC.v1_2.Formal.main relation
          (rlcInterface program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)))
          (rlcOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))) := by
    have rows := p4.rows
    rw [ops4, ops3, ops2, ops1] at rows
    simpa only [Sequence.empty, List.nil_append] using rows
  have rows4 := holdsFlat_implies_holds _ _ filledRows
  simp only [holds_append] at rows4
  obtain ⟨⟨⟨_, decRows⟩, piccsRows⟩, rlcRows⟩ := rows4
  have rep4 := repAt p4.current (agreeOf p4)
  have active4 := activeAt p4.current (agreeOf p4)
  -- The folded slot that the Π_RLC words carry is the honest fold result.
  have foldEq := RefreshFold.spec_implies_fold (circuitKeys selected).relation ajtai
    (input.running selected).1 input.fresh proof (decInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount))
    (decOffset program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) (piccsLeaves program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))
    (rlcInterface program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))
    (rlcOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) p4.current
    (RefreshStepLayout.decSpec_any (Lifecycle.PiDEC.v1_2.Formal.soundness relation
      (decInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)) (decOffset program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) p4.current
      (RefreshStepBounds.decAssumptions program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount) below p4.current)
      decRows))
    (RefreshStepSoundness.decWired program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) ajtai circuitKeys circuit input output proof
      p4.current rep4 selected active4 (sharedEnd base program.witnessWordCount))
    (RefreshPiCCSLayout.soundness relation (piccsOffset program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))
      (piccsInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)) p4.current
      (RefreshPiCCSBounds.assumptions relation _ _ piccsBelow p4.current) piccsRows
      (circuitKeys selected).relation)
    (RefreshPiCCSLayout.wired relation (piccsOffset program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))
      (circuitKeys selected).relation ajtai
      (RefreshFold.children (circuitKeys selected).relation ajtai (input.running selected).1
        proof.split) input.fresh proof.nifs (piccsInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)) p4.current
      (RefreshStepSoundness.piccsRepresents program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) ajtai circuitKeys circuit input output
        proof p4.current rep4 selected active4 (circuitKeys selected).relation
        (piccsOffset program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))))
    (RefreshStepLayout.rlcSpec_any (Lifecycle.PiRLC.v1_2.Formal.soundness relation
      (rlcInterface program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))
      (rlcOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) p4.current
      (RefreshStepBounds.rlcAssumptions program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount) below p4.current)
      rlcRows))
    (RefreshStepSoundness.rlcWired program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) ajtai circuitKeys circuit input output proof
      p4.current rep4 relation (sharedEnd base program.witnessWordCount))
  have nextEq : RefreshFold.nextSlot relation (rlcInterface program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))
      (rlcOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) p4.current = folded :=
    LightFold.toClaim_injective _ (Option.some.inj (foldEq.symm.trans foldedEq))
  -- The slot-select rows on the filled values.
  have counter4 := fun (word : Nat) (bound : word < 5) =>
    (starAbove p4.current (base + word) (Nat.le_add_right _ _)).symm.trans
      (target_counter ajtai circuitKeys input output proof selected foldedSlot base
        program.witnessWordCount _ (SharedWordsAbove.withFront_agree (agreeOf p4)) word bound)
  have inverseWord : (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).inverse.eval p4.current = Hint.inverse (natWord input.iteration) :=
    counter4 1 (by decide)
  have recursiveWord : (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).recursive.eval p4.current =
      if input.iteration = 0 then 0 else 1 := counter4 2 (by decide)
  have selectorWord : (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).selector.eval p4.current = natWord selected.val :=
    counter4 4 (by decide)
  have selectorValues : ∀ chosen : Fin 2,
      (natWord chosen.val = 0 ∨ natWord chosen.val = 1) ∧
        (if natWord chosen.val = 1 then (1 : Fin 2) else 0) = chosen := by decide
  have selectedEq : SlotSelect.selected (selectInterface program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))
      (selectOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) p4.current = selected := by
    unfold SlotSelect.selected
    change (if (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).selector.eval p4.current = 1 then (1 : Fin 2) else 0) = selected
    rw [selectorWord]
    exact (selectorValues selected).2
  have recursiveZero : (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).recursive.eval p4.current = 0 → input.iteration = 0 := by
    intro zero
    by_contra nonzero
    rw [recursiveWord] at zero
    simp only [nonzero, ↓reduceIte] at zero
    exact absurd zero (by decide)
  have recursiveOne : (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).recursive.eval p4.current = 1 → input.iteration ≠ 0 := by
    intro one zero
    rw [recursiveWord] at one
    simp only [zero, ↓reduceIte] at one
    exact absurd one (by decide)
  have recursiveFacts : input.iteration ≠ 0 →
      (output.runningNext selected).1 = folded ∧ (output.runningNext selected).2 = 0 ∧
        ∀ slot, slot ≠ selected → output.runningNext slot = input.running slot := by
    intro nonzero
    rcases cases with ⟨zero, _⟩ | ⟨priorPcValid, facts⟩
    · exact absurd zero nonzero
    · dsimp only at facts
      obtain ⟨_, _, _, foldsEq, depthEq, others⟩ := facts
      have indexEq : Paper.selectedIndex priorPcValid = selected :=
        Fin.ext (by simp [Paper.selectedIndex, selectedPc nonzero])
      rw [indexEq, refresh, proofEq nonzero] at foldsEq
      rw [indexEq, refresh] at depthEq
      rw [indexEq] at others
      exact ⟨LightFold.toClaim_injective _ (Option.some.inj (foldsEq.symm.trans foldedEq)),
        depthEq, others⟩
  have flags : ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).iteration.eval p4.current = 0 ∧ (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).recursive.eval p4.current = 0) ∨
      ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).iteration.eval p4.current * (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).inverse.eval p4.current = 1 ∧
        (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).recursive.eval p4.current = 1) := by
    rw [rep4.iteration, inverseWord, recursiveWord]
    by_cases zero : input.iteration = 0
    · left
      simp only [zero, ↓reduceIte, and_true]
      decide
    · right
      simp only [zero, ↓reduceIte, and_true]
      exact Gadgets.Range.CanonicalU64.mul_hintInverse_eq_one _
        (natWord_ne_zero iterationBound zero)
  have selectorBit : (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).selector.eval p4.current = 0 ∨ (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).selector.eval p4.current = 1 := by
    rw [selectorWord]
    exact (selectorValues selected).1
  have pcLink : (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).recursive.eval p4.current = 1 →
      (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).priorPc.eval p4.current = (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).selector.eval p4.current + 1 := by
    intro one
    rw [rep4.priorPc, selectorWord, selectedPc (recursiveOne one), natWord_succ]
  have baseState : (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).recursive.eval p4.current = 0 → ∀ index,
      ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).initialState index).eval p4.current = ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).currentState index).eval p4.current := by
    intro zero index
    rcases cases with ⟨_, sameState, _⟩ | ⟨_, facts⟩
    · exact congrFun (List.ofFn_injective
        (rep4.initialState.trans (sameState.trans rep4.currentState.symm))) index
    · dsimp only at facts
      exact absurd (recursiveZero zero) (Nat.pos_iff_ne_zero.mp facts.1)
  have groupWords := fun (slot : Fin 7) depth =>
    (SharedWordsAbove.slotAt_words_eq (fun column above => (starAbove p4.current column above).symm)
      (by omega)).trans (target_slot_words ajtai circuitKeys input output proof selected
        foldedSlot base program.witnessWordCount _ (SharedWordsAbove.withFront_agree (agreeOf p4))
        slot depth)
  have activeWords : Hash.evalList p4.current (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).active.words =
      Hash.evalList p4.current ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).prior (SlotSelect.selected
        (selectInterface program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))
        (selectOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) p4.current)).words := by
    rw [selectedEq, rep4.priorSlots selected]
    have words := groupWords ⟨2, by decide⟩ (by
      rw [active_group]
      exact priorDepth selected)
    rw [active_group] at words
    exact words
  have innerWords : ∀ slot : Fin 2, Hash.evalList p4.current ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).inner slot).words =
      serializeSlot (if slot = selected then foldedSlot else input.running slot) := by
    intro slot
    have words := groupWords ⟨3 + slot.val, by omega⟩ (by
      rw [inner_group]
      split_ifs
      · exact foldedDepth
      · exact priorDepth slot)
    rw [inner_group] at words
    exact words
  have newWords : Hash.evalList p4.current
      (newSlot program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)).words = serializeSlot foldedSlot := by
    rw [SlotExpr.words_eval, RefreshStepSoundness.newSlot_value, foldedSlotEq, nextEq]
  have innerSelected : Hash.evalList p4.current
      ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).inner (SlotSelect.selected (selectInterface program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))
        (selectOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) p4.current)).words =
        Hash.evalList p4.current (newSlot program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)).words := by
    rw [selectedEq, innerWords selected, newWords]
    simp only [↓reduceIte]
  have innerOther : ∀ slot, slot ≠ SlotSelect.selected
      (selectInterface program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))
      (selectOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) p4.current →
      Hash.evalList p4.current ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).inner slot).words =
        Hash.evalList p4.current ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).prior slot).words := by
    intro slot other
    rw [selectedEq] at other
    rw [innerWords slot, rep4.priorSlots slot]
    simp only [other, ↓reduceIte]
  have nextBase : (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).recursive.eval p4.current = 0 → ∀ slot,
      Hash.evalList p4.current ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).next slot).words =
        Hash.evalList p4.current (SlotSelect.defaultWords (logicalWidth := logicalWidth)
          (publicFits := publicFits)) := by
    intro zero slot
    rw [rep4.nextSlots slot, defaultWords_eval]
    rcases cases with ⟨_, _, sameNext⟩ | ⟨_, facts⟩
    · rw [sameNext]
    · dsimp only at facts
      exact absurd (recursiveZero zero) (Nat.pos_iff_ne_zero.mp facts.1)
  have nextRecursive : (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).recursive.eval p4.current = 1 → ∀ slot,
      Hash.evalList p4.current ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).next slot).words =
        Hash.evalList p4.current ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).inner slot).words := by
    intro one slot
    obtain ⟨claimEq, depthEq, others⟩ := recursiveFacts (recursiveOne one)
    rw [rep4.nextSlots slot, innerWords slot]
    by_cases same : slot = selected
    · subst same
      simp only [↓reduceIte, foldedSlotEq]
      exact congrArg serializeSlot (Prod.ext claimEq depthEq)
    · simp only [same, ↓reduceIte]
      rw [others slot same]
  have selectRows := SlotSelect.rows_of_select (selectInterface program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)
    (sharedEnd base program.witnessWordCount)) (selectOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) p4.current flags selectorBit pcLink
    baseState activeWords innerSelected innerOther nextBase nextRecursive
  have selectLength : localLength (Circuit.ops
      (selectCircuit program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)).main
      (selectOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))) = 0 :=
    StepFootprint.assertZero_localLength _
  have lengthEq : localLength (RefreshStepLayout.opsAt program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) =
      localLength p4.operations := by
    rw [ops4, ops3, ops2, ops1]
    unfold RefreshStepLayout.opsAt
    rw [Sequence.localLength_append, Sequence.localLength_singleton,
      Formal.CompletenessSupport.childOp_localLength]
    unfold Formal.childLength
    rw [selectLength, Nat.add_zero]
    simp only [Sequence.empty, List.nil_append]
    rfl
  refine ⟨p4.current, fun column below => ?_, ?_, ?_, rep4⟩
  · rw [p4.agrees column (Or.inl (by omega)), sourceEq]
    exact SharedWordsAbove.withFront_below below
  · change ∀ expression ∈ flatConstraints (RefreshStepLayout.opsAt program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)),
      expression.VarsBelow (sharedEnd base program.witnessWordCount + localLength (RefreshStepLayout.opsAt program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)))
    rw [lengthEq, end4]
    intro expression member
    unfold RefreshStepLayout.opsAt at member
    rw [flatConstraints_append, List.mem_append] at member
    rcases member with filled | select
    · have scope := p4.scope expression (by
        rw [ops4, ops3, ops2, ops1]
        simp only [Sequence.empty, List.nil_append]
        exact filled)
      rwa [end4] at scope
    · simp only [flatConstraints, List.flatMap_cons, List.flatMap_nil, List.append_nil] at select
      change expression ∈ flatConstraints ((SlotSelect.rows (selectInterface program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) (selectOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))).map Op.assertZero) at select
      simp only [flatConstraints, List.flatMap_map, List.mem_flatMap, Op.flatConstraints,
        List.mem_singleton] at select
      obtain ⟨original, inside, rfl⟩ := select
      exact StepFootprint.refresh_assembler_select_below program relation circuit base
        (le_of_le_of_eq (Nat.le_add_right _ _) end4) _ inside
  change holdsFlat p4.current (RefreshStepLayout.opsAt program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))
  unfold RefreshStepLayout.opsAt holdsFlat
  rw [flatConstraints_append]
  refine (constraintsHold_append _ _ _).mpr ⟨filledRows, ?_⟩
  intro row member
  simp only [flatConstraints, List.flatMap_cons, List.flatMap_nil, List.append_nil] at member
  change row ∈ flatConstraints ((SlotSelect.rows (selectInterface program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)
    (sharedEnd base program.witnessWordCount)) (selectOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))).map Op.assertZero) at member
  simp only [flatConstraints, List.flatMap_map, List.mem_flatMap, Op.flatConstraints,
    List.mem_singleton] at member
  obtain ⟨original, inside, rfl⟩ := member
  exact selectRows _ inside

/-- Completeness of the refresh step circuit at the source layout. -/
theorem complete (program : Stage1.Application.Program)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) (base : Nat)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
    (step : AppState → AppWitness → AppState) (control : AppState → AppWitness → Fin 2)
    (circuit : Fin 2) (refresh : BoundedDepth.circuitStep circuit = .refresh)
    (stepEq : program.step = step)
    (input : StepInput logicalWidth publicFits) (output : StepOutput logicalWidth publicFits)
    (proof : Spec.Folding.Refresh.Proof PaperAlgebra.Commitment productionShape 8)
    (selected : Fin 2) (folded : SlotClaim logicalWidth publicFits)
    (holds : StepHolds Folding.Schedule.productionLightFoldCap ajtai circuitKeys step control
      circuit input output)
    (valid : program.valid input.zi input.witness)
    (selectedPc : input.iteration ≠ 0 → input.priorPc = selected.val + 1)
    (freshPublic : input.fresh.publicInputs ⟨0, by decide⟩ =
      encHash (publicFits := publicFits) (stateHash (toState (Paper.priorHashPreimage
        ((setup Folding.Schedule.productionLightFoldCap ajtai circuitKeys).forCircuit circuit)
        input))))
    (proofEq : input.iteration ≠ 0 → input.nifsProof = .refresh proof)
    (accepted : OptionBFolds.folds .refresh (keys (circuitKeys selected).relation ajtai)
      (toClaim ((circuitKeys selected).system ajtai) (input.running selected).1) input.fresh
      (.refresh proof) = some (toClaim ((circuitKeys selected).system ajtai) folded))
    (iterationBound : input.iteration < goldilocksModulus)
    (priorPcBound : input.priorPc < goldilocksModulus)
    (priorDepth : ∀ slot, (input.running slot).2 < goldilocksModulus)
    (nextDepth : ∀ slot, (output.runningNext slot).2 < goldilocksModulus)
    (keyLength : ∀ key, (circuitKeys key).digest.length = 4)
    (z0Length : input.z0.length = 4) (ziLength : input.zi.length = 4)
    (zNextLength : output.zNext.length = 4)
    (witnessLength : input.witness.length = program.witnessWordCount) :
    ∃ env, holdsFlat env (Circuit.ops (RefreshStepLayout.main program relation circuit
        (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)) (sharedEnd base program.witnessWordCount)) ∧
      RefreshStepSoundness.StepRepresents program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) ajtai circuitKeys circuit input output
        proof env := by
  obtain ⟨env, _, _, rows, represents⟩ := completeWith program relation base ajtai circuitKeys step
    control circuit refresh stepEq input output proof selected folded holds valid selectedPc
    freshPublic proofEq accepted iterationBound priorPcBound priorDepth nextDepth keyLength
    z0Length ziLength zNextLength witnessLength (fun _ => 0)
  exact ⟨env, rows, represents⟩

end NightstreamFPrime.Layout.OptionB.RefreshAssemblerCompleteness
