import NightstreamFPrime.Layout.OptionB.RefreshAssemblerSource
import NightstreamFPrime.Layout.OptionB.RefreshStepCompleteness
import NightstreamFPrime.Layout.OptionB.RefreshPiCCSCompleteness

/-! Completeness of the refresh step circuit (CW-3.8). An honest refresh step (`StepHolds` with
the application's validity, the prior-state binding of the fresh instance, and a refresh proof
that the refresh fold accepts for the selected slot) has an assignment of the circuit at the
source layout: the honest source values (`RefreshAssemblerSource.honestValues`), the seven step
gadgets (`completeGadgetsFrom`), and the fold phase (`completeFoldFrom`). The fold phase fills
today's Π_DEC (`completeDecFrom`, whose phase the accepted fold and the honest digit words give),
the refresh Π_CCS (`RefreshPiCCSCompleteness.completeFrom`) and today's Π_RLC
(`completeRlcPhaseFrom`); then the slot-select rows hold (`SlotSelect.rows_of_select` from
`SelectSpec`), and the slot-select gadget's completeness gives the values. Every row holds, and
the words carry the step (`complete`). The finish step's completeness reuses the prior hashes'
specs (`priorSlot_spec`, `priorState_spec`) and the fold phase. -/

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


section Words

variable (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
  (input : StepInput logicalWidth publicFits) (output : StepOutput logicalWidth publicFits)
  (proof : Spec.Folding.Refresh.Proof PaperAlgebra.Commitment productionShape 8)
  (selected : Fin 2) (folded : SlotData logicalWidth publicFits)
  (program : Stage1.Application.Program) (base : Nat) (target : Env)
  (agree : ∀ column, column < sharedEnd base program.witnessWordCount →
    target column = honestEnv ajtai circuitKeys input output proof selected folded base
      program.witnessWordCount column)

include agree

private theorem digest_lanes (block : Fin 5) (start : Nat)
    (startEq : start = digestsStart base program.witnessWordCount + 4 * block.val)
    (length : ((honestValues ajtai circuitKeys input output proof selected folded).digests
      block).length = 4) :
    List.ofFn (fun lane : Fin 4 => target (start + lane.val)) =
      (honestValues ajtai circuitKeys input output proof selected folded).digests block := by
  rw [← LightAssemblerSource.ofFn_getD 0 _ length]
  apply congrArg List.ofFn
  funext lane
  have blockBound := block.isLt
  have laneBound := lane.isLt
  rw [startEq, Nat.add_assoc]
  exact (target_digest ajtai circuitKeys input output proof selected folded base
    program.witnessWordCount target agree (4 * block.val + lane.val) (by omega)).trans
    (LightAssemblerSource.quadColumn_eq _ _ block _ laneBound)

private theorem priorDigest_words (slot : Fin 2) :
    List.ofFn (fun lane => ((interface (logicalWidth := logicalWidth) (publicFits := publicFits)
        base program.witnessWordCount).priorDigest slot lane).eval target) =
      slotDigest (input.running slot) := by
  have lanes := digest_lanes ajtai circuitKeys input output proof selected folded program base
    target agree ⟨slot.val, by omega⟩ _ rfl (by
      rw [priorDigest_group]
      exact Spec.Poseidon2.hash_length _)
  rw [priorDigest_group] at lanes
  exact lanes

private theorem nextDigest_words (slot : Fin 2) :
    List.ofFn (fun lane => ((interface (logicalWidth := logicalWidth) (publicFits := publicFits)
        base program.witnessWordCount).nextDigest slot lane).eval target) =
      slotDigest (output.runningNext slot) := by
  have lanes := digest_lanes ajtai circuitKeys input output proof selected folded program base
    target agree ⟨2 + slot.val, by omega⟩
    (digestsStart base program.witnessWordCount + 8 + 4 * slot.val) (by simp only; omega) (by
      rw [nextDigest_group]
      exact Spec.Poseidon2.hash_length _)
  rw [nextDigest_group] at lanes
  exact lanes

/-- On values that agree with the honest source columns and carry the step, each prior slot
hash's spec holds. -/
theorem priorSlot_spec (circuit : Fin 2)
    (represents : RefreshStepSoundness.StepRepresents program
      (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
        program.witnessWordCount) ajtai circuitKeys circuit input output proof target)
    (offset : Nat) (slot : Fin 2) :
    OutputHash.SpecHolds (priorSlotInterface program (interface (logicalWidth := logicalWidth)
      (publicFits := publicFits) base program.witnessWordCount) slot) offset target := by
  change List.ofFn (fun lane => ((interface (logicalWidth := logicalWidth)
      (publicFits := publicFits) base program.witnessWordCount).priorDigest slot lane).eval
        target) =
    Spec.Poseidon2.hash (Hash.evalList target (slotDomainChunk.map Expr.const ++
      ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
        program.witnessWordCount).prior slot).words))
  rw [slotPreimage_eval, represents.priorSlots slot]
  exact priorDigest_words ajtai circuitKeys input output proof selected folded program base
    target agree slot

/-- On values that agree with the honest source columns and carry the step, the prior state
hash's spec holds when the fresh instance's public input is the hash of the prior state. -/
theorem priorState_spec (circuit : Fin 2)
    (represents : RefreshStepSoundness.StepRepresents program
      (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
        program.witnessWordCount) ajtai circuitKeys circuit input output proof target)
    (freshPublic : input.fresh.publicInputs ⟨0, by decide⟩ =
      encHash (publicFits := publicFits) (stateHash (toState (Paper.priorHashPreimage
        ((setup Folding.Schedule.productionLightFoldCap ajtai circuitKeys).forCircuit circuit)
        input))))
    (offset : Nat) :
    PriorStateHash.SpecHolds (priorStateInterface program
      (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
        program.witnessWordCount)) offset target := by
  have publicEq := congrArg (fun fresh => fresh.publicInputs ⟨0, by decide⟩) represents.fresh
  change (fun column => ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
      program.witnessWordCount).fresh.publicInput ⟨0, by decide⟩ column).eval target) =
    PriorStateHash.encodedHash (Spec.Poseidon2.hash (Hash.evalList target
      (statePreimageExpr program (interface (logicalWidth := logicalWidth)
          (publicFits := publicFits) base program.witnessWordCount)
        (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
          program.witnessWordCount).iteration
        (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
          program.witnessWordCount).currentState
        (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
          program.witnessWordCount).priorPc
        (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
          program.witnessWordCount).priorDigest)))
  refine publicEq.trans (freshPublic.trans ?_)
  rw [RefreshStepSoundness.statePreimageExpr_eval, represents.iteration, represents.initialState,
    represents.currentState, represents.priorPc, represents.verifierKeys 0,
    represents.verifierKeys 1,
    priorDigest_words ajtai circuitKeys input output proof selected folded program base target
      agree 0,
    priorDigest_words ajtai circuitKeys input output proof selected folded program base target
      agree 1]
  rfl

end Words

/-- The fold phase of the refresh step circuit at the source layout, which the finish circuit
also places. After a prefix that ends at today's Π_DEC offset, on values that agree with the
honest source columns below `sharedEnd`, an accepted refresh fold of the fresh instance into the
selected slot fills today's Π_DEC, the refresh Π_CCS and today's Π_RLC. When `SelectSpec` relates
the output to the input, the slot-select rows hold on the filled values, and the slot-select
gadget's completeness gives values on which they hold. These values keep the prefix's values
below today's Π_DEC offset, the prefix's rows and the phases' rows hold on them, and the words
carry the step. -/
theorem completeFoldFrom (program : Stage1.Application.Program)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) (base : Nat)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits) (circuit : Fin 2)
    (input : StepInput logicalWidth publicFits) (output : StepOutput logicalWidth publicFits)
    (proof : Spec.Folding.Refresh.Proof PaperAlgebra.Commitment productionShape 8)
    (selected : Fin 2) (folded : SlotData logicalWidth publicFits) (foldedDepth : folded.2 = 0)
    {initial : Env} {start : Nat} (before : Sequence.Prefix initial start)
    (startEq : start + localLength before.operations = decOffset program circuit
      (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
        program.witnessWordCount) (sharedEnd base program.witnessWordCount))
    (agree : ∀ column, column < sharedEnd base program.witnessWordCount →
      before.current column = honestEnv ajtai circuitKeys input output proof selected folded
        base program.witnessWordCount column)
    (select : SelectSpec input output selected folded.1 0)
    (accepted : OptionBFolds.folds .refresh (keys (circuitKeys selected).relation ajtai)
      (toClaim ((circuitKeys selected).system ajtai) (input.running selected).1) input.fresh
      (.refresh proof) = some (toClaim ((circuitKeys selected).system ajtai) folded.1))
    (iterationBound : input.iteration < goldilocksModulus)
    (priorPcBound : input.priorPc < goldilocksModulus)
    (priorDepth : ∀ slot, (input.running slot).2 < goldilocksModulus)
    (nextDepth : ∀ slot, (output.runningNext slot).2 < goldilocksModulus)
    (keyLength : ∀ key, (circuitKeys key).digest.length = 4)
    (z0Length : input.z0.length = 4) (ziLength : input.zi.length = 4)
    (zNextLength : output.zNext.length = 4)
    (witnessLength : input.witness.length = program.witnessWordCount)
    (xLength : output.x.length = 4) (pcNext : output.pcNext = circuit)
    (proofEq : input.iteration ≠ 0 → input.nifsProof = .refresh proof) :
    ∃ env, (∀ index, index < decOffset program circuit (interface (logicalWidth := logicalWidth)
          (publicFits := publicFits) base program.witnessWordCount)
          (sharedEnd base program.witnessWordCount) → env index = before.current index) ∧
      holdsFlat env (before.operations ++
        Circuit.ops (Lifecycle.PiDEC.v1_2.Formal.main relation (decInterface program
          (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
            program.witnessWordCount)))
          (decOffset program circuit (interface (logicalWidth := logicalWidth)
            (publicFits := publicFits) base program.witnessWordCount)
            (sharedEnd base program.witnessWordCount)) ++
        Circuit.ops (RefreshPiCCSLayout.main relation (piccsInterface program
          (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
            program.witnessWordCount)))
          (piccsOffset program circuit (interface (logicalWidth := logicalWidth)
            (publicFits := publicFits) base program.witnessWordCount)
            (sharedEnd base program.witnessWordCount)) ++
        Circuit.ops (Lifecycle.PiRLC.v1_2.Formal.main relation (rlcInterface program relation
          circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
            program.witnessWordCount) (sharedEnd base program.witnessWordCount)))
          (rlcOffset program relation circuit (interface (logicalWidth := logicalWidth)
            (publicFits := publicFits) base program.witnessWordCount)
            (sharedEnd base program.witnessWordCount)) ++
        [Formal.childOp "refresh.slot_select" (selectCircuit program relation circuit
            (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
              program.witnessWordCount) (sharedEnd base program.witnessWordCount))
          (selectOffset program relation circuit (interface (logicalWidth := logicalWidth)
            (publicFits := publicFits) base program.witnessWordCount)
            (sharedEnd base program.witnessWordCount))]) ∧
      RefreshStepSoundness.StepRepresents program (interface (logicalWidth := logicalWidth)
        (publicFits := publicFits) base program.witnessWordCount) ajtai circuitKeys circuit input
        output proof env := by
  have below := inputsBelow (logicalWidth := logicalWidth) (publicFits := publicFits) base
    program.witnessWordCount
  have sharedLe := RefreshStepBounds.dec_le program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)
  have agreeOf : ∀ completed : Sequence.Prefix initial start,
      Sequence.PreservesPrefix before completed → ∀ column,
        column < sharedEnd base program.witnessWordCount →
        completed.current column = honestEnv ajtai circuitKeys input output proof selected folded
          base program.witnessWordCount column :=
    fun completed preserves column bound =>
      (preserves.values column (by omega)).trans (agree column bound)
  have repAt := fun (target : Env) (agreeTarget : ∀ column, column < (sharedEnd base program.witnessWordCount) → target column =
      honestEnv ajtai circuitKeys input output proof selected folded base
        program.witnessWordCount column) =>
    stepRepresents ajtai circuitKeys input output proof selected folded base target program
      circuit agreeTarget iterationBound priorPcBound priorDepth nextDepth keyLength z0Length
      ziLength zNextLength witnessLength xLength pcNext proofEq
  have activeAt : ∀ (target : Env) (agreeTarget : ∀ column, column < (sharedEnd base program.witnessWordCount) → target column =
      honestEnv ajtai circuitKeys input output proof selected folded base
        program.witnessWordCount column),
      (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).active.value target = input.running selected :=
    fun target agreeTarget => (target_slot_value ajtai circuitKeys input output proof selected
      folded base program.witnessWordCount target agreeTarget ⟨2, by decide⟩ (by
        rw [active_group]
        exact priorDepth selected)).trans
      (active_group ajtai circuitKeys input output proof selected folded)
  -- The accepted fold: Π_DEC on the slot, the refresh Π_CCS check, and the Π_RLC parent.
  have foldedEq := accepted
  obtain ⟨decAccepted, piccsAccepted, -⟩ :=
    (Spec.Folding.Refresh.verify_eq_some_iff _ _ _ _ _).mp accepted
  -- Today's Π_DEC: the accepted fold and the honest digit words give its phase.
  have rep1 := repAt before.current agree
  have active1 := activeAt before.current agree
  have attemptEq := RefreshFold.decAttempt_eq relation ajtai (input.running selected).1
    proof.split (decInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)) (decOffset program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) before.current
    (RefreshStepSoundness.decWired program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) ajtai circuitKeys circuit input output proof
      before.current rep1 selected active1 (sharedEnd base program.witnessWordCount))
  have digitWord : ∀ child column,
      ((decInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)).digit (decOffset program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) child
        column).eval before.current =
        (PaperAlgebra.publicInputSplit ajtai).split (input.running selected).1.publicInput child
          column := by
    intro child column
    exact digit_eval before.current _ (fun word bound => target_digit ajtai circuitKeys input
      output proof selected folded base program.witnessWordCount before.current agree word bound)
      (childOf child) column
  have decPhase : Lifecycle.PiDEC.v1_2.Semantics.PhaseHolds relation ajtai
      (decInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)) (decOffset program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) before.current := by
    have outputEq : Lifecycle.PiDEC.v1_2.Semantics.output relation (decInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount))
        (decOffset program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) before.current =
        Spec.Folding.PiDEC.PaperVerifier.children (PaperAlgebra.publicInputSplit ajtai)
          (Lifecycle.PiDEC.v1_2.Semantics.inputAttempt relation (decInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount))
            (decOffset program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) before.current) := by
      have parentPublic : (Lifecycle.PiDEC.v1_2.Semantics.inputAttempt relation
          (decInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)) (decOffset program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))
          before.current).parent.publicInput = (input.running selected).1.publicInput :=
        funext (RefreshStepSoundness.decWired program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) ajtai circuitKeys circuit input output
          proof before.current rep1 selected active1 (sharedEnd base program.witnessWordCount)).parentPublicInput
      funext child
      refine output_ext ?_ ?_ ?_ ?_ ?_ ?_
      · rfl
      · rfl
      · change (fun column => ((decInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)).digit
            (decOffset program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) child column).eval before.current) =
          (PaperAlgebra.publicInputSplit ajtai).split
            (Lifecycle.PiDEC.v1_2.Semantics.inputAttempt relation (decInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount))
              (decOffset program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) before.current).parent.publicInput child
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
          (decOffset program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) before.current)
      rw [attemptEq]
      exact ⟨decAccepted.parentBounded, decAccepted.parentCombined,
        decAccepted.parentEvaluationSize, decAccepted.messageEvaluationSize,
        decAccepted.commitmentEquation, decAccepted.evaluationEquation⟩
  obtain ⟨p2, ops2, end2, preserves2⟩ := RefreshStepCompleteness.completeDecFrom program circuit
    (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount) relation ajtai before startEq below decPhase
  -- The refresh Π_CCS and today's Π_RLC.
  have piccsBelow := RefreshStepBounds.piccsBelow program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount) below
  obtain ⟨p3, ops3, end3, preserves3⟩ := RefreshPiCCSCompleteness.completeFrom relation
    (piccsInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)) (piccsOffset program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) ajtai
    (RefreshFold.children (circuitKeys selected).relation ajtai (input.running selected).1
      proof.split) input.fresh proof.nifs p2 end2 piccsBelow (circuitKeys selected).relation
    (RefreshStepSoundness.piccsRepresents program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) ajtai circuitKeys circuit input output
      proof p2.current (repAt p2.current (agreeOf p2 preserves2)) selected
      (activeAt p2.current (agreeOf p2 preserves2)) (circuitKeys selected).relation _)
    piccsAccepted
  have through3 := preserves2.trans preserves3
  obtain ⟨p4, ops4, end4, preserves4, -⟩ :=
    RefreshStepCompleteness.completeRlcPhaseFrom relation ajtai
      (rlcInterface program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))
      (rlcOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) p3 end3
      (RefreshStepBounds.rlcAssumptions program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount) below p3.current)
  have through4 := through3.trans preserves4
  -- The phases' rows on the filled values.
  have filledRows : holdsFlat p4.current
      (before.operations ++
        Circuit.ops (Lifecycle.PiDEC.v1_2.Formal.main relation (decInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)))
          (decOffset program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) ++
        RefreshPiCCSLayout.opsAt relation (piccsInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount))
          (piccsOffset program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) ++
        Circuit.ops (Lifecycle.PiRLC.v1_2.Formal.main relation
          (rlcInterface program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)))
          (rlcOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))) := by
    have rows := p4.rows
    rw [ops4, ops3, ops2] at rows
    exact rows
  have rows4 := holdsFlat_implies_holds _ _ filledRows
  simp only [holds_append] at rows4
  obtain ⟨⟨⟨_, decRows⟩, piccsRows⟩, rlcRows⟩ := rows4
  have rep4 := repAt p4.current (agreeOf p4 through4)
  have active4 := activeAt p4.current (agreeOf p4 through4)
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
      (rlcOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) p4.current = folded.1 :=
    LightFold.toClaim_injective _ (Option.some.inj (foldEq.symm.trans foldedEq))
  -- The slot-select rows on the filled values.
  have counter4 := fun (word : Nat) (bound : word < 5) => target_counter ajtai circuitKeys input
    output proof selected folded base program.witnessWordCount p4.current (agreeOf p4 through4)
    word bound
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
    rw [rep4.priorPc, selectorWord, (select.recursive (recursiveOne one)).1, natWord_succ]
  have baseState : (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).recursive.eval p4.current = 0 → ∀ index,
      ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).initialState index).eval p4.current = ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).currentState index).eval p4.current := by
    intro zero index
    exact congrFun (List.ofFn_injective (rep4.initialState.trans
      ((select.base (recursiveZero zero)).1.trans rep4.currentState.symm))) index
  have groupWords := fun (slot : Fin 7) depth => target_slot_words ajtai circuitKeys input output
    proof selected folded base program.witnessWordCount p4.current (agreeOf p4 through4) slot
    depth
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
      serializeSlot (if slot = selected then folded else input.running slot) := by
    intro slot
    have words := groupWords ⟨3 + slot.val, by omega⟩ (by
      rw [inner_group]
      split_ifs
      · rw [foldedDepth]
        exact (by decide : (0 : Nat) < goldilocksModulus)
      · exact priorDepth slot)
    rw [inner_group] at words
    exact words
  have newWords : Hash.evalList p4.current
      (newSlot program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)).words = serializeSlot folded := by
    rw [SlotExpr.words_eval, RefreshStepSoundness.newSlot_value]
    exact congrArg serializeSlot (Prod.ext nextEq foldedDepth.symm)
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
    rw [rep4.nextSlots slot, defaultWords_eval, (select.base (recursiveZero zero)).2]
  have nextRecursive : (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).recursive.eval p4.current = 1 → ∀ slot,
      Hash.evalList p4.current ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).next slot).words =
        Hash.evalList p4.current ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).inner slot).words := by
    intro one slot
    obtain ⟨_, claimEq, depthEq, others⟩ := select.recursive (recursiveOne one)
    rw [rep4.nextSlots slot, innerWords slot]
    by_cases same : slot = selected
    · subst same
      simp only [↓reduceIte]
      exact congrArg serializeSlot (Prod.ext claimEq (depthEq.trans foldedDepth.symm))
    · simp only [same, ↓reduceIte]
      rw [others slot same]
  have selectRows := SlotSelect.rows_of_select (selectInterface program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)
    (sharedEnd base program.witnessWordCount)) (selectOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) p4.current flags selectorBit pcLink
    baseState activeWords innerSelected innerOther nextBase nextRecursive
  -- The slot-select gadget fills its rows from its spec.
  obtain ⟨completed, selectAgrees, selectFlat⟩ := (selectCircuit program relation circuit
    (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
      program.witnessWordCount) (sharedEnd base program.witnessWordCount)).completeness p4.current
    (selectOffset program relation circuit (interface (logicalWidth := logicalWidth)
      (publicFits := publicFits) base program.witnessWordCount)
      (sharedEnd base program.witnessWordCount)) trivial selectRows
  have keep4 : ∀ index, index < start + localLength p4.operations →
      completed index = p4.current index := by
    intro index bound
    rw [end4] at bound
    exact selectAgrees index (Or.inl bound)
  have decLe : decOffset program circuit (interface (logicalWidth := logicalWidth)
      (publicFits := publicFits) base program.witnessWordCount)
      (sharedEnd base program.witnessWordCount) ≤ start + localLength p4.operations := by
    rw [← startEq]
    exact Nat.add_le_add_left through4.length_le start
  refine ⟨completed, fun index bound => (keep4 index (by omega)).trans
      (through4.values index (by omega)), ?_,
    repAt completed fun column bound => (keep4 column (by omega)).trans
      (agreeOf p4 through4 column bound)⟩
  unfold holdsFlat
  rw [flatConstraints_append]
  refine (constraintsHold_append _ _ _).mpr ⟨?_, ?_⟩
  · have rows := constraintsHold_of_agree_below p4.current completed _ _ p4.scope keep4 p4.rows
    rw [ops4, ops3, ops2] at rows
    exact rows
  · rw [flatConstraints_singleton]
    exact selectFlat

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
  obtain ⟨_, pcNext, zNextEq, xEq, cases⟩ := (stepHolds_iff _ _ _ _ _ _ _ _).mp holds
  have xLength : output.x.length = 4 := by
    rw [xEq]
    exact Spec.Poseidon2.hash_length _
  obtain ⟨foldedSlot, foldedSlotEq⟩ : ∃ slot : SlotData logicalWidth publicFits,
      slot = (folded, 0) := ⟨_, rfl⟩
  obtain ⟨source, sourceEq⟩ : ∃ env : Env, env = honestEnv ajtai circuitKeys input output proof
      selected foldedSlot base program.witnessWordCount := ⟨_, rfl⟩
  have sourceAgree : ∀ column, column < (sharedEnd base program.witnessWordCount) →
      source column = honestEnv ajtai circuitKeys input output proof selected foldedSlot base
        program.witnessWordCount column := fun column _ => congrFun sourceEq column
  have agreeOf : ∀ completed : Sequence.Prefix source (sharedEnd base program.witnessWordCount),
      ∀ column, column < (sharedEnd base program.witnessWordCount) →
        completed.current column = honestEnv ajtai circuitKeys input output proof selected
          foldedSlot base program.witnessWordCount column :=
    fun completed column bound => (completed.agrees column (Or.inl bound)).trans
      (congrFun sourceEq column)
  have rep0 := stepRepresents ajtai circuitKeys input output proof selected foldedSlot base source
    program circuit sourceAgree iterationBound priorPcBound priorDepth nextDepth keyLength z0Length
    ziLength zNextLength witnessLength xLength pcNext proofEq
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
  have nextSlotSpec : ∀ slot, OutputHash.SpecHolds (nextSlotInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) slot)
      (sharedEnd base program.witnessWordCount) source := by
    intro slot
    change List.ofFn (fun lane => ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).nextDigest slot lane).eval source) =
      Spec.Poseidon2.hash (Hash.evalList source (slotDomainChunk.map Expr.const ++
        ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).next slot).words))
    rw [slotPreimage_eval, rep0.nextSlots slot]
    exact nextDigest_words ajtai circuitKeys input output proof selected foldedSlot program base
      source sourceAgree slot
  have nextStateSpec : OutputHash.SpecHolds (nextStateInterface program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)) (sharedEnd base program.witnessWordCount)
      source := by
    change List.ofFn (fun lane => ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).outputDigest lane).eval source) =
      Spec.Poseidon2.hash (Hash.evalList source
        (statePreimageExpr program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).iteration + 1) (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).nextState
          (Expr.const (natWord (circuit.val + 1))) (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).nextDigest))
    rw [← rep0.outputDigest, xEq, RefreshStepSoundness.statePreimageExpr_eval, Expr.eval_hadd,
      eval_one, rep0.iteration, ← natWord_succ, rep0.initialState, rep0.nextState,
      rep0.verifierKeys 0, rep0.verifierKeys 1, Expr.eval_const, ← pcNext,
      nextDigest_words ajtai circuitKeys input output proof selected foldedSlot program base
        source sourceAgree 0,
      nextDigest_words ajtai circuitKeys input output proof selected foldedSlot program base
        source sourceAgree 1]
    rfl
  have below := inputsBelow (logicalWidth := logicalWidth) (publicFits := publicFits) base
    program.witnessWordCount
  obtain ⟨p1, ops1, end1, -⟩ := RefreshStepCompleteness.completeGadgetsFrom program
    circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount) relation (Sequence.empty source (sharedEnd base program.witnessWordCount)) rfl below application
    (priorSlot_spec ajtai circuitKeys input output proof selected foldedSlot program base source
      sourceAgree circuit rep0 _)
    (priorState_spec ajtai circuitKeys input output proof selected foldedSlot program base source
      sourceAgree circuit rep0 freshPublic _)
    nextSlotSpec nextStateSpec
  -- The step relation gives the slot select's spec for the selected slot and the fold's claim.
  have select : SelectSpec input output selected foldedSlot.1 0 := by
    rw [foldedSlotEq]
    refine ⟨fun zero => ?_, fun nonzero => ?_⟩
    · rcases cases with ⟨_, sameState, sameNext⟩ | ⟨_, facts⟩
      · exact ⟨sameState, sameNext⟩
      · dsimp only at facts
        exact absurd zero (Nat.pos_iff_ne_zero.mp facts.1)
    · rcases cases with ⟨zero, _⟩ | ⟨priorPcValid, facts⟩
      · exact absurd zero nonzero
      · dsimp only at facts
        obtain ⟨_, _, _, foldsEq, depthEq, others⟩ := facts
        have indexEq : Paper.selectedIndex priorPcValid = selected :=
          Fin.ext (by simp [Paper.selectedIndex, selectedPc nonzero])
        rw [indexEq, refresh, proofEq nonzero] at foldsEq
        rw [indexEq, refresh] at depthEq
        rw [indexEq] at others
        exact ⟨selectedPc nonzero,
          LightFold.toClaim_injective _ (Option.some.inj (foldsEq.symm.trans accepted)), depthEq,
          others⟩
  -- The fold phase and the slot select.
  obtain ⟨env, -, rows, rep⟩ := completeFoldFrom program relation base ajtai circuitKeys circuit
    input output proof selected foldedSlot (by rw [foldedSlotEq]) p1 end1 (agreeOf p1) select
    (by rw [foldedSlotEq]; exact accepted) iterationBound priorPcBound priorDepth nextDepth
    keyLength z0Length ziLength zNextLength witnessLength xLength pcNext proofEq
  refine ⟨env, ?_, rep⟩
  change holdsFlat env (RefreshStepLayout.opsAt program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))
  unfold RefreshStepLayout.opsAt
  rw [ops1] at rows
  simpa only [Sequence.empty, List.nil_append] using rows

end NightstreamFPrime.Layout.OptionB.RefreshAssemblerCompleteness
