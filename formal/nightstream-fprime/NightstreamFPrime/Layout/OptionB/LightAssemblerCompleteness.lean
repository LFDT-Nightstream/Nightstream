import NightstreamFPrime.Layout.OptionB.LightAssemblerSource
import NightstreamFPrime.Layout.OptionB.LightStepCompleteness
import NightstreamFPrime.Layout.OptionB.LightPiCCSCompleteness
import NightstreamFPrime.Layout.OptionB.LightRlcCompleteness

/-! Completeness of the light step circuit (assembler A3d-2). An honest light step (`StepHolds`
with the application's validity, the prior-state binding of the fresh instance, a light proof
that the light fold accepts for the selected slot, and an active depth below the cap) has an
assignment of the circuit at the source layout: the honest source values
(`LightAssemblerSource.honestValues`), then the nine step gadgets (`completeGadgetsFrom`), the
light Π_CCS (`LightPiCCSCompleteness.completeFrom`), the light Π_RLC
(`LightRlcCompleteness.completeFrom`), and the slot-select rows (`SlotSelect.rows_of_select`).
Every row holds, and the words carry the step (`complete`). -/

namespace NightstreamFPrime.Layout.OptionB.LightAssemblerCompleteness

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
open NightstreamFPrime.Lifecycle.OptionB.LightStepLayout
open NightstreamFPrime.Layout.OptionB.LightAssembler
open NightstreamFPrime.Layout.OptionB.LightAssemblerSource

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

section Facts

variable (program : Stage1.Application.Program)
  (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
  (input : StepInput logicalWidth publicFits) (output : StepOutput logicalWidth publicFits)
  (proof : Spec.Folding.Light.Proof LightShape 8) (selected : Fin 2)
  (folded : SlotData logicalWidth publicFits) (base : Nat) (target : Env)
  (agree : ∀ column, column < sharedEnd base program.witnessWordCount →
    target column = honestEnv circuitKeys input output proof selected folded base
      program.witnessWordCount column)

include agree

theorem counter_eval (word : Nat) (bound : word < 5) :
    target (base + word) =
      (honestValues circuitKeys input output proof selected folded).counters.getD word 0 :=
  target_counter circuitKeys input output proof selected folded base program.witnessWordCount
    target agree word bound

theorem digest_lanes (block : Fin 5) (start : Nat)
    (startEq : start = digestsStart base program.witnessWordCount + 4 * block.val)
    (length : ((honestValues circuitKeys input output proof selected folded).digests block).length =
      4) :
    List.ofFn (fun lane : Fin 4 => target (start + lane.val)) =
      (honestValues circuitKeys input output proof selected folded).digests block := by
  rw [← ofFn_getD 0 _ length]
  apply congrArg List.ofFn
  funext lane
  have blockBound := block.isLt
  have laneBound := lane.isLt
  rw [startEq, Nat.add_assoc]
  exact (target_digest circuitKeys input output proof selected folded base
    program.witnessWordCount target agree (4 * block.val + lane.val) (by omega)).trans
    (quadColumn_eq _ _ block _ laneBound)

theorem group_words (slot : Fin 7)
    (depth : ((honestValues circuitKeys input output proof selected folded).slots slot).2 <
      goldilocksModulus) :
    Hash.evalList target (slotAt (logicalWidth := logicalWidth) (publicFits := publicFits)
        (slotsStart base program.witnessWordCount + slot.val * slotWordCount)).words =
      serializeSlot ((honestValues circuitKeys input output proof selected folded).slots slot) :=
  target_slot_words circuitKeys input output proof selected folded base program.witnessWordCount
    target agree slot depth

theorem group_value (slot : Fin 7)
    (depth : ((honestValues circuitKeys input output proof selected folded).slots slot).2 <
      goldilocksModulus) :
    (slotAt (logicalWidth := logicalWidth) (publicFits := publicFits)
        (slotsStart base program.witnessWordCount + slot.val * slotWordCount)).value target =
      (honestValues circuitKeys input output proof selected folded).slots slot :=
  target_slot_value circuitKeys input output proof selected folded base program.witnessWordCount
    target agree slot depth

end Facts

section Groups

variable (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
  (input : StepInput logicalWidth publicFits) (output : StepOutput logicalWidth publicFits)
  (proof : Spec.Folding.Light.Proof LightShape 8) (selected : Fin 2)
  (folded : SlotData logicalWidth publicFits)

theorem active_group :
    (honestValues circuitKeys input output proof selected folded).slots ⟨2, by decide⟩ =
      input.running selected := by
  simp only [honestValues, show ¬(2 : Nat) < 2 by decide,
    ↓reduceIte]

theorem inner_group (slot : Fin 2) :
    (honestValues circuitKeys input output proof selected folded).slots ⟨3 + slot.val, by omega⟩ =
      if slot = selected then folded else input.running slot := by
  have bound := slot.isLt
  simp only [honestValues, show ¬3 + slot.val < 2 by omega, show ¬3 + slot.val = 2 by omega,
    show 3 + slot.val < 5 by omega, ↓reduceIte, Nat.add_sub_cancel_left]
  by_cases same : slot = selected
  · simp only [same, ↓reduceIte]
  · have different : slot.val ≠ selected.val := fun equal => same (Fin.ext equal)
    simp only [same, different, ↓reduceIte]
    exact congrArg input.running (finAt_eq _ slot _ (Nat.mod_eq_of_lt bound))

theorem priorDigest_group (slot : Fin 2) :
    (honestValues circuitKeys input output proof selected folded).digests ⟨slot.val, by omega⟩ =
      slotDigest (input.running slot) := by
  have bound := slot.isLt
  simp only [honestValues, show slot.val < 2 from bound, ↓reduceIte]
  exact congrArg (fun index => slotDigest (input.running index))
    (finAt_eq _ slot _ (Nat.mod_eq_of_lt bound))

theorem nextDigest_group (slot : Fin 2) :
    (honestValues circuitKeys input output proof selected folded).digests ⟨2 + slot.val, by omega⟩ =
      slotDigest (output.runningNext slot) := by
  have bound := slot.isLt
  simp only [honestValues, show ¬2 + slot.val < 2 by omega, show 2 + slot.val < 4 by omega,
    ↓reduceIte, Nat.add_sub_cancel_left]
  exact congrArg (fun index => slotDigest (output.runningNext index))
    (finAt_eq _ slot _ (Nat.mod_eq_of_lt bound))

end Groups

private theorem eval_one (env : Env) : (1 : Expr).eval env = 1 := rfl

private theorem flatConstraints_append (left right : List Op) :
    flatConstraints (left ++ right) = flatConstraints left ++ flatConstraints right := by
  simp [flatConstraints, List.flatMap_append]

/-- Completeness of the light step circuit at the source layout: an honest light step has an
assignment where every row holds and the words carry the step. The honest step is `StepHolds`
with the application's validity predicate, the fresh instance's binding to the prior state
(which the circuit checks at every iteration), a light proof that the light fold accepts for the
selected slot with result `folded`, and an active depth below the cap. -/
theorem complete (program : Stage1.Application.Program)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) (base : Nat)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
    (step : AppState → AppWitness → AppState) (control : AppState → AppWitness → Fin 2)
    (circuit : Fin 2) (light : BoundedDepth.circuitStep circuit = .light)
    (stepEq : program.step = step)
    (input : StepInput logicalWidth publicFits) (output : StepOutput logicalWidth publicFits)
    (proof : Spec.Folding.Light.Proof LightShape 8) (selected : Fin 2)
    (folded : SlotClaim logicalWidth publicFits)
    (holds : StepHolds Folding.Schedule.productionLightFoldCap ajtai circuitKeys step control
      circuit input output)
    (valid : program.valid input.zi input.witness)
    (selectedPc : input.iteration ≠ 0 → input.priorPc = selected.val + 1)
    (freshPublic : input.fresh.publicInputs ⟨0, by decide⟩ =
      encHash (publicFits := publicFits) (stateHash (toState (Paper.priorHashPreimage
        ((setup Folding.Schedule.productionLightFoldCap ajtai circuitKeys).forCircuit circuit)
        input))))
    (proofEq : input.iteration ≠ 0 → input.nifsProof = .light proof)
    (accepted : OptionBFolds.folds .light (keys (circuitKeys selected).relation ajtai)
      (toClaim ((circuitKeys selected).system ajtai) (input.running selected).1) input.fresh
      (.light proof) = some (toClaim ((circuitKeys selected).system ajtai) folded))
    (depthSmall : (input.running selected).2 < Folding.Schedule.productionLightFoldCap)
    (iterationBound : input.iteration < goldilocksModulus)
    (priorPcBound : input.priorPc < goldilocksModulus)
    (priorDepth : ∀ slot, (input.running slot).2 < goldilocksModulus)
    (nextDepth : ∀ slot, (output.runningNext slot).2 < goldilocksModulus)
    (keyLength : ∀ key, (circuitKeys key).digest.length = 4)
    (z0Length : input.z0.length = 4) (ziLength : input.zi.length = 4)
    (zNextLength : output.zNext.length = 4)
    (witnessLength : input.witness.length = program.witnessWordCount) :
    ∃ env, holdsFlat env (Circuit.ops (LightStepLayout.main program relation circuit
        (interface base program.witnessWordCount)) (sharedEnd base program.witnessWordCount)) ∧
      LightStepSoundness.StepRepresents program (interface base program.witnessWordCount) ajtai
        circuitKeys circuit input output proof env := by
  obtain ⟨_, pcNext, zNextEq, xEq, cases⟩ := (stepHolds_iff _ _ _ _ _ _ _ _).mp holds
  have xLength : output.x.length = 4 := by
    rw [xEq]
    exact Spec.Poseidon2.hash_length _
  obtain ⟨foldedSlot, foldedSlotEq⟩ : ∃ slot : SlotData logicalWidth publicFits,
      slot = (folded, (input.running selected).2 + 1) := ⟨_, rfl⟩
  have foldedDepth : foldedSlot.2 < goldilocksModulus := by
    rw [foldedSlotEq]
    have cap := depthSmall
    simp only [Folding.Schedule.productionLightFoldCap] at cap
    have modulus : 65 ≤ goldilocksModulus := by decide
    show (input.running selected).2 + 1 < goldilocksModulus
    omega
  obtain ⟨source, sourceEq⟩ : ∃ env : Env, env = honestEnv circuitKeys input output proof
      selected foldedSlot base program.witnessWordCount := ⟨_, rfl⟩
  have sourceAgree : ∀ column, column < sharedEnd base program.witnessWordCount →
      source column = honestEnv circuitKeys input output proof selected foldedSlot base
        program.witnessWordCount column := fun column _ => congrFun sourceEq column
  have agreeOf : ∀ completed : Sequence.Prefix source (sharedEnd base program.witnessWordCount),
      ∀ column, column < sharedEnd base program.witnessWordCount →
        completed.current column = honestEnv circuitKeys input output proof selected foldedSlot
          base program.witnessWordCount column :=
    fun completed column bound => (completed.agrees column (Or.inl bound)).trans
      (congrFun sourceEq column)
  have repAt := fun (target : Env) (agree : ∀ column,
      column < sharedEnd base program.witnessWordCount → target column =
        honestEnv circuitKeys input output proof selected foldedSlot base program.witnessWordCount
          column) =>
    stepRepresents circuitKeys input output proof selected foldedSlot base target program ajtai
      circuit agree iterationBound priorPcBound priorDepth nextDepth keyLength z0Length ziLength
      zNextLength witnessLength xLength pcNext proofEq
  have activeAt : ∀ (target : Env) (agree : ∀ column,
      column < sharedEnd base program.witnessWordCount → target column =
        honestEnv circuitKeys input output proof selected foldedSlot base program.witnessWordCount
          column),
      (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
        program.witnessWordCount).active.value target = input.running selected :=
    fun target agree => (group_value program circuitKeys input output proof selected foldedSlot
      base target agree ⟨2, by decide⟩ (by
        rw [active_group]
        exact priorDepth selected)).trans
      (active_group circuitKeys input output proof selected foldedSlot)
  have digestLength : ∀ block, ((honestValues circuitKeys input output proof selected
      foldedSlot).digests block).length = 4 := by
    intro block
    simp only [honestValues]
    split_ifs
    · exact Spec.Poseidon2.hash_length _
    · exact Spec.Poseidon2.hash_length _
    · exact xLength
  have priorDigestWords : ∀ (target : Env) (agree : ∀ column,
      column < sharedEnd base program.witnessWordCount → target column =
        honestEnv circuitKeys input output proof selected foldedSlot base program.witnessWordCount
          column) (slot : Fin 2),
      List.ofFn (fun lane => ((interface (logicalWidth := logicalWidth) (publicFits := publicFits)
        base program.witnessWordCount).priorDigest slot lane).eval target) =
          slotDigest (input.running slot) := by
    intro target agree slot
    have lanes := digest_lanes program circuitKeys input output proof selected foldedSlot base
      target agree ⟨slot.val, by omega⟩ _ rfl (digestLength _)
    rw [priorDigest_group] at lanes
    exact lanes
  have nextDigestWords : ∀ (target : Env) (agree : ∀ column,
      column < sharedEnd base program.witnessWordCount → target column =
        honestEnv circuitKeys input output proof selected foldedSlot base program.witnessWordCount
          column) (slot : Fin 2),
      List.ofFn (fun lane => ((interface (logicalWidth := logicalWidth) (publicFits := publicFits)
        base program.witnessWordCount).nextDigest slot lane).eval target) =
          slotDigest (output.runningNext slot) := by
    intro target agree slot
    have lanes := digest_lanes program circuitKeys input output proof selected foldedSlot base
      target agree ⟨2 + slot.val, by omega⟩ (digestsStart base program.witnessWordCount + 8 +
        4 * slot.val) (by simp only; omega) (digestLength _)
    rw [nextDigest_group] at lanes
    exact lanes
  have rep0 := repAt source sourceAgree
  -- The step gadgets on the source values.
  have application : (applicationCircuit program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)).spec
      (sharedEnd base program.witnessWordCount) source := by
    apply (program.spec_iff _ _ _).mpr
    constructor
    · change List.ofFn (fun lane => ((interface (logicalWidth := logicalWidth)
          (publicFits := publicFits) base program.witnessWordCount).nextState lane).eval source) =
        program.step (List.ofFn fun lane => ((interface (logicalWidth := logicalWidth)
          (publicFits := publicFits) base program.witnessWordCount).currentState lane).eval
            source)
          (List.ofFn fun index => ((interface (logicalWidth := logicalWidth)
            (publicFits := publicFits) base program.witnessWordCount).witness index).eval source)
      rw [rep0.nextState, rep0.currentState, rep0.witness, stepEq]
      exact zNextEq
    · change program.valid (List.ofFn fun lane => ((interface (logicalWidth := logicalWidth)
          (publicFits := publicFits) base program.witnessWordCount).currentState lane).eval
            source)
          (List.ofFn fun index => ((interface (logicalWidth := logicalWidth)
            (publicFits := publicFits) base program.witnessWordCount).witness index).eval source)
      rw [rep0.currentState, rep0.witness]
      exact valid
  have depth0 : ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
      program.witnessWordCount).active.depth.eval source).val <
        Folding.Schedule.productionLightFoldCap := by
    have depthEq := congrArg Prod.snd (activeAt source sourceAgree)
    change ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
      program.witnessWordCount).active.value source).2 < _
    rw [depthEq]
    exact depthSmall
  have priorSlotSpec : ∀ slot, OutputHash.SpecHolds
      (priorSlotInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) slot)
      (sharedEnd base program.witnessWordCount) source := by
    intro slot
    change List.ofFn (fun lane => ((interface (logicalWidth := logicalWidth)
        (publicFits := publicFits) base program.witnessWordCount).priorDigest slot lane).eval
          source) =
      Spec.Poseidon2.hash (Hash.evalList source (slotDomainChunk.map Expr.const ++
        ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
          program.witnessWordCount).prior slot).words))
    rw [slotPreimage_eval, rep0.priorSlots slot]
    exact priorDigestWords source sourceAgree slot
  have nextSlotSpec : ∀ slot, OutputHash.SpecHolds
      (nextSlotInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) slot)
      (sharedEnd base program.witnessWordCount) source := by
    intro slot
    change List.ofFn (fun lane => ((interface (logicalWidth := logicalWidth)
        (publicFits := publicFits) base program.witnessWordCount).nextDigest slot lane).eval
          source) =
      Spec.Poseidon2.hash (Hash.evalList source (slotDomainChunk.map Expr.const ++
        ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
          program.witnessWordCount).next slot).words))
    rw [slotPreimage_eval, rep0.nextSlots slot]
    exact nextDigestWords source sourceAgree slot
  have priorStateSpec : PriorStateHash.SpecHolds
      (priorStateInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount))
      (sharedEnd base program.witnessWordCount) source := by
    have publicEq := congrArg (fun fresh => fresh.publicInputs ⟨0, by decide⟩) rep0.fresh
    change (fun column => ((interface (logicalWidth := logicalWidth) (publicFits := publicFits)
        base program.witnessWordCount).fresh.publicInput ⟨0, by decide⟩ column).eval source) =
      PriorStateHash.encodedHash (Spec.Poseidon2.hash (Hash.evalList source
        (statePreimageExpr program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)
          (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
            program.witnessWordCount).iteration
          (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
            program.witnessWordCount).currentState
          (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
            program.witnessWordCount).priorPc
          (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
            program.witnessWordCount).priorDigest)))
    refine publicEq.trans (freshPublic.trans ?_)
    rw [LightStepSoundness.statePreimageExpr_eval, rep0.iteration, rep0.initialState,
      rep0.currentState, rep0.priorPc, rep0.verifierKeys 0, rep0.verifierKeys 1,
      priorDigestWords source sourceAgree 0, priorDigestWords source sourceAgree 1]
    rfl
  have nextStateSpec : OutputHash.SpecHolds
      (nextStateInterface program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount))
      (sharedEnd base program.witnessWordCount) source := by
    change List.ofFn (fun lane => ((interface (logicalWidth := logicalWidth)
        (publicFits := publicFits) base program.witnessWordCount).outputDigest lane).eval
          source) =
      Spec.Poseidon2.hash (Hash.evalList source
        (statePreimageExpr program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)
          ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
            program.witnessWordCount).iteration + 1)
          (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
            program.witnessWordCount).nextState
          (Expr.const (natWord (circuit.val + 1)))
          (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
            program.witnessWordCount).nextDigest))
    rw [← rep0.outputDigest, xEq, LightStepSoundness.statePreimageExpr_eval, Expr.eval_hadd,
      eval_one, rep0.iteration, ← natWord_succ, rep0.initialState, rep0.nextState,
      rep0.verifierKeys 0, rep0.verifierKeys 1, Expr.eval_const, ← pcNext,
      nextDigestWords source sourceAgree 0, nextDigestWords source sourceAgree 1]
    rfl
  -- Fill the gadgets, the light Π_CCS and the light Π_RLC.
  have below := inputsBelow (logicalWidth := logicalWidth) (publicFits := publicFits) base
    program.witnessWordCount
  obtain ⟨p1, ops1, end1, preserves1⟩ := LightStepCompleteness.completeGadgetsFrom program circuit
    (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount) relation
    (Sequence.empty source (sharedEnd base program.witnessWordCount)) rfl below application
    depth0 priorSlotSpec priorStateSpec nextSlotSpec nextStateSpec
  have piccsBelow := LightStepBounds.piccsBelow program circuit
    (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount) below
  have rlcBelow := LightStepBounds.rlcBelow program relation circuit
    (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount) below
  obtain ⟨p2, ops2, end2, preserves2⟩ := LightPiCCSCompleteness.completeFrom relation
    (piccsInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount))
    (piccsOffset program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)
      (sharedEnd base program.witnessWordCount)) ajtai
    (LightFold.lightRunning (input.running selected).1) (OptionBFolds.freshToLight input.fresh)
    proof p1 end1 piccsBelow (circuitKeys selected).relation
    (LightStepSoundness.piccsRepresents program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) ajtai
      circuitKeys circuit input output proof p1.current (repAt p1.current (agreeOf p1)) selected
      (activeAt p1.current (agreeOf p1)) _)
    (LightFold.check_of_folds (circuitKeys selected).relation ajtai _ _ input.fresh proof _
      accepted)
  obtain ⟨p3, ops3, end3, preserves3⟩ := LightRlcCompleteness.completeFrom
    (piccsLeaves program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)
      (sharedEnd base program.witnessWordCount))
    (rlcInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount))
    (rlcOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)
      (sharedEnd base program.witnessWordCount)) p2 end2 rlcBelow
  -- The rows so far and the words on the filled values.
  have filledRows : holdsFlat p3.current
      (gadgetOps program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)
          (sharedEnd base program.witnessWordCount) ++
        LightPiCCSLayout.opsAt relation (piccsInterface program
          (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount))
          (piccsOffset program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)
            (sharedEnd base program.witnessWordCount)) ++
        LightRlcLayout.opsAt (piccsLeaves program relation circuit
          (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))
          (rlcInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount))
          (rlcOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)
            (sharedEnd base program.witnessWordCount))) := by
    have rows := p3.rows
    rw [ops3, ops2, ops1] at rows
    simpa only [Sequence.empty, List.nil_append] using rows
  have rows3 := holdsFlat_implies_holds _ _ filledRows
  simp only [holds_append] at rows3
  obtain ⟨⟨_, piccsRows⟩, rlcRows⟩ := rows3
  have rep3 := repAt p3.current (agreeOf p3)
  have active3 := activeAt p3.current (agreeOf p3)
  -- The folded slot that the Π_RLC words carry is the honest fold result.
  have piccsSpec := LightPiCCSLayout.soundness relation
    (piccsOffset program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)
      (sharedEnd base program.witnessWordCount))
    (piccsInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)) p3.current
    (LightPiCCSBounds.assumptions relation _ _ piccsBelow p3.current) piccsRows
    (circuitKeys selected).relation
  have piccsWired := LightPiCCSLayout.wired relation
    (piccsOffset program circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)
      (sharedEnd base program.witnessWordCount)) (circuitKeys selected).relation ajtai
    (LightFold.lightRunning (input.running selected).1) (OptionBFolds.freshToLight input.fresh)
    proof (piccsInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)) p3.current
    (LightStepSoundness.piccsRepresents program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) ajtai
      circuitKeys circuit input output proof p3.current rep3 selected active3 _)
  have rlcSpec := LightRlcLayout.soundness
    (piccsLeaves program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)
      (sharedEnd base program.witnessWordCount))
    (rlcInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount))
    (rlcOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)
      (sharedEnd base program.witnessWordCount)) p3.current
    (LightRlcBounds.assumptions _ _ _ rlcBelow p3.current) rlcRows
  have rlcWired := LightRlcLayout.wired
    (piccsLeaves program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)
      (sharedEnd base program.witnessWordCount))
    (rlcInterface program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount))
    (rlcOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)
      (sharedEnd base program.witnessWordCount)) (input.running selected).1 input.fresh
    proof.piCcsOutput p3.current
    (LightStepSoundness.rlcRepresents program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) relation
      circuit (sharedEnd base program.witnessWordCount) ajtai circuitKeys input output proof
      p3.current rep3 selected active3)
  have foldEq := LightFold.spec_implies_fold (circuitKeys selected).relation ajtai
    (input.running selected).1 input.fresh proof
    (piccsLeaves program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)
      (sharedEnd base program.witnessWordCount))
    (rlcLeaves program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)
      (sharedEnd base program.witnessWordCount)) p3.current piccsSpec piccsWired rlcSpec rlcWired
  have nextEq : LightFold.nextSlot
      (piccsLeaves program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)
        (sharedEnd base program.witnessWordCount))
      (rlcLeaves program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount)
        (sharedEnd base program.witnessWordCount)) p3.current = folded :=
    LightFold.toClaim_injective _ (Option.some.inj (foldEq.symm.trans accepted))
  -- The slot-select rows on the filled values.
  have succBound : (input.running selected).2 + 1 < goldilocksModulus := by
    have cap := depthSmall
    simp only [Folding.Schedule.productionLightFoldCap] at cap
    have modulus : 65 ≤ goldilocksModulus := by decide
    omega
  have counter3 := fun (word : Nat) (bound : word < 5) => counter_eval program circuitKeys input
    output proof selected foldedSlot base p3.current (agreeOf p3) word bound
  have inverseWord : (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).inverse.eval p3.current = Hint.inverse (natWord input.iteration) :=
    counter3 1 (by decide)
  have recursiveWord : (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).recursive.eval p3.current =
      if input.iteration = 0 then 0 else 1 := counter3 2 (by decide)
  have selectorWord : (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).selector.eval p3.current = natWord selected.val :=
    counter3 4 (by decide)
  have selectorValues : ∀ chosen : Fin 2,
      (natWord chosen.val = 0 ∨ natWord chosen.val = 1) ∧
        (if natWord chosen.val = 1 then (1 : Fin 2) else 0) = chosen := by decide
  have selectedEq : SlotSelect.selected (selectInterface program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) (selectOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) p3.current = selected := by
    unfold SlotSelect.selected
    change (if (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).selector.eval p3.current = 1 then (1 : Fin 2) else 0) = selected
    rw [selectorWord]
    exact (selectorValues selected).2
  have recursiveZero : (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).recursive.eval p3.current = 0 → input.iteration = 0 := by
    intro zero
    by_contra nonzero
    rw [recursiveWord] at zero
    simp only [nonzero, ↓reduceIte] at zero
    exact absurd zero (by decide)
  have recursiveOne : (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).recursive.eval p3.current = 1 → input.iteration ≠ 0 := by
    intro one zero
    rw [recursiveWord] at one
    simp only [zero, ↓reduceIte] at one
    exact absurd one (by decide)
  have recursiveFacts : input.iteration ≠ 0 →
      (output.runningNext selected).1 = folded ∧
        (output.runningNext selected).2 = (input.running selected).2 + 1 ∧
        ∀ slot, slot ≠ selected → output.runningNext slot = input.running slot := by
    intro nonzero
    rcases cases with ⟨zero, _⟩ | ⟨priorPcValid, facts⟩
    · exact absurd zero nonzero
    · dsimp only at facts
      obtain ⟨_, _, _, foldsEq, depthEq, others⟩ := facts
      have indexEq : Paper.selectedIndex priorPcValid = selected :=
        Fin.ext (by simp [Paper.selectedIndex, selectedPc nonzero])
      rw [indexEq, light, proofEq nonzero] at foldsEq
      rw [indexEq, light] at depthEq
      rw [indexEq] at others
      exact ⟨LightFold.toClaim_injective _ (Option.some.inj (foldsEq.symm.trans accepted)),
        depthEq, others⟩
  have flags : ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).iteration.eval p3.current = 0 ∧ (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).recursive.eval p3.current = 0) ∨
      ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).iteration.eval p3.current * (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).inverse.eval p3.current = 1 ∧
        (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).recursive.eval p3.current = 1) := by
    rw [rep3.iteration, inverseWord, recursiveWord]
    by_cases zero : input.iteration = 0
    · left
      simp only [zero, ↓reduceIte, and_true]
      decide
    · right
      simp only [zero, ↓reduceIte, and_true]
      exact Gadgets.Range.CanonicalU64.mul_hintInverse_eq_one _ (natWord_ne_zero iterationBound zero)
  have selectorBit : (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).selector.eval p3.current = 0 ∨ (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).selector.eval p3.current = 1 := by
    rw [selectorWord]
    exact (selectorValues selected).1
  have pcLink : (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).recursive.eval p3.current = 1 →
      (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).priorPc.eval p3.current = (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).selector.eval p3.current + 1 := by
    intro one
    rw [rep3.priorPc, selectorWord, selectedPc (recursiveOne one), natWord_succ]
  have baseState : (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).recursive.eval p3.current = 0 → ∀ index,
      ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).initialState index).eval p3.current = ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).currentState index).eval p3.current := by
    intro zero index
    rcases cases with ⟨_, sameState, _⟩ | ⟨_, facts⟩
    · exact congrFun (List.ofFn_injective
        (rep3.initialState.trans (sameState.trans rep3.currentState.symm))) index
    · dsimp only at facts
      exact absurd (recursiveZero zero) (Nat.pos_iff_ne_zero.mp facts.1)
  have groupWords := fun (slot : Fin 7) depth => group_words program circuitKeys input output
    proof selected foldedSlot base p3.current (agreeOf p3) slot depth
  have activeWords : Hash.evalList p3.current (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).active.words =
      Hash.evalList p3.current ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).prior (SlotSelect.selected (selectInterface program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) (selectOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) p3.current)).words := by
    rw [selectedEq, rep3.priorSlots selected]
    have words := groupWords ⟨2, by decide⟩ (by
      rw [active_group]
      exact priorDepth selected)
    rw [active_group] at words
    exact words
  have innerWords : ∀ slot : Fin 2, Hash.evalList p3.current ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).inner slot).words =
      serializeSlot (if slot = selected then foldedSlot else input.running slot) := by
    intro slot
    have words := groupWords ⟨3 + slot.val, by omega⟩ (by
      rw [inner_group]
      split_ifs
      · exact foldedDepth
      · exact priorDepth slot)
    rw [inner_group] at words
    exact words
  have depthWord : (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).active.depth.eval p3.current = natWord (input.running selected).2 := by
    have depthEq := congrArg Prod.snd active3
    change ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).active.depth.eval p3.current).val = _ at depthEq
    rw [← depthEq, SlotExpr.natWord_val]
  have newWords : Hash.evalList p3.current (newSlot program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)).words = serializeSlot foldedSlot := by
    rw [SlotExpr.words_eval]
    apply congrArg serializeSlot
    rw [foldedSlotEq]
    apply Prod.ext
    · exact (LightStepSoundness.newSlot_value program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) relation circuit (sharedEnd base program.witnessWordCount) p3.current).trans
        nextEq
    · change (((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).active.depth + 1).eval p3.current).val = (input.running selected).2 + 1
      rw [Expr.eval_hadd, eval_one, depthWord, ← natWord_succ]
      simp only [natWord, Poseidon2.ofNat]
      exact Nat.mod_eq_of_lt succBound
  have innerSelected : Hash.evalList p3.current
      ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).inner (SlotSelect.selected (selectInterface program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) (selectOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) p3.current)).words =
        Hash.evalList p3.current (newSlot program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)).words := by
    rw [selectedEq, innerWords selected, newWords]
    simp only [↓reduceIte]
  have innerOther : ∀ slot, slot ≠ SlotSelect.selected (selectInterface program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) (selectOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) p3.current →
      Hash.evalList p3.current ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).inner slot).words =
        Hash.evalList p3.current ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).prior slot).words := by
    intro slot other
    rw [selectedEq] at other
    rw [innerWords slot, rep3.priorSlots slot]
    simp only [other, ↓reduceIte]
  have nextBase : (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).recursive.eval p3.current = 0 → ∀ slot,
      Hash.evalList p3.current ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).next slot).words =
        Hash.evalList p3.current (SlotSelect.defaultWords (logicalWidth := logicalWidth)
          (publicFits := publicFits)) := by
    intro zero slot
    rw [rep3.nextSlots slot, defaultWords_eval]
    rcases cases with ⟨_, _, sameNext⟩ | ⟨_, facts⟩
    · rw [sameNext]
    · dsimp only at facts
      exact absurd (recursiveZero zero) (Nat.pos_iff_ne_zero.mp facts.1)
  have nextRecursive : (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).recursive.eval p3.current = 1 → ∀ slot,
      Hash.evalList p3.current ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).next slot).words =
        Hash.evalList p3.current ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount).inner slot).words := by
    intro one slot
    obtain ⟨claimEq, depthEq, others⟩ := recursiveFacts (recursiveOne one)
    rw [rep3.nextSlots slot, innerWords slot]
    by_cases same : slot = selected
    · subst same
      simp only [↓reduceIte, foldedSlotEq]
      exact congrArg serializeSlot (Prod.ext claimEq depthEq)
    · simp only [same, ↓reduceIte]
      rw [others slot same]
  have selectRows := SlotSelect.rows_of_select (selectInterface program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) (selectOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) p3.current flags selectorBit pcLink
    baseState activeWords innerSelected innerOther nextBase nextRecursive
  refine ⟨p3.current, ?_, rep3⟩
  change holdsFlat p3.current (LightStepLayout.opsAt program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))
  unfold LightStepLayout.opsAt holdsFlat
  rw [flatConstraints_append]
  refine (constraintsHold_append _ _ _).mpr ⟨filledRows, ?_⟩
  intro row member
  simp only [flatConstraints, List.flatMap_cons, List.flatMap_nil, List.append_nil] at member
  change row ∈ flatConstraints ((SlotSelect.rows (selectInterface program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount)) (selectOffset program relation circuit (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base program.witnessWordCount) (sharedEnd base program.witnessWordCount))).map Op.assertZero) at member
  simp only [flatConstraints, List.flatMap_map, List.mem_flatMap, Op.flatConstraints,
    List.mem_singleton] at member
  obtain ⟨original, inside, rfl⟩ := member
  exact selectRows _ inside


end NightstreamFPrime.Layout.OptionB.LightAssemblerCompleteness
