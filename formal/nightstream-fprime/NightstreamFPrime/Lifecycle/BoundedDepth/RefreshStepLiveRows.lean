import NightstreamFPrime.Lifecycle.BoundedDepth.RefreshStepLayout
import NightstreamFPrime.Lifecycle.BoundedDepth.LightStepLiveRows

/-! The refresh step circuit's rows for the exporter's blocks (SPEC.md section 4.4). The step
places its children one after another: the application, the two prior slot hashes, the prior and
next state hashes, today's Π_DEC, the refresh Π_CCS layout and today's Π_RLC inline, the new
slot's hash and the digest select. Each child's marks lie in its locals and ascend (`placed`).
When the shared words that the children read are affine and live before the layout
(`InputsLive`), and the layout's locals outside its marks' inner locals are live, each child is
has live rows (`liveRows`): the words that the new slot hashes (the Π_CCS round point and the Π_RLC outputs)
are affine and live before it. -/

namespace NightstreamFPrime.Lifecycle.BoundedDepth.RefreshStepLayout

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Lifecycle.PiCCS.v1_2
open NightstreamFPrime.Lifecycle.BoundedDepth.SlotWords
open NightstreamFPrime.Gadgets.Poseidon2.Duplex.Formal (WordFacts StateFacts)
open NightstreamFPrime.Lifecycle.BoundedDepth.LightStepLayout (slotWords_forall
  slotFields_of_words slotPreimage_forall statePreimage_forall wordFacts_const wordReads
  stateFacts_before)

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    RingCarrierLayout.carrierWidth logicalWidth}

/-! ### Layout ends -/

/-- Today's Π_DEC ends at its final offset. -/
theorem dec_ends (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (dec : Lifecycle.PiDEC.v1_2.Formal.Interface logicalWidth publicFits) (start : Nat) :
    start + localLength (Circuit.ops (Lifecycle.PiDEC.v1_2.Formal.main relation dec) start) =
      Lifecycle.PiDEC.v1_2.Formal.finalOffset start := by
  rw [Lifecycle.PiDEC.v1_2.Formal.localLength_eq]
  rfl

/-- The refresh Π_CCS layout's children end at its final offset. -/
theorem piccs_ends {degreeBound : Nat}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (piccs : RefreshPiCCSLayout.Interface logicalWidth degreeBound publicFits) (start : Nat) :
    start + localLength (Circuit.ops (RefreshPiCCSLayout.main relation piccs) start) =
      RefreshPiCCSLayout.finalOffset relation piccs start :=
  localLength_ends_cons <| localLength_ends_cons <| localLength_ends_cons <|
    localLength_ends_cons <| localLength_ends_cons <| localLength_ends_cons <|
    localLength_ends_cons <| localLength_ends_cons <| localLength_ends_cons <|
    localLength_ends_cons <| localLength_ends_cons <| localLength_ends_nil _

/-- Today's Π_RLC ends at its final offset. -/
theorem rlc_ends (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (rlc : Lifecycle.PiRLC.v1_2.Formal.Interface logicalWidth publicFits) (start : Nat) :
    start + localLength (Circuit.ops (Lifecycle.PiRLC.v1_2.Formal.main relation rlc) start) =
      Lifecycle.PiRLC.v1_2.Formal.finalOffset start := by
  rw [Lifecycle.PiRLC.v1_2.Formal.localLength_eq,
    Lifecycle.PiRLC.v1_2.Formal.logicalPrivateCount_eq]
  unfold Lifecycle.PiRLC.v1_2.Formal.finalOffset Lifecycle.PiRLC.v1_2.Formal.outputBindingOffset
    Lifecycle.PiRLC.v1_2.Formal.evalAOffset Lifecycle.PiRLC.v1_2.Formal.evalKOffset
    Lifecycle.PiRLC.v1_2.Formal.publicInputOffset Lifecycle.PiRLC.v1_2.Formal.commitmentOffset
    Lifecycle.PiRLC.v1_2.Formal.samplerOffset
  have := Lifecycle.PiRLC.v1_2.SamplerChain.counts.1
  omega

/-- The refresh Π_CCS round point's bound lies before the layout's end. -/
theorem initial_le_final {degreeBound : Nat}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (piccs : RefreshPiCCSLayout.Interface logicalWidth degreeBound publicFits) (start : Nat) :
    RefreshPiCCSLayout.initialOffset piccs start ≤
      RefreshPiCCSLayout.finalOffset relation piccs start := by
  unfold RefreshPiCCSLayout.finalOffset RefreshPiCCSLayout.outputOffset
    RefreshPiCCSLayout.finalIdentityOffset RefreshPiCCSLayout.normOffset
    RefreshPiCCSLayout.ccsOffset RefreshPiCCSLayout.evalAOffset RefreshPiCCSLayout.evalKOffset
    RefreshPiCCSLayout.sumcheckOffset Formal.nextOffset
  omega

/-! ### The refresh step -/

variable (program : Stage1.Application.Program)
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits) (circuit : Fin 2)
  (interface : Interface logicalWidth publicFits program.witnessWordCount) (offset : Nat)

/-- The shared words that the step's children read: each is affine and reads only live columns
before the layout. The refresh step does not read the inner and next slots' words. -/
structure InputsLive (live : Nat → Prop) : Prop where
  iteration : WordFacts live offset interface.iteration
  inverse : WordFacts live offset interface.inverse
  recursive : WordFacts live offset interface.recursive
  priorPc : WordFacts live offset interface.priorPc
  selector : WordFacts live offset interface.selector
  verifierKeys : ∀ key lane, WordFacts live offset (interface.verifierKeys key lane)
  initialState : ∀ lane, WordFacts live offset (interface.initialState lane)
  currentState : ∀ lane, WordFacts live offset (interface.currentState lane)
  nextState : ∀ lane, WordFacts live offset (interface.nextState lane)
  witness : ∀ index, WordFacts live offset (interface.witness index)
  prior : ∀ slot, ∀ word ∈ (interface.prior slot).words, WordFacts live offset word
  active : ∀ word ∈ interface.active.words, WordFacts live offset word
  priorDigest : ∀ slot lane, WordFacts live offset (interface.priorDigest slot lane)
  nextDigest : ∀ slot lane, WordFacts live offset (interface.nextDigest slot lane)
  outputDigest : ∀ lane, WordFacts live offset (interface.outputDigest lane)
  newDigest : ∀ lane, WordFacts live offset (interface.newDigest lane)
  innerDigest : ∀ slot lane, WordFacts live offset (interface.innerDigest slot lane)
  freshCommitment : ∀ index row lane,
    WordFacts live offset (interface.fresh.commitment index row lane)
  freshPublicInput : ∀ index column,
    WordFacts live offset (interface.fresh.publicInput index column)
  childCommitment : ∀ child row lane,
    WordFacts live offset (interface.childCommitment child row lane)
  childPad : ∀ child coefficient,
    WordFacts live offset ((interface.childEvaluation child).eval_K coefficient).c0 ∧
      WordFacts live offset ((interface.childEvaluation child).eval_K coefficient).c1
  childMatrix : ∀ child matrix coefficient,
    WordFacts live offset ((interface.childEvaluation child).eval_A matrix coefficient).c0 ∧
      WordFacts live offset ((interface.childEvaluation child).eval_A matrix coefficient).c1
  digit : ∀ child column, WordFacts live offset (interface.digit child column)
  round : ∀ roundIndex coefficient,
    WordFacts live offset ((interface.round roundIndex).coefficient coefficient).c0 ∧
      WordFacts live offset ((interface.round roundIndex).coefficient coefficient).c1
  outputPad : ∀ source coefficient,
    WordFacts live offset (interface.output.padCoordinate source coefficient).c0 ∧
      WordFacts live offset (interface.output.padCoordinate source coefficient).c1
  outputMatrix : ∀ source matrix coefficient,
    WordFacts live offset (interface.output.matrixCoordinate source matrix coefficient).c0 ∧
      WordFacts live offset (interface.output.matrixCoordinate source matrix coefficient).c1

/-- The step's operations as one list of children, with Π_DEC, Π_CCS and Π_RLC inline. -/
theorem opsAt_eq : opsAt program relation circuit interface offset =
    Formal.childOp "refresh.application" (applicationCircuit program interface) offset ::
      Formal.childOp "refresh.prior_slot_0" (priorSlotCircuit program interface 0)
        (priorSlotZeroOffset program interface offset) ::
      Formal.childOp "refresh.prior_slot_1" (priorSlotCircuit program interface 1)
        (priorSlotOneOffset program interface offset) ::
      Formal.childOp "refresh.prior_state" (priorStateCircuit program interface)
        (priorStateOffset program interface offset) ::
      Formal.childOp "refresh.next_state" (nextStateCircuit program circuit interface)
        (nextStateOffset program interface offset) ::
      (Circuit.ops (Lifecycle.PiDEC.v1_2.Formal.main relation (decInterface program interface))
          (decOffset program circuit interface offset) ++
        (Circuit.ops (RefreshPiCCSLayout.main relation (piccsInterface program interface))
            (piccsOffset program circuit interface offset) ++
          (Circuit.ops (Lifecycle.PiRLC.v1_2.Formal.main relation
              (rlcInterface program relation circuit interface offset))
              (rlcOffset program relation circuit interface offset) ++
            [Formal.childOp "refresh.new_slot"
                (newSlotCircuit program relation circuit interface offset)
                (newSlotOffset program relation circuit interface offset),
              Formal.childOp "refresh.digest_select" (selectCircuit program interface)
                (selectOffset program relation circuit interface offset)]))) := by
  simp only [opsAt, gadgetOps, List.cons_append, List.nil_append, List.append_assoc]

/-- The step's marks lie in its locals and ascend. -/
theorem placed :
    Placed offset (localLength (opsAt program relation circuit interface offset))
      (opsAt program relation circuit interface offset) := by
  rw [opsAt_eq]
  exact Placed.child_cons rfl (program.marksPlaced _ _) <|
    Placed.child_cons rfl (OutputHash.placed _ _) <|
    Placed.child_cons rfl (OutputHash.placed _ _) <|
    Placed.child_cons rfl (PriorStateHash.placed _ _) <|
    Placed.child_cons rfl (OutputHash.placed _ _) <|
    Placed.append_ends (dec_ends _ _ _)
      (.of_marks_nil (Lifecycle.PiDEC.v1_2.Formal.marks_nil _ _ _)) <|
    Placed.append_ends (piccs_ends _ _ _) (RefreshPiCCSLayout.placed _ _ _) <|
    Placed.append_ends (rlc_ends _ _ _) (Lifecycle.PiRLC.v1_2.Formal.placed _ _ _) <|
    Placed.child_cons rfl (OutputHash.placed _ _) <|
    Placed.child_cons rfl (.of_marks_nil (DigestSelect.marks_nil _ _)) <|
    Placed.nil _

/-- Today's Π_DEC reads the active slot, the children and the digit words. -/
theorem decInputs {live : Nat → Prop} (inputs : InputsLive program interface offset live) :
    Lifecycle.PiDEC.v1_2.Formal.InputsSatisfy (decInterface program interface)
      (decOffset program circuit interface offset) live := by
  obtain ⟨commitment, pad, matrix, -, publicInput, -⟩ :=
    slotFields_of_words interface.active inputs.active
  exact {
    parentCommitment := fun row lane => (commitment row lane).varsSatisfy
    parentPublicInput := fun coordinate => (publicInput coordinate).varsSatisfy
    parentEval_K := fun coefficient =>
      ⟨(pad coefficient).1.varsSatisfy, (pad coefficient).2.varsSatisfy⟩
    parentEval_A := fun matrixIndex coefficient =>
      ⟨(matrix matrixIndex coefficient).1.varsSatisfy,
        (matrix matrixIndex coefficient).2.varsSatisfy⟩
    messageCommitment := fun child row lane =>
      (inputs.childCommitment (childOf child) row lane).varsSatisfy
    messageEval_K := fun child coefficient =>
      ⟨(inputs.childPad (childOf child) coefficient).1.varsSatisfy,
        (inputs.childPad (childOf child) coefficient).2.varsSatisfy⟩
    messageEval_A := fun child matrixIndex coefficient =>
      ⟨(inputs.childMatrix (childOf child) matrixIndex coefficient).1.varsSatisfy,
        (inputs.childMatrix (childOf child) matrixIndex coefficient).2.varsSatisfy⟩
    digit := fun child column => (inputs.digit (childOf child) column).varsSatisfy }

/-- The refresh Π_CCS layout's inputs: the fresh instance, the children, the round messages and
the Π_CCS output live before the layout, and the active slot's point. -/
theorem piccsInputs {live : Nat → Prop} (inputs : InputsLive program interface offset live) :
    RefreshPiCCSLayout.InputsLive (piccsInterface program interface)
      (piccsOffset program circuit interface offset) live := by
  have le := fun (child : FormalCircuit) (start : Nat) =>
    (Nat.le_add_right start _ : start ≤ Formal.nextOffset child start)
  have decLe : offset ≤ decOffset program circuit interface offset :=
    Nat.le_trans (le _ _) (Nat.le_trans (le _ _) (Nat.le_trans (le _ _) (Nat.le_trans (le _ _)
      (le _ _))))
  have piccsLe : offset ≤ piccsOffset program circuit interface offset := by
    unfold piccsOffset Lifecycle.PiDEC.v1_2.Formal.finalOffset
      Lifecycle.PiDEC.v1_2.Formal.recompositionOffset
    omega
  obtain ⟨-, -, -, point, -, -⟩ := slotFields_of_words interface.active inputs.active
  exact {
    freshCommitment := fun index row lane => (inputs.freshCommitment index row lane).mono piccsLe
    freshPublicInput := fun index column => (inputs.freshPublicInput index column).mono piccsLe
    round := fun roundIndex coefficient => ⟨(inputs.round roundIndex coefficient).1.mono piccsLe,
      (inputs.round roundIndex coefficient).2.mono piccsLe⟩
    runningCommitment := fun child row lane => (inputs.childCommitment child row lane).mono piccsLe
    runningEvalK := fun child coefficient => ⟨(inputs.childPad child coefficient).1.mono piccsLe,
      (inputs.childPad child coefficient).2.mono piccsLe⟩
    runningEvalA := fun child matrixIndex coefficient =>
      ⟨(inputs.childMatrix child matrixIndex coefficient).1.mono piccsLe,
        (inputs.childMatrix child matrixIndex coefficient).2.mono piccsLe⟩
    runningPoint := fun coordinate =>
      ⟨(point coordinate).1.varsSatisfy, (point coordinate).2.varsSatisfy⟩
    outputPad := fun source coefficient => ⟨(inputs.outputPad source coefficient).1.mono piccsLe,
      (inputs.outputPad source coefficient).2.mono piccsLe⟩
    outputMatrix := fun source matrixIndex coefficient =>
      ⟨(inputs.outputMatrix source matrixIndex coefficient).1.mono piccsLe,
        (inputs.outputMatrix source matrixIndex coefficient).2.mono piccsLe⟩ }

/-- Today's Π_RLC's source words, for any `P` that holds on the live columns before the step: the
fresh instance, the children, the digit words and the Π_CCS output. -/
theorem rlcInputs {live P : Nat → Prop} (inputs : InputsLive program interface offset live)
    (lift : ∀ column, column < offset → live column → P column) :
    Lifecycle.PiRLC.v1_2.Formal.InputsLive (rlcInterface program relation circuit interface offset)
      (rlcOffset program relation circuit interface offset) P := by
  refine {
    commitment := fun source row lane => ?_
    publicInput := fun source column => ?_
    eval_K := fun source coefficient => ⟨wordReads (inputs.outputPad source coefficient).1 lift,
      wordReads (inputs.outputPad source coefficient).2 lift⟩
    eval_A := fun source matrixIndex coefficient =>
      ⟨wordReads (inputs.outputMatrix source matrixIndex coefficient).1 lift,
        wordReads (inputs.outputMatrix source matrixIndex coefficient).2 lift⟩ }
  · refine Fin.addCases (fun index => ?_) (fun child => ?_) source
    · simp only [rlcInterface, Fin.addCases_left]
      exact wordReads (inputs.freshCommitment index row lane) lift
    · simp only [rlcInterface, Fin.addCases_right]
      exact wordReads (inputs.childCommitment child row lane) lift
  · refine Fin.addCases (fun index => ?_) (fun child => ?_) source
    · simp only [rlcInterface, Fin.addCases_left]
      exact wordReads (inputs.freshPublicInput index column) lift
    · simp only [rlcInterface, Fin.addCases_right]
      exact wordReads (inputs.digit child column) lift

/-- The step's rows: each child has live rows, when the shared words are live before the layout
and the layout's locals outside its marks' inner locals are live. -/
theorem liveRows {live : Nat → Prop} (inputs : InputsLive program interface offset live)
    (own : OwnLive live offset (localLength (opsAt program relation circuit interface offset))
      (opsAt program relation circuit interface offset)) :
    LiveRows live (opsAt program relation circuit interface offset) := by
  rw [opsAt_eq] at own ⊢
  -- Each child's tail, from the last child back.
  have selectTail := Placed.child_cons (name := "refresh.digest_select") rfl
    (.of_marks_nil (DigestSelect.marks_nil (selectInterface program interface)
      (selectOffset program relation circuit interface offset))) (Placed.nil _)
  have newSlotTail := Placed.child_cons (name := "refresh.new_slot") rfl
    (OutputHash.placed (newSlotInterface program relation circuit interface offset)
      (newSlotOffset program relation circuit interface offset)) selectTail
  have rlcEnds := rlc_ends relation (rlcInterface program relation circuit interface offset)
    (rlcOffset program relation circuit interface offset)
  have rlcPlaced := Lifecycle.PiRLC.v1_2.Formal.placed relation
    (rlcInterface program relation circuit interface offset)
    (rlcOffset program relation circuit interface offset)
  have rlcTail := Placed.append_ends rlcEnds rlcPlaced newSlotTail
  have piccsEnds := piccs_ends relation (piccsInterface program interface)
    (piccsOffset program circuit interface offset)
  have piccsPlaced := RefreshPiCCSLayout.placed relation (piccsInterface program interface)
    (piccsOffset program circuit interface offset)
  have piccsTail := Placed.append_ends piccsEnds piccsPlaced rlcTail
  have decEnds := dec_ends relation (decInterface program interface)
    (decOffset program circuit interface offset)
  have decMarks := Lifecycle.PiDEC.v1_2.Formal.marks_nil relation (decInterface program interface)
    (decOffset program circuit interface offset)
  have decPlaced : Placed (decOffset program circuit interface offset)
      (localLength (Circuit.ops (Lifecycle.PiDEC.v1_2.Formal.main relation
        (decInterface program interface)) (decOffset program circuit interface offset)))
      (Circuit.ops (Lifecycle.PiDEC.v1_2.Formal.main relation (decInterface program interface))
        (decOffset program circuit interface offset)) :=
    .of_marks_nil decMarks
  have decTail := Placed.append_ends decEnds decPlaced piccsTail
  have nextTail := Placed.child_cons (name := "refresh.next_state") rfl
    (OutputHash.placed (nextStateInterface program circuit interface)
      (nextStateOffset program interface offset)) decTail
  have priorTail := Placed.child_cons (name := "refresh.prior_state") rfl
    (PriorStateHash.placed (priorStateInterface program interface)
      (priorStateOffset program interface offset)) nextTail
  have priorOneTail := Placed.child_cons (name := "refresh.prior_slot_1") rfl
    (OutputHash.placed (priorSlotInterface program interface 1)
      (priorSlotOneOffset program interface offset)) priorTail
  have priorZeroTail := Placed.child_cons (name := "refresh.prior_slot_0") rfl
    (OutputHash.placed (priorSlotInterface program interface 0)
      (priorSlotZeroOffset program interface offset)) priorOneTail
  -- Each child's own locals.
  obtain ⟨applicationOwn, afterApplication⟩ :=
    own.child_cons rfl (program.marksPlaced _ _) priorZeroTail
  obtain ⟨priorZeroOwn, afterPriorZero⟩ :=
    afterApplication.child_cons rfl (OutputHash.placed _ _) priorOneTail
  obtain ⟨priorOneOwn, afterPriorOne⟩ :=
    afterPriorZero.child_cons rfl (OutputHash.placed _ _) priorTail
  obtain ⟨priorOwn, afterPrior⟩ :=
    afterPriorOne.child_cons rfl (PriorStateHash.placed _ _) nextTail
  obtain ⟨nextOwn, afterNext⟩ := afterPrior.child_cons rfl (OutputHash.placed _ _) decTail
  obtain ⟨decOwn, afterDec⟩ := afterNext.append_ends decEnds decPlaced piccsTail
  obtain ⟨piccsOwn, afterPiccs⟩ := afterDec.append_ends piccsEnds piccsPlaced rlcTail
  obtain ⟨rlcOwn, afterRlc⟩ := afterPiccs.append_ends rlcEnds rlcPlaced newSlotTail
  obtain ⟨newSlotOwn, -⟩ := afterRlc.child_cons rfl (OutputHash.placed _ _) selectTail
  -- Offsets.
  have le := fun (child : FormalCircuit) (start : Nat) =>
    (Nat.le_add_right start _ : start ≤ Formal.nextOffset child start)
  have priorZeroLe : offset ≤ priorSlotZeroOffset program interface offset := le _ _
  have priorOneLe : offset ≤ priorSlotOneOffset program interface offset :=
    Nat.le_trans priorZeroLe (le _ _)
  have priorLe : offset ≤ priorStateOffset program interface offset :=
    Nat.le_trans priorOneLe (le _ _)
  have nextLe : offset ≤ nextStateOffset program interface offset := Nat.le_trans priorLe (le _ _)
  have decLe : offset ≤ decOffset program circuit interface offset := Nat.le_trans nextLe (le _ _)
  have piccsLe : offset ≤ piccsOffset program circuit interface offset :=
    Nat.le_trans decLe (Nat.le_trans (Nat.le_add_right _ _) (Nat.le_of_eq decEnds))
  have rlcAfter : piccsOffset program circuit interface offset ≤
      rlcOffset program relation circuit interface offset :=
    Nat.le_trans (Nat.le_add_right _ _) (Nat.le_of_eq piccsEnds)
  have rlcLe : rlcOffset program relation circuit interface offset ≤
      newSlotOffset program relation circuit interface offset :=
    Nat.le_trans (Nat.le_add_right _ _) (Nat.le_of_eq rlcEnds)
  have newSlotLe : offset ≤ newSlotOffset program relation circuit interface offset :=
    Nat.le_trans piccsLe (Nat.le_trans rlcAfter rlcLe)
  have pointLe : RefreshPiCCSLayout.initialOffset (piccsInterface program interface)
      (piccsOffset program circuit interface offset) ≤
      newSlotOffset program relation circuit interface offset :=
    Nat.le_trans (initial_le_final relation _ _) rlcLe
  -- The application.
  have applicationLiveRows := program.marksLiveRows (applicationInterface program interface) offset live
    ⟨fun index => (inputs.currentState index).1, fun index => (inputs.witness index).1,
      fun index => (inputs.nextState index).1⟩
    ⟨fun index => (inputs.currentState index).2, fun index => (inputs.witness index).2,
      fun index => (inputs.nextState index).2⟩ applicationOwn
  -- The prior slot hashes and the state hashes.
  have priorZeroPreimage := slotPreimage_forall (wordFacts_const live _) fun word member =>
    (inputs.prior 0 word member).mono priorZeroLe
  have priorZeroLiveRows := OutputHash.liveRows (priorSlotInterface program interface 0)
    (priorSlotZeroOffset program interface offset)
    (fun word member => (priorZeroPreimage word member).1)
    (fun word member => (priorZeroPreimage word member).2)
    (fun lane => (inputs.priorDigest 0 lane).varsSatisfy) priorZeroOwn
  have priorOnePreimage := slotPreimage_forall (wordFacts_const live _) fun word member =>
    (inputs.prior 1 word member).mono priorOneLe
  have priorOneLiveRows := OutputHash.liveRows (priorSlotInterface program interface 1)
    (priorSlotOneOffset program interface offset)
    (fun word member => (priorOnePreimage word member).1)
    (fun word member => (priorOnePreimage word member).2)
    (fun lane => (inputs.priorDigest 1 lane).varsSatisfy) priorOneOwn
  have priorPreimage : ∀ word ∈ statePreimageExpr program interface interface.iteration
      interface.currentState interface.priorPc interface.priorDigest,
      WordFacts live (priorStateOffset program interface offset) word :=
    statePreimage_forall (wordFacts_const live _) _ _ _ _ _ _ _
      (fun key lane => (inputs.verifierKeys key lane).mono priorLe) (inputs.iteration.mono priorLe)
      (fun lane => (inputs.initialState lane).mono priorLe)
      (fun lane => (inputs.currentState lane).mono priorLe) (inputs.priorPc.mono priorLe)
      (fun slot lane => (inputs.priorDigest slot lane).mono priorLe)
  have priorLiveRows := PriorStateHash.liveRows (priorStateInterface program interface)
    (priorStateOffset program interface offset) (fun word member => (priorPreimage word member).1)
    (fun word member => (priorPreimage word member).2)
    (fun column => (inputs.freshPublicInput _ column).varsSatisfy) priorOwn
  have iterationNext := inputs.iteration.mono nextLe
  have nextPreimage : ∀ word ∈ statePreimageExpr program interface (interface.iteration + 1)
      interface.nextState (Expr.const (natWord (circuit.val + 1))) interface.nextDigest,
      WordFacts live (nextStateOffset program interface offset) word :=
    statePreimage_forall (wordFacts_const live _) _ _ _ _ _ _ _
      (fun key lane => (inputs.verifierKeys key lane).mono nextLe)
      ⟨.add iterationNext.1 (affine_ofNat 1), iterationNext.2, trivial⟩
      (fun lane => (inputs.initialState lane).mono nextLe)
      (fun lane => (inputs.nextState lane).mono nextLe) (wordFacts_const live _ _)
      (fun slot lane => (inputs.nextDigest slot lane).mono nextLe)
  have nextLiveRows := OutputHash.liveRows (nextStateInterface program circuit interface)
    (nextStateOffset program interface offset) (fun word member => (nextPreimage word member).1)
    (fun word member => (nextPreimage word member).2)
    (fun lane => (inputs.outputDigest lane).varsSatisfy) nextOwn
  -- Today's Π_DEC, the refresh Π_CCS layout and today's Π_RLC.
  have decFree := Lifecycle.PiDEC.v1_2.Formal.free relation (decInterface program interface)
    (decOffset program circuit interface offset)
    (decInputs program circuit interface offset inputs)
    (OwnLive.of_marks_nil decMarks decOwn)
  obtain ⟨piccsLiveRows, roundPoint, piccsState⟩ := RefreshPiCCSLayout.liveRows relation
    (piccsInterface program interface) (piccsOffset program circuit interface offset)
    (piccsInputs program circuit interface offset inputs) piccsOwn
  have rlcState : StateFacts live (rlcOffset program relation circuit interface offset)
      (OutputBinding.finalState (piccsLeaves program relation circuit interface offset).output
        (piccsLeaves program relation circuit interface offset).outputOffset) :=
    piccsState
  obtain ⟨rlcLiveRows, -⟩ := Lifecycle.PiRLC.v1_2.Formal.liveRows relation
    (rlcInterface program relation circuit interface offset)
    (rlcOffset program relation circuit interface offset)
    (by dsimp only [rlcInterface]; exact rlcState)
    (rlcInputs program relation circuit interface offset inputs fun _ _ alive => alive) rlcOwn
  obtain ⟨-, rlcOutputs⟩ := Lifecycle.PiRLC.v1_2.Formal.liveRows
    (live := fun column => column < newSlotOffset program relation circuit interface offset ∧
      live column) relation
    (rlcInterface program relation circuit interface offset)
    (rlcOffset program relation circuit interface offset)
    (by dsimp only [rlcInterface]; exact stateFacts_before rlcState rlcLe)
    (rlcInputs program relation circuit interface offset inputs
      fun _ below alive => ⟨Nat.lt_of_lt_of_le below newSlotLe, alive⟩)
    (rlcOwn.before (Nat.le_of_eq rlcEnds))
  -- The new slot's hash.
  have newWords : ∀ word ∈ (newSlot program relation circuit interface offset).words,
      WordFacts live (newSlotOffset program relation circuit interface offset) word :=
    slotWords_forall _ (fun row lane => ⟨.var _, rlcOutputs.commitment
        (Lifecycle.PiRLC.v1_2.Formal.outputBindingOffset
          (rlcOffset program relation circuit interface offset)) row lane⟩)
      (fun coefficient => ⟨⟨.var _, (rlcOutputs.eval_K
          (Lifecycle.PiRLC.v1_2.Formal.outputBindingOffset
            (rlcOffset program relation circuit interface offset)) coefficient).1⟩,
        ⟨.var _, (rlcOutputs.eval_K (Lifecycle.PiRLC.v1_2.Formal.outputBindingOffset
            (rlcOffset program relation circuit interface offset)) coefficient).2⟩⟩)
      (fun matrix coefficient => ⟨⟨.var _, (rlcOutputs.eval_A
          (Lifecycle.PiRLC.v1_2.Formal.outputBindingOffset
            (rlcOffset program relation circuit interface offset)) matrix coefficient).1⟩,
        ⟨.var _, (rlcOutputs.eval_A (Lifecycle.PiRLC.v1_2.Formal.outputBindingOffset
            (rlcOffset program relation circuit interface offset)) matrix coefficient).2⟩⟩)
      (fun coordinate => ⟨(roundPoint coordinate).1.mono pointLe,
        (roundPoint coordinate).2.mono pointLe⟩)
      (fun column => ⟨.var _, rlcOutputs.publicInput
        (Lifecycle.PiRLC.v1_2.Formal.outputBindingOffset
          (rlcOffset program relation circuit interface offset)) column⟩)
      ⟨affine_ofNat 0, trivial⟩
  have newPreimage := slotPreimage_forall (wordFacts_const live _) newWords
  have newSlotLiveRows := OutputHash.liveRows (newSlotInterface program relation circuit interface offset)
    (newSlotOffset program relation circuit interface offset)
    (fun word member => (newPreimage word member).1)
    (fun word member => (newPreimage word member).2)
    (fun lane => (inputs.newDigest lane).varsSatisfy) newSlotOwn
  -- The digest select.
  have selectFree := DigestSelect.free (selectInterface program interface)
    (selectOffset program relation circuit interface offset)
    ⟨inputs.iteration.varsSatisfy, inputs.inverse.varsSatisfy, inputs.recursive.varsSatisfy,
      inputs.priorPc.varsSatisfy, inputs.selector.varsSatisfy,
      fun lane => (inputs.initialState lane).varsSatisfy,
      fun lane => (inputs.currentState lane).varsSatisfy,
      fun slot lane => (inputs.priorDigest slot lane).varsSatisfy,
      fun lane => (inputs.newDigest lane).varsSatisfy,
      fun slot lane => (inputs.innerDigest slot lane).varsSatisfy,
      fun slot lane => (inputs.nextDigest slot lane).varsSatisfy⟩
    (fun slot word member => (inputs.prior slot word member).varsSatisfy)
    (fun word member => (inputs.active word member).varsSatisfy)
  exact LiveRows.child_cons applicationLiveRows <| LiveRows.child_cons priorZeroLiveRows <|
    LiveRows.child_cons priorOneLiveRows <| LiveRows.child_cons priorLiveRows <|
    LiveRows.child_cons nextLiveRows <|
    (LiveRows.of_free decMarks decFree).append <| piccsLiveRows.append <| rlcLiveRows.append <|
    LiveRows.child_cons newSlotLiveRows <|
    LiveRows.child_cons (.of_free (DigestSelect.marks_nil _ _) selectFree) LiveRows.nil

end NightstreamFPrime.Lifecycle.BoundedDepth.RefreshStepLayout
