import NightstreamFPrime.Lifecycle.BoundedDepth.LightStepLayout

/-! The light step circuit's rows for the exporter's blocks (SPEC.md section 4.4). The step places
its children one after another: the application, the depth guard, the active slot's hash, the
prior and next state hashes, the light Π_CCS and Π_RLC layouts inline, the new slot's hash and the
digest select. Each child's marks lie in its locals and ascend (`placed`). When the shared words
that the children read are affine and live before the layout (`InputsLive`), and the layout's
locals outside its marks' inner locals are live, each child has live rows (`liveRows`): the words that a
later child hashes (the Π_CCS round point and the Π_RLC outputs) are affine and live before it. -/

namespace NightstreamFPrime.Lifecycle.BoundedDepth.LightStepLayout

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Range
open NightstreamFPrime.Lifecycle.PiCCS.v1_2
open NightstreamFPrime.Lifecycle.PiRLC.v1_2
open NightstreamFPrime.Lifecycle.BoundedDepth.SlotWords
open NightstreamFPrime.Gadgets.Poseidon2.Duplex.Formal (WordFacts StateFacts)

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    RingCarrierLayout.carrierWidth logicalWidth}

/-! ### Words -/

/-- A slot's words satisfy `P` when each of its fields does. -/
theorem slotWords_forall {P : Expr → Prop} (slot : SlotExpr logicalWidth publicFits)
    (commitment : ∀ row lane, P (slot.commitment row lane))
    (pad : ∀ coefficient, P (slot.evaluation.eval_K coefficient).c0 ∧
      P (slot.evaluation.eval_K coefficient).c1)
    (matrix : ∀ matrix coefficient, P (slot.evaluation.eval_A matrix coefficient).c0 ∧
      P (slot.evaluation.eval_A matrix coefficient).c1)
    (point : ∀ coordinate, P (slot.point coordinate).c0 ∧ P (slot.point coordinate).c1)
    (publicInput : ∀ column, P (slot.publicInput column)) (depth : P slot.depth) :
    ∀ word ∈ slot.words, P word := by
  intro word member
  simp only [SlotExpr.words, StatementAbsorption.serializeCommitmentExpr,
    StatementAbsorption.serializeEvalKExpr, StatementAbsorption.serializeEvalAExpr,
    StatementAbsorption.serializePointExpr, StatementAbsorption.serializePublicInputExpr,
    StatementAbsorption.serializeKExpr, List.mem_append, List.mem_flatMap, List.mem_map,
    List.mem_finRange, List.mem_cons, List.not_mem_nil, or_false, true_and] at member
  rcases member with ((((commitmentWord | padWord | matrixWord) | pointWord) | publicWord) |
    depthWord)
  · obtain ⟨row, lane, rfl⟩ := commitmentWord
    exact commitment row lane
  · obtain ⟨coefficient, rfl | rfl⟩ := padWord
    · exact (pad coefficient).1
    · exact (pad coefficient).2
  · obtain ⟨matrixIndex, coefficient, rfl | rfl⟩ := matrixWord
    · exact (matrix matrixIndex coefficient).1
    · exact (matrix matrixIndex coefficient).2
  · obtain ⟨coordinate, rfl | rfl⟩ := pointWord
    · exact (point coordinate).1
    · exact (point coordinate).2
  · obtain ⟨column, rfl⟩ := publicWord
    exact publicInput column
  · subst depthWord
    exact depth

/-- Each field of a slot is among its words. -/
theorem slotFields_of_words {P : Expr → Prop} (slot : SlotExpr logicalWidth publicFits)
    (words : ∀ word ∈ slot.words, P word) :
    (∀ row lane, P (slot.commitment row lane)) ∧
      (∀ coefficient, P (slot.evaluation.eval_K coefficient).c0 ∧
        P (slot.evaluation.eval_K coefficient).c1) ∧
      (∀ matrix coefficient, P (slot.evaluation.eval_A matrix coefficient).c0 ∧
        P (slot.evaluation.eval_A matrix coefficient).c1) ∧
      (∀ coordinate, P (slot.point coordinate).c0 ∧ P (slot.point coordinate).c1) ∧
      (∀ column, P (slot.publicInput column)) ∧ P slot.depth := by
  have claims : ∀ word, word ∈ StatementAbsorption.serializeCommitmentExpr slot.commitment ++
      (StatementAbsorption.serializeEvalKExpr slot.evaluation ++
        StatementAbsorption.serializeEvalAExpr slot.evaluation) ++
      StatementAbsorption.serializePointExpr slot.point → P word :=
    fun word member => words word (List.mem_append_left _ (List.mem_append_left _ member))
  have kWords : ∀ value : KExpr, value.c0 ∈ StatementAbsorption.serializeKExpr value ∧
      value.c1 ∈ StatementAbsorption.serializeKExpr value :=
    fun _ => ⟨List.mem_cons_self, List.mem_cons_of_mem _ List.mem_cons_self⟩
  refine ⟨fun row lane => claims _ (List.mem_append_left _ (List.mem_append_left _
      (List.mem_flatMap.mpr ⟨row, List.mem_finRange _,
        List.mem_map.mpr ⟨lane, List.mem_finRange _, rfl⟩⟩))),
    fun coefficient => ⟨claims _ (List.mem_append_left _ (List.mem_append_right _
        (List.mem_append_left _ (List.mem_flatMap.mpr ⟨coefficient, List.mem_finRange _,
          (kWords _).1⟩)))),
      claims _ (List.mem_append_left _ (List.mem_append_right _
        (List.mem_append_left _ (List.mem_flatMap.mpr ⟨coefficient, List.mem_finRange _,
          (kWords _).2⟩))))⟩,
    fun matrix coefficient => ⟨claims _ (List.mem_append_left _ (List.mem_append_right _
        (List.mem_append_right _ (List.mem_flatMap.mpr ⟨matrix, List.mem_finRange _,
          List.mem_flatMap.mpr ⟨coefficient, List.mem_finRange _, (kWords _).1⟩⟩)))),
      claims _ (List.mem_append_left _ (List.mem_append_right _
        (List.mem_append_right _ (List.mem_flatMap.mpr ⟨matrix, List.mem_finRange _,
          List.mem_flatMap.mpr ⟨coefficient, List.mem_finRange _, (kWords _).2⟩⟩))))⟩,
    fun coordinate => ⟨claims _ (List.mem_append_right _ (List.mem_flatMap.mpr
        ⟨coordinate, List.mem_finRange _, (kWords _).1⟩)),
      claims _ (List.mem_append_right _ (List.mem_flatMap.mpr
        ⟨coordinate, List.mem_finRange _, (kWords _).2⟩))⟩,
    fun column => words _ (List.mem_append_left _ (List.mem_append_right _
      (List.mem_map.mpr ⟨column, List.mem_finRange _, rfl⟩))),
    words _ (List.mem_append_right _ (List.mem_singleton.mpr rfl))⟩

/-- A slot hash's preimage: the domain chunk's constants, then the slot's words. -/
theorem slotPreimage_forall {P : Expr → Prop} (constant : ∀ value, P (Expr.const value))
    {words : List Expr} (facts : ∀ word ∈ words, P word) :
    ∀ word ∈ slotDomainChunk.map Expr.const ++ words, P word := by
  intro word member
  rcases List.mem_append.mp member with member | member
  · obtain ⟨value, _, rfl⟩ := List.mem_map.mp member
    exact constant value
  · exact facts word member

/-- A state hash's preimage (`statePreimage`): the domain chunk's constants, the key digests, the
iteration, the start and current states, the pc and the slot digests. -/
theorem statePreimage_forall {P : Expr → Prop} (constant : ∀ value, P (Expr.const value))
    (chunk : List F) (keys : Fin 2 → Fin 4 → Expr) (iteration : Expr)
    (initial current : Fin 4 → Expr) (pc : Expr) (digests : Fin 2 → Fin 4 → Expr)
    (keysFacts : ∀ key lane, P (keys key lane)) (iterationFacts : P iteration)
    (initialFacts : ∀ lane, P (initial lane)) (currentFacts : ∀ lane, P (current lane))
    (pcFacts : P pc) (digestFacts : ∀ slot lane, P (digests slot lane)) :
    ∀ word ∈ chunk.map Expr.const ++ List.ofFn (keys 0) ++ List.ofFn (keys 1) ++ [iteration] ++
      List.ofFn initial ++ List.ofFn current ++ [pc] ++ List.ofFn (digests 0) ++
      List.ofFn (digests 1), P word := by
  intro word member
  simp only [List.mem_append, List.mem_map, List.mem_ofFn, List.mem_singleton] at member
  rcases member with ((((((((⟨value, _, rfl⟩ | ⟨lane, rfl⟩) | ⟨lane, rfl⟩) | rfl) |
    ⟨lane, rfl⟩) | ⟨lane, rfl⟩) | rfl) | ⟨lane, rfl⟩) | ⟨lane, rfl⟩)
  · exact constant value
  · exact keysFacts 0 lane
  · exact keysFacts 1 lane
  · exact iterationFacts
  · exact initialFacts lane
  · exact currentFacts lane
  · exact pcFacts
  · exact digestFacts 0 lane
  · exact digestFacts 1 lane

/-- A constant word is affine and reads no column. -/
theorem wordFacts_const (live : Nat → Prop) (bound : Nat) (value : F) :
    WordFacts live bound (Expr.const value) :=
  ⟨.const value, trivial⟩

/-- A word that is affine and live before `bound` reads only columns that `P` holds on, when `P`
holds on the live columns before `bound`. -/
theorem wordReads {live P : Nat → Prop} {bound : Nat} {word : Expr}
    (facts : WordFacts live bound word)
    (lift : ∀ column, column < bound → live column → P column) :
    word.VarsSatisfy P :=
  facts.2.mono _ fun column inside => lift column inside.1 inside.2

/-- A state live before `start` reads only columns before `bound ≥ start`. -/
theorem stateFacts_before {live : Nat → Prop} {start bound : Nat} {state : Layer.EState}
    (facts : StateFacts live start state) (le : start ≤ bound) :
    StateFacts (fun column => column < bound ∧ live column) start state :=
  fun lane => ⟨(facts lane).1, (facts lane).2.mono _ fun _ inside =>
    ⟨inside.1, Nat.lt_of_lt_of_le inside.1 le, inside.2⟩⟩

/-! ### Layout ends -/

/-- The light Π_CCS layout's children end at its final offset. -/
theorem piccs_ends {degreeBound : Nat}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (piccs : LightPiCCSLayout.Interface logicalWidth degreeBound publicFits) (start : Nat) :
    start + localLength (Circuit.ops (LightPiCCSLayout.main relation piccs) start) =
      LightPiCCSLayout.finalOffset relation piccs start :=
  localLength_ends_cons <| localLength_ends_cons <| localLength_ends_cons <|
    localLength_ends_cons <| localLength_ends_cons <| localLength_ends_cons <|
    localLength_ends_cons <| localLength_ends_cons <| localLength_ends_cons <|
    localLength_ends_cons <| localLength_ends_cons <| localLength_ends_nil _

/-- The light Π_RLC layout's children end at its final offset. -/
theorem rlc_ends {degreeBound : Nat}
    (piccs : LightPiCCSPhase.Leaves logicalWidth degreeBound publicFits)
    (rlc : LightRlcLayout.Interface logicalWidth publicFits) (start : Nat) :
    start + localLength (Circuit.ops (LightRlcLayout.main piccs rlc) start) =
      LightRlcLayout.finalOffset piccs rlc start :=
  localLength_ends_cons <| localLength_ends_cons <| localLength_ends_cons <|
    localLength_ends_cons <| localLength_ends_nil _

/-- The light Π_CCS round point's bound lies before the layout's end. -/
theorem initial_le_final {degreeBound : Nat}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (piccs : LightPiCCSLayout.Interface logicalWidth degreeBound publicFits) (start : Nat) :
    LightPiCCSLayout.initialOffset piccs start ≤
      LightPiCCSLayout.finalOffset relation piccs start := by
  unfold LightPiCCSLayout.finalOffset LightPiCCSLayout.outputOffset
    LightPiCCSLayout.finalIdentityOffset LightPiCCSLayout.normOffset LightPiCCSLayout.ccsOffset
    LightPiCCSLayout.evalAOffset LightPiCCSLayout.evalKOffset LightPiCCSLayout.sumcheckOffset
    Formal.nextOffset
  omega

/-! ### The light step -/

variable (program : Stage1.Application.Program)
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits) (circuit : Fin 2)
  (interface : Interface logicalWidth publicFits program.witnessWordCount) (offset : Nat)

/-- The shared words that the step's children read: each is affine and reads only live columns
before the layout. The light step does not read the prior slots' words. -/
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
  active : ∀ word ∈ interface.active.words, WordFacts live offset word
  priorDigest : ∀ slot lane, WordFacts live offset (interface.priorDigest slot lane)
  nextDigest : ∀ slot lane, WordFacts live offset (interface.nextDigest slot lane)
  outputDigest : ∀ lane, WordFacts live offset (interface.outputDigest lane)
  newDigest : ∀ lane, WordFacts live offset (interface.newDigest lane)
  innerDigest : ∀ slot lane, WordFacts live offset (interface.innerDigest slot lane)
  activeDigest : ∀ lane, WordFacts live offset (interface.activeDigest lane)
  freshCommitment : ∀ index row lane,
    WordFacts live offset (interface.fresh.commitment index row lane)
  freshPublicInput : ∀ index column,
    WordFacts live offset (interface.fresh.publicInput index column)
  round : ∀ roundIndex coefficient,
    WordFacts live offset ((interface.round roundIndex).coefficient coefficient).c0 ∧
      WordFacts live offset ((interface.round roundIndex).coefficient coefficient).c1
  outputPad : ∀ source coefficient,
    WordFacts live offset (interface.output.padCoordinate source coefficient).c0 ∧
      WordFacts live offset (interface.output.padCoordinate source coefficient).c1
  outputMatrix : ∀ source matrix coefficient,
    WordFacts live offset (interface.output.matrixCoordinate source matrix coefficient).c0 ∧
      WordFacts live offset (interface.output.matrixCoordinate source matrix coefficient).c1

/-- The step's operations as one list of children, with the two fold layouts inline. -/
theorem opsAt_eq : opsAt program relation circuit interface offset =
    Formal.childOp "light.application" (applicationCircuit program interface) offset ::
      Formal.childOp "light.depth_range" (rangeCircuit program interface)
        (rangeOffset program interface offset) ::
      Formal.childOp "light.depth_high_bits" (highBitsCircuit program interface offset)
        (highBitsOffset program interface offset) ::
      Formal.childOp "light.active_slot" (activeSlotCircuit program interface)
        (activeSlotOffset program interface offset) ::
      Formal.childOp "light.prior_state" (priorStateCircuit program interface)
        (priorStateOffset program interface offset) ::
      Formal.childOp "light.next_state" (nextStateCircuit program circuit interface)
        (nextStateOffset program interface offset) ::
      (Circuit.ops (LightPiCCSLayout.main relation (piccsInterface program interface))
          (piccsOffset program circuit interface offset) ++
        (Circuit.ops (LightRlcLayout.main (piccsLeaves program relation circuit interface offset)
            (rlcInterface program interface))
            (rlcOffset program relation circuit interface offset) ++
          [Formal.childOp "light.new_slot"
              (newSlotCircuit program relation circuit interface offset)
              (newSlotOffset program relation circuit interface offset),
            Formal.childOp "light.digest_select" (selectCircuit program interface)
              (selectOffset program relation circuit interface offset)])) := by
  simp only [opsAt, gadgetOps, List.cons_append, List.nil_append, List.append_assoc]
  rfl

/-- The step's marks lie in its locals and ascend. -/
theorem placed :
    Placed offset (localLength (opsAt program relation circuit interface offset))
      (opsAt program relation circuit interface offset) := by
  rw [opsAt_eq]
  exact Placed.child_cons rfl (program.marksPlaced _ _) <|
    Placed.child_cons rfl (.of_marks_nil (CanonicalU64.marks_nil _ _)) <|
    Placed.child_cons rfl (.of_marks_nil (DepthGuard.marks_nil _ _)) <|
    Placed.child_cons rfl (OutputHash.placed _ _) <|
    Placed.child_cons rfl (PriorStateHash.placed _ _) <|
    Placed.child_cons rfl (OutputHash.placed _ _) <|
    Placed.append_ends (piccs_ends _ _ _) (LightPiCCSLayout.placed _ _ _) <|
    Placed.append_ends (rlc_ends _ _ _) (LightRlcLayout.placed _ _ _) <|
    Placed.child_cons rfl (OutputHash.placed _ _) <|
    Placed.child_cons rfl (.of_marks_nil (LightSelect.marks_nil _ _)) <|
    Placed.nil _

/-- The light Π_CCS layout's inputs: the fresh instance, the round messages and the Π_CCS output
live before the layout, and the active slot's point and evaluations. -/
theorem piccsInputs {live : Nat → Prop} (inputs : InputsLive program interface offset live) :
    LightPiCCSLayout.InputsLive (piccsInterface program interface)
      (piccsOffset program circuit interface offset) live := by
  have le := fun (child : FormalCircuit) (start : Nat) =>
    (Nat.le_add_right start _ : start ≤ Formal.nextOffset child start)
  have piccsLe : offset ≤ piccsOffset program circuit interface offset :=
    Nat.le_trans (le _ _) (Nat.le_trans (le _ _) (Nat.le_trans (le _ _) (Nat.le_trans (le _ _)
      (Nat.le_trans (le _ _) (le _ _)))))
  obtain ⟨-, pad, matrix, point, -, -⟩ := slotFields_of_words interface.active inputs.active
  exact {
    freshCommitment := fun index row lane => (inputs.freshCommitment index row lane).mono piccsLe
    freshPublicInput := fun index column => (inputs.freshPublicInput index column).mono piccsLe
    round := fun roundIndex coefficient => ⟨(inputs.round roundIndex coefficient).1.mono piccsLe,
      (inputs.round roundIndex coefficient).2.mono piccsLe⟩
    runningPoint := fun coordinate =>
      ⟨(point coordinate).1.varsSatisfy, (point coordinate).2.varsSatisfy⟩
    runningEvalK := fun coefficient =>
      ⟨(pad coefficient).1.varsSatisfy, (pad coefficient).2.varsSatisfy⟩
    runningEvalA := fun matrixIndex coefficient =>
      ⟨(matrix matrixIndex coefficient).1.varsSatisfy,
        (matrix matrixIndex coefficient).2.varsSatisfy⟩
    outputPad := fun source coefficient => ⟨(inputs.outputPad source coefficient).1.mono piccsLe,
      (inputs.outputPad source coefficient).2.mono piccsLe⟩
    outputMatrix := fun source matrixIndex coefficient =>
      ⟨(inputs.outputMatrix source matrixIndex coefficient).1.mono piccsLe,
        (inputs.outputMatrix source matrixIndex coefficient).2.mono piccsLe⟩ }

/-- The light Π_RLC layout's inputs, for any `P` that holds on the live columns before the step:
the Π_CCS final state, the active slot's commitment and public input, the fresh instance and the
Π_CCS output. -/
theorem rlcInputs {live P : Nat → Prop} (inputs : InputsLive program interface offset live)
    (lift : ∀ column, column < offset → live column → P column)
    (state : StateFacts P (rlcOffset program relation circuit interface offset)
      (OutputBinding.finalState (piccsLeaves program relation circuit interface offset).output
        (piccsLeaves program relation circuit interface offset).outputOffset)) :
    LightRlcLayout.InputsLive (piccsLeaves program relation circuit interface offset)
      (rlcInterface program interface) (rlcOffset program relation circuit interface offset) P := by
  obtain ⟨commitment, -, -, -, publicInput, -⟩ :=
    slotFields_of_words interface.active inputs.active
  exact {
    samplerState := state
    runningCommitment := fun block lane => wordReads (commitment block lane) lift
    runningPublicInput := fun column => wordReads (publicInput column) lift
    freshCommitment := fun block lane => wordReads (inputs.freshCommitment _ block lane) lift
    freshPublicInput := fun column => wordReads (inputs.freshPublicInput _ column) lift
    outputPad := fun source coefficient =>
      ⟨wordReads (inputs.outputPad source coefficient).1 lift,
        wordReads (inputs.outputPad source coefficient).2 lift⟩
    outputMatrix := fun source matrixIndex coefficient =>
      ⟨wordReads (inputs.outputMatrix source matrixIndex coefficient).1 lift,
        wordReads (inputs.outputMatrix source matrixIndex coefficient).2 lift⟩ }

/-- The step's rows: each child has live rows, when the shared words are live before the layout
and the layout's locals outside its marks' inner locals are live. -/
theorem liveRows {live : Nat → Prop} (inputs : InputsLive program interface offset live)
    (own : OwnLive live offset (localLength (opsAt program relation circuit interface offset))
      (opsAt program relation circuit interface offset)) :
    LiveRows live (opsAt program relation circuit interface offset) := by
  rw [opsAt_eq] at own ⊢
  -- Each child's tail, from the last child back.
  have selectTail := Placed.child_cons (name := "light.digest_select") rfl
    (.of_marks_nil (LightSelect.marks_nil (selectInterface program interface)
      (selectOffset program relation circuit interface offset))) (Placed.nil _)
  have newSlotTail := Placed.child_cons (name := "light.new_slot") rfl
    (OutputHash.placed (newSlotInterface program relation circuit interface offset)
      (newSlotOffset program relation circuit interface offset)) selectTail
  have rlcEnds := rlc_ends (piccsLeaves program relation circuit interface offset)
    (rlcInterface program interface) (rlcOffset program relation circuit interface offset)
  have rlcPlaced := LightRlcLayout.placed (piccsLeaves program relation circuit interface offset)
    (rlcInterface program interface) (rlcOffset program relation circuit interface offset)
  have rlcTail := Placed.append_ends rlcEnds rlcPlaced newSlotTail
  have piccsEnds := piccs_ends relation (piccsInterface program interface)
    (piccsOffset program circuit interface offset)
  have piccsPlaced := LightPiCCSLayout.placed relation (piccsInterface program interface)
    (piccsOffset program circuit interface offset)
  have piccsTail := Placed.append_ends piccsEnds piccsPlaced rlcTail
  have nextTail := Placed.child_cons (name := "light.next_state") rfl
    (OutputHash.placed (nextStateInterface program circuit interface)
      (nextStateOffset program interface offset)) piccsTail
  have priorTail := Placed.child_cons (name := "light.prior_state") rfl
    (PriorStateHash.placed (priorStateInterface program interface)
      (priorStateOffset program interface offset)) nextTail
  have activeTail := Placed.child_cons (name := "light.active_slot") rfl
    (OutputHash.placed (activeSlotInterface program interface)
      (activeSlotOffset program interface offset)) priorTail
  have highTail := Placed.child_cons (name := "light.depth_high_bits") rfl
    (.of_marks_nil (DepthGuard.marks_nil (rangeOffset program interface offset)
      (highBitsOffset program interface offset))) activeTail
  have rangeTail := Placed.child_cons (name := "light.depth_range") rfl
    (.of_marks_nil (CanonicalU64.marks_nil (rangeInterface program interface)
      (rangeOffset program interface offset))) highTail
  -- Each child's own locals.
  obtain ⟨applicationOwn, afterApplication⟩ :=
    own.child_cons rfl (program.marksPlaced _ _) rangeTail
  obtain ⟨rangeOwn, afterRange⟩ :=
    afterApplication.child_cons rfl (.of_marks_nil (CanonicalU64.marks_nil _ _)) highTail
  obtain ⟨-, afterHigh⟩ :=
    afterRange.child_cons rfl (.of_marks_nil (DepthGuard.marks_nil _ _)) activeTail
  obtain ⟨activeOwn, afterActive⟩ := afterHigh.child_cons rfl (OutputHash.placed _ _) priorTail
  obtain ⟨priorOwn, afterPrior⟩ :=
    afterActive.child_cons rfl (PriorStateHash.placed _ _) nextTail
  obtain ⟨nextOwn, afterNext⟩ := afterPrior.child_cons rfl (OutputHash.placed _ _) piccsTail
  obtain ⟨piccsOwn, afterPiccs⟩ := afterNext.append_ends piccsEnds piccsPlaced rlcTail
  obtain ⟨rlcOwn, afterRlc⟩ := afterPiccs.append_ends rlcEnds rlcPlaced newSlotTail
  obtain ⟨newSlotOwn, -⟩ := afterRlc.child_cons rfl (OutputHash.placed _ _) selectTail
  -- Offsets.
  have le := fun (child : FormalCircuit) (start : Nat) =>
    (Nat.le_add_right start _ : start ≤ Formal.nextOffset child start)
  have rangeLe : offset ≤ rangeOffset program interface offset := le _ _
  have activeLe : offset ≤ activeSlotOffset program interface offset :=
    Nat.le_trans rangeLe (Nat.le_trans (le _ _) (le _ _))
  have priorLe : offset ≤ priorStateOffset program interface offset :=
    Nat.le_trans activeLe (le _ _)
  have nextLe : offset ≤ nextStateOffset program interface offset := Nat.le_trans priorLe (le _ _)
  have piccsLe : offset ≤ piccsOffset program circuit interface offset :=
    Nat.le_trans nextLe (le _ _)
  have rlcAfter : piccsOffset program circuit interface offset ≤
      rlcOffset program relation circuit interface offset :=
    Nat.le_trans (Nat.le_add_right _ _) (Nat.le_of_eq piccsEnds)
  have rlcLe : rlcOffset program relation circuit interface offset ≤
      newSlotOffset program relation circuit interface offset :=
    Nat.le_trans (Nat.le_add_right _ _) (Nat.le_of_eq rlcEnds)
  have newSlotLe : offset ≤ newSlotOffset program relation circuit interface offset :=
    Nat.le_trans piccsLe (Nat.le_trans rlcAfter rlcLe)
  have pointLe : LightPiCCSLayout.initialOffset (piccsInterface program interface)
      (piccsOffset program circuit interface offset) ≤
      newSlotOffset program relation circuit interface offset :=
    Nat.le_trans (initial_le_final relation _ _) rlcLe
  obtain ⟨-, -, -, -, -, activeDepth⟩ := slotFields_of_words interface.active inputs.active
  -- The application and the depth guard.
  have applicationLiveRows := program.marksLiveRows (applicationInterface program interface) offset live
    ⟨fun index => (inputs.currentState index).1, fun index => (inputs.witness index).1,
      fun index => (inputs.nextState index).1⟩
    ⟨fun index => (inputs.currentState index).2, fun index => (inputs.witness index).2,
      fun index => (inputs.nextState index).2⟩ applicationOwn
  have rangeLocals := OwnLive.of_marks_nil (CanonicalU64.marks_nil _ _) rangeOwn
  have rangeFree := CanonicalU64.free (rangeInterface program interface)
    (rangeOffset program interface offset) activeDepth.varsSatisfy rangeLocals
  have highFree := DepthGuard.free (rangeOffset program interface offset)
    (highBitsOffset program interface offset)
    (CanonicalU64.bit_live (rangeInterface program interface)
      (rangeOffset program interface offset) rangeLocals)
  -- The active slot's hash and the state hashes.
  have activePreimage := slotPreimage_forall (wordFacts_const live _) fun word member =>
    (inputs.active word member).mono activeLe
  have activeLiveRows := OutputHash.liveRows (activeSlotInterface program interface)
    (activeSlotOffset program interface offset) (fun word member => (activePreimage word member).1)
    (fun word member => (activePreimage word member).2)
    (fun lane => (inputs.activeDigest lane).varsSatisfy) activeOwn
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
  -- The light Π_CCS and Π_RLC layouts.
  obtain ⟨piccsLiveRows, roundPoint, piccsState⟩ := LightPiCCSLayout.liveRows relation
    (piccsInterface program interface) (piccsOffset program circuit interface offset)
    (piccsInputs program circuit interface offset inputs) piccsOwn
  obtain ⟨rlcLiveRows, -⟩ := LightRlcLayout.liveRows
    (piccsLeaves program relation circuit interface offset) (rlcInterface program interface)
    (rlcOffset program relation circuit interface offset)
    (rlcInputs program relation circuit interface offset inputs (fun _ _ alive => alive) piccsState)
    rlcOwn
  obtain ⟨-, rlcOutputs⟩ := LightRlcLayout.liveRows
    (live := fun column => column < newSlotOffset program relation circuit interface offset ∧
      live column)
    (piccsLeaves program relation circuit interface offset)
    (rlcInterface program interface) (rlcOffset program relation circuit interface offset)
    (rlcInputs program relation circuit interface offset inputs
      (fun _ below alive => ⟨Nat.lt_of_lt_of_le below newSlotLe, alive⟩)
      (stateFacts_before piccsState rlcLe))
    (rlcOwn.before (Nat.le_of_eq rlcEnds))
  -- The new slot's hash.
  have newWords : ∀ word ∈ (newSlot program relation circuit interface offset).words,
      WordFacts live (newSlotOffset program relation circuit interface offset) word :=
    slotWords_forall _ (fun row lane => ⟨.var _, rlcOutputs.commitment row lane⟩)
      (fun coefficient => ⟨⟨.var _, (rlcOutputs.evaluation ⟨0, by decide⟩ coefficient).1⟩,
        ⟨.var _, (rlcOutputs.evaluation ⟨0, by decide⟩ coefficient).2⟩⟩)
      (fun matrix coefficient =>
        ⟨⟨.var _, (rlcOutputs.evaluation ⟨matrix.val + 1, Nat.succ_lt_succ matrix.isLt⟩
            coefficient).1⟩,
          ⟨.var _, (rlcOutputs.evaluation ⟨matrix.val + 1, Nat.succ_lt_succ matrix.isLt⟩
            coefficient).2⟩⟩)
      (fun coordinate => ⟨(roundPoint coordinate).1.mono pointLe,
        (roundPoint coordinate).2.mono pointLe⟩)
      (fun column => ⟨.var _, rlcOutputs.publicInput column⟩)
      ⟨.add activeDepth.1 (affine_ofNat 1), (activeDepth.mono newSlotLe).2, trivial⟩
  have newPreimage := slotPreimage_forall (wordFacts_const live _) newWords
  have newSlotLiveRows := OutputHash.liveRows (newSlotInterface program relation circuit interface offset)
    (newSlotOffset program relation circuit interface offset)
    (fun word member => (newPreimage word member).1)
    (fun word member => (newPreimage word member).2)
    (fun lane => (inputs.newDigest lane).varsSatisfy) newSlotOwn
  -- The digest select.
  have selectFree := LightSelect.free (selectInterface program interface)
    (selectOffset program relation circuit interface offset)
    ⟨inputs.iteration.varsSatisfy, inputs.inverse.varsSatisfy, inputs.recursive.varsSatisfy,
      inputs.priorPc.varsSatisfy, inputs.selector.varsSatisfy,
      fun lane => (inputs.initialState lane).varsSatisfy,
      fun lane => (inputs.currentState lane).varsSatisfy,
      fun slot lane => (inputs.priorDigest slot lane).varsSatisfy,
      fun lane => (inputs.newDigest lane).varsSatisfy,
      fun slot lane => (inputs.innerDigest slot lane).varsSatisfy,
      fun slot lane => (inputs.nextDigest slot lane).varsSatisfy⟩
    (fun lane => (inputs.activeDigest lane).varsSatisfy)
  exact LiveRows.child_cons applicationLiveRows <|
    LiveRows.child_cons (.of_free (CanonicalU64.marks_nil _ _) rangeFree) <|
    LiveRows.child_cons (.of_free (DepthGuard.marks_nil _ _) highFree) <|
    LiveRows.child_cons activeLiveRows <| LiveRows.child_cons priorLiveRows <|
    LiveRows.child_cons nextLiveRows <| piccsLiveRows.append <| rlcLiveRows.append <|
    LiveRows.child_cons newSlotLiveRows <|
    LiveRows.child_cons (.of_free (LightSelect.marks_nil _ _) selectFree) LiveRows.nil

end NightstreamFPrime.Lifecycle.BoundedDepth.LightStepLayout
