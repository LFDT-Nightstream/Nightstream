import NightstreamFPrime.Layout.OptionB.LightPiCCSBounds
import NightstreamFPrime.Layout.OptionB.LightRlcBounds
import NightstreamFPrime.Lifecycle.OptionB.LightStepLayout

/-! Input scoping of the light step layout (assembler A2c): when the step's shared words lie below
the parent offset, `LightStepLayout.Assumptions` holds (`assumptions`). The two fold layouts get
their own input scoping (`LightPiCCSBounds`, `LightRlcBounds`); the Π_RLC sampler starts from the
Π_CCS output binding's final state, which lies below the Π_RLC layout's offset. -/

namespace NightstreamFPrime.Layout.OptionB.LightStepBounds

open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.OptionB

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle.PiCCS.v1_2
open NightstreamFPrime.Lifecycle.OptionB.SlotWords
open NightstreamFPrime.Lifecycle.OptionB.LightStepLayout

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- The shared words that the step's gadgets read lie below the parent offset. -/
structure InputsBelow {witnessWordCount : Nat}
    (interface : LightStepLayout.Interface logicalWidth publicFits witnessWordCount)
    (offset : Nat) : Prop where
  iteration : interface.iteration.VarsBelow offset
  priorPc : interface.priorPc.VarsBelow offset
  verifierKeys : ∀ key lane, (interface.verifierKeys key lane).VarsBelow offset
  initialState : ∀ lane, (interface.initialState lane).VarsBelow offset
  currentState : ∀ lane, (interface.currentState lane).VarsBelow offset
  nextState : ∀ lane, (interface.nextState lane).VarsBelow offset
  witness : ∀ index, (interface.witness index).VarsBelow offset
  activePoint : ∀ coordinate, (interface.active.point coordinate).VarsBelow offset
  activePad : ∀ coefficient, (interface.active.evaluation.eval_K coefficient).VarsBelow offset
  activeMatrix : ∀ matrix coefficient,
    (interface.active.evaluation.eval_A matrix coefficient).VarsBelow offset
  activeCommitment : ∀ row lane, (interface.active.commitment row lane).VarsBelow offset
  activePublicInput : ∀ column, (interface.active.publicInput column).VarsBelow offset
  activeDepth : interface.active.depth.VarsBelow offset
  priorDigest : ∀ slot lane, (interface.priorDigest slot lane).VarsBelow offset
  nextDigest : ∀ slot lane, (interface.nextDigest slot lane).VarsBelow offset
  outputDigest : ∀ lane, (interface.outputDigest lane).VarsBelow offset
  newDigest : ∀ lane, (interface.newDigest lane).VarsBelow offset
  innerDigest : ∀ slot lane, (interface.innerDigest slot lane).VarsBelow offset
  activeDigest : ∀ lane, (interface.activeDigest lane).VarsBelow offset
  freshCommitment : ∀ source row lane,
    (interface.fresh.commitment source row lane).VarsBelow offset
  freshPublicInput : ∀ source column, (interface.fresh.publicInput source column).VarsBelow offset
  round : ∀ roundIndex coefficient,
    ((interface.round roundIndex).coefficient coefficient).VarsBelow offset
  outputPad : ∀ source coefficient,
    (interface.output.padCoordinate source coefficient).VarsBelow offset
  outputMatrix : ∀ source matrix coefficient,
    (interface.output.matrixCoordinate source matrix coefficient).VarsBelow offset

variable (program : Stage1.Application.Program)
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits) (circuit : Fin 2)
  (interface : LightStepLayout.Interface logicalWidth publicFits program.witnessWordCount)

theorem statePreimage_below (iteration : Expr) (current : Fin 4 → Expr) (pc : Expr)
    (digests : Fin 2 → Fin 4 → Expr) (bound : Nat)
    (keys : ∀ key lane, (interface.verifierKeys key lane).VarsBelow bound)
    (initial : ∀ lane, (interface.initialState lane).VarsBelow bound)
    (iterationBelow : iteration.VarsBelow bound)
    (currentBelow : ∀ lane, (current lane).VarsBelow bound) (pcBelow : pc.VarsBelow bound)
    (digestBelow : ∀ slot lane, (digests slot lane).VarsBelow bound) :
    ∀ word ∈ statePreimageExpr program interface iteration current pc digests,
      word.VarsBelow bound := by
  intro word member
  simp only [statePreimageExpr, List.mem_append, List.mem_map, List.mem_ofFn,
    List.mem_singleton] at member
  rcases member with ((((((((⟨_, _, rfl⟩ | ⟨lane, rfl⟩) | ⟨lane, rfl⟩) | rfl) |
    ⟨lane, rfl⟩) | ⟨lane, rfl⟩) | rfl) | ⟨lane, rfl⟩) | ⟨lane, rfl⟩)
  · trivial
  · exact keys 0 lane
  · exact keys 1 lane
  · exact iterationBelow
  · exact initial lane
  · exact currentBelow lane
  · exact pcBelow
  · exact digestBelow 0 lane
  · exact digestBelow 1 lane

theorem slotPreimage_below (words : List Expr) (bound : Nat)
    (below : ∀ word ∈ words, word.VarsBelow bound) :
    ∀ word ∈ slotDomainChunk.map Expr.const ++ words, word.VarsBelow bound := by
  intro word member
  rcases List.mem_append.mp member with member | member
  · obtain ⟨_, _, rfl⟩ := List.mem_map.mp member
    trivial
  · exact below word member

/-- A slot's words lie below `bound` when each of its fields does. -/
theorem slotWords_below (slot : SlotExpr logicalWidth publicFits) (bound : Nat)
    (commitment : ∀ row lane, (slot.commitment row lane).VarsBelow bound)
    (pad : ∀ coefficient, (slot.evaluation.eval_K coefficient).VarsBelow bound)
    (matrix : ∀ matrix coefficient, (slot.evaluation.eval_A matrix coefficient).VarsBelow bound)
    (point : ∀ coordinate, (slot.point coordinate).VarsBelow bound)
    (publicInput : ∀ column, (slot.publicInput column).VarsBelow bound)
    (depth : slot.depth.VarsBelow bound) :
    ∀ word ∈ slot.words, word.VarsBelow bound := by
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

variable (offset : Nat) (below : InputsBelow interface offset) (env : Env)

include below

/-- The light Π_CCS layout's inputs lie below its offset. -/
theorem piccsBelow : LightPiCCSBounds.InputsBelow (piccsInterface program interface)
    (piccsOffset program circuit interface offset) := by
  have le := fun (child : FormalCircuit) (start : Nat) =>
    (Nat.le_add_right start _ : start ≤ Formal.nextOffset child start)
  have rangeLe : offset ≤ rangeOffset program interface offset := le _ _
  have activeLe : offset ≤ activeSlotOffset program interface offset :=
    Nat.le_trans rangeLe (Nat.le_trans (le _ _) (le _ _))
  have priorStateLe : offset ≤ priorStateOffset program interface offset :=
    Nat.le_trans activeLe (le _ _)
  have nextStateLe : offset ≤ nextStateOffset program interface offset :=
    Nat.le_trans priorStateLe (le _ _)
  have piccsLe : offset ≤ piccsOffset program circuit interface offset :=
    Nat.le_trans nextStateLe (le _ _)
  exact
    ⟨fun coordinate => KExpr.varsBelow_mono _ (below.activePoint coordinate) piccsLe,
      fun coefficient => KExpr.varsBelow_mono _ (below.activePad coefficient) piccsLe,
      fun matrix coefficient => KExpr.varsBelow_mono _ (below.activeMatrix matrix coefficient)
        piccsLe,
      fun source row lane => Expr.VarsBelow.mono _ (below.freshCommitment source row lane) piccsLe,
      fun source column => Expr.VarsBelow.mono _ (below.freshPublicInput source column) piccsLe,
      fun roundIndex coefficient => KExpr.varsBelow_mono _ (below.round roundIndex coefficient)
        piccsLe,
      fun source coefficient => KExpr.varsBelow_mono _ (below.outputPad source coefficient) piccsLe,
      fun source matrix coefficient =>
        KExpr.varsBelow_mono _ (below.outputMatrix source matrix coefficient) piccsLe⟩

/-- The light Π_RLC layout's inputs lie below its offset. -/
theorem rlcBelow : LightRlcBounds.InputsBelow (piccsLeaves program relation circuit interface offset)
    (rlcInterface program interface) (rlcOffset program relation circuit interface offset) := by
  have le := fun (child : FormalCircuit) (start : Nat) =>
    (Nat.le_add_right start _ : start ≤ Formal.nextOffset child start)
  have rangeLe : offset ≤ rangeOffset program interface offset := le _ _
  have activeLe : offset ≤ activeSlotOffset program interface offset :=
    Nat.le_trans rangeLe (Nat.le_trans (le _ _) (le _ _))
  have priorStateLe : offset ≤ priorStateOffset program interface offset :=
    Nat.le_trans activeLe (le _ _)
  have nextStateLe : offset ≤ nextStateOffset program interface offset :=
    Nat.le_trans priorStateLe (le _ _)
  have piccsLe : offset ≤ piccsOffset program circuit interface offset :=
    Nat.le_trans nextStateLe (le _ _)
  have piccsBelow := piccsBelow program circuit interface offset below
  have outputLe : piccsOffset program circuit interface offset ≤
      LightPiCCSLayout.outputOffset relation (piccsInterface program interface)
        (piccsOffset program circuit interface offset) := by
    exact Nat.le_trans (LightPiCCSBounds.challenge_le _ _) (Nat.le_trans
      (LightPiCCSBounds.rounds_le _ _) (Nat.le_trans (LightPiCCSBounds.initial_le _ _)
        (Nat.le_trans (LightPiCCSBounds.sumcheck_le _ _) (Nat.le_trans
          (LightPiCCSBounds.evalK_le _ _) (Nat.le_trans (LightPiCCSBounds.evalA_le _ _)
            (Nat.le_trans (LightPiCCSBounds.ccs_le _ _) (Nat.le_trans
              (LightPiCCSBounds.norm_le relation _ _) (Nat.le_trans
                (LightPiCCSBounds.finalIdentity_le relation _ _)
                (LightPiCCSBounds.output_le relation _ _)))))))))
  have rlcLe : offset ≤ rlcOffset program relation circuit interface offset :=
    Nat.le_trans piccsLe (Nat.le_trans outputLe (le _ _))
  exact
    ⟨fun lane => OutputBinding.finalState_varsBelow
        (LightPiCCSLayout.outputInterface (piccsInterface program interface)
          (piccsOffset program circuit interface offset))
        (LightPiCCSLayout.outputOffset relation (piccsInterface program interface)
          (piccsOffset program circuit interface offset)) (fun _ => 0)
        (LightPiCCSBounds.output relation (piccsInterface program interface)
          (piccsOffset program circuit interface offset) piccsBelow (fun _ => 0)) lane,
      fun source coefficient => KExpr.varsBelow_mono _ (below.outputPad source coefficient) rlcLe,
      fun source matrix coefficient =>
        KExpr.varsBelow_mono _ (below.outputMatrix source matrix coefficient) rlcLe,
      fun row lane => Expr.VarsBelow.mono _ (below.activeCommitment row lane) rlcLe,
      fun column => Expr.VarsBelow.mono _ (below.activePublicInput column) rlcLe,
      fun source row lane => Expr.VarsBelow.mono _ (below.freshCommitment source row lane) rlcLe,
      fun source column => Expr.VarsBelow.mono _ (below.freshPublicInput source column) rlcLe⟩

omit below in
/-- The light Π_RLC layout's three output groups: 1,188 commitment words, 270 public-input words
and 540 evaluation words, in that order, up to the new slot hash's offset. -/
theorem rlcOutputs_eq :
    LightRlcLayout.publicInputOffset (piccsLeaves program relation circuit interface offset)
        (rlcInterface program interface) (rlcOffset program relation circuit interface offset) =
      LightRlcLayout.commitmentOffset (piccsLeaves program relation circuit interface offset)
        (rlcOffset program relation circuit interface offset) + 1188 ∧
    LightRlcLayout.evaluationOffset (piccsLeaves program relation circuit interface offset)
        (rlcInterface program interface) (rlcOffset program relation circuit interface offset) =
      LightRlcLayout.publicInputOffset (piccsLeaves program relation circuit interface offset)
        (rlcInterface program interface) (rlcOffset program relation circuit interface offset) +
        270 ∧
    newSlotOffset program relation circuit interface offset =
      LightRlcLayout.evaluationOffset (piccsLeaves program relation circuit interface offset)
        (rlcInterface program interface) (rlcOffset program relation circuit interface offset) +
        540 := by
  refine ⟨?_, ?_, ?_⟩
  · unfold LightRlcLayout.publicInputOffset Formal.nextOffset Formal.childLength
      LightRlcLayout.commitmentCircuit
    change _ + localLength (Lifecycle.PiRLC.v1_2.CombinationStep.operations _ _) = _
    rw [Lifecycle.PiRLC.v1_2.CombinationStep.localLength_eq, LightCommitment.rowCount]
  · unfold LightRlcLayout.evaluationOffset Formal.nextOffset Formal.childLength
      LightRlcLayout.publicInputCircuit
    change _ + localLength (Lifecycle.PiRLC.v1_2.CombinationStep.operations _ _) = _
    rw [Lifecycle.PiRLC.v1_2.CombinationStep.localLength_eq, LightPublicInput.rowCount]
  · unfold newSlotOffset LightRlcLayout.finalOffset Formal.nextOffset Formal.childLength
      LightRlcLayout.evaluationCircuit
    change _ + localLength (Lifecycle.PiRLC.v1_2.CombinationStep.operations _ _) = _
    rw [Lifecycle.PiRLC.v1_2.CombinationStep.localLength_eq, LightEvaluation.rowCount]

omit below in
/-- The Π_CCS round point lies below the Π_RLC layout's offset, which lies below the new slot
hash's offset. -/
theorem newSlot_le :
    LightPiCCSLayout.initialOffset (piccsInterface program interface)
        (piccsOffset program circuit interface offset) ≤
      rlcOffset program relation circuit interface offset ∧
    offset ≤ rlcOffset program relation circuit interface offset ∧
    rlcOffset program relation circuit interface offset ≤
      LightRlcLayout.commitmentOffset (piccsLeaves program relation circuit interface offset)
        (rlcOffset program relation circuit interface offset) ∧
    offset ≤ newSlotOffset program relation circuit interface offset := by
  have le := fun (child : FormalCircuit) (start : Nat) =>
    (Nat.le_add_right start _ : start ≤ Formal.nextOffset child start)
  have priorStateLe : offset ≤ priorStateOffset program interface offset :=
    Nat.le_trans (le _ _) (Nat.le_trans (le _ _) (Nat.le_trans (le _ _) (le _ _)))
  have piccsLe : offset ≤ piccsOffset program circuit interface offset :=
    Nat.le_trans (Nat.le_trans priorStateLe (le _ _)) (le _ _)
  have initialRlcLe : LightPiCCSLayout.initialOffset (piccsInterface program interface)
      (piccsOffset program circuit interface offset) ≤
      rlcOffset program relation circuit interface offset :=
    Nat.le_trans (LightPiCCSBounds.sumcheck_le _ _) (Nat.le_trans
      (LightPiCCSBounds.evalK_le _ _) (Nat.le_trans (LightPiCCSBounds.evalA_le _ _)
        (Nat.le_trans (LightPiCCSBounds.ccs_le _ _) (Nat.le_trans
          (LightPiCCSBounds.norm_le relation _ _) (Nat.le_trans
            (LightPiCCSBounds.finalIdentity_le relation _ _) (Nat.le_trans
              (LightPiCCSBounds.output_le relation _ _) (le _ _)))))))
  have rlcLe : offset ≤ rlcOffset program relation circuit interface offset :=
    Nat.le_trans piccsLe (Nat.le_trans (LightPiCCSBounds.initial_offset_le _ _) initialRlcLe)
  have commitmentStart := LightRlcBounds.commitment_le
    (piccsLeaves program relation circuit interface offset)
    (rlcOffset program relation circuit interface offset)
  obtain ⟨commitmentEnd, publicEnd, evaluationEnd⟩ :=
    rlcOutputs_eq program relation circuit interface offset
  exact ⟨initialRlcLe, rlcLe, commitmentStart, by omega⟩

/-- The new slot's words, the Π_RLC outputs, the Π_CCS round point and the active depth plus one,
lie below the new slot hash's offset. -/
theorem newSlot_below : ∀ word ∈ (newSlot program relation circuit interface offset).words,
    word.VarsBelow (newSlotOffset program relation circuit interface offset) := by
  have piccsBelow := piccsBelow program circuit interface offset below
  obtain ⟨initialRlcLe, rlcLe, commitmentStart, slotLe⟩ :=
    newSlot_le program relation circuit interface offset
  obtain ⟨commitmentEnd, publicEnd, evaluationEnd⟩ :=
    rlcOutputs_eq program relation circuit interface offset
  apply slotWords_below
  · intro row lane
    show LightRlcLayout.commitmentOffset _ _ + _ < _
    exact Nat.lt_of_lt_of_le (Nat.add_lt_add_left
      (lt_of_lt_of_eq (Fin.isLt _) LightCommitment.rowCount) _) (by omega)
  · intro coefficient
    refine ⟨?_, ?_⟩ <;>
    · show LightRlcLayout.evaluationOffset _ _ _ + _ < _
      exact Nat.lt_of_lt_of_le (Nat.add_lt_add_left
        (lt_of_lt_of_eq (Fin.isLt _) LightEvaluation.rowCount) _) (by omega)
  · intro matrix coefficient
    refine ⟨?_, ?_⟩ <;>
    · show LightRlcLayout.evaluationOffset _ _ _ + _ < _
      exact Nat.lt_of_lt_of_le (Nat.add_lt_add_left
        (lt_of_lt_of_eq (Fin.isLt _) LightEvaluation.rowCount) _) (by omega)
  · intro coordinate
    exact KExpr.varsBelow_mono _
      (LightPiCCSBounds.roundPoint_below (piccsInterface program interface)
        (piccsOffset program circuit interface offset) piccsBelow coordinate)
      (by omega)
  · intro column
    show LightRlcLayout.publicInputOffset _ _ _ + _ < _
    exact Nat.lt_of_lt_of_le (Nat.add_lt_add_left
      (lt_of_lt_of_eq (Fin.isLt _) LightPublicInput.rowCount) _) (by omega)
  · exact Expr.VarsBelow.add _ _ _ (Expr.VarsBelow.mono _ below.activeDepth slotLe) trivial

/-- Input scoping of the light step layout. -/
theorem assumptions :
    LightStepLayout.Assumptions program relation circuit interface offset env := by
  have le := fun (child : FormalCircuit) (start : Nat) =>
    (Nat.le_add_right start _ : start ≤ Formal.nextOffset child start)
  have rangeLe : offset ≤ rangeOffset program interface offset := le _ _
  have activeLe : offset ≤ activeSlotOffset program interface offset :=
    Nat.le_trans rangeLe (Nat.le_trans (le _ _) (le _ _))
  have priorStateLe : offset ≤ priorStateOffset program interface offset :=
    Nat.le_trans activeLe (le _ _)
  have nextStateLe : offset ≤ nextStateOffset program interface offset :=
    Nat.le_trans priorStateLe (le _ _)
  have piccsLe : offset ≤ piccsOffset program circuit interface offset :=
    Nat.le_trans nextStateLe (le _ _)
  have piccsBelow := piccsBelow program circuit interface offset below
  have rlcBelow := rlcBelow program relation circuit interface offset below
  have keysAt := fun (bound : Nat) (le : offset ≤ bound) key lane =>
    Expr.VarsBelow.mono _ (below.verifierKeys key lane) le
  have initialAt := fun (bound : Nat) (le : offset ≤ bound) lane =>
    Expr.VarsBelow.mono _ (below.initialState lane) le
  exact {
    application := Stage1.Application.Program.assumptions program
      (applicationInterface program interface) offset env
      ⟨below.currentState, below.witness, below.nextState⟩
    range := Expr.VarsBelow.mono _ below.activeDepth rangeLe
    activeSlot := ⟨slotPreimage_below _ _ (slotWords_below _ _
        (fun row lane => Expr.VarsBelow.mono _ (below.activeCommitment row lane) activeLe)
        (fun coefficient => KExpr.varsBelow_mono _ (below.activePad coefficient) activeLe)
        (fun matrix coefficient =>
          KExpr.varsBelow_mono _ (below.activeMatrix matrix coefficient) activeLe)
        (fun coordinate => KExpr.varsBelow_mono _ (below.activePoint coordinate) activeLe)
        (fun column => Expr.VarsBelow.mono _ (below.activePublicInput column) activeLe)
        (Expr.VarsBelow.mono _ below.activeDepth activeLe)),
      fun lane => Expr.VarsBelow.mono _ (below.activeDigest lane) activeLe⟩
    priorState := ⟨statePreimage_below program interface _ _ _ _ _
        (keysAt _ priorStateLe) (initialAt _ priorStateLe)
        (Expr.VarsBelow.mono _ below.iteration priorStateLe)
        (fun lane => Expr.VarsBelow.mono _ (below.currentState lane) priorStateLe)
        (Expr.VarsBelow.mono _ below.priorPc priorStateLe)
        (fun slot lane => Expr.VarsBelow.mono _ (below.priorDigest slot lane) priorStateLe),
      fun column => Expr.VarsBelow.mono _ (below.freshPublicInput _ column) priorStateLe⟩
    nextState := ⟨statePreimage_below program interface _ _ _ _ _
        (keysAt _ nextStateLe) (initialAt _ nextStateLe)
        (Expr.VarsBelow.add _ _ _ (Expr.VarsBelow.mono _ below.iteration nextStateLe) trivial)
        (fun lane => Expr.VarsBelow.mono _ (below.nextState lane) nextStateLe) trivial
        (fun slot lane => Expr.VarsBelow.mono _ (below.nextDigest slot lane) nextStateLe),
      fun lane => Expr.VarsBelow.mono _ (below.outputDigest lane) nextStateLe⟩
    piccs := LightPiCCSBounds.assumptions relation (piccsInterface program interface)
      (piccsOffset program circuit interface offset) piccsBelow env
    rlc := LightRlcBounds.assumptions (piccsLeaves program relation circuit interface offset)
      (rlcInterface program interface) (rlcOffset program relation circuit interface offset)
      rlcBelow env
    newSlot := ⟨slotPreimage_below _ _ (newSlot_below program relation circuit interface offset
        below),
      fun lane => Expr.VarsBelow.mono _ (below.newDigest lane)
        (newSlot_le program relation circuit interface offset).2.2.2⟩ }

end NightstreamFPrime.Layout.OptionB.LightStepBounds
