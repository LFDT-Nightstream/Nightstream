import NightstreamFPrime.Layout.OptionB.LightAssembler
import NightstreamFPrime.Layout.OptionB.RefreshAssembler

/-! The slot-select rows of both step circuits read below their own offset (`light_select_below`,
`refresh_select_below`): they read the shared words, and the new slot that the Π_RLC outputs and
the SumCheck point carry. With the prefixes of `LightAssemblerCompleteness.completeWith` and
`RefreshAssemblerCompleteness.completeWith`, this puts every constraint of a step circuit below
its last local, the footprint that the export's fresh columns need. -/

namespace NightstreamFPrime.Layout.OptionB.StepFootprint

open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.OptionB
open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle.PiCCS.v1_2
open NightstreamFPrime.Lifecycle.PiRLC.v1_2
open NightstreamFPrime.Lifecycle.OptionB.SlotWords

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-! ### Slot words and select rows -/

/-- A slot's words read below `bound` when each of its parts does. -/
theorem words_below (slot : SlotExpr logicalWidth publicFits) (bound : Nat)
    (commitment : ∀ row lane, (slot.commitment row lane).VarsBelow bound)
    (evalK : ∀ coefficient, (slot.evaluation.eval_K coefficient).VarsBelow bound)
    (evalA : ∀ matrix coefficient, (slot.evaluation.eval_A matrix coefficient).VarsBelow bound)
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
  rcases member with ((((inCommitment | pad | matrix) | inPoint) | inPublic) | inDepth)
  · obtain ⟨row, lane, rfl⟩ := inCommitment
    exact commitment row lane
  · obtain ⟨coefficient, rfl | rfl⟩ := pad
    · exact (evalK coefficient).1
    · exact (evalK coefficient).2
  · obtain ⟨matrixIndex, coefficient, rfl | rfl⟩ := matrix
    · exact (evalA matrixIndex coefficient).1
    · exact (evalA matrixIndex coefficient).2
  · obtain ⟨coordinate, rfl | rfl⟩ := inPoint
    · exact (point coordinate).1
    · exact (point coordinate).2
  · obtain ⟨column, rfl⟩ := inPublic
    exact publicInput column
  · subst inDepth
    exact depth

theorem selectRows_below {condition : Expr} {bound : Nat} (conditionBelow : condition.VarsBelow bound)
    {left right out : List Expr} (leftBelow : ∀ word ∈ left, word.VarsBelow bound)
    (rightBelow : ∀ word ∈ right, word.VarsBelow bound)
    (outBelow : ∀ word ∈ out, word.VarsBelow bound) :
    ∀ row ∈ selectRows condition left right out, row.VarsBelow bound := by
  intro row member
  simp only [selectRows, List.mem_map] at member
  obtain ⟨⟨⟨first, second⟩, third⟩, inZip, rfl⟩ := member
  have outer := List.of_mem_zip inZip
  have inner := List.of_mem_zip outer.1
  exact Expr.VarsBelow.sub _ _ _
    (Expr.VarsBelow.sub _ _ _ (outBelow _ outer.2) (leftBelow _ inner.1))
    ⟨conditionBelow, Expr.VarsBelow.sub _ _ _ (rightBelow _ inner.2) (leftBelow _ inner.1)⟩

/-- The slot-select rows read below `bound` when every word of their interface does. -/
theorem select_rows_below (interface : SlotSelect.Interface logicalWidth publicFits)
    (offset bound : Nat)
    (iteration : (interface.iteration offset).VarsBelow bound)
    (inverse : (interface.inverse offset).VarsBelow bound)
    (recursive : (interface.recursive offset).VarsBelow bound)
    (priorPc : (interface.priorPc offset).VarsBelow bound)
    (selector : (interface.selector offset).VarsBelow bound)
    (initialState : ∀ index, (interface.initialState offset index).VarsBelow bound)
    (currentState : ∀ index, (interface.currentState offset index).VarsBelow bound)
    (prior : ∀ slot, ∀ word ∈ (interface.prior offset slot).words, word.VarsBelow bound)
    (active : ∀ word ∈ (interface.active offset).words, word.VarsBelow bound)
    (new : ∀ word ∈ (interface.new offset).words, word.VarsBelow bound)
    (inner : ∀ slot, ∀ word ∈ (interface.inner offset slot).words, word.VarsBelow bound)
    (next : ∀ slot, ∀ word ∈ (interface.next offset slot).words, word.VarsBelow bound) :
    ∀ row ∈ SlotSelect.rows interface offset, row.VarsBelow bound := by
  have one : (1 : Expr).VarsBelow bound := trivial
  have constants : ∀ word ∈ SlotSelect.defaultWords (logicalWidth := logicalWidth)
      (publicFits := publicFits), word.VarsBelow bound := by
    intro word member
    simp only [SlotSelect.defaultWords, List.mem_map] at member
    obtain ⟨_, _, rfl⟩ := member
    trivial
  have isSelected : ∀ slot, (SlotSelect.isSelected interface offset slot).VarsBelow bound := by
    intro slot
    unfold SlotSelect.isSelected
    split
    · exact Expr.VarsBelow.sub _ _ _ one selector
    · exact selector
  intro row member
  simp only [SlotSelect.rows, SlotSelect.flagRows, SlotSelect.baseStateRows, SlotSelect.wordRows,
    List.mem_append, List.mem_cons, List.not_mem_nil, or_false, List.mem_ofFn] at member
  rcases member with ((flag | base) | words)
  · rcases flag with rfl | rfl | rfl | rfl
    · exact Expr.VarsBelow.sub _ _ _ recursive ⟨iteration, inverse⟩
    · exact ⟨iteration, Expr.VarsBelow.sub _ _ _ one recursive⟩
    · exact ⟨selector, Expr.VarsBelow.sub _ _ _ selector one⟩
    · exact ⟨recursive, Expr.VarsBelow.sub _ _ _ (Expr.VarsBelow.sub _ _ _ priorPc selector) one⟩
  · obtain ⟨index, rfl⟩ := base
    exact ⟨Expr.VarsBelow.sub _ _ _ one recursive,
      Expr.VarsBelow.sub _ _ _ (initialState index) (currentState index)⟩
  · rcases words with (((first | second) | third) | fourth) | fifth
    · exact selectRows_below selector (prior 0) (prior 1) active _ first
    · exact selectRows_below (isSelected 0) (prior 0) new (inner 0) _ second
    · exact selectRows_below (isSelected 1) (prior 1) new (inner 1) _ third
    · exact selectRows_below recursive constants (inner 0) (next 0) _ fourth
    · exact selectRows_below recursive constants (inner 1) (next 1) _ fifth

theorem assertZero_localLength (expressions : List Expr) :
    localLength (expressions.map Op.assertZero) = 0 := by
  induction expressions with
  | nil => rfl
  | cons _ rest ih =>
      simp only [List.map_cons, localLength, Op.localLength, List.sum_cons, Nat.zero_add]
      simpa [localLength] using ih

/-! ### Π_RLC outputs -/

theorem combination_nextOffset {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : CombinationStep.Interface blockCount cellCount) (offset : Nat) :
    Formal.nextOffset (CombinationStep.circuit interface) offset =
      offset + CombinationStep.privateCount blockCount cellCount := by
  unfold Formal.nextOffset Formal.childLength
  change offset + localLength (CombinationStep.operations interface offset) = _
  rw [CombinationStep.localLength_eq]

theorem combination_output_below {blockCount cellCount : Nat} [NeZero cellCount] (offset : Nat)
    (index : Fin (CombinationStep.privateCount blockCount cellCount)) :
    (CombinationStep.output offset index).VarsBelow
      (offset + CombinationStep.privateCount blockCount cellCount) := by
  show offset + index.val < _
  have := index.isLt
  omega

/-- The last step's output of a combination family (`CombinationFamily.output`) lies below the
family's end. -/
theorem family_output_below {blockCount cellCount : Nat} [NeZero cellCount] (offset : Nat)
    (block : Fin blockCount) (lane : Fin ringDegree) (cell : Fin cellCount) :
    (CombinationStep.output (CombinationFamily.stepOffset offset
        CombinationFamily.finalSource.val blockCount cellCount)
      (CombinationStep.indexOf block lane cell)).VarsBelow
      (offset + CombinationFamily.logicalPrivateCount blockCount cellCount) := by
  have index := (CombinationStep.indexOf block lane cell).isLt
  show CombinationFamily.stepOffset offset CombinationFamily.finalSource.val blockCount cellCount +
    (CombinationStep.indexOf block lane cell).val < _
  simp only [CombinationFamily.stepOffset, CombinationFamily.logicalPrivateCount,
    CombinationFamily.stepSize, CombinationFamily.finalSource, CombinationFamily.sourceCount_eq]
  omega

/-! ### The light step -/

section Light

variable (program : Stage1.Application.Program)
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits) (circuit : Fin 2)
  (interface : LightStepLayout.Interface logicalWidth publicFits program.witnessWordCount)
  (offset : Nat)

/-- The light Π_CCS phase and the light Π_RLC phase end at the slot-select rows; the SumCheck
point lies before the Π_RLC phase. -/
theorem light_offsets :
    LightPiCCSLayout.initialOffset (LightStepLayout.piccsInterface program interface)
        (LightStepLayout.piccsOffset program circuit interface offset) ≤
      LightStepLayout.rlcOffset program relation circuit interface offset ∧
    LightStepLayout.rlcOffset program relation circuit interface offset ≤
      LightRlcLayout.commitmentOffset (LightStepLayout.piccsLeaves program relation circuit
        interface offset) (LightStepLayout.rlcOffset program relation circuit interface offset) ∧
    LightRlcLayout.publicInputOffset (LightStepLayout.piccsLeaves program relation circuit
        interface offset) (LightStepLayout.rlcInterface program interface)
        (LightStepLayout.rlcOffset program relation circuit interface offset) ≤
      LightRlcLayout.evaluationOffset (LightStepLayout.piccsLeaves program relation circuit
        interface offset) (LightStepLayout.rlcInterface program interface)
        (LightStepLayout.rlcOffset program relation circuit interface offset) := by
  let piccsInterface := LightStepLayout.piccsInterface program interface
  let piccsOffset := LightStepLayout.piccsOffset program circuit interface offset
  refine ⟨?_, LightRlcBounds.commitment_le _ _, LightRlcBounds.evaluation_le _ _ _⟩
  have chain := Nat.le_trans (LightPiCCSBounds.sumcheck_le piccsInterface piccsOffset)
    (Nat.le_trans (LightPiCCSBounds.evalK_le piccsInterface piccsOffset)
      (Nat.le_trans (LightPiCCSBounds.evalA_le piccsInterface piccsOffset)
        (Nat.le_trans (LightPiCCSBounds.ccs_le piccsInterface piccsOffset)
          (Nat.le_trans (LightPiCCSBounds.norm_le relation piccsInterface piccsOffset)
            (Nat.le_trans (LightPiCCSBounds.finalIdentity_le relation piccsInterface piccsOffset)
              (LightPiCCSBounds.output_le relation piccsInterface piccsOffset))))))
  exact Nat.le_trans chain (LightPiCCSBounds.le_nextOffset _ _)

/-- The light new slot reads below the slot-select rows. -/
theorem light_newSlot_below (below : LightStepBounds.InputsBelow interface offset)
    (endLe : offset ≤ LightStepLayout.selectOffset program relation circuit interface offset) :
    ∀ word ∈ (LightStepLayout.newSlot program relation circuit interface offset).words,
      word.VarsBelow (LightStepLayout.selectOffset program relation circuit interface offset) := by
  obtain ⟨pointLe, rlcLe, publicLe⟩ := light_offsets program relation circuit interface offset
  have commitmentEq := combination_nextOffset (LightCommitment.stepInterface
    (LightRlcLayout.commitmentInterface (LightStepLayout.rlcInterface program interface)
      (LightStepLayout.rlcOffset program relation circuit interface offset)))
    (LightRlcLayout.commitmentOffset (LightStepLayout.piccsLeaves program relation circuit
      interface offset) (LightStepLayout.rlcOffset program relation circuit interface offset))
  have publicEq := combination_nextOffset (LightPublicInput.stepInterface
    (LightRlcLayout.publicInputInterface (LightStepLayout.rlcInterface program interface)
      (LightStepLayout.rlcOffset program relation circuit interface offset)))
    (LightRlcLayout.publicInputOffset (LightStepLayout.piccsLeaves program relation circuit
      interface offset) (LightStepLayout.rlcInterface program interface)
      (LightStepLayout.rlcOffset program relation circuit interface offset))
  have evaluationEq := combination_nextOffset (LightEvaluation.stepInterface
    (LightRlcLayout.evaluationInterface (LightStepLayout.piccsLeaves program relation circuit
      interface offset) (LightStepLayout.rlcOffset program relation circuit interface offset)))
    (LightRlcLayout.evaluationOffset (LightStepLayout.piccsLeaves program relation circuit
      interface offset) (LightStepLayout.rlcInterface program interface)
      (LightStepLayout.rlcOffset program relation circuit interface offset))
  change LightRlcLayout.publicInputOffset _ _ _ = _ at commitmentEq
  change LightRlcLayout.evaluationOffset _ _ _ = _ at publicEq
  change LightRlcLayout.finalOffset _ _ _ = _ at evaluationEq
  have pointBelow := LightPiCCSBounds.roundPoint_below
    (LightStepLayout.piccsInterface program interface)
    (LightStepLayout.piccsOffset program circuit interface offset)
    (LightStepBounds.piccsBelow program circuit interface offset below)
  have commitmentLe : LightRlcLayout.commitmentOffset (LightStepLayout.piccsLeaves program
      relation circuit interface offset) (LightStepLayout.rlcOffset program relation circuit
      interface offset) ≤ LightRlcLayout.publicInputOffset (LightStepLayout.piccsLeaves program
      relation circuit interface offset) (LightStepLayout.rlcInterface program interface)
      (LightStepLayout.rlcOffset program relation circuit interface offset) := by
    rw [commitmentEq]
    exact Nat.le_add_right _ _
  have evaluationLe : LightRlcLayout.evaluationOffset (LightStepLayout.piccsLeaves program
      relation circuit interface offset) (LightStepLayout.rlcInterface program interface)
      (LightStepLayout.rlcOffset program relation circuit interface offset) ≤
      LightRlcLayout.finalOffset (LightStepLayout.piccsLeaves program relation circuit interface
        offset) (LightStepLayout.rlcInterface program interface)
        (LightStepLayout.rlcOffset program relation circuit interface offset) := by
    rw [evaluationEq]
    exact Nat.le_add_right _ _
  have publicEnd := publicLe.trans evaluationLe
  apply words_below
  · intro row lane
    exact Expr.VarsBelow.mono _ (combination_output_below _ _)
      ((Nat.le_of_eq commitmentEq.symm).trans publicEnd)
  · intro coefficient
    exact ⟨Expr.VarsBelow.mono _ (combination_output_below _ _) (Nat.le_of_eq evaluationEq.symm),
      Expr.VarsBelow.mono _ (combination_output_below _ _) (Nat.le_of_eq evaluationEq.symm)⟩
  · intro matrix coefficient
    exact ⟨Expr.VarsBelow.mono _ (combination_output_below _ _) (Nat.le_of_eq evaluationEq.symm),
      Expr.VarsBelow.mono _ (combination_output_below _ _) (Nat.le_of_eq evaluationEq.symm)⟩
  · intro coordinate
    exact KExpr.varsBelow_mono _ (pointBelow coordinate)
      (pointLe.trans (rlcLe.trans (commitmentLe.trans publicEnd)))
  · intro column
    exact Expr.VarsBelow.mono _ (combination_output_below _ _)
      ((Nat.le_of_eq publicEq.symm).trans evaluationLe)
  · exact ⟨Expr.VarsBelow.mono _ below.activeDepth endLe, trivial⟩

/-- The light slot-select rows read below their own offset. -/
theorem light_select_below (below : LightStepBounds.InputsBelow interface offset)
    (endLe : offset ≤ LightStepLayout.selectOffset program relation circuit interface offset)
    (inverse : interface.inverse.VarsBelow offset) (recursive : interface.recursive.VarsBelow offset)
    (selector : interface.selector.VarsBelow offset)
    (active : ∀ word ∈ interface.active.words, word.VarsBelow offset)
    (inner : ∀ slot, ∀ word ∈ (interface.inner slot).words, word.VarsBelow offset) :
    ∀ row ∈ SlotSelect.rows (LightStepLayout.selectInterface program relation circuit interface
        offset) (LightStepLayout.selectOffset program relation circuit interface offset),
      row.VarsBelow (LightStepLayout.selectOffset program relation circuit interface offset) :=
  select_rows_below _ _ _ (Expr.VarsBelow.mono _ below.iteration endLe)
    (Expr.VarsBelow.mono _ inverse endLe) (Expr.VarsBelow.mono _ recursive endLe)
    (Expr.VarsBelow.mono _ below.priorPc endLe) (Expr.VarsBelow.mono _ selector endLe)
    (fun index => Expr.VarsBelow.mono _ (below.initialState index) endLe)
    (fun index => Expr.VarsBelow.mono _ (below.currentState index) endLe)
    (fun slot word member => Expr.VarsBelow.mono _ (below.prior slot word member) endLe)
    (fun word member => Expr.VarsBelow.mono _ (active word member) endLe)
    (light_newSlot_below program relation circuit interface offset below endLe)
    (fun slot word member => Expr.VarsBelow.mono _ (inner slot word member) endLe)
    (fun slot word member => Expr.VarsBelow.mono _ (below.next slot word member) endLe)

/-- `light_select_below` at the light source layout. -/
theorem light_assembler_select_below (base : Nat)
    (endLe : LightAssembler.sharedEnd base program.witnessWordCount ≤
      LightStepLayout.selectOffset program relation circuit
        (LightAssembler.interface base program.witnessWordCount)
        (LightAssembler.sharedEnd base program.witnessWordCount)) :
    ∀ row ∈ SlotSelect.rows (LightStepLayout.selectInterface program relation circuit
        (LightAssembler.interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
          program.witnessWordCount) (LightAssembler.sharedEnd base program.witnessWordCount))
        (LightStepLayout.selectOffset program relation circuit
          (LightAssembler.interface base program.witnessWordCount)
          (LightAssembler.sharedEnd base program.witnessWordCount)),
      row.VarsBelow (LightStepLayout.selectOffset program relation circuit
        (LightAssembler.interface base program.witnessWordCount)
        (LightAssembler.sharedEnd base program.witnessWordCount)) := by
  have endEq := LightAssembler.sharedEnd_eq base program.witnessWordCount
  have slots := fun (group : Nat) (bound : group < 7) =>
    LightAssembler.slotAt_below (logicalWidth := logicalWidth) (publicFits := publicFits)
      (LightAssembler.slotsStart base program.witnessWordCount + group * LightAssembler.slotWordCount)
      (LightAssembler.sharedEnd base program.witnessWordCount) (by
        simp only [LightAssembler.slotsStart, LightAssembler.witnessStart,
          LightAssembler.slotWordCount] at endEq ⊢
        rw [endEq]
        have : group * 2055 ≤ 6 * 2055 := Nat.mul_le_mul_right _ (by omega)
        omega)
  exact light_select_below program relation circuit _ _
    (LightAssembler.inputsBelow base program.witnessWordCount) endLe
    (show base + 1 < _ by omega) (show base + 2 < _ by omega) (show base + 4 < _ by omega)
    (slots 2 (by decide)) (fun slot => slots (3 + slot.val) (by omega))

end Light

/-! ### The refresh step -/

section Refresh

variable (program : Stage1.Application.Program)
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits) (circuit : Fin 2)
  (interface : RefreshStepLayout.Interface logicalWidth publicFits program.witnessWordCount)
  (offset : Nat)

/-- The refresh Π_CCS phase ends where today's Π_RLC starts, and the SumCheck point lies before
it. -/
theorem refresh_point_le :
    RefreshPiCCSLayout.initialOffset (RefreshStepLayout.piccsInterface program interface)
        (RefreshStepLayout.piccsOffset program circuit interface offset) ≤
      RefreshStepLayout.rlcOffset program relation circuit interface offset := by
  let piccsInterface := RefreshStepLayout.piccsInterface program interface
  let piccsOffset := RefreshStepLayout.piccsOffset program circuit interface offset
  have chain := Nat.le_trans (RefreshPiCCSBounds.sumcheck_le piccsInterface piccsOffset)
    (Nat.le_trans (RefreshPiCCSBounds.evalK_le piccsInterface piccsOffset)
      (Nat.le_trans (RefreshPiCCSBounds.evalA_le piccsInterface piccsOffset)
        (Nat.le_trans (RefreshPiCCSBounds.ccs_le piccsInterface piccsOffset)
          (Nat.le_trans (RefreshPiCCSBounds.norm_le relation piccsInterface piccsOffset)
            (Nat.le_trans (RefreshPiCCSBounds.finalIdentity_le relation piccsInterface piccsOffset)
              (RefreshPiCCSBounds.output_le relation piccsInterface piccsOffset))))))
  exact Nat.le_trans chain (RefreshPiCCSBounds.le_nextOffset _ _)

/-- The refresh new slot reads below the slot-select rows. -/
theorem refresh_newSlot_below (below : RefreshStepBounds.InputsBelow interface offset) :
    ∀ word ∈ (RefreshStepLayout.newSlot program relation circuit interface offset).words,
      word.VarsBelow (RefreshStepLayout.selectOffset program relation circuit interface offset) := by
  have pointLe := refresh_point_le program relation circuit interface offset
  have pointBelow := RefreshPiCCSBounds.roundPoint_below
    (RefreshStepLayout.piccsInterface program interface)
    (RefreshStepLayout.piccsOffset program circuit interface offset)
    (RefreshStepBounds.piccsBelow program circuit interface offset below)
  have offsets : ∀ rlcAt : Nat,
      PiRLC.v1_2.Formal.commitmentOffset rlcAt + 20196 ≤ PiRLC.v1_2.Formal.finalOffset rlcAt ∧
        PiRLC.v1_2.Formal.publicInputOffset rlcAt + 4590 ≤ PiRLC.v1_2.Formal.finalOffset rlcAt ∧
        PiRLC.v1_2.Formal.evalKOffset rlcAt + 1836 ≤ PiRLC.v1_2.Formal.finalOffset rlcAt ∧
        PiRLC.v1_2.Formal.evalAOffset rlcAt + 7344 ≤ PiRLC.v1_2.Formal.finalOffset rlcAt ∧
        rlcAt ≤ PiRLC.v1_2.Formal.finalOffset rlcAt := by
    intro rlcAt
    simp only [PiRLC.v1_2.Formal.finalOffset, PiRLC.v1_2.Formal.outputBindingOffset,
      PiRLC.v1_2.Formal.evalAOffset, PiRLC.v1_2.Formal.evalKOffset,
      PiRLC.v1_2.Formal.publicInputOffset, PiRLC.v1_2.Formal.commitmentOffset,
      PiRLC.v1_2.Formal.samplerOffset]
    omega
  obtain ⟨commitmentEnd, publicEnd, evalKEnd, evalAEnd, rlcEnd⟩ :=
    offsets (RefreshStepLayout.rlcOffset program relation circuit interface offset)
  apply words_below
  · intro row lane
    refine Expr.VarsBelow.mono _ (family_output_below _ _ _ _) ?_
    rw [CommitmentCombination.logicalPrivateCount_eq]
    exact commitmentEnd
  · intro coefficient
    refine ⟨Expr.VarsBelow.mono _ (family_output_below _ _ _ _) ?_,
      Expr.VarsBelow.mono _ (family_output_below _ _ _ _) ?_⟩ <;>
    · rw [EvalKCombination.logicalPrivateCount_eq]
      exact evalKEnd
  · intro matrix coefficient
    refine ⟨Expr.VarsBelow.mono _ (family_output_below _ _ _ _) ?_,
      Expr.VarsBelow.mono _ (family_output_below _ _ _ _) ?_⟩ <;>
    · rw [EvalACombination.logicalPrivateCount_eq]
      exact evalAEnd
  · intro coordinate
    exact KExpr.varsBelow_mono _ (pointBelow coordinate) (pointLe.trans rlcEnd)
  · intro column
    refine Expr.VarsBelow.mono _ (family_output_below _ _ _ _) ?_
    rw [PublicInputCombination.logicalPrivateCount_eq]
    exact publicEnd
  · trivial

/-- The refresh slot-select rows read below their own offset. -/
theorem refresh_select_below (below : RefreshStepBounds.InputsBelow interface offset)
    (endLe : offset ≤ RefreshStepLayout.selectOffset program relation circuit interface offset)
    (inverse : interface.inverse.VarsBelow offset) (recursive : interface.recursive.VarsBelow offset)
    (selector : interface.selector.VarsBelow offset)
    (active : ∀ word ∈ interface.active.words, word.VarsBelow offset)
    (inner : ∀ slot, ∀ word ∈ (interface.inner slot).words, word.VarsBelow offset) :
    ∀ row ∈ SlotSelect.rows (RefreshStepLayout.selectInterface program relation circuit interface
        offset) (RefreshStepLayout.selectOffset program relation circuit interface offset),
      row.VarsBelow (RefreshStepLayout.selectOffset program relation circuit interface offset) :=
  select_rows_below _ _ _ (Expr.VarsBelow.mono _ below.iteration endLe)
    (Expr.VarsBelow.mono _ inverse endLe) (Expr.VarsBelow.mono _ recursive endLe)
    (Expr.VarsBelow.mono _ below.priorPc endLe) (Expr.VarsBelow.mono _ selector endLe)
    (fun index => Expr.VarsBelow.mono _ (below.initialState index) endLe)
    (fun index => Expr.VarsBelow.mono _ (below.currentState index) endLe)
    (fun slot word member => Expr.VarsBelow.mono _ (below.prior slot word member) endLe)
    (fun word member => Expr.VarsBelow.mono _ (active word member) endLe)
    (refresh_newSlot_below program relation circuit interface offset below)
    (fun slot word member => Expr.VarsBelow.mono _ (inner slot word member) endLe)
    (fun slot word member => Expr.VarsBelow.mono _ (below.next slot word member) endLe)

/-- `refresh_select_below` at the refresh source layout. -/
theorem refresh_assembler_select_below (base : Nat)
    (endLe : RefreshAssembler.sharedEnd base program.witnessWordCount ≤
      RefreshStepLayout.selectOffset program relation circuit
        (RefreshAssembler.interface base program.witnessWordCount)
        (RefreshAssembler.sharedEnd base program.witnessWordCount)) :
    ∀ row ∈ SlotSelect.rows (RefreshStepLayout.selectInterface program relation circuit
        (RefreshAssembler.interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
          program.witnessWordCount) (RefreshAssembler.sharedEnd base program.witnessWordCount))
        (RefreshStepLayout.selectOffset program relation circuit
          (RefreshAssembler.interface base program.witnessWordCount)
          (RefreshAssembler.sharedEnd base program.witnessWordCount)),
      row.VarsBelow (RefreshStepLayout.selectOffset program relation circuit
        (RefreshAssembler.interface base program.witnessWordCount)
        (RefreshAssembler.sharedEnd base program.witnessWordCount)) := by
  have endEq := RefreshAssembler.sharedEnd_eq base program.witnessWordCount
  have slots := fun (group : Nat) (bound : group < 7) =>
    LightAssembler.slotAt_below (logicalWidth := logicalWidth) (publicFits := publicFits)
      (LightAssembler.slotsStart base program.witnessWordCount + group * LightAssembler.slotWordCount)
      (RefreshAssembler.sharedEnd base program.witnessWordCount) (by
        simp only [LightAssembler.slotsStart, LightAssembler.witnessStart,
          LightAssembler.slotWordCount] at endEq ⊢
        rw [endEq]
        have : group * 2055 ≤ 6 * 2055 := Nat.mul_le_mul_right _ (by omega)
        omega)
  exact refresh_select_below program relation circuit _ _
    (RefreshAssembler.inputsBelow base program.witnessWordCount) endLe
    (show base + 1 < _ by omega) (show base + 2 < _ by omega) (show base + 4 < _ by omega)
    (slots 2 (by decide)) (fun slot => slots (3 + slot.val) (by omega))

end Refresh

end NightstreamFPrime.Layout.OptionB.StepFootprint
