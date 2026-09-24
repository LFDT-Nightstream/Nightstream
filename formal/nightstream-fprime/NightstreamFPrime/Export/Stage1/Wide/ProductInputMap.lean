import NightstreamFPrime.Export.Stage1.Wide.MatrixProjection
import NightstreamFPrime.Export.Stage1.Wide.CheckedForm
import NightstreamFPrime.Layout.MatrixProgram.RetainedMap
import NightstreamFPrime.Export.Stage1.PiRLCValueMatrixProgram

/-! Compact relocation of the ring product's retained input blocks.
Only their starts move; source-key grids and term order are unchanged. -/

namespace NightstreamFPrime.Export.Stage1.Wide.ProductInputMap

open NightstreamFPrime.Spec NightstreamFPrime.Layout
open MatrixProgram ProductionRelation

abbrev Program := RetainedLayout.Program

def move : RetainedBlock → RetainedBlock := RetainedBlock.shift 7389186

private theorem move_slotCount (block : RetainedBlock) : (move block).slotCount = block.slotCount :=
  RetainedBlock.shift_slotCount 7389186 block

def Inside (program : Program) (block : RetainedBlock) : Prop :=
  RetainedLayout.sharedStart program ≤ block.start ∧
    block.start + block.coordinateCount ≤ RetainedLayout.sharedEnd program

private theorem shared_column (program : Program) (source : Nat)
    (lower : RetainedLayout.sharedStart program ≤ source)
    (upper : source < RetainedLayout.sharedEnd program) :
    RetainedLayout.column? program source = some (source - 7389186) := by
  obtain ⟨hash, first, last, _⟩ := RetainedLayout.boundaries program
  rw [first] at lower
  rw [last] at upper
  unfold RetainedLayout.column?
  rw [hash, first, last, if_neg (by omega), if_pos (by omega)]
  exact congrArg some (by omega)

/-- The wire block at its shifted start decodes the exact checked projection,
including rejection of an invalid slot. -/
theorem block_form (program : Program) (block : RetainedBlock) (inside : Inside program block) (slot : Nat) :
    (move block).form? (RetainedLayout.logicalWidth program) slot =
      (block.form? (PerApplicationFixedPoint.logicalWidth program) slot).bind
        ((MatrixProjection.projection program).sparseForm? (RetainedLayout.logicalWidth program)) := by
  have lower := inside.1
  have upper := inside.2
  rw [(RetainedLayout.boundaries program).2.1] at lower
  rw [(RetainedLayout.boundaries program).2.2.1] at upper
  have oldFits : block.start + block.coordinateCount ≤ PerApplicationFixedPoint.logicalWidth program := by
    rw [RetainedLayout.referenceWidth_eq]
    omega
  have newFits : block.start - 7389186 + block.coordinateCount ≤ RetainedLayout.logicalWidth program := by
    rw [RetainedLayout.logicalWidth_eq]
    omega
  by_cases bounded : slot < block.slotCount
  · let index : Fin block.slotCount := ⟨slot, bounded⟩
    have live (digit : Fin block.kind.width) : RetainedLayout.Live program
        (block.semantic.column block.start oldFits index digit).val := by
      have offset := (block.semantic.coordinateOffset index digit).isLt
      change _ < block.coordinateCount at offset
      apply Or.inr ∘ Or.inl
      change RetainedLayout.sharedStart program ≤ block.start + _ ∧ block.start + _ < RetainedLayout.sharedEnd program
      rw [(RetainedLayout.boundaries program).2.1, (RetainedLayout.boundaries program).2.2.1]
      constructor <;> omega
    have supported : ReadSupport.Form program (block.semantic.form block.start oldFits index) := by
      apply FormSupport.block
      intro column low high
      exact Or.inr (Or.inl ⟨le_trans inside.1 low, lt_of_lt_of_le high inside.2⟩)
    have decoded := SourceProjection.sparseForm?_checked (MatrixProjection.projection program)
      (RetainedLayout.column program) (block.semantic.form block.start oldFits index) supported
      (fun column valid => (MatrixProjection.column_eq program column.val).trans
        (RetainedLayout.column_mapped program column valid))
    have forms : RetainedLayout.renameForm program (block.semantic.form block.start oldFits index) supported =
        block.semantic.form (block.start - 7389186) newFits index := by
      unfold LowNormBlock.Block.form
      refine (FormSupport.rename_recompose_ofFn program _
        (fun digit => FormSupport.singleton _ 1 (live digit)) _).trans ?_
      apply congrArg RetainedSlot.recomposeForms
      apply congrArg List.ofFn
      funext digit
      rw [FormSupport.rename_singleton program _ _ (live digit)]
      apply congrArg (fun column => SparseForm.singleton column (1 : F))
      apply Fin.ext
      have offset := (block.semantic.coordinateOffset index digit).isLt
      change _ < block.coordinateCount at offset
      have lookup := shared_column program (block.semantic.column block.start oldFits index digit).val
        (by change RetainedLayout.sharedStart program ≤ block.start + _; exact le_trans inside.1 (Nat.le_add_right _ _))
        (by change block.start + _ < RetainedLayout.sharedEnd program; rw [(RetainedLayout.boundaries program).2.2.1]; omega)
      rw [RetainedLayout.column_of_some program _ _ _ lookup]
      change (block.start + (block.semantic.coordinateOffset index digit).val) - 7389186 =
        block.start - 7389186 + (block.semantic.coordinateOffset index digit).val
      omega
    have result := decoded.trans (congrArg some forms)
    have oldLoaded := RetainedBlock.form?_ofSemantic block.semantic block.start oldFits index
    have newLoaded := RetainedBlock.form?_ofSemantic block.semantic (block.start - 7389186) newFits index
    change block.form? (PerApplicationFixedPoint.logicalWidth program) slot = _ at oldLoaded
    change (move block).form? (RetainedLayout.logicalWidth program) slot = _ at newLoaded
    rw [oldLoaded, newLoaded]
    exact result.symm
  · have movedBound : ¬slot < (move block).slotCount := by
      rw [move_slotCount]
      exact bounded
    have newNone : (move block).form? (RetainedLayout.logicalWidth program) slot = none := by
      apply RetainedBlock.form?_of_not_lt
      exact movedBound
    have oldNone : block.form? (PerApplicationFixedPoint.logicalWidth program) slot = none := by
      apply RetainedBlock.form?_of_not_lt
      exact bounded
    exact newNone.trans (congrArg (fun value : Option (SparseForm (PerApplicationFixedPoint.logicalWidth program)) =>
      value.bind ((MatrixProjection.projection program).sparseForm? (RetainedLayout.logicalWidth program))) oldNone).symm

private def within (program : Program) : PiCCSOrdinaryRetainedGeometry.Geometry program (RetainedLayout.sharedEnd program) :=
  ⟨by rw [PiCCSOrdinaryRetainedGeometry.completeLogicalWidth_eq,
    (RetainedLayout.boundaries program).2.2.1]; decide⟩

private theorem prior_inside (program : Program) : Inside program
    (RetainedBlock.ofSemantic (PiCCSOrdinaryRetainedBlocks.priorInputBlock program)
      (PiCCSOrdinaryRetainedGeometry.priorInputStart program)) :=
  ⟨Nat.le_refl _, PiCCSOrdinaryRetainedGeometry.priorInputFits (within program)⟩

private theorem fresh_lower (program : Program) : RetainedLayout.sharedStart program ≤
    PiCCSOrdinaryRetainedGeometry.freshPublicInputStart program := by
  rw [(RetainedLayout.boundaries program).2.1]
  unfold PiCCSOrdinaryRetainedGeometry.freshPublicInputStart PiCCSOrdinaryRetainedGeometry.prefixLogicalWidth
  rw [RunningTransitionReducedRetainedBlocks.nextStart_eq]
  decide

private theorem fresh_inside (program : Program) : Inside program
    (RetainedBlock.ofSemantic (PiCCSOrdinaryRetainedBlocks.freshPublicInputBlock program)
      (PiCCSOrdinaryRetainedGeometry.freshPublicInputStart program)) :=
  ⟨fresh_lower program, PiCCSOrdinaryRetainedGeometry.freshPublicInputFits (within program)⟩

private theorem proof_inside (program : Program) : Inside program
    (RetainedBlock.ofSemantic (PiCCSOrdinaryRetainedBlocks.proofLogicalBlock program)
      (PiCCSOrdinaryRetainedGeometry.proofLogicalStart program)) := by
  refine ⟨?_, PiCCSOrdinaryRetainedGeometry.proofLogicalFits (within program)⟩
  have lower := fresh_lower program
  change RetainedLayout.sharedStart program ≤ PiCCSOrdinaryRetainedGeometry.proofLogicalStart program
  unfold PiCCSOrdinaryRetainedGeometry.proofLogicalStart PiCCSOrdinaryRetainedGeometry.expectedContextStart
    PiCCSOrdinaryRetainedGeometry.outputLastStart PiCCSOrdinaryRetainedGeometry.priorLastStart
  omega

def substitution (program : Program) : SourceSubstitution :=
  (PiRLCValueMatrixProgram.substitution program).mapRetained move

/-- Each of the six source maps decodes the exact candidate input form. -/
theorem form (program : Program) (ring : PiRLCGeometry.RingIndex) (lane : Fin ringDegree) :
    (substitution program).form? (RetainedLayout.logicalWidth program)
        (PiRLCProductRingSchedule.laneInvocation ring lane).val = some (Stage1Plan.value program ring lane) := by
  apply SourceSubstitution.mapRetained_form (before := PerApplicationFixedPoint.logicalWidth program)
    (PiRLCValueMatrixProgram.substitution program) move
    ((MatrixProjection.projection program).sparseForm? (RetainedLayout.logicalWidth program))
  · intro range member slot
    simp only [PiRLCValueMatrixProgram.substitution, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl
    · exact block_form program _ (proof_inside program) slot
    · exact block_form program _ (fresh_inside program) slot
  · intro grid member
    simp only [PiRLCValueMatrixProgram.substitution, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl | rfl | rfl
    all_goals refine ⟨rfl, fun slot => ?_⟩
    · exact block_form program _ (prior_inside program) _
    · exact block_form program _ (prior_inside program) _
    · exact block_form program _ (proof_inside program) _
    · exact block_form program _ (proof_inside program) _
  · exact PiRLCValueMatrixProgram.substitution_form? (Stage1Plan.piCcsGeometry program)
      (PiRLCProductRingSchedule.laneInvocation ring lane)
  · unfold Stage1Plan.value RetainedLayout.renameForm
    apply SourceProjection.sparseForm?_checked (column := RetainedLayout.column program)
    intro column live
    rw [MatrixProjection.column_eq]
    exact RetainedLayout.column_mapped program column live

end NightstreamFPrime.Export.Stage1.Wide.ProductInputMap
