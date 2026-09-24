import NightstreamFPrime.Export.Stage1.Wide.Stage1Witness
import NightstreamFPrime.Export.Stage1.Wide.CheckedForm

/-! The PiDEC parent views read the newly retained PiRLC outputs, including
the extension-cell permutation. No copied parent fields are allocated. -/

namespace NightstreamFPrime.Export.Stage1.Wide.PiDECOutput

open NightstreamFPrime.Spec NightstreamFPrime.Layout
open ProductionRelation
open Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

theorem referenceOutputStart (program : RetainedLayout.Program) :
    PiRLCRetainedGeometry.productOutputStart program = 119147994 := by
  have total := PiRLCRetainedGeometry.prefixLogicalWidth_eq program
  unfold PiRLCRetainedGeometry.prefixLogicalWidth at total
  change PiRLCRetainedGeometry.productOutputStart program +
    (PiRLCProductSourceBlocks.outputBlock program).coordinateCount = 121293360 at total
  rw [PiRLCProductSourceBlocks.outputBlock_coordinateCount] at total
  omega

def referenceOutput (program : RetainedLayout.Program) (ring : PiRLCGeometry.RingIndex)
    (lane : Fin ringDegree) : SparseForm (PerApplicationFixedPoint.logicalWidth program) :=
  (PiRLCRetainedGeometry.productOutputBlock program).form
    (PiRLCRetainedGeometry.productOutputStart program)
    (PiRLCRetainedGeometry.productOutputFits
      (PiCCSPoseidonPlan.prefixGeometry (Stage1Plan.poseidonGeometry program)))
    (PiRLCProductRingSchedule.laneInvocation ring lane)

theorem referenceOutput_live (program : RetainedLayout.Program) (ring : PiRLCGeometry.RingIndex)
    (lane : Fin ringDegree) : ReadSupport.Form program (referenceOutput program ring lane) := by
  exact ReadSupport.product_block program _ _ _ _ (Nat.le_refl _) (Nat.le_refl _)

theorem output_column_live (program : RetainedLayout.Program) (ring : PiRLCGeometry.RingIndex)
    (lane : Fin ringDegree) (digit : Fin 41) :
    RetainedLayout.Live program
      ((PiRLCRetainedGeometry.productOutputBlock program).column
        (PiRLCRetainedGeometry.productOutputStart program)
        (PiRLCRetainedGeometry.productOutputFits
          (PiCCSPoseidonPlan.prefixGeometry (Stage1Plan.poseidonGeometry program)))
        (PiRLCProductRingSchedule.laneInvocation ring lane) digit).val := by
  have slotBound := (PiRLCProductRingSchedule.laneInvocation ring lane).isLt
  change (PiRLCProductRingSchedule.laneInvocation ring lane).val < 52326 at slotBound
  have digitBound := digit.isLt
  apply Or.inr ∘ Or.inr ∘ Or.inr ∘ Or.inl
  change 119147994 ≤ PiRLCRetainedGeometry.productOutputStart program +
      ((PiRLCProductRingSchedule.laneInvocation ring lane).val * 41 + digit.val) ∧
    PiRLCRetainedGeometry.productOutputStart program +
      ((PiRLCProductRingSchedule.laneInvocation ring lane).val * 41 + digit.val) < 121293360
  rw [referenceOutputStart]
  constructor <;> omega

theorem output_column (program : RetainedLayout.Program) (ring : PiRLCGeometry.RingIndex)
    (lane : Fin ringDegree) (digit : Fin 41) :
    RetainedLayout.column program
      ((PiRLCRetainedGeometry.productOutputBlock program).column
        (PiRLCRetainedGeometry.productOutputStart program)
        (PiRLCRetainedGeometry.productOutputFits
          (PiCCSPoseidonPlan.prefixGeometry (Stage1Plan.poseidonGeometry program)))
        (PiRLCProductRingSchedule.laneInvocation ring lane) digit)
      (output_column_live program ring lane digit) =
      PiRLCGeometry.fieldBlock.column (PiRLCGeometry.fieldStart (Stage1Plan.piRlcInterface program))
        (PiRLCGeometry.fieldFits (Stage1Plan.piRlcInterface program))
        (PiRLCGeometry.outputSlot ring lane) digit := by
  let oldSlot := PiRLCProductRingSchedule.laneInvocation ring lane
  let coordinate : Fin 2145366 := Fin.encodeProd (oldSlot, digit)
  have bound := coordinate.isLt
  have mapped : RetainedLayout.column? program (119147994 + coordinate.val) =
      some (RetainedLayout.outputStart program + (ProductCoordinates.coordinate coordinate).val) := by
    obtain ⟨hash, start, stop, app⟩ := RetainedLayout.boundaries program
    unfold RetainedLayout.column?
    rw [hash, start, stop, app]
    rw [if_neg (by omega), if_neg (by omega), if_neg (by omega), dif_pos (by omega)]
    have indexEq : (⟨119147994 + coordinate.val - 119147994, by omega⟩ : Fin 2145366) = coordinate := by
      apply Fin.ext
      dsimp only
      omega
    exact congrArg (fun index : Fin 2145366 =>
      some (RetainedLayout.outputStart program + (ProductCoordinates.coordinate index).val)) indexEq
  apply Fin.ext
  have oldColumn :
      ((PiRLCRetainedGeometry.productOutputBlock program).column
        (PiRLCRetainedGeometry.productOutputStart program)
        (PiRLCRetainedGeometry.productOutputFits
          (PiCCSPoseidonPlan.prefixGeometry (Stage1Plan.poseidonGeometry program)))
        oldSlot digit).val = 119147994 + coordinate.val := by
    change PiRLCRetainedGeometry.productOutputStart program + (oldSlot.val * 41 + digit.val) = _
    rw [referenceOutputStart]
    simp only [coordinate, Fin.encodeProd, Fin.coe_mkDivMod]
    omega
  have lookup : RetainedLayout.column? program
      ((PiRLCRetainedGeometry.productOutputBlock program).column
        (PiRLCRetainedGeometry.productOutputStart program)
        (PiRLCRetainedGeometry.productOutputFits
          (PiCCSPoseidonPlan.prefixGeometry (Stage1Plan.poseidonGeometry program)))
        oldSlot digit).val =
      some (RetainedLayout.outputStart program + (ProductCoordinates.coordinate coordinate).val) := by
    rw [oldColumn]
    exact mapped
  calc
    _ = RetainedLayout.outputStart program + (ProductCoordinates.coordinate coordinate).val :=
      RetainedLayout.column_of_some program _ _ _ lookup
    _ = _ := by
      dsimp only [coordinate, oldSlot]
      rw [ProductCoordinates.coordinate_lane]
      change RetainedLayout.outputStart program + (41 * (ringDegree * ring.val + lane.val) + digit.val) =
        PiRLCGeometry.fieldStart (Stage1Plan.piRlcInterface program) +
          ((ring.val * ringDegree + lane.val) * 41 + digit.val)
      unfold RetainedLayout.outputStart PiRLCGeometry.fieldStart Stage1Plan.piRlcInterface
      ring

theorem output_form (program : RetainedLayout.Program) (ring : PiRLCGeometry.RingIndex)
    (lane : Fin ringDegree) :
    RetainedLayout.renameForm program (referenceOutput program ring lane) (referenceOutput_live program ring lane) =
      PiRLCGeometry.output (Stage1Plan.piRlcInterface program) ring lane := by
  unfold referenceOutput PiRLCGeometry.output LowNormBlock.Block.form
  refine (FormSupport.rename_recompose_ofFn program _ (fun digit =>
    FormSupport.singleton _ 1 (output_column_live program ring lane digit)) _).trans ?_
  apply congrArg RetainedSlot.recomposeForms
  apply congrArg List.ofFn
  funext digit
  rw [FormSupport.rename_singleton program _ _ (output_column_live program ring lane digit)]
  exact congrArg (fun column => SparseForm.singleton column (1 : F)) (output_column program ring lane digit)

theorem reference_output_value (program : RetainedLayout.Program)
    (base : Assignment F (RetainedLayout.logicalWidth program))
    (family : PiRLCOutput.Family) (block : Fin family.blockCount) (cell : Fin family.cellCount)
    (lane : Fin ringDegree) :
    (RetainedLayout.renameForm program (referenceOutput program (PiRLCOutput.terminal family block cell) lane)
      (referenceOutput_live program (PiRLCOutput.terminal family block cell) lane)).eval (Stage1Witness.assignment program base) =
      PiRLCOutput.ordered (PiRLCWitness.initial (Stage1Plan.piRlcInterface program) base)
        (PiRLCWitness.inputValues (Stage1Plan.piRlcInterface program) base) family block cell lane := by
  rw [output_form]
  exact congrFun (Stage1Witness.output program base family block cell) lane

def parentView (family : PiRLCOutput.Family) (block : Fin family.blockCount)
    (cell : Fin family.cellCount) (lane : Fin ringDegree) : PiDECDirectPlan.Location :=
  match family with
  | .commitment => .parentCommitment ⟨block.val * 54 + lane.val, by
      have b : block.val < 22 := block.isLt
      have l : lane.val < 54 := lane.isLt
      change _ < 1188
      omega⟩
  | .publicInput => .parentPublicInput ⟨block.val * 54 + lane.val, by
      have b : block.val < 5 := block.isLt
      have l : lane.val < 54 := lane.isLt
      change _ < 270
      omega⟩
  | .evalK => .parentEvalK ⟨lane.val * 2 + cell.val, by
      have l : lane.val < 54 := lane.isLt
      have c : cell.val < 2 := cell.isLt
      change _ < 108
      omega⟩
  | .evalA => .parentEvalA ⟨block.val * 108 + lane.val * 2 + cell.val, by
      have b : block.val < 14 := block.isLt
      have l : lane.val < 54 := lane.isLt
      have c : cell.val < 2 := cell.isLt
      change _ < 1512
      omega⟩

theorem terminal_lane (family : PiRLCOutput.Family)
    (block : Fin family.blockCount) (cell : Fin family.cellCount) (lane : Fin ringDegree) :
    PiRLCProductRingSchedule.laneInvocation (PiRLCOutput.terminal family block cell) lane =
      ({ family, source := ⟨16, by decide⟩, block, lane, cell } : PiRLCProductSchedule.Descriptor).invocation := by
  unfold PiRLCProductRingSchedule.laneInvocation PiRLCOutput.terminal
  rw [PiRLCProductRingSchedule.descriptor_invocation]
  rfl

theorem parent_reference_form (program : RetainedLayout.Program) (family : PiRLCOutput.Family)
    (block : Fin family.blockCount) (cell : Fin family.cellCount) (lane : Fin ringDegree) :
    (parentView family block cell lane).form (Stage1Plan.piDecGeometry program) =
      referenceOutput program (PiRLCOutput.terminal family block cell) lane := by
  have blockBound := block.isLt
  have cellBound := cell.isLt
  cases family <;>
    dsimp only [parentView, PiDECDirectPlan.Location.form, referenceOutput] <;>
    rw [terminal_lane] <;>
    refine LowNormBlock.Block.form_eq_of_coordinates _ _ _ _ _ _ _ _ rfl ?_
  all_goals
    norm_num [PiDECRetainedGeometry.parentCommitmentStart,
      PiDECRetainedGeometry.parentPublicInputStart, PiDECRetainedGeometry.parentEvalKStart,
      PiDECRetainedGeometry.parentEvalAStart, PiDECRetainedGeometry.parentCommitmentSlot,
      PiDECRetainedGeometry.parentPublicInputSlot, PiDECRetainedGeometry.parentEvalKSlot,
      PiDECRetainedGeometry.parentEvalASlot, PiRLCProductSchedule.Descriptor.invocation,
      PiRLCProductSchedule.Descriptor.familyIndex, PiRLCProductSchedule.Family.invocationCount,
      PiRLCProductSchedule.Family.privateCount, PiRLCProductSchedule.Family.blockCount,
      PiRLCProductSchedule.Family.cellCount, PiRLCCombinationInvocations.sourceCount,
      Lifecycle.PiRLC.v1_1.CombinationStep.privateCount, Lifecycle.PiRLC.v1_1.CombinationStep.indexOf,
      Fin.encodeProd, Fin.coe_mkDivMod,
      PiDECRetainedBlocks.parentCommitmentBlock, PiDECRetainedBlocks.parentPublicInputBlock,
      PiDECRetainedBlocks.parentEvalKBlock, PiDECRetainedBlocks.parentEvalABlock,
      PiDECRetainedBlocks.sourceFieldBlock, PiRLCRetainedGeometry.productOutputBlock,
      PiRLCProductSourceBlocks.outputBlock_kind, LowNormBlock.Block.lift_kind,
      LowNormSlot.Kind.width, BalancedTernary.width, ringDegree] at blockBound cellBound ⊢
    omega

theorem parent_form (program : RetainedLayout.Program) (family : PiRLCOutput.Family)
    (block : Fin family.blockCount) (cell : Fin family.cellCount) (lane : Fin ringDegree) :
    RetainedLayout.renameForm program ((parentView family block cell lane).form (Stage1Plan.piDecGeometry program))
      (ReadSupport.piDec_location program (Stage1Plan.piDecGeometry program) _) =
      PiRLCGeometry.output (Stage1Plan.piRlcInterface program)
        (PiRLCOutput.terminal family block cell) lane := by
  exact (RetainedLayout.renameForm_congr program (parent_reference_form program family block cell lane) _ _).trans
    (output_form program (PiRLCOutput.terminal family block cell) lane)

end NightstreamFPrime.Export.Stage1.Wide.PiDECOutput
