import NightstreamFPrime.Export.Stage1.Wide.Stage1Witness

/-! The PiDEC parent views read the newly retained PiRLC outputs, including
the extension-cell permutation. No copied parent fields are allocated. -/

namespace NightstreamFPrime.Export.Stage1.Wide.PiDECOutput

open NightstreamFPrime.Spec NightstreamFPrime.Layout
open ProductionRelation
open Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

private theorem recompose_map {source target : Nat} (column : Fin source → Fin target)
    (forms : List (SparseForm source)) :
    (RetainedSlot.recomposeForms forms).mapColumns column =
      RetainedSlot.recomposeForms (forms.map (SparseForm.mapColumns column)) := by
  induction forms with
  | nil => rfl
  | cons head tail ih =>
    change (SparseForm.add head (SparseForm.scale _ (RetainedSlot.recomposeForms tail))).mapColumns column = _
    have addMap (left right : SparseForm source) :
        (SparseForm.add left right).mapColumns column =
          SparseForm.add (left.mapColumns column) (right.mapColumns column) := by
      simp [SparseForm.mapColumns, SparseForm.add, List.map_append]
    have scaleMap (coefficient : F) (form : SparseForm source) :
        (SparseForm.scale coefficient form).mapColumns column =
          SparseForm.scale coefficient (form.mapColumns column) := by
      simp [SparseForm.mapColumns, SparseForm.scale, List.map_map]
    rw [addMap, scaleMap, ih]
    rfl

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

theorem output_column (program : RetainedLayout.Program) (ring : PiRLCGeometry.RingIndex)
    (lane : Fin ringDegree) (digit : Fin 41) :
    RetainedLayout.column program
      ((PiRLCRetainedGeometry.productOutputBlock program).column
        (PiRLCRetainedGeometry.productOutputStart program)
        (PiRLCRetainedGeometry.productOutputFits
          (PiCCSPoseidonPlan.prefixGeometry (Stage1Plan.poseidonGeometry program)))
        (PiRLCProductRingSchedule.laneInvocation ring lane) digit) =
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
      RetainedLayout.column_of_some program _ _ lookup
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
    (referenceOutput program ring lane).mapColumns (RetainedLayout.column program) =
      PiRLCGeometry.output (Stage1Plan.piRlcInterface program) ring lane := by
  unfold referenceOutput PiRLCGeometry.output LowNormBlock.Block.form
  rw [recompose_map, List.map_ofFn]
  apply congrArg RetainedSlot.recomposeForms
  apply congrArg List.ofFn
  funext digit
  change SparseForm.singleton _ 1 = SparseForm.singleton _ 1
  exact congrArg (fun column => SparseForm.singleton column (1 : F)) (output_column program ring lane digit)

theorem reference_output_value (program : RetainedLayout.Program)
    (base : Assignment F (RetainedLayout.logicalWidth program))
    (family : PiRLCOutput.Family) (block : Fin family.blockCount) (cell : Fin family.cellCount)
    (lane : Fin ringDegree) :
    ((referenceOutput program (PiRLCOutput.terminal family block cell) lane).mapColumns
      (RetainedLayout.column program)).eval (Stage1Witness.assignment program base) =
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
    ((parentView family block cell lane).form (Stage1Plan.piDecGeometry program)).mapColumns
      (RetainedLayout.column program) =
      PiRLCGeometry.output (Stage1Plan.piRlcInterface program)
        (PiRLCOutput.terminal family block cell) lane := by
  rw [parent_reference_form, output_form]

end NightstreamFPrime.Export.Stage1.Wide.PiDECOutput
