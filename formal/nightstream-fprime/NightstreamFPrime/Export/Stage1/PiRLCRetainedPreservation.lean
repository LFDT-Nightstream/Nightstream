import NightstreamFPrime.Export.Stage1.PiRLCRetainedInputs

/-!
Owns the preservation contract for the canonical retained PiRLC geometry.
One nested source assignment contains package values, product-group values,
and First54 accepted-symbol products. Each compact block encodes that same
source assignment at its fixed interval.

This module does not assume that matrix rows vanish.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCRetainedPreservation

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open PiRLCRetainedGeometry
open PiRLCRetainedInputs

def sourceAssignment (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F) :
    Fin (sourceWidth program) → F :=
  PiRLCFirst54DirectPlan.sourceAssignment program base groupValue products

/-- Canonical embedding of the complete application-package source prefix
through the two derived PiRLC suffixes. -/
def baseSourceColumn (program : Lifecycle.Stage1.Application.Program)
    (column : Fin (PiRLCProductPlan.baseSourceWidth program)) :
    Fin (sourceWidth program) :=
  PiRLCFirst54DirectPlan.prefixColumn program <|
    ProductRetainedBlock.baseColumn (PiRLCProductPlan.baseSourceWidth program)
      PiRLCProductSchedule.invocationCount column

/-- The nested retained source assignment preserves every package-prefix
column exactly. -/
theorem sourceAssignment_base
    (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (column : Fin (PiRLCProductPlan.baseSourceWidth program)) :
    sourceAssignment program base groupValue products
        (baseSourceColumn program column) = base column := by
  rw [sourceAssignment, baseSourceColumn,
    PiRLCFirst54DirectPlan.sourceAssignment_prefix]
  exact ProductRetainedBlock.sourceAssignment_base _ _ _ _ column

theorem sourceAssignment_package
    (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (column : Nat) (bound : column < PiRLCProductPlan.basePackage.layout.constantColumn) :
    sourceAssignment program base groupValue products
        (PiRLCFirst54DirectPlan.packageColumn program column bound) =
      PiRLCProductPlan.baseEnv program base column := by
  unfold PiRLCFirst54DirectPlan.packageColumn
  rw [sourceAssignment, PiRLCFirst54DirectPlan.sourceAssignment_prefix]
  unfold PiRLCProductPlan.sourceAssignment
  unfold PiRLCProductPlan.baseColumn
  rw [ProductRetainedBlock.sourceAssignment_base]
  exact (PiRLCProductPlan.baseEnv_eq_mappedPackageColumn
    program base column bound).symm

theorem sourceAssignment_group
    (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (invocation : Fin PiRLCProductSchedule.invocationCount)
    (group : Fin 33) :
    sourceAssignment program base groupValue products
        (PiRLCFirst54DirectPlan.prefixColumn program
          (PiRLCProductPlan.groupColumn program invocation group)) =
      groupValue invocation group := by
  rw [sourceAssignment, PiRLCFirst54DirectPlan.sourceAssignment_prefix]
  exact ProductRetainedBlock.sourceAssignment_group _ _ _ _ invocation group

theorem sourceAssignment_valueColumn
    (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (descriptor : PiRLCProductSchedule.Descriptor)
    (lane : Fin ringDegree) :
    sourceAssignment program base groupValue products
        (PiRLCFirst54DirectPlan.prefixColumn program
          (PiRLCProductPlan.valueColumn program descriptor lane)) =
      PiRLCProductPlan.baseEnv program base (descriptor.valueColumn lane) := by
  rw [sourceAssignment, PiRLCFirst54DirectPlan.sourceAssignment_prefix]
  unfold PiRLCProductPlan.valueColumn PiRLCProductPlan.baseColumn
  unfold PiRLCProductPlan.sourceAssignment
  rw [ProductRetainedBlock.sourceAssignment_base]
  exact (PiRLCProductPlan.baseEnv_valueColumn
    program base descriptor lane).symm

theorem sourceAssignment_outputColumn
    (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (descriptor : PiRLCProductSchedule.Descriptor) :
    sourceAssignment program base groupValue products
        (PiRLCFirst54DirectPlan.prefixColumn program
          (PiRLCProductPlan.outputColumn program descriptor)) =
      PiRLCProductPlan.baseEnv program base descriptor.outputColumn := by
  rw [sourceAssignment, PiRLCFirst54DirectPlan.sourceAssignment_prefix]
  unfold PiRLCProductPlan.outputColumn PiRLCProductPlan.baseColumn
  unfold PiRLCProductPlan.sourceAssignment
  rw [ProductRetainedBlock.sourceAssignment_base]
  exact (PiRLCProductPlan.baseEnv_outputColumn
    program base descriptor).symm

@[simp] theorem sourceAssignment_product
    (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (candidate : Fin PiRLCFirst54DirectSchedule.candidateCount) :
    sourceAssignment program base groupValue products
        (PiRLCFirst54DirectPlan.productColumn program candidate) =
      products candidate := by
  exact PiRLCFirst54DirectPlan.sourceAssignment_product
    program base groupValue products candidate

theorem productGroupBlock_source
    (program : Lifecycle.Stage1.Application.Program)
    (invocation : Fin PiRLCProductSchedule.invocationCount)
    (group : Fin 33) :
    (productGroupBlock program).source (Fin.encodeProd (invocation, group)) =
      PiRLCFirst54DirectPlan.prefixColumn program
        (PiRLCProductPlan.groupColumn program invocation group) := by
  apply Fin.ext
  simp [productGroupBlock, LowNormBlock.Block.lift,
    ProductRetainedBlock.block, PiRLCFirst54DirectPlan.prefixColumn,
    PiRLCProductPlan.groupColumn, FieldSuffixBlock.baseColumn]

theorem productInputBlock_source
    (program : Lifecycle.Stage1.Application.Program)
    (descriptor : PiRLCProductSchedule.Descriptor)
    (lane : Fin ringDegree) :
    (productInputBlock program).source
        (descriptor.withLane lane).invocation =
      PiRLCFirst54DirectPlan.prefixColumn program
        (PiRLCProductPlan.valueColumn program descriptor lane) := by
  apply Fin.ext
  simp [productInputBlock, LowNormBlock.Block.lift,
    PiRLCProductSourceBlocks.inputBlock,
    PiRLCFirst54DirectPlan.prefixColumn, PiRLCProductPlan.valueColumn,
    FieldSuffixBlock.baseColumn]

theorem productOutputBlock_source
    (program : Lifecycle.Stage1.Application.Program)
    (invocation : Fin PiRLCProductSchedule.invocationCount) :
    (productOutputBlock program).source invocation =
      PiRLCFirst54DirectPlan.prefixColumn program
        (PiRLCProductPlan.outputColumn program
          (PiRLCProductSchedule.descriptor invocation)) := by
  apply Fin.ext
  rfl

structure Encodes {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F) : Prop where
  priorPoseidon : (priorPoseidonBlock program).EncodesAt
    (priorPoseidonStart program) (priorPoseidonFits geometry) assignment
      (sourceAssignment program base groupValue products)
  outputPoseidon : (outputPoseidonBlock program).EncodesAt
    (outputPoseidonStart program) (outputPoseidonFits geometry) assignment
      (sourceAssignment program base groupValue products)
  laterPoseidon : (laterPoseidonBlock program).EncodesAt
    (laterPoseidonStart program) (laterPoseidonFits geometry) assignment
      (sourceAssignment program base groupValue products)
  productGroup : (productGroupBlock program).EncodesAt
    (productGroupStart program) (productGroupFits geometry) assignment
      (sourceAssignment program base groupValue products)
  reject : (PiRLCFirst54RetainedBlocks.rejectBlock program).EncodesAt
    (rejectStart program) (rejectFits geometry) assignment
      (sourceAssignment program base groupValue products)
  symbol : (PiRLCFirst54RetainedBlocks.symbolBlock program).EncodesAt
    (symbolStart program) (symbolFits geometry) assignment
      (sourceAssignment program base groupValue products)
  position : (PiRLCFirst54RetainedBlocks.positionBlock program).EncodesAt
    (positionStart program) (positionFits geometry) assignment
      (sourceAssignment program base groupValue products)
  value : (PiRLCFirst54RetainedBlocks.valueBlock program).EncodesAt
    (valueStart program) (valueFits geometry) assignment
      (sourceAssignment program base groupValue products)
  first54Product : (PiRLCFirst54RetainedBlocks.productBlock program).EncodesAt
    (first54ProductStart program) (first54ProductFits geometry) assignment
      (sourceAssignment program base groupValue products)
  productInput : (productInputBlock program).EncodesAt
    (productInputStart program) (productInputFits geometry) assignment
      (sourceAssignment program base groupValue products)
  productOutput : (productOutputBlock program).EncodesAt
    (productOutputStart program) (productOutputFits geometry) assignment
      (sourceAssignment program base groupValue products)

theorem productInputs_preserves
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (encodes : Encodes geometry assignment base groupValue products) :
    PiRLCProductPlan.Preserves (productInputs geometry)
      assignment base groupValue := by
  refine
    { challenge := ?_
      value := ?_
      prior := ?_
      output := ?_
      group := ?_ }
  · intro invocation lane
    let descriptor := PiRLCProductSchedule.descriptor invocation
    let valueDescriptor :=
      PiRLCProductSourceBlocks.challengeValueDescriptor descriptor.source lane
    change
      ((PiRLCFirst54RetainedBlocks.valueBlock program).form
        (valueStart program) (valueFits geometry)
          (PiRLCFirst54DirectSchedule.valueIndex valueDescriptor)).eval
        assignment = _
    rw [LowNormBlock.Block.form_eval _ _ _ assignment _ encodes.value]
    rw [PiRLCFirst54RetainedBlocks.valueBlock_source]
    rw [PiRLCFirst54DirectSchedule.value_valueIndex]
    unfold PiRLCFirst54DirectPlan.retainedValueColumn
    rw [sourceAssignment_package]
    rw [← PiRLCProductSourceBlocks.challengeColumn_eq_first54Value]
  · intro invocation lane
    let descriptor := PiRLCProductSchedule.descriptor invocation
    change
      ((productInputBlock program).form
        (productInputStart program) (productInputFits geometry)
          (descriptor.withLane lane).invocation).eval assignment = _
    rw [LowNormBlock.Block.form_eval _ _ _ assignment _ encodes.productInput]
    rw [productInputBlock_source]
    rw [sourceAssignment_valueColumn]
  · intro invocation
    let descriptor := PiRLCProductSchedule.descriptor invocation
    unfold PiRLCProductPlan.priorForm productInputs
      PiRLCProductPlan.priorValue
    dsimp only
    split
    · simp
    · rw [LowNormBlock.Block.form_eval _ _ _ assignment _
          encodes.productOutput]
      rw [productOutputBlock_source]
      rw [sourceAssignment_outputColumn]
      rw [PiRLCProductSchedule.descriptor_invocation]
      rw [PiRLCProductSchedule.Descriptor.previousSource_outputColumn]
  · intro invocation
    change
      ((productOutputBlock program).form
        (productOutputStart program) (productOutputFits geometry)
          invocation).eval assignment = _
    rw [LowNormBlock.Block.form_eval _ _ _ assignment _ encodes.productOutput]
    rw [productOutputBlock_source]
    rw [sourceAssignment_outputColumn]
    rfl
  · intro invocation group
    change
      ((productGroupBlock program).form
        (productGroupStart program) (productGroupFits geometry)
          (Fin.encodeProd (invocation, group))).eval assignment = _
    rw [LowNormBlock.Block.form_eval _ _ _ assignment _ encodes.productGroup]
    rw [productGroupBlock_source]
    rw [sourceAssignment_group]

theorem first54Inputs_preserves
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (one : assignment (PiRLCRetainedGeometry.oneColumn geometry) = 1)
    (encodes : Encodes geometry assignment base groupValue products) :
    PiRLCFirst54DirectPlan.Preserves (first54Inputs geometry)
      assignment base products := by
  refine
    { reject := ?_
      symbol := ?_
      priorPosition := ?_
      positionOutput := ?_
      product := ?_
      priorValue := ?_
      outputValue := ?_
      final := ?_ }
  · intro candidate
    change
      ((PiRLCFirst54RetainedBlocks.rejectBlock program).form
        (rejectStart program) (rejectFits geometry)
          (PiRLCFirst54DirectSchedule.candidateIndex candidate)).eval
        assignment = _
    rw [LowNormBlock.Block.form_eval _ _ _ assignment _ encodes.reject]
    rw [PiRLCFirst54RetainedBlocks.rejectBlock_source]
    rw [PiRLCFirst54DirectSchedule.candidate_candidateIndex]
    unfold PiRLCFirst54DirectPlan.retainedRejectColumn
      PiRLCFirst54DirectPlan.rejectValue
    rw [sourceAssignment_package]
    rfl
  · intro candidate
    change
      ((PiRLCFirst54RetainedBlocks.symbolBlock program).form
        (symbolStart program) (symbolFits geometry)
          (PiRLCFirst54DirectSchedule.candidateIndex candidate)).eval
        assignment = _
    rw [LowNormBlock.Block.form_eval _ _ _ assignment _ encodes.symbol]
    rw [PiRLCFirst54RetainedBlocks.symbolBlock_source]
    rw [PiRLCFirst54DirectSchedule.candidate_candidateIndex]
    unfold PiRLCFirst54DirectPlan.retainedSymbolColumn
      PiRLCFirst54DirectPlan.symbolValue
    rw [sourceAssignment_package]
    rfl
  · intro candidate slot
    by_cases first : candidate.round.val = 0
    · by_cases firstSlot : slot.val = 0
      · simp [PiRLCFirst54DirectPlan.priorPositionForm,
          PiRLCFirst54DirectPlan.priorPositionValue,
          PiRLCFirst54DirectPlan.initialPositionForm,
          PiRLCFirst54DirectPlan.oneForm, first54Inputs, first, firstSlot, one]
      · simp [PiRLCFirst54DirectPlan.priorPositionForm,
          PiRLCFirst54DirectPlan.priorPositionValue,
          PiRLCFirst54DirectPlan.initialPositionForm,
          first, firstSlot]
    · simp only [PiRLCFirst54DirectPlan.priorPositionForm,
        PiRLCFirst54DirectPlan.priorPositionValue, first, dite_false]
      change
        ((PiRLCFirst54RetainedBlocks.positionBlock program).form
          (positionStart program) (positionFits geometry)
            (PiRLCFirst54DirectSchedule.positionIndex
              ⟨PiRLCFirst54DirectPlan.previousCandidate candidate first,
                slot⟩)).eval assignment = _
      rw [LowNormBlock.Block.form_eval _ _ _ assignment _ encodes.position]
      rw [PiRLCFirst54RetainedBlocks.positionBlock_source]
      rw [PiRLCFirst54DirectSchedule.position_positionIndex]
      unfold PiRLCFirst54DirectPlan.retainedPositionColumn
      rw [sourceAssignment_package]
      rw [PiRLCFirst54DirectPlan.previousCandidate_positionColumn]
      simp [PiRLCFirst54DirectPlan.baseEnv]
  · intro descriptor
    change
      ((PiRLCFirst54RetainedBlocks.positionBlock program).form
        (positionStart program) (positionFits geometry)
          (PiRLCFirst54DirectSchedule.positionIndex descriptor)).eval
        assignment = _
    rw [LowNormBlock.Block.form_eval _ _ _ assignment _ encodes.position]
    rw [PiRLCFirst54RetainedBlocks.positionBlock_source]
    rw [PiRLCFirst54DirectSchedule.position_positionIndex]
    unfold PiRLCFirst54DirectPlan.retainedPositionColumn
      PiRLCFirst54DirectPlan.positionOutputValue
    rw [sourceAssignment_package]
    rfl
  · intro candidate
    change
      ((PiRLCFirst54RetainedBlocks.productBlock program).form
        (first54ProductStart program) (first54ProductFits geometry)
          candidate).eval assignment = _
    rw [LowNormBlock.Block.form_eval _ _ _ assignment _ encodes.first54Product]
    rw [PiRLCFirst54RetainedBlocks.productBlock_source]
    rw [sourceAssignment_product]
  · intro descriptor
    by_cases first : descriptor.candidate.round.val = 0
    · simp [PiRLCFirst54DirectPlan.priorValueForm,
        PiRLCFirst54DirectPlan.priorValue, first]
    · simp only [PiRLCFirst54DirectPlan.priorValueForm,
        PiRLCFirst54DirectPlan.priorValue, first, dite_false]
      change
        ((PiRLCFirst54RetainedBlocks.valueBlock program).form
          (valueStart program) (valueFits geometry)
            (PiRLCFirst54DirectSchedule.valueIndex
              ⟨PiRLCFirst54DirectPlan.previousCandidate
                  descriptor.candidate first,
                descriptor.slot⟩)).eval assignment = _
      rw [LowNormBlock.Block.form_eval _ _ _ assignment _ encodes.value]
      rw [PiRLCFirst54RetainedBlocks.valueBlock_source]
      rw [PiRLCFirst54DirectSchedule.value_valueIndex]
      unfold PiRLCFirst54DirectPlan.retainedValueColumn
      rw [sourceAssignment_package]
      rw [PiRLCFirst54DirectPlan.previousCandidate_valueColumn]
      simp [PiRLCFirst54DirectPlan.baseEnv]
  · intro descriptor
    change
      ((PiRLCFirst54RetainedBlocks.valueBlock program).form
        (valueStart program) (valueFits geometry)
          (PiRLCFirst54DirectSchedule.valueIndex descriptor)).eval
        assignment = _
    rw [LowNormBlock.Block.form_eval _ _ _ assignment _ encodes.value]
    rw [PiRLCFirst54RetainedBlocks.valueBlock_source]
    rw [PiRLCFirst54DirectSchedule.value_valueIndex]
    unfold PiRLCFirst54DirectPlan.retainedValueColumn
      PiRLCFirst54DirectPlan.outputValue
    rw [sourceAssignment_package]
    rfl
  · intro source
    change
      ((PiRLCFirst54RetainedBlocks.positionBlock program).form
        (positionStart program) (positionFits geometry)
          (PiRLCFirst54DirectSchedule.positionIndex
            (PiRLCFirst54DirectPlan.finalPositionDescriptor source))).eval
        assignment = _
    rw [LowNormBlock.Block.form_eval _ _ _ assignment _ encodes.position]
    rw [PiRLCFirst54RetainedBlocks.positionBlock_source]
    rw [PiRLCFirst54DirectSchedule.position_positionIndex]
    unfold PiRLCFirst54DirectPlan.retainedPositionColumn
      PiRLCFirst54DirectPlan.finalValue
    rw [sourceAssignment_package]
    rw [PiRLCFirst54DirectPlan.finalPositionDescriptor_positionColumn]
    rfl

end NightstreamFPrime.Export.Stage1.PiRLCRetainedPreservation
