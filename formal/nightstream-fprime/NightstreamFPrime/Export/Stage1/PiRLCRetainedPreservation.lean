import NightstreamFPrime.Export.Stage1.PiRLCRetainedInputs

/-!
Owns the preservation contract for the canonical retained PiRLC geometry.
One source assignment contains package values and product-group values.
Each compact block encodes that same assignment at its fixed interval.

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
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 1 → F) :
    Fin (sourceWidth program) → F :=
  PiRLCProductPlan.sourceAssignment program base groupValue

/-- Canonical embedding of the complete application-package source prefix
through the product-group suffix. -/
def baseSourceColumn (program : Lifecycle.Stage1.Application.Program)
    (column : Fin (PiRLCProductPlan.baseSourceWidth program)) :
    Fin (sourceWidth program) :=
  ProductRetainedBlock.baseColumn (PiRLCProductPlan.baseSourceWidth program)
      PiRLCProductSchedule.invocationCount column

/-- The nested retained source assignment preserves every package-prefix
column exactly. -/
theorem sourceAssignment_base
    (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 1 → F)
    (column : Fin (PiRLCProductPlan.baseSourceWidth program)) :
    sourceAssignment program base groupValue
        (baseSourceColumn program column) = base column := by
  rw [sourceAssignment, baseSourceColumn]
  exact ProductRetainedBlock.sourceAssignment_base _ _ _ _ column

theorem sourceAssignment_package
    (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 1 → F)
    (column : Nat) (bound : column < PiRLCProductPlan.basePackage.layout.constantColumn) :
    sourceAssignment program base groupValue
        (PiRLCProductPlan.baseColumn program column bound) =
      PiRLCProductPlan.baseEnv program base column := by
  unfold sourceAssignment PiRLCProductPlan.sourceAssignment PiRLCProductPlan.baseColumn
  rw [ProductRetainedBlock.sourceAssignment_base]
  exact (PiRLCProductPlan.baseEnv_eq_mappedPackageColumn
    program base column bound).symm

theorem sourceAssignment_group
    (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 1 → F)
    (invocation : Fin PiRLCProductSchedule.invocationCount)
    (group : Fin 1) :
    sourceAssignment program base groupValue
        (PiRLCProductPlan.groupColumn program invocation group) =
      groupValue invocation group := by
  rw [sourceAssignment]
  exact ProductRetainedBlock.sourceAssignment_group _ _ _ _ invocation group

theorem sourceAssignment_valueColumn
    (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 1 → F)
    (descriptor : PiRLCProductSchedule.Descriptor)
    (lane : Fin ringDegree) :
    sourceAssignment program base groupValue
        (PiRLCProductPlan.valueColumn program descriptor lane) =
      PiRLCProductPlan.baseEnv program base (descriptor.valueColumn lane) := by
  rw [sourceAssignment]
  unfold PiRLCProductPlan.valueColumn PiRLCProductPlan.baseColumn
  unfold PiRLCProductPlan.sourceAssignment
  rw [ProductRetainedBlock.sourceAssignment_base]
  exact (PiRLCProductPlan.baseEnv_valueColumn
    program base descriptor lane).symm

theorem sourceAssignment_outputColumn
    (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 1 → F)
    (descriptor : PiRLCProductSchedule.Descriptor) :
    sourceAssignment program base groupValue
        (PiRLCProductPlan.outputColumn program descriptor) =
      PiRLCProductPlan.baseEnv program base descriptor.outputColumn := by
  rw [sourceAssignment]
  unfold PiRLCProductPlan.outputColumn PiRLCProductPlan.baseColumn
  unfold PiRLCProductPlan.sourceAssignment
  rw [ProductRetainedBlock.sourceAssignment_base]
  exact (PiRLCProductPlan.baseEnv_outputColumn
    program base descriptor).symm

theorem sourceAssignment_challengeColumn
    (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 1 → F)
    (descriptor : PiRLCProductSchedule.Descriptor) (lane : Fin ringDegree) :
    sourceAssignment program base groupValue
        (PiRLCProductPlan.challengeColumn program descriptor lane) =
      PiRLCProductPlan.baseEnv program base (descriptor.challengeColumn lane) := by
  unfold PiRLCProductPlan.challengeColumn
  exact sourceAssignment_package program base groupValue _ _

theorem productGroupBlock_source
    (program : Lifecycle.Stage1.Application.Program)
    (invocation : Fin PiRLCProductSchedule.invocationCount)
    (group : Fin 1) :
    (productGroupBlock program).source (Fin.encodeProd (invocation, group)) =
      PiRLCProductPlan.groupColumn program invocation group := by
  apply Fin.ext
  simp [productGroupBlock, LowNormBlock.Block.lift,
    ProductRetainedBlock.block, PiRLCProductPlan.groupColumn]

theorem productOutputBlock_source
    (program : Lifecycle.Stage1.Application.Program)
    (invocation : Fin PiRLCProductSchedule.invocationCount) :
    (productOutputBlock program).source invocation =
      PiRLCProductPlan.outputColumn program
        (PiRLCProductSchedule.descriptor invocation) := by
  apply Fin.ext
  rfl

structure Encodes {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 1 → F) : Prop where
  priorPoseidon : (priorPoseidonBlock program).EncodesAt
    (priorPoseidonStart program) (priorPoseidonFits geometry) assignment
      (sourceAssignment program base groupValue)
  outputPoseidon : (outputPoseidonBlock program).EncodesAt
    (outputPoseidonStart program) (outputPoseidonFits geometry) assignment
      (sourceAssignment program base groupValue)
  laterPoseidon : (laterPoseidonBlock program).EncodesAt
    (laterPoseidonStart program) (laterPoseidonFits geometry) assignment
      (sourceAssignment program base groupValue)
  productGroup : (productGroupBlock program).EncodesAt
    (productGroupStart program) (productGroupFits geometry) assignment
      (sourceAssignment program base groupValue)
  challenge : (challengeBlock program).EncodesAt
    (challengeStart program) (challengeFits geometry) assignment
      (sourceAssignment program base groupValue)
  productOutput : (productOutputBlock program).EncodesAt
    (productOutputStart program) (productOutputFits geometry) assignment
      (sourceAssignment program base groupValue)

theorem productInputs_preserves
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (values : Values logicalWidth)
    (geometry : Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 1 → F)
    (valuePreserves : ∀ invocation,
      (values invocation).eval assignment =
        PiRLCProductPlan.baseEnv program base
          ((PiRLCProductSchedule.descriptor invocation).valueColumn
            (PiRLCProductSchedule.descriptor invocation).lane))
    (encodes : Encodes geometry assignment base groupValue) :
    PiRLCProductPlan.Preserves (productInputs values geometry)
      assignment base groupValue := by
  refine
    { challenge := ?_
      value := ?_
      prior := ?_
      output := ?_
      group := ?_ }
  · intro invocation lane
    change
      ((challengeBlock program).form (challengeStart program) (challengeFits geometry)
        (PiRLCProductSourceBlocks.challengeIndex
          (PiRLCProductSchedule.descriptor invocation).source lane)).eval assignment = _
    rw [LowNormBlock.Block.form_eval _ _ _ assignment _ encodes.challenge]
    exact (congrArg (sourceAssignment program base groupValue)
      (PiRLCProductSourceBlocks.challengeBlock_source program
        (PiRLCProductSchedule.descriptor invocation) lane)).trans
      (sourceAssignment_challengeColumn program base groupValue
        (PiRLCProductSchedule.descriptor invocation) lane)
  · intro invocation lane
    have value := valuePreserves
      ((PiRLCProductSchedule.descriptor invocation).withLane lane).invocation
    simpa only [PiRLCProductSchedule.descriptor_invocation,
      PiRLCProductSchedule.Descriptor.withLane_valueColumn] using! value
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

end NightstreamFPrime.Export.Stage1.PiRLCRetainedPreservation
