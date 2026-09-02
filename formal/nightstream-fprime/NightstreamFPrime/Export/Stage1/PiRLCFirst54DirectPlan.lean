import NightstreamFPrime.Export.Stage1.PiRLCFirst54DirectSchedule
import NightstreamFPrime.Export.Stage1.PiRLCProductPlan
import NightstreamFPrime.Layout.ProductionRelation.FieldSuffixBlock
import NightstreamFPrime.Layout.ProductionRelation.MultiplicationFamilyPlan
import NightstreamFPrime.Layout.ProductionRelation.PinFamilyPlan

/-!
Owns the direct 14-matrix First54 selector plan for one Lean-authored
application package. Position, accepted-symbol, value, and final-pin rows use
fixed indexed families. One derived field value per candidate is shared by
all 54 value rows.

This module does not select the final retained-slot set or close PiRLC status.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCFirst54DirectPlan

open NightstreamFPrime.Spec
open NightstreamFPrime.Gadgets.Sampling
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

def basePackage (_delay : Unit := ()) := PiRLCProductPlan.basePackage ()

def prefixSourceWidth (program : Lifecycle.Stage1.Application.Program) : Nat :=
  PiRLCProductPlan.sourceWidth program

def sourceWidth (program : Lifecycle.Stage1.Application.Program) : Nat :=
  FieldSuffixBlock.sourceWidth (prefixSourceWidth program)
    PiRLCFirst54DirectSchedule.candidateCount

private theorem basePackage_constantColumn :
    basePackage.layout.constantColumn = 29336446 := by
  exact NightstreamFPrime.Export.Stage1.Package.circuitPackage_layout_values.2.2.1

private theorem rejectColumn_lt_basePackage
    (descriptor : PiRLCFirst54DirectSchedule.Candidate) :
    descriptor.rejectColumn < basePackage.layout.constantColumn := by
  rw [basePackage_constantColumn]
  rcases descriptor with ⟨source, round⟩
  have sourceBound := source.isLt
  have roundBound := round.isLt
  norm_num [PiRLCFirst54DirectSchedule.Candidate.rejectColumn,
    PiRLCFirst54Invocations.rejectSourceColumn,
    PiRLCFirst54Invocations.decoderLogicalStart,
    PiRLCFirst54Invocations.candidateDigestRound,
    PiRLCFirst54Invocations.candidateLane,
    PiRLCFirst54Invocations.candidatePart,
    PiRLCFirst54DirectSchedule.sourceCount,
    PiRLCFirst54DirectSchedule.roundCount,
    PiRLCFirst54Invocations.sourceCount,
    PiRLCFirst54Invocations.roundCount, First54.candidateCount,
    PiRLCStarts.digestLaneLogicalStart, PiRLCStarts.windowLogicalStart,
    PiRLCStarts.samplerSourceLogicalStart, PiRLCStarts.samplerLogicalStart,
    PiRLCStarts.phaseLogicalStart, PiRLCInputs.phaseOffset,
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerOffset,
    NightstreamFPrime.Gadgets.Range.CanonicalU64.auxiliaryCount,
    Candidate16Five.auxiliaryCount] at sourceBound roundBound ⊢
  omega

private theorem symbolColumn_lt_basePackage
    (descriptor : PiRLCFirst54DirectSchedule.Candidate) :
    descriptor.symbolColumn < basePackage.layout.constantColumn := by
  rw [basePackage_constantColumn]
  rcases descriptor with ⟨source, round⟩
  have sourceBound := source.isLt
  have roundBound := round.isLt
  norm_num [PiRLCFirst54DirectSchedule.Candidate.symbolColumn,
    PiRLCFirst54Invocations.remainderSourceColumn,
    PiRLCFirst54Invocations.decoderLogicalStart,
    PiRLCFirst54Invocations.candidateDigestRound,
    PiRLCFirst54Invocations.candidateLane,
    PiRLCFirst54Invocations.candidatePart,
    PiRLCFirst54DirectSchedule.sourceCount,
    PiRLCFirst54DirectSchedule.roundCount,
    PiRLCFirst54Invocations.sourceCount,
    PiRLCFirst54Invocations.roundCount, First54.candidateCount,
    PiRLCStarts.digestLaneLogicalStart, PiRLCStarts.windowLogicalStart,
    PiRLCStarts.samplerSourceLogicalStart, PiRLCStarts.samplerLogicalStart,
    PiRLCStarts.phaseLogicalStart, PiRLCInputs.phaseOffset,
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerOffset,
    NightstreamFPrime.Gadgets.Range.CanonicalU64.auxiliaryCount,
    Candidate16Five.auxiliaryCount] at sourceBound roundBound ⊢
  omega

private theorem positionColumn_lt_basePackage
    (descriptor : PiRLCFirst54DirectSchedule.Position) :
    descriptor.positionColumn < basePackage.layout.constantColumn := by
  rw [basePackage_constantColumn]
  rcases descriptor with ⟨⟨source, round⟩, slot⟩
  have sourceBound := source.isLt
  have roundBound := round.isLt
  have slotBound := slot.isLt
  norm_num [PiRLCFirst54DirectSchedule.Position.positionColumn,
    PiRLCFirst54Invocations.positionSourceStart, First54.positionOffset,
    PiRLCStarts.selectorLogicalStart, PiRLCStarts.samplerSourceLogicalStart,
    PiRLCStarts.samplerLogicalStart, PiRLCStarts.phaseLogicalStart,
    PiRLCInputs.phaseOffset,
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerOffset,
    PiRLCFirst54DirectSchedule.sourceCount,
    PiRLCFirst54DirectSchedule.roundCount,
    PiRLCFirst54Invocations.sourceCount,
    PiRLCFirst54Invocations.roundCount, First54.candidateCount,
    First54.roundPrivateCount, First54Step.slotCount,
    First54ValueStep.outputCount] at sourceBound roundBound slotBound ⊢
  omega

private theorem valueColumn_lt_basePackage
    (descriptor : PiRLCFirst54DirectSchedule.Value) :
    descriptor.valueColumn < basePackage.layout.constantColumn := by
  rw [basePackage_constantColumn]
  rcases descriptor with ⟨⟨source, round⟩, slot⟩
  have sourceBound := source.isLt
  have roundBound := round.isLt
  have slotBound := slot.isLt
  norm_num [PiRLCFirst54DirectSchedule.Value.valueColumn,
    PiRLCFirst54Invocations.valueSourceStart, First54.valueOffset,
    First54.positionOffset, PiRLCStarts.selectorLogicalStart,
    PiRLCStarts.samplerSourceLogicalStart, PiRLCStarts.samplerLogicalStart,
    PiRLCStarts.phaseLogicalStart, PiRLCInputs.phaseOffset,
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerOffset,
    PiRLCFirst54DirectSchedule.sourceCount,
    PiRLCFirst54DirectSchedule.roundCount,
    PiRLCFirst54Invocations.sourceCount,
    PiRLCFirst54Invocations.roundCount, First54.candidateCount,
    First54.roundPrivateCount, First54Step.slotCount,
    First54ValueStep.outputCount] at sourceBound roundBound slotBound ⊢
  omega

private theorem priorPositionColumn_lt_basePackage
    (descriptor : PiRLCFirst54DirectSchedule.Position)
    (notFirst : descriptor.candidate.round.val ≠ 0) :
    descriptor.priorPositionColumn < basePackage.layout.constantColumn := by
  rw [basePackage_constantColumn]
  rcases descriptor with ⟨⟨source, round⟩, slot⟩
  have sourceBound := source.isLt
  have roundBound := round.isLt
  have slotBound := slot.isLt
  norm_num [PiRLCFirst54DirectSchedule.Position.priorPositionColumn,
    PiRLCFirst54Invocations.previousPositionSourceStart,
    PiRLCFirst54Invocations.positionSourceStart, First54.positionOffset,
    PiRLCStarts.selectorLogicalStart, PiRLCStarts.samplerSourceLogicalStart,
    PiRLCStarts.samplerLogicalStart, PiRLCStarts.phaseLogicalStart,
    PiRLCInputs.phaseOffset,
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerOffset,
    PiRLCFirst54DirectSchedule.sourceCount,
    PiRLCFirst54DirectSchedule.roundCount,
    PiRLCFirst54Invocations.sourceCount,
    PiRLCFirst54Invocations.roundCount, First54.candidateCount,
    First54.roundPrivateCount, First54Step.slotCount,
    First54ValueStep.outputCount] at sourceBound roundBound slotBound ⊢
  omega

private theorem priorValueColumn_lt_basePackage
    (descriptor : PiRLCFirst54DirectSchedule.Value)
    (notFirst : descriptor.candidate.round.val ≠ 0) :
    descriptor.priorValueColumn < basePackage.layout.constantColumn := by
  rw [basePackage_constantColumn]
  rcases descriptor with ⟨⟨source, round⟩, slot⟩
  have sourceBound := source.isLt
  have roundBound := round.isLt
  have slotBound := slot.isLt
  norm_num [PiRLCFirst54DirectSchedule.Value.priorValueColumn,
    PiRLCFirst54Invocations.previousValueSourceStart,
    PiRLCFirst54Invocations.valueSourceStart, First54.valueOffset,
    First54.positionOffset, PiRLCStarts.selectorLogicalStart,
    PiRLCStarts.samplerSourceLogicalStart, PiRLCStarts.samplerLogicalStart,
    PiRLCStarts.phaseLogicalStart, PiRLCInputs.phaseOffset,
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerOffset,
    PiRLCFirst54DirectSchedule.sourceCount,
    PiRLCFirst54DirectSchedule.roundCount,
    PiRLCFirst54Invocations.sourceCount,
    PiRLCFirst54Invocations.roundCount, First54.candidateCount,
    First54.roundPrivateCount, First54Step.slotCount,
    First54ValueStep.outputCount] at sourceBound roundBound slotBound ⊢
  omega

private theorem finalColumn_lt_basePackage
    (source : Fin PiRLCFirst54DirectSchedule.finalCount) :
    PiRLCFirst54DirectSchedule.finalColumn source <
      basePackage.layout.constantColumn := by
  rw [basePackage_constantColumn]
  have sourceBound := source.isLt
  norm_num [PiRLCFirst54DirectSchedule.finalColumn,
    PiRLCFirst54Invocations.positionSourceStart, First54.positionOffset,
    PiRLCStarts.selectorLogicalStart, PiRLCStarts.samplerSourceLogicalStart,
    PiRLCStarts.samplerLogicalStart, PiRLCStarts.phaseLogicalStart,
    PiRLCInputs.phaseOffset,
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerOffset,
    PiRLCFirst54DirectSchedule.finalCount,
    PiRLCFirst54DirectSchedule.sourceCount,
    PiRLCFirst54DirectSchedule.roundCount,
    PiRLCFirst54Invocations.sourceCount,
    PiRLCFirst54Invocations.roundCount, First54.candidateCount,
    First54.roundPrivateCount, First54Step.slotCount, First54Step.fullSlot,
    First54ValueStep.outputCount] at sourceBound ⊢
  omega

def prefixColumn (program : Lifecycle.Stage1.Application.Program)
    (column : Fin (prefixSourceWidth program)) : Fin (sourceWidth program) :=
  FieldSuffixBlock.baseColumn (prefixSourceWidth program)
    PiRLCFirst54DirectSchedule.candidateCount column

def packageColumn (program : Lifecycle.Stage1.Application.Program)
    (column : Nat) (bound : column < basePackage.layout.constantColumn) :
    Fin (sourceWidth program) :=
  prefixColumn program (PiRLCProductPlan.baseColumn program column bound)

def productColumn (program : Lifecycle.Stage1.Application.Program)
    (candidate : Fin PiRLCFirst54DirectSchedule.candidateCount) :
    Fin (sourceWidth program) :=
  FieldSuffixBlock.derivedColumn (prefixSourceWidth program)
    PiRLCFirst54DirectSchedule.candidateCount candidate

def retainedRejectColumn (program : Lifecycle.Stage1.Application.Program)
    (candidate : PiRLCFirst54DirectSchedule.Candidate) :
    Fin (sourceWidth program) :=
  packageColumn program candidate.rejectColumn
    (rejectColumn_lt_basePackage candidate)

def retainedSymbolColumn (program : Lifecycle.Stage1.Application.Program)
    (candidate : PiRLCFirst54DirectSchedule.Candidate) :
    Fin (sourceWidth program) :=
  packageColumn program candidate.symbolColumn
    (symbolColumn_lt_basePackage candidate)

def retainedPositionColumn (program : Lifecycle.Stage1.Application.Program)
    (descriptor : PiRLCFirst54DirectSchedule.Position) :
    Fin (sourceWidth program) :=
  packageColumn program descriptor.positionColumn
    (positionColumn_lt_basePackage descriptor)

def retainedValueColumn (program : Lifecycle.Stage1.Application.Program)
    (descriptor : PiRLCFirst54DirectSchedule.Value) :
    Fin (sourceWidth program) :=
  packageColumn program descriptor.valueColumn
    (valueColumn_lt_basePackage descriptor)

def sourceAssignment (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F) :
    Fin (sourceWidth program) → F :=
  FieldSuffixBlock.sourceAssignment (prefixSourceWidth program)
    PiRLCFirst54DirectSchedule.candidateCount
      (PiRLCProductPlan.sourceAssignment program base groupValue) products

@[simp] theorem sourceAssignment_prefix
    (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (column : Fin (prefixSourceWidth program)) :
    sourceAssignment program base groupValue products
        (prefixColumn program column) =
      PiRLCProductPlan.sourceAssignment program base groupValue column := by
  exact FieldSuffixBlock.sourceAssignment_base _ _ _ _ column

@[simp] theorem sourceAssignment_product
    (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (candidate : Fin PiRLCFirst54DirectSchedule.candidateCount) :
    sourceAssignment program base groupValue products
        (productColumn program candidate) = products candidate := by
  exact FieldSuffixBlock.sourceAssignment_derived _ _ _ _ candidate

structure Inputs (program : Lifecycle.Stage1.Application.Program)
    (logicalWidth : Nat) where
  oneColumn : Fin logicalWidth
  reject : PiRLCFirst54DirectSchedule.Candidate → SparseForm logicalWidth
  symbol : PiRLCFirst54DirectSchedule.Candidate → SparseForm logicalWidth
  position : PiRLCFirst54DirectSchedule.Position → SparseForm logicalWidth
  value : PiRLCFirst54DirectSchedule.Value → SparseForm logicalWidth
  product : Fin PiRLCFirst54DirectSchedule.candidateCount →
    SparseForm logicalWidth

def oneForm {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (inputs : Inputs program logicalWidth) :
    SparseForm logicalWidth :=
  SparseForm.singleton inputs.oneColumn 1

def subtract {logicalWidth : Nat} (left right : SparseForm logicalWidth) :
    SparseForm logicalWidth :=
  SparseForm.add left (SparseForm.scale (-1) right)

def productForm {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (inputs : Inputs program logicalWidth)
    (candidate : Fin PiRLCFirst54DirectSchedule.candidateCount) :
    SparseForm logicalWidth :=
  inputs.product candidate

def rejectForm {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (inputs : Inputs program logicalWidth)
    (candidate : PiRLCFirst54DirectSchedule.Candidate) :
    SparseForm logicalWidth :=
  inputs.reject candidate

def acceptedForm {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (inputs : Inputs program logicalWidth)
    (candidate : PiRLCFirst54DirectSchedule.Candidate) :
    SparseForm logicalWidth :=
  subtract (oneForm inputs) (rejectForm inputs candidate)

def symbolForm {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (inputs : Inputs program logicalWidth)
    (candidate : PiRLCFirst54DirectSchedule.Candidate) :
    SparseForm logicalWidth :=
  inputs.symbol candidate

def previousCandidate
    (candidate : PiRLCFirst54DirectSchedule.Candidate)
    (notFirst : candidate.round.val ≠ 0) :
    PiRLCFirst54DirectSchedule.Candidate :=
  { source := candidate.source
    round := ⟨candidate.round.val - 1, by
      have roundBound := candidate.round.isLt
      omega⟩ }

@[simp] theorem previousCandidate_positionColumn
    (candidate : PiRLCFirst54DirectSchedule.Candidate)
    (notFirst : candidate.round.val ≠ 0)
    (slot : Fin First54Step.slotCount) :
    (PiRLCFirst54DirectSchedule.Position.positionColumn
      ⟨previousCandidate candidate notFirst, slot⟩) =
      (PiRLCFirst54DirectSchedule.Position.priorPositionColumn
        ⟨candidate, slot⟩) := by
  rfl

@[simp] theorem previousCandidate_valueColumn
    (candidate : PiRLCFirst54DirectSchedule.Candidate)
    (notFirst : candidate.round.val ≠ 0)
    (slot : Fin First54ValueStep.outputCount) :
    (PiRLCFirst54DirectSchedule.Value.valueColumn
      ⟨previousCandidate candidate notFirst, slot⟩) =
      (PiRLCFirst54DirectSchedule.Value.priorValueColumn
        ⟨candidate, slot⟩) := by
  rfl

def initialPositionForm {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (inputs : Inputs program logicalWidth)
    (slot : Fin First54Step.slotCount) : SparseForm logicalWidth :=
  if slot.val = 0 then oneForm inputs else .empty

def priorPositionForm {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (inputs : Inputs program logicalWidth)
    (candidate : PiRLCFirst54DirectSchedule.Candidate)
    (slot : Fin First54Step.slotCount) : SparseForm logicalWidth :=
  if first : candidate.round.val = 0 then
    initialPositionForm inputs slot
  else
    inputs.position ⟨previousCandidate candidate first, slot⟩

def positionOutputForm {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (inputs : Inputs program logicalWidth)
    (descriptor : PiRLCFirst54DirectSchedule.Position) :
    SparseForm logicalWidth :=
  inputs.position descriptor

def previousPositionForm {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (inputs : Inputs program logicalWidth)
    (descriptor : PiRLCFirst54DirectSchedule.Position) :
    SparseForm logicalWidth :=
  if first : descriptor.slot.val = 0 then
    .empty
  else
    priorPositionForm inputs descriptor.candidate
      (First54Step.previousSlot descriptor.slot (by omega))

def positionDeltaForm {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (inputs : Inputs program logicalWidth)
    (descriptor : PiRLCFirst54DirectSchedule.Position) :
    SparseForm logicalWidth :=
  if descriptor.slot.val = First54Step.fullSlot then
    previousPositionForm inputs descriptor
  else
    subtract (previousPositionForm inputs descriptor)
      (priorPositionForm inputs descriptor.candidate descriptor.slot)

def positionDifferenceForm {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (inputs : Inputs program logicalWidth)
    (descriptor : PiRLCFirst54DirectSchedule.Position) :
    SparseForm logicalWidth :=
  subtract (positionOutputForm inputs descriptor)
    (priorPositionForm inputs descriptor.candidate descriptor.slot)

def priorValueForm {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (inputs : Inputs program logicalWidth)
    (descriptor : PiRLCFirst54DirectSchedule.Value) :
    SparseForm logicalWidth :=
  if first : descriptor.candidate.round.val = 0 then
    .empty
  else
    inputs.value
      ⟨previousCandidate descriptor.candidate first, descriptor.slot⟩

def valueOutputForm {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (inputs : Inputs program logicalWidth)
    (descriptor : PiRLCFirst54DirectSchedule.Value) :
    SparseForm logicalWidth :=
  inputs.value descriptor

def valueDifferenceForm {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (inputs : Inputs program logicalWidth)
    (descriptor : PiRLCFirst54DirectSchedule.Value) :
    SparseForm logicalWidth :=
  subtract (valueOutputForm inputs descriptor) (priorValueForm inputs descriptor)

def finalPositionDescriptor
    (source : Fin PiRLCFirst54DirectSchedule.finalCount) :
    PiRLCFirst54DirectSchedule.Position :=
  { candidate :=
      { source := ⟨source.val, by
          simpa [PiRLCFirst54DirectSchedule.finalCount] using source.isLt⟩
        round := ⟨PiRLCFirst54DirectSchedule.roundCount - 1, by
          rw [PiRLCFirst54DirectSchedule.roundCount_eq]
          norm_num⟩ }
    slot := ⟨First54Step.fullSlot, by
      norm_num [First54Step.fullSlot, First54Step.slotCount]⟩ }

@[simp] theorem finalPositionDescriptor_positionColumn
    (source : Fin PiRLCFirst54DirectSchedule.finalCount) :
    (finalPositionDescriptor source).positionColumn =
      PiRLCFirst54DirectSchedule.finalColumn source := by
  rfl

def finalForm {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (inputs : Inputs program logicalWidth)
    (source : Fin PiRLCFirst54DirectSchedule.finalCount) :
    SparseForm logicalWidth :=
  inputs.position (finalPositionDescriptor source)

def positionInterface {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (inputs : Inputs program logicalWidth) :
    MultiplicationFamilyPlan.Interface logicalWidth
      PiRLCFirst54DirectSchedule.positionCount :=
  { oneColumn := inputs.oneColumn
    left := fun row => acceptedForm inputs (PiRLCFirst54DirectSchedule.position row).candidate
    right := fun row => positionDeltaForm inputs
      (PiRLCFirst54DirectSchedule.position row)
    output := fun row => positionDifferenceForm inputs
      (PiRLCFirst54DirectSchedule.position row) }

def acceptedProductInterface {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (inputs : Inputs program logicalWidth) :
    MultiplicationFamilyPlan.Interface logicalWidth
      PiRLCFirst54DirectSchedule.candidateCount :=
  { oneColumn := inputs.oneColumn
    left := fun row => acceptedForm inputs (PiRLCFirst54DirectSchedule.candidate row)
    right := fun row => symbolForm inputs (PiRLCFirst54DirectSchedule.candidate row)
    output := productForm inputs }

def valueInterface {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (inputs : Inputs program logicalWidth) :
    MultiplicationFamilyPlan.Interface logicalWidth
      PiRLCFirst54DirectSchedule.valueCount :=
  { oneColumn := inputs.oneColumn
    left := fun row =>
      let descriptor := PiRLCFirst54DirectSchedule.value row
      priorPositionForm inputs descriptor.candidate
        (First54ValueStep.positionSlot descriptor.slot)
    right := fun row => productForm inputs <|
      PiRLCFirst54DirectSchedule.candidateIndex
        (PiRLCFirst54DirectSchedule.value row).candidate
    output := fun row => valueDifferenceForm inputs
      (PiRLCFirst54DirectSchedule.value row) }

def finalInterface {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (inputs : Inputs program logicalWidth) :
    PinFamilyPlan.Interface logicalWidth
      PiRLCFirst54DirectSchedule.finalCount :=
  { oneColumn := inputs.oneColumn
    value := fun source =>
      subtract (finalForm inputs source) (oneForm inputs) }

theorem positionCount_le : PiRLCFirst54DirectSchedule.positionCount ≤
    2 ^ NightstreamFPrime.Lifecycle.cubeVariables := by
  rw [PiRLCFirst54DirectSchedule.positionCount_eq]
  norm_num [NightstreamFPrime.Lifecycle.cubeVariables]

theorem candidateCount_le : PiRLCFirst54DirectSchedule.candidateCount ≤
    2 ^ NightstreamFPrime.Lifecycle.cubeVariables := by
  rw [PiRLCFirst54DirectSchedule.candidateCount_eq]
  norm_num [NightstreamFPrime.Lifecycle.cubeVariables]

theorem valueCount_le : PiRLCFirst54DirectSchedule.valueCount ≤
    2 ^ NightstreamFPrime.Lifecycle.cubeVariables := by
  rw [PiRLCFirst54DirectSchedule.valueCount_eq]
  norm_num [NightstreamFPrime.Lifecycle.cubeVariables]

theorem finalCount_le : PiRLCFirst54DirectSchedule.finalCount ≤
    2 ^ NightstreamFPrime.Lifecycle.cubeVariables := by
  rw [PiRLCFirst54DirectSchedule.finalCount_eq]
  norm_num [NightstreamFPrime.Lifecycle.cubeVariables]

def positionPlan {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (inputs : Inputs program logicalWidth) :=
  MultiplicationFamilyPlan.plan (positionInterface inputs) positionCount_le

def acceptedProductPlan {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (inputs : Inputs program logicalWidth) :=
  MultiplicationFamilyPlan.plan (acceptedProductInterface inputs)
    candidateCount_le

def valuePlan {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (inputs : Inputs program logicalWidth) :=
  MultiplicationFamilyPlan.plan (valueInterface inputs) valueCount_le

def finalPlan {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (inputs : Inputs program logicalWidth) :=
  PinFamilyPlan.plan (finalInterface inputs) finalCount_le

@[simp] theorem positionPlan_rowCount
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (inputs : Inputs program logicalWidth) :
    (positionPlan inputs).rowCount = 59840 := by
  rfl

@[simp] theorem acceptedProductPlan_rowCount
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (inputs : Inputs program logicalWidth) :
    (acceptedProductPlan inputs).rowCount = 1088 := by
  rfl

@[simp] theorem valuePlan_rowCount
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (inputs : Inputs program logicalWidth) :
    (valuePlan inputs).rowCount = 58752 := by
  rfl

@[simp] theorem finalPlan_rowCount
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (inputs : Inputs program logicalWidth) :
    (finalPlan inputs).rowCount = 17 := by
  rfl

theorem valueFinalCount_le {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (inputs : Inputs program logicalWidth) :
    (valuePlan inputs).rowCount + (finalPlan inputs).rowCount ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables := by
  norm_num [valuePlan, finalPlan, MultiplicationFamilyPlan.plan,
    PinFamilyPlan.plan, PiRLCFirst54DirectSchedule.valueCount,
    PiRLCFirst54DirectSchedule.finalCount,
    PiRLCFirst54DirectSchedule.candidateCount,
    PiRLCFirst54DirectSchedule.sourceCount,
    PiRLCFirst54DirectSchedule.roundCount,
    PiRLCFirst54Invocations.sourceCount,
    PiRLCFirst54Invocations.roundCount, First54.candidateCount,
    First54ValueStep.outputCount, NightstreamFPrime.Lifecycle.cubeVariables]

def valueFinalPlan {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (inputs : Inputs program logicalWidth) :=
  ProductionRelation.Plan.append (valuePlan inputs) (finalPlan inputs)
    (valueFinalCount_le inputs)

@[simp] theorem valueFinalPlan_rowCount
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (inputs : Inputs program logicalWidth) :
    (valueFinalPlan inputs).rowCount = 58769 := by
  rfl

theorem acceptedTailCount_le {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (inputs : Inputs program logicalWidth) :
    (acceptedProductPlan inputs).rowCount + (valueFinalPlan inputs).rowCount ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables := by
  norm_num [acceptedProductPlan, valueFinalPlan, valuePlan, finalPlan,
    MultiplicationFamilyPlan.plan, PinFamilyPlan.plan,
    ProductionRelation.Plan.append,
    PiRLCFirst54DirectSchedule.candidateCount,
    PiRLCFirst54DirectSchedule.valueCount,
    PiRLCFirst54DirectSchedule.finalCount,
    PiRLCFirst54DirectSchedule.sourceCount,
    PiRLCFirst54DirectSchedule.roundCount,
    PiRLCFirst54Invocations.sourceCount,
    PiRLCFirst54Invocations.roundCount, First54.candidateCount,
    First54ValueStep.outputCount, NightstreamFPrime.Lifecycle.cubeVariables]

def acceptedTailPlan {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (inputs : Inputs program logicalWidth) :=
  ProductionRelation.Plan.append (acceptedProductPlan inputs)
    (valueFinalPlan inputs) (acceptedTailCount_le inputs)

@[simp] theorem acceptedTailPlan_rowCount
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (inputs : Inputs program logicalWidth) :
    (acceptedTailPlan inputs).rowCount = 59857 := by
  rfl

theorem totalCount_le {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (inputs : Inputs program logicalWidth) :
    (positionPlan inputs).rowCount + (acceptedTailPlan inputs).rowCount ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables := by
  norm_num [positionPlan, acceptedTailPlan, acceptedProductPlan,
    valueFinalPlan, valuePlan, finalPlan, MultiplicationFamilyPlan.plan,
    PinFamilyPlan.plan, ProductionRelation.Plan.append,
    PiRLCFirst54DirectSchedule.positionCount,
    PiRLCFirst54DirectSchedule.candidateCount,
    PiRLCFirst54DirectSchedule.valueCount,
    PiRLCFirst54DirectSchedule.finalCount,
    PiRLCFirst54DirectSchedule.sourceCount,
    PiRLCFirst54DirectSchedule.roundCount,
    PiRLCFirst54Invocations.sourceCount,
    PiRLCFirst54Invocations.roundCount, First54.candidateCount,
    First54Step.slotCount, First54ValueStep.outputCount,
    NightstreamFPrime.Lifecycle.cubeVariables]

/-- Family-major canonical direct First54 plan. -/
def plan {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (inputs : Inputs program logicalWidth) :=
  ProductionRelation.Plan.append (positionPlan inputs)
    (acceptedTailPlan inputs) (totalCount_le inputs)

@[simp] theorem plan_rowCount {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (inputs : Inputs program logicalWidth) :
    (plan inputs).rowCount = 119697 := by
  norm_num [plan, positionPlan, acceptedTailPlan, acceptedProductPlan,
    valueFinalPlan, valuePlan, finalPlan, MultiplicationFamilyPlan.plan,
    PinFamilyPlan.plan, ProductionRelation.Plan.append,
    PiRLCFirst54DirectSchedule.positionCount,
    PiRLCFirst54DirectSchedule.candidateCount,
    PiRLCFirst54DirectSchedule.valueCount,
    PiRLCFirst54DirectSchedule.finalCount,
    PiRLCFirst54DirectSchedule.sourceCount,
    PiRLCFirst54DirectSchedule.roundCount,
    PiRLCFirst54Invocations.sourceCount,
    PiRLCFirst54Invocations.roundCount, First54.candidateCount,
    First54Step.slotCount, First54ValueStep.outputCount]

def baseEnv (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F) :
    NightstreamFPrime.Circuit.Env :=
  PiRLCProductPlan.baseEnv program base

def rejectValue (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (candidate : PiRLCFirst54DirectSchedule.Candidate) : F :=
  baseEnv program base candidate.rejectColumn

def acceptedValue (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (candidate : PiRLCFirst54DirectSchedule.Candidate) : F :=
  1 - rejectValue program base candidate

def symbolValue (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (candidate : PiRLCFirst54DirectSchedule.Candidate) : F :=
  baseEnv program base candidate.symbolColumn

def priorPositionValue (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (candidate : PiRLCFirst54DirectSchedule.Candidate)
    (slot : Fin First54Step.slotCount) : F :=
  if candidate.round.val = 0 then
    if slot.val = 0 then 1 else 0
  else
    baseEnv program base
      (PiRLCFirst54DirectSchedule.Position.priorPositionColumn
        ⟨candidate, slot⟩)

def positionOutputValue (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (descriptor : PiRLCFirst54DirectSchedule.Position) : F :=
  baseEnv program base descriptor.positionColumn

def previousPositionValue (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (descriptor : PiRLCFirst54DirectSchedule.Position) : F :=
  if first : descriptor.slot.val = 0 then 0
  else priorPositionValue program base descriptor.candidate
    (First54Step.previousSlot descriptor.slot (by omega))

def priorValue (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (descriptor : PiRLCFirst54DirectSchedule.Value) : F :=
  if descriptor.candidate.round.val = 0 then 0
  else baseEnv program base descriptor.priorValueColumn

def outputValue (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (descriptor : PiRLCFirst54DirectSchedule.Value) : F :=
  baseEnv program base descriptor.valueColumn

def finalValue (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (source : Fin PiRLCFirst54DirectSchedule.finalCount) : F :=
  baseEnv program base (PiRLCFirst54DirectSchedule.finalColumn source)

/-- The one shared accepted-symbol product required by each candidate's 54
value rows. -/
def honestProducts (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (candidate : Fin PiRLCFirst54DirectSchedule.candidateCount) : F :=
  acceptedValue program base
      (PiRLCFirst54DirectSchedule.candidate candidate) *
    symbolValue program base
      (PiRLCFirst54DirectSchedule.candidate candidate)

@[simp] theorem subtract_eval {logicalWidth : Nat}
    (left right : SparseForm logicalWidth)
    (assignment : Assignment F logicalWidth) :
    (subtract left right).eval assignment =
      left.eval assignment - right.eval assignment := by
  simp [subtract, sub_eq_add_neg]

structure Preserves {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (inputs : Inputs program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F) : Prop where
  reject : ∀ candidate,
    (rejectForm inputs candidate).eval assignment =
      rejectValue program base candidate
  symbol : ∀ candidate,
    (symbolForm inputs candidate).eval assignment =
      symbolValue program base candidate
  priorPosition : ∀ candidate slot,
    (priorPositionForm inputs candidate slot).eval assignment =
      priorPositionValue program base candidate slot
  positionOutput : ∀ descriptor,
    (positionOutputForm inputs descriptor).eval assignment =
      positionOutputValue program base descriptor
  product : ∀ candidate,
    (productForm inputs candidate).eval assignment = products candidate
  priorValue : ∀ descriptor,
    (priorValueForm inputs descriptor).eval assignment =
      priorValue program base descriptor
  outputValue : ∀ descriptor,
    (valueOutputForm inputs descriptor).eval assignment =
      outputValue program base descriptor
  final : ∀ source,
    (finalForm inputs source).eval assignment = finalValue program base source

private theorem oneForm_eval
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (inputs : Inputs program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment inputs.oneColumn = 1) :
    (oneForm inputs).eval assignment = 1 := by
  simp [oneForm, one]

private theorem acceptedForm_eval
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (inputs : Inputs program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (one : assignment inputs.oneColumn = 1)
    (preserves : Preserves inputs assignment base products)
    (candidate : PiRLCFirst54DirectSchedule.Candidate) :
    (acceptedForm inputs candidate).eval assignment =
      acceptedValue program base candidate := by
  simp [acceptedForm, acceptedValue, oneForm_eval inputs assignment one,
    preserves.reject candidate]

private theorem previousPositionForm_eval
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (inputs : Inputs program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (preserves : Preserves inputs assignment base products)
    (descriptor : PiRLCFirst54DirectSchedule.Position) :
    (previousPositionForm inputs descriptor).eval assignment =
      previousPositionValue program base descriptor := by
  unfold previousPositionForm previousPositionValue
  split
  · simp
  · exact preserves.priorPosition _ _

private theorem positionDeltaForm_eval
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (inputs : Inputs program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (preserves : Preserves inputs assignment base products)
    (descriptor : PiRLCFirst54DirectSchedule.Position) :
    (positionDeltaForm inputs descriptor).eval assignment =
      if descriptor.slot.val = First54Step.fullSlot then
        previousPositionValue program base descriptor
      else
        previousPositionValue program base descriptor -
          priorPositionValue program base descriptor.candidate descriptor.slot := by
  unfold positionDeltaForm
  split
  · exact previousPositionForm_eval inputs assignment base products preserves
      descriptor
  · rw [subtract_eval,
      previousPositionForm_eval inputs assignment base products preserves]
    rw [preserves.priorPosition]

private theorem positionDifferenceForm_eval
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (inputs : Inputs program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (preserves : Preserves inputs assignment base products)
    (descriptor : PiRLCFirst54DirectSchedule.Position) :
    (positionDifferenceForm inputs descriptor).eval assignment =
      positionOutputValue program base descriptor -
        priorPositionValue program base descriptor.candidate descriptor.slot := by
  rw [positionDifferenceForm, subtract_eval, preserves.positionOutput,
    preserves.priorPosition]

private theorem valueDifferenceForm_eval
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (inputs : Inputs program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (preserves : Preserves inputs assignment base products)
    (descriptor : PiRLCFirst54DirectSchedule.Value) :
    (valueDifferenceForm inputs descriptor).eval assignment =
      outputValue program base descriptor - priorValue program base descriptor := by
  rw [valueDifferenceForm, subtract_eval, preserves.outputValue,
    preserves.priorValue]

def PositionEquations {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (inputs : Inputs program logicalWidth)
    (assignment : Assignment F logicalWidth) : Prop :=
  ∀ row,
    ((positionInterface inputs).left row).eval assignment *
        ((positionInterface inputs).right row).eval assignment =
      ((positionInterface inputs).output row).eval assignment

def AcceptedProductEquations
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (inputs : Inputs program logicalWidth)
    (assignment : Assignment F logicalWidth) : Prop :=
  ∀ row,
    ((acceptedProductInterface inputs).left row).eval assignment *
        ((acceptedProductInterface inputs).right row).eval assignment =
      ((acceptedProductInterface inputs).output row).eval assignment

def ValueEquations {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (inputs : Inputs program logicalWidth)
    (assignment : Assignment F logicalWidth) : Prop :=
  ∀ row,
    ((valueInterface inputs).left row).eval assignment *
        ((valueInterface inputs).right row).eval assignment =
      ((valueInterface inputs).output row).eval assignment

def FinalEquations {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (inputs : Inputs program logicalWidth)
    (assignment : Assignment F logicalWidth) : Prop :=
  ∀ source, ((finalInterface inputs).value source).eval assignment = 0

/-- The composed First54 plan vanishes exactly for its four indexed equation
families. -/
theorem planRowsZero_iff
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (inputs : Inputs program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment inputs.oneColumn = 1) :
    (plan inputs).RowsZero assignment ↔
      PositionEquations inputs assignment ∧
        AcceptedProductEquations inputs assignment ∧
          ValueEquations inputs assignment ∧
            FinalEquations inputs assignment := by
  unfold plan acceptedTailPlan valueFinalPlan positionPlan
    acceptedProductPlan valuePlan finalPlan
  rw [ProductionRelation.Plan.append_rowsZero_iff]
  rw [ProductionRelation.Plan.append_rowsZero_iff]
  rw [ProductionRelation.Plan.append_rowsZero_iff]
  rw [MultiplicationFamilyPlan.planRowsZero_iff
    (positionInterface inputs) positionCount_le assignment one]
  rw [MultiplicationFamilyPlan.planRowsZero_iff
    (acceptedProductInterface inputs) candidateCount_le assignment one]
  rw [MultiplicationFamilyPlan.planRowsZero_iff
    (valueInterface inputs) valueCount_le assignment one]
  rw [PinFamilyPlan.planRowsZero_iff
    (finalInterface inputs) finalCount_le assignment one]
  rfl

private theorem output_eq_add_of_product_eq_sub
    (product output prior : F) (equation : product = output - prior) :
    output = prior + product := by
  have moved := congrArg (fun value : F => value + prior) equation
  simpa [sub_eq_add_neg, add_assoc, add_comm, add_left_comm] using moved.symm

theorem positionEquation_implies_update
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (inputs : Inputs program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (one : assignment inputs.oneColumn = 1)
    (preserves : Preserves inputs assignment base products)
    (equations : PositionEquations inputs assignment)
    (descriptor : PiRLCFirst54DirectSchedule.Position) :
    positionOutputValue program base descriptor =
      First54Step.update
        (acceptedValue program base descriptor.candidate)
        (priorPositionValue program base descriptor.candidate)
        descriptor.slot := by
  have equation := equations
    (PiRLCFirst54DirectSchedule.positionIndex descriptor)
  simp only [positionInterface,
    PiRLCFirst54DirectSchedule.position_positionIndex] at equation
  rw [acceptedForm_eval inputs assignment base products one preserves,
    positionDeltaForm_eval inputs assignment base products preserves,
    positionDifferenceForm_eval inputs assignment base products preserves]
      at equation
  by_cases first : descriptor.slot.val = 0
  · simp [First54Step.update, first, First54Step.fullSlot,
      previousPositionValue] at equation ⊢
    have outputEquation := output_eq_add_of_product_eq_sub _ _ _ equation
    simpa [sub_eq_add_neg, mul_add, add_mul, mul_neg, neg_mul, mul_comm,
      mul_left_comm, mul_assoc] using outputEquation
  · by_cases full : descriptor.slot.val = First54Step.fullSlot
    · simp [First54Step.update, first, full, First54Step.fullSlot,
        previousPositionValue] at equation ⊢
      have outputEquation := output_eq_add_of_product_eq_sub _ _ _ equation
      simpa [mul_comm] using outputEquation
    · have notFull54 : descriptor.slot.val ≠ 54 := by
        simpa [First54Step.fullSlot] using full
      simp [First54Step.update, first, notFull54, First54Step.fullSlot,
        previousPositionValue] at equation ⊢
      have outputEquation := output_eq_add_of_product_eq_sub _ _ _ equation
      simpa [sub_eq_add_neg, mul_add, add_mul, mul_neg, neg_mul, mul_comm,
        mul_left_comm, mul_assoc, add_assoc, add_comm, add_left_comm] using
          outputEquation

theorem acceptedProductEquation
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (inputs : Inputs program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (one : assignment inputs.oneColumn = 1)
    (preserves : Preserves inputs assignment base products)
    (equations : AcceptedProductEquations inputs assignment)
    (candidate : Fin PiRLCFirst54DirectSchedule.candidateCount) :
    acceptedValue program base (PiRLCFirst54DirectSchedule.candidate candidate) *
        symbolValue program base (PiRLCFirst54DirectSchedule.candidate candidate) =
      products candidate := by
  have equation := equations candidate
  simp only [acceptedProductInterface] at equation
  rw [acceptedForm_eval inputs assignment base products one preserves,
    preserves.symbol, preserves.product] at equation
  exact equation

theorem valueEquation_implies_update
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (inputs : Inputs program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (preserves : Preserves inputs assignment base products)
    (productEquations : AcceptedProductEquations inputs assignment)
    (valueEquations : ValueEquations inputs assignment)
    (one : assignment inputs.oneColumn = 1)
    (descriptor : PiRLCFirst54DirectSchedule.Value) :
    outputValue program base descriptor =
      First54ValueStep.update
        (acceptedValue program base descriptor.candidate)
        (symbolValue program base descriptor.candidate)
        (priorPositionValue program base descriptor.candidate)
        (fun slot => priorValue program base ⟨descriptor.candidate, slot⟩)
        descriptor.slot := by
  have valueEquation := valueEquations
    (PiRLCFirst54DirectSchedule.valueIndex descriptor)
  simp only [valueInterface,
    PiRLCFirst54DirectSchedule.value_valueIndex] at valueEquation
  rw [preserves.priorPosition, preserves.product,
    valueDifferenceForm_eval inputs assignment base products preserves]
      at valueEquation
  have productEquation := acceptedProductEquation inputs assignment base
    products one preserves productEquations
      (PiRLCFirst54DirectSchedule.candidateIndex descriptor.candidate)
  simp only [PiRLCFirst54DirectSchedule.candidate_candidateIndex]
    at productEquation
  rw [← productEquation] at valueEquation
  simp only [First54ValueStep.update]
  have outputEquation := output_eq_add_of_product_eq_sub _ _ _ valueEquation
  simpa [mul_assoc] using outputEquation

theorem finalEquation_implies_full
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (inputs : Inputs program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (one : assignment inputs.oneColumn = 1)
    (preserves : Preserves inputs assignment base products)
    (equations : FinalEquations inputs assignment)
    (source : Fin PiRLCFirst54DirectSchedule.finalCount) :
    finalValue program base source = 1 := by
  have equation := equations source
  simp only [finalInterface] at equation
  rw [subtract_eval, preserves.final, oneForm_eval inputs assignment one]
    at equation
  exact Lean.Grind.AddCommGroup.sub_eq_zero_iff.mp equation

structure SourceHolds (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (source : Fin PiRLCFirst54DirectSchedule.sourceCount) : Prop where
  position : ∀ round slot,
    let candidate : PiRLCFirst54DirectSchedule.Candidate := ⟨source, round⟩
    let descriptor : PiRLCFirst54DirectSchedule.Position := ⟨candidate, slot⟩
    positionOutputValue program base descriptor =
      First54Step.update (acceptedValue program base candidate)
        (priorPositionValue program base candidate) slot
  value : ∀ round slot,
    let candidate : PiRLCFirst54DirectSchedule.Candidate := ⟨source, round⟩
    let descriptor : PiRLCFirst54DirectSchedule.Value := ⟨candidate, slot⟩
    outputValue program base descriptor =
      First54ValueStep.update (acceptedValue program base candidate)
        (symbolValue program base candidate)
        (priorPositionValue program base candidate)
        (fun current => priorValue program base ⟨candidate, current⟩) slot
  full : finalValue program base ⟨source.val, by simpa using source.isLt⟩ = 1

/-- Zero direct rows force the complete First54 transition semantics for
every source. -/
theorem rowsZero_implies_sourceHolds
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (inputs : Inputs program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (one : assignment inputs.oneColumn = 1)
    (preserves : Preserves inputs assignment base products)
    (rowsZero : (plan inputs).RowsZero assignment)
    (source : Fin PiRLCFirst54DirectSchedule.sourceCount) :
    SourceHolds program base source := by
  rcases (planRowsZero_iff inputs assignment one).mp rowsZero with
    ⟨positionEquations, productEquations, valueEquations, finalEquations⟩
  refine ⟨?_, ?_, ?_⟩
  · intro round slot
    exact positionEquation_implies_update inputs assignment base products one
      preserves positionEquations ⟨⟨source, round⟩, slot⟩
  · intro round slot
    exact valueEquation_implies_update inputs assignment base products preserves
      productEquations valueEquations one ⟨⟨source, round⟩, slot⟩
  · exact finalEquation_implies_full inputs assignment base products one
      preserves finalEquations ⟨source.val, by simpa using source.isLt⟩

/-- The complete First54 source semantics, with the honest shared products,
make every row of the direct plan vanish. -/
theorem sourceHolds_imply_rowsZero
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (inputs : Inputs program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (one : assignment inputs.oneColumn = 1)
    (preserves : Preserves inputs assignment base
      (honestProducts program base))
    (sourceHolds : ∀ source, SourceHolds program base source) :
    (plan inputs).RowsZero assignment := by
  apply (planRowsZero_iff inputs assignment one).mpr
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro row
    let descriptor := PiRLCFirst54DirectSchedule.position row
    have semantic :=
      (sourceHolds descriptor.candidate.source).position
        descriptor.candidate.round descriptor.slot
    change positionOutputValue program base descriptor =
      First54Step.update
        (acceptedValue program base descriptor.candidate)
        (priorPositionValue program base descriptor.candidate)
        descriptor.slot at semantic
    simp only [positionInterface]
    change (acceptedForm inputs descriptor.candidate).eval assignment *
        (positionDeltaForm inputs descriptor).eval assignment =
      (positionDifferenceForm inputs descriptor).eval assignment
    rw [acceptedForm_eval inputs assignment base (honestProducts program base)
      one preserves]
    rw [positionDeltaForm_eval inputs assignment base
      (honestProducts program base) preserves]
    rw [positionDifferenceForm_eval inputs assignment base
      (honestProducts program base) preserves]
    rw [semantic]
    by_cases first : descriptor.slot.val = 0
    · simp [First54Step.update, previousPositionValue, first,
        First54Step.fullSlot, sub_eq_add_neg, mul_add, add_mul, mul_neg,
        neg_mul, mul_comm, mul_left_comm, mul_assoc, add_assoc, add_comm,
        add_left_comm]
    · by_cases full : descriptor.slot.val = First54Step.fullSlot
      · simp [First54Step.update, previousPositionValue, full,
          First54Step.fullSlot, sub_eq_add_neg, mul_add, add_mul, mul_neg,
          neg_mul, mul_comm, mul_left_comm, mul_assoc, add_assoc, add_comm,
          add_left_comm]
      · simp [First54Step.update, previousPositionValue, first, full,
          sub_eq_add_neg, mul_add, add_mul, mul_neg, neg_mul, mul_comm,
          mul_left_comm, mul_assoc, add_assoc, add_comm, add_left_comm]
  · intro row
    simp only [acceptedProductInterface]
    rw [acceptedForm_eval inputs assignment base (honestProducts program base)
      one preserves]
    rw [preserves.symbol, preserves.product]
    rfl
  · intro row
    let descriptor := PiRLCFirst54DirectSchedule.value row
    have semantic :=
      (sourceHolds descriptor.candidate.source).value
        descriptor.candidate.round descriptor.slot
    change outputValue program base descriptor =
      First54ValueStep.update
        (acceptedValue program base descriptor.candidate)
        (symbolValue program base descriptor.candidate)
        (priorPositionValue program base descriptor.candidate)
        (fun slot => priorValue program base ⟨descriptor.candidate, slot⟩)
        descriptor.slot at semantic
    simp only [valueInterface]
    change
      (priorPositionForm inputs descriptor.candidate
          (First54ValueStep.positionSlot descriptor.slot)).eval assignment *
        (productForm inputs
          (PiRLCFirst54DirectSchedule.candidateIndex
            descriptor.candidate)).eval assignment =
      (valueDifferenceForm inputs descriptor).eval assignment
    rw [preserves.priorPosition, preserves.product]
    rw [valueDifferenceForm_eval inputs assignment base
      (honestProducts program base) preserves]
    rw [semantic]
    simp only [honestProducts,
      PiRLCFirst54DirectSchedule.candidate_candidateIndex,
      First54ValueStep.update]
    simp [sub_eq_add_neg, mul_assoc, add_assoc, add_comm, add_left_comm]
  · intro source
    simp only [finalInterface]
    rw [subtract_eval, preserves.final, oneForm_eval inputs assignment one]
    have full := (sourceHolds
      ⟨source.val, by simpa using source.isLt⟩).full
    simpa using congrArg (fun value : F => value - 1) full

end NightstreamFPrime.Export.Stage1.PiRLCFirst54DirectPlan
