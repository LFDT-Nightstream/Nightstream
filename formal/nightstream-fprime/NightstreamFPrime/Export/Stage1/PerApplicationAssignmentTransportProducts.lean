import NightstreamFPrime.Export.Stage1.PerApplicationAssignmentTransport
import NightstreamFPrime.Export.Stage1.PerApplicationCanonicalEncodes
import NightstreamFPrime.Export.Stage1.PiRLCProductMatrixProgramSemantics
import NightstreamFPrime.Export.Stage1.PiRLCRetainedInputs

/-!
Owns the value-level interpreter for the two compact derived-product recipes
in the per-application assignment transport. The interpreter reads only the
physical base assignment. It does not construct retained coordinates or the
final 31-block assignment.
-/

namespace NightstreamFPrime.Export.Stage1.PerApplicationAssignmentTransportProducts

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Stage1.PerApplicationAssignmentTransport
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open PerApplicationAssignmentPlan

abbrev Program := Lifecycle.Stage1.Application.Program

variable {program : Program}

/-- The physical-base assignment read by every derived-product recipe. -/
abbrev BaseValues (program : Program) :=
  Fin (PiRLCProductPlan.baseSourceWidth program) → F

def familyOrdinal : PiRLCProductSchedule.Family → Nat
  | .commitment => 0
  | .publicInput => 1
  | .evalK => 2
  | .evalA => 3

def emptyFamilyShape : Phi81FamilyShape := ⟨0, 0, 0⟩

/-- Select one family shape without expanding its invocations. Invalid recipe
indices select the empty shape and are rejected by the sealed-package parser. -/
def familyShape (recipe : Phi81GroupRecipe)
    (family : PiRLCProductSchedule.Family) : Phi81FamilyShape :=
  recipe.familyShapes.getD (familyOrdinal family) emptyFamilyShape

def shapeInvocationCount (recipe : Phi81GroupRecipe)
    (shape : Phi81FamilyShape) : Nat :=
  shape.sourceCount * shape.blockCount * recipe.ringDegree * shape.cellCount

/-- Prefix count of invocations in the recipe's fixed family order. -/
def familyOffset (recipe : Phi81GroupRecipe) :
    PiRLCProductSchedule.Family → Nat
  | .commitment => 0
  | .publicInput =>
      shapeInvocationCount recipe (familyShape recipe .commitment)
  | .evalK =>
      shapeInvocationCount recipe (familyShape recipe .commitment) +
        shapeInvocationCount recipe (familyShape recipe .publicInput)
  | .evalA =>
      shapeInvocationCount recipe (familyShape recipe .commitment) +
        shapeInvocationCount recipe (familyShape recipe .publicInput) +
          shapeInvocationCount recipe (familyShape recipe .evalK)

/-- Flat source-major, block-major, lane-major, cell-major recipe index. -/
def invocationIndex (recipe : Phi81GroupRecipe)
    (descriptor : PiRLCProductSchedule.Descriptor) : Nat :=
  let shape := familyShape recipe descriptor.family
  familyOffset recipe descriptor.family +
    descriptor.source.val * shape.blockCount * recipe.ringDegree *
        shape.cellCount +
      descriptor.block.val * recipe.ringDegree * shape.cellCount +
        descriptor.lane.val * shape.cellCount + descriptor.cell.val

@[simp] private theorem canonical_familyShape
    (family : PiRLCProductSchedule.Family) :
    familyShape (phi81GroupRecipe program) family =
      match family with
      | .commitment => ⟨17, 22, 1⟩
      | .publicInput => ⟨17, 5, 1⟩
      | .evalK => ⟨17, 1, 2⟩
      | .evalA => ⟨17, 14, 2⟩ := by
  cases family <;> rfl

/-- The assignment recipe uses the authoritative flat product index. -/
private theorem canonical_invocationIndex
    (descriptor : PiRLCProductSchedule.Descriptor) :
    invocationIndex (phi81GroupRecipe program) descriptor = descriptor.invocation.val := by
  rw [PiRLCProductSchedule.Descriptor.invocation_val]
  rcases descriptor with ⟨family, source, block, lane, cell⟩
  cases family <;>
    simp only [invocationIndex, familyOffset, canonical_familyShape, shapeInvocationCount,
      PiRLCProductSchedule.Family.blockCount, PiRLCProductSchedule.Family.cellCount,
      show (phi81GroupRecipe program).ringDegree = 54 from rfl]
  all_goals omega

/-- Read one compact block source from the physical base. An invalid slot or
a source outside the physical base evaluates to zero; the sealed decoder
rejects both conditions before execution. -/
def baseBlockValue (program : Program) (base : BaseValues program)
    (kind : BlockKind) (slot : Nat) : F :=
  if slotBound : slot <
      (PerApplicationAssignmentBlocks.entry program kind).block.slotCount then
    let source := PerApplicationAssignmentBlocks.sourceIndex program kind
      ⟨slot, slotBound⟩
    if sourceBound : source < PiRLCProductPlan.baseSourceWidth program then
      base ⟨source, sourceBound⟩
    else
      0
  else
    0

private theorem baseBlockValue_eq_source (program : Program) (base : BaseValues program)
    (kind : BlockKind) (slot : Nat)
    (slotBound : slot <
      (PerApplicationAssignmentBlocks.entry program kind).block.slotCount)
    (sourceBound :
      PerApplicationAssignmentBlocks.sourceIndex program kind
          ⟨slot, slotBound⟩ < PiRLCProductPlan.baseSourceWidth program) :
    baseBlockValue program base kind slot =
      base ⟨PerApplicationAssignmentBlocks.sourceIndex program kind
        ⟨slot, slotBound⟩, sourceBound⟩ := by
  unfold baseBlockValue
  rw [dif_pos slotBound]
  dsimp only
  rw [dif_pos sourceBound]

/-- Recipe arithmetic selects the final First54 value for the same source and
coefficient lane. -/
private theorem canonical_challengeSlot
    (source : Fin PiRLCCombinationInvocations.sourceCount)
    (lane : Fin ringDegree) :
    (phi81GroupRecipe program).challengeSlotBase +
        source.val * (phi81GroupRecipe program).challengeSourceStride + lane.val =
      (PiRLCFirst54DirectSchedule.valueIndex
        (PiRLCProductSourceBlocks.challengeValueDescriptor source lane)).val := by
  simpa [phi81GroupRecipe,
    PiRLCProductMatrixProgram.challengeSlotStart,
    PiRLCProductMatrixProgram.challengeSourceStride] using
      PiRLCProductMatrixProgram.challengeSlot_eq source lane

/-- The retained First54 value block exposes its physical package source at
the same value descriptor. -/
private theorem first54Value_sourceIndex (program : Program)
    (descriptor : PiRLCFirst54DirectSchedule.Value) :
    PerApplicationAssignmentBlocks.sourceIndex program .first54Value
        (PiRLCFirst54DirectSchedule.valueIndex descriptor) =
      (PiRLCFirst54DirectPlan.retainedValueColumn program descriptor).val := by
  unfold PerApplicationAssignmentBlocks.sourceIndex
    PerApplicationAssignmentBlocks.entry
    PerApplicationAssignmentBlocks.zeroRaw
    PerApplicationAssignmentPlan.BlockKind.expand
    PerApplicationCanonicalAssignment.Canonical.ofBlock
    CanonicalBlockAssignment.ofBlock
  change ((PiRLCFirst54RetainedBlocks.valueBlock program).source
    (PiRLCFirst54DirectSchedule.valueIndex descriptor)).val = _
  rw [PiRLCFirst54RetainedBlocks.valueBlock_source]
  rw [PiRLCFirst54DirectSchedule.value_valueIndex]

private theorem first54Reject_sourceIndex (program : Program)
    (candidate : PiRLCFirst54DirectSchedule.Candidate) :
    PerApplicationAssignmentBlocks.sourceIndex program .first54Reject
        (PiRLCFirst54DirectSchedule.candidateIndex candidate) =
      (PiRLCFirst54DirectPlan.retainedRejectColumn program candidate).val := by
  unfold PerApplicationAssignmentBlocks.sourceIndex
    PerApplicationAssignmentBlocks.entry
    PerApplicationAssignmentBlocks.zeroRaw
    PerApplicationAssignmentPlan.BlockKind.expand
    PerApplicationCanonicalAssignment.Canonical.ofBlock
    CanonicalBlockAssignment.ofBlock
  change ((PiRLCFirst54RetainedBlocks.rejectBlock program).source
    (PiRLCFirst54DirectSchedule.candidateIndex candidate)).val = _
  rw [PiRLCFirst54RetainedBlocks.rejectBlock_source]
  rw [PiRLCFirst54DirectSchedule.candidate_candidateIndex]

private theorem first54Symbol_sourceIndex (program : Program)
    (candidate : PiRLCFirst54DirectSchedule.Candidate) :
    PerApplicationAssignmentBlocks.sourceIndex program .first54Symbol
        (PiRLCFirst54DirectSchedule.candidateIndex candidate) =
      (PiRLCFirst54DirectPlan.retainedSymbolColumn program candidate).val := by
  unfold PerApplicationAssignmentBlocks.sourceIndex
    PerApplicationAssignmentBlocks.entry
    PerApplicationAssignmentBlocks.zeroRaw
    PerApplicationAssignmentPlan.BlockKind.expand
    PerApplicationCanonicalAssignment.Canonical.ofBlock
    CanonicalBlockAssignment.ofBlock
  change ((PiRLCFirst54RetainedBlocks.symbolBlock program).source
    (PiRLCFirst54DirectSchedule.candidateIndex candidate)).val = _
  rw [PiRLCFirst54RetainedBlocks.symbolBlock_source]
  rw [PiRLCFirst54DirectSchedule.candidate_candidateIndex]

private theorem canonical_first54Value_read
    {program : Program}
    (raw : PerApplicationCanonicalAssignment.RawValues program)
    (descriptor : PiRLCFirst54DirectSchedule.Value) :
    baseBlockValue program raw.base .first54Value
        (PiRLCFirst54DirectSchedule.valueIndex descriptor).val =
      PiRLCFirst54DirectPlan.outputValue program raw.base descriptor := by
  have slotBound :
      (PiRLCFirst54DirectSchedule.valueIndex descriptor).val <
        (PerApplicationAssignmentBlocks.entry program
          .first54Value).block.slotCount := by
    change (PiRLCFirst54DirectSchedule.valueIndex descriptor).val <
      PiRLCFirst54DirectSchedule.valueCount
    exact (PiRLCFirst54DirectSchedule.valueIndex descriptor).isLt
  have sourceIndexEq :
      PerApplicationAssignmentBlocks.sourceIndex program .first54Value
          ⟨(PiRLCFirst54DirectSchedule.valueIndex descriptor).val,
            slotBound⟩ =
        (PiRLCFirst54DirectPlan.retainedValueColumn program descriptor).val := by
    simpa only using! first54Value_sourceIndex program descriptor
  have sourceBound :
      PerApplicationAssignmentBlocks.sourceIndex program .first54Value
          ⟨(PiRLCFirst54DirectSchedule.valueIndex descriptor).val,
            slotBound⟩ < PiRLCProductPlan.baseSourceWidth program := by
    rw [sourceIndexEq]
    exact PiRLCFirst54DirectPlan.retainedValueColumn_val_lt_baseSourceWidth
      program descriptor
  have sourceEq :
      PiRLCRetainedPreservation.baseSourceColumn program
          ⟨PerApplicationAssignmentBlocks.sourceIndex program
            .first54Value
              ⟨(PiRLCFirst54DirectSchedule.valueIndex descriptor).val,
                slotBound⟩, sourceBound⟩ =
        (PerApplicationAssignmentBlocks.entry program
          .first54Value).block.source
            ⟨(PiRLCFirst54DirectSchedule.valueIndex descriptor).val,
              slotBound⟩ := by
    apply Fin.ext
    rfl
  let geometry := PerApplicationCanonicalEncodes.retainedGeometry program
  have encodes := PerApplicationCanonicalEncodes.retainedEncodes raw
  have preserves := PiRLCRetainedPreservation.first54Inputs_preserves
    geometry raw.assignment raw.base raw.groupValue raw.products
      (PerApplicationCanonicalAssignment.assignment_one raw) encodes
  calc
    baseBlockValue program raw.base .first54Value
        (PiRLCFirst54DirectSchedule.valueIndex descriptor).val =
      raw.base
        ⟨PerApplicationAssignmentBlocks.sourceIndex program .first54Value
          ⟨(PiRLCFirst54DirectSchedule.valueIndex descriptor).val,
            slotBound⟩, sourceBound⟩ :=
      baseBlockValue_eq_source program raw.base .first54Value _ slotBound
        sourceBound
    _ =
      raw.retainedSource
        ((PerApplicationAssignmentBlocks.entry program
          .first54Value).block.source
            ⟨(PiRLCFirst54DirectSchedule.valueIndex descriptor).val,
              slotBound⟩) := by
      unfold PerApplicationCanonicalAssignment.RawValues.retainedSource
      rw [← sourceEq]
      exact (PiRLCRetainedPreservation.sourceAssignment_base program raw.base
        raw.groupValue raw.products ⟨_, sourceBound⟩).symm
    _ = ((PiRLCFirst54RetainedBlocks.valueBlock program).form
          (PiRLCRetainedGeometry.valueStart program)
          (PiRLCRetainedGeometry.valueFits geometry)
          (PiRLCFirst54DirectSchedule.valueIndex descriptor)).eval
        raw.assignment := by
      symm
      simpa [PerApplicationAssignmentBlocks.entry,
        PerApplicationAssignmentBlocks.zeroRaw,
        PerApplicationAssignmentPlan.BlockKind.expand,
        PerApplicationCanonicalAssignment.Canonical.ofBlock,
        CanonicalBlockAssignment.ofBlock] using!
          (LowNormBlock.Block.form_eval
            (PiRLCFirst54RetainedBlocks.valueBlock program)
            (PiRLCRetainedGeometry.valueStart program)
            (PiRLCRetainedGeometry.valueFits geometry) raw.assignment
            raw.retainedSource encodes.value
            (PiRLCFirst54DirectSchedule.valueIndex descriptor))
    _ = PiRLCFirst54DirectPlan.outputValue program raw.base descriptor := by
      simpa [PiRLCFirst54DirectPlan.valueOutputForm,
        PiRLCRetainedInputs.first54Inputs] using preserves.outputValue descriptor

private theorem canonical_first54Reject_read
    {program : Program}
    (raw : PerApplicationCanonicalAssignment.RawValues program)
    (candidate : PiRLCFirst54DirectSchedule.Candidate) :
    baseBlockValue program raw.base .first54Reject
        (PiRLCFirst54DirectSchedule.candidateIndex candidate).val =
      PiRLCFirst54DirectPlan.rejectValue program raw.base candidate := by
  have slotBound :
      (PiRLCFirst54DirectSchedule.candidateIndex candidate).val <
        (PerApplicationAssignmentBlocks.entry program
          .first54Reject).block.slotCount := by
    change (PiRLCFirst54DirectSchedule.candidateIndex candidate).val <
      PiRLCFirst54DirectSchedule.candidateCount
    exact (PiRLCFirst54DirectSchedule.candidateIndex candidate).isLt
  have sourceIndexEq :
      PerApplicationAssignmentBlocks.sourceIndex program .first54Reject
          ⟨(PiRLCFirst54DirectSchedule.candidateIndex candidate).val,
            slotBound⟩ =
        (PiRLCFirst54DirectPlan.retainedRejectColumn program candidate).val := by
    simpa only using! first54Reject_sourceIndex program candidate
  have sourceBound :
      PerApplicationAssignmentBlocks.sourceIndex program .first54Reject
          ⟨(PiRLCFirst54DirectSchedule.candidateIndex candidate).val,
            slotBound⟩ < PiRLCProductPlan.baseSourceWidth program := by
    rw [sourceIndexEq]
    exact PiRLCFirst54DirectPlan.retainedRejectColumn_val_lt_baseSourceWidth
      program candidate
  have sourceEq :
      PiRLCRetainedPreservation.baseSourceColumn program
          ⟨PerApplicationAssignmentBlocks.sourceIndex program
            .first54Reject
              ⟨(PiRLCFirst54DirectSchedule.candidateIndex candidate).val,
                slotBound⟩, sourceBound⟩ =
        (PerApplicationAssignmentBlocks.entry program
          .first54Reject).block.source
            ⟨(PiRLCFirst54DirectSchedule.candidateIndex candidate).val,
              slotBound⟩ := by
    apply Fin.ext
    rfl
  let geometry := PerApplicationCanonicalEncodes.retainedGeometry program
  have encodes := PerApplicationCanonicalEncodes.retainedEncodes raw
  have preserves := PiRLCRetainedPreservation.first54Inputs_preserves
    geometry raw.assignment raw.base raw.groupValue raw.products
      (PerApplicationCanonicalAssignment.assignment_one raw) encodes
  calc
    baseBlockValue program raw.base .first54Reject
        (PiRLCFirst54DirectSchedule.candidateIndex candidate).val =
      raw.base
        ⟨PerApplicationAssignmentBlocks.sourceIndex program .first54Reject
          ⟨(PiRLCFirst54DirectSchedule.candidateIndex candidate).val,
            slotBound⟩, sourceBound⟩ :=
      baseBlockValue_eq_source program raw.base .first54Reject _ slotBound
        sourceBound
    _ =
      raw.retainedSource
        ((PerApplicationAssignmentBlocks.entry program
          .first54Reject).block.source
            ⟨(PiRLCFirst54DirectSchedule.candidateIndex candidate).val,
              slotBound⟩) := by
      unfold PerApplicationCanonicalAssignment.RawValues.retainedSource
      rw [← sourceEq]
      exact (PiRLCRetainedPreservation.sourceAssignment_base program raw.base
        raw.groupValue raw.products ⟨_, sourceBound⟩).symm
    _ = ((PiRLCFirst54RetainedBlocks.rejectBlock program).form
          (PiRLCRetainedGeometry.rejectStart program)
          (PiRLCRetainedGeometry.rejectFits geometry)
          (PiRLCFirst54DirectSchedule.candidateIndex candidate)).eval
        raw.assignment := by
      symm
      simpa [PerApplicationAssignmentBlocks.entry,
        PerApplicationAssignmentBlocks.zeroRaw,
        PerApplicationAssignmentPlan.BlockKind.expand,
        PerApplicationCanonicalAssignment.Canonical.ofBlock,
        CanonicalBlockAssignment.ofBlock] using!
          (LowNormBlock.Block.form_eval
            (PiRLCFirst54RetainedBlocks.rejectBlock program)
            (PiRLCRetainedGeometry.rejectStart program)
            (PiRLCRetainedGeometry.rejectFits geometry) raw.assignment
            raw.retainedSource encodes.reject
            (PiRLCFirst54DirectSchedule.candidateIndex candidate))
    _ = PiRLCFirst54DirectPlan.rejectValue program raw.base candidate := by
      simpa [PiRLCFirst54DirectPlan.rejectForm,
        PiRLCRetainedInputs.first54Inputs] using preserves.reject candidate

private theorem canonical_first54Symbol_read
    {program : Program}
    (raw : PerApplicationCanonicalAssignment.RawValues program)
    (candidate : PiRLCFirst54DirectSchedule.Candidate) :
    baseBlockValue program raw.base .first54Symbol
        (PiRLCFirst54DirectSchedule.candidateIndex candidate).val =
      PiRLCFirst54DirectPlan.symbolValue program raw.base candidate := by
  have slotBound :
      (PiRLCFirst54DirectSchedule.candidateIndex candidate).val <
        (PerApplicationAssignmentBlocks.entry program
          .first54Symbol).block.slotCount := by
    change (PiRLCFirst54DirectSchedule.candidateIndex candidate).val <
      PiRLCFirst54DirectSchedule.candidateCount
    exact (PiRLCFirst54DirectSchedule.candidateIndex candidate).isLt
  have sourceIndexEq :
      PerApplicationAssignmentBlocks.sourceIndex program .first54Symbol
          ⟨(PiRLCFirst54DirectSchedule.candidateIndex candidate).val,
            slotBound⟩ =
        (PiRLCFirst54DirectPlan.retainedSymbolColumn program candidate).val := by
    simpa only using! first54Symbol_sourceIndex program candidate
  have sourceBound :
      PerApplicationAssignmentBlocks.sourceIndex program .first54Symbol
          ⟨(PiRLCFirst54DirectSchedule.candidateIndex candidate).val,
            slotBound⟩ < PiRLCProductPlan.baseSourceWidth program := by
    rw [sourceIndexEq]
    exact PiRLCFirst54DirectPlan.retainedSymbolColumn_val_lt_baseSourceWidth
      program candidate
  have sourceEq :
      PiRLCRetainedPreservation.baseSourceColumn program
          ⟨PerApplicationAssignmentBlocks.sourceIndex program
            .first54Symbol
              ⟨(PiRLCFirst54DirectSchedule.candidateIndex candidate).val,
                slotBound⟩, sourceBound⟩ =
        (PerApplicationAssignmentBlocks.entry program
          .first54Symbol).block.source
            ⟨(PiRLCFirst54DirectSchedule.candidateIndex candidate).val,
              slotBound⟩ := by
    apply Fin.ext
    rfl
  let geometry := PerApplicationCanonicalEncodes.retainedGeometry program
  have encodes := PerApplicationCanonicalEncodes.retainedEncodes raw
  have preserves := PiRLCRetainedPreservation.first54Inputs_preserves
    geometry raw.assignment raw.base raw.groupValue raw.products
      (PerApplicationCanonicalAssignment.assignment_one raw) encodes
  calc
    baseBlockValue program raw.base .first54Symbol
        (PiRLCFirst54DirectSchedule.candidateIndex candidate).val =
      raw.base
        ⟨PerApplicationAssignmentBlocks.sourceIndex program .first54Symbol
          ⟨(PiRLCFirst54DirectSchedule.candidateIndex candidate).val,
            slotBound⟩, sourceBound⟩ :=
      baseBlockValue_eq_source program raw.base .first54Symbol _ slotBound
        sourceBound
    _ =
      raw.retainedSource
        ((PerApplicationAssignmentBlocks.entry program
          .first54Symbol).block.source
            ⟨(PiRLCFirst54DirectSchedule.candidateIndex candidate).val,
              slotBound⟩) := by
      unfold PerApplicationCanonicalAssignment.RawValues.retainedSource
      rw [← sourceEq]
      exact (PiRLCRetainedPreservation.sourceAssignment_base program raw.base
        raw.groupValue raw.products ⟨_, sourceBound⟩).symm
    _ = ((PiRLCFirst54RetainedBlocks.symbolBlock program).form
          (PiRLCRetainedGeometry.symbolStart program)
          (PiRLCRetainedGeometry.symbolFits geometry)
          (PiRLCFirst54DirectSchedule.candidateIndex candidate)).eval
        raw.assignment := by
      symm
      simpa [PerApplicationAssignmentBlocks.entry,
        PerApplicationAssignmentBlocks.zeroRaw,
        PerApplicationAssignmentPlan.BlockKind.expand,
        PerApplicationCanonicalAssignment.Canonical.ofBlock,
        CanonicalBlockAssignment.ofBlock] using!
          (LowNormBlock.Block.form_eval
            (PiRLCFirst54RetainedBlocks.symbolBlock program)
            (PiRLCRetainedGeometry.symbolStart program)
            (PiRLCRetainedGeometry.symbolFits geometry) raw.assignment
            raw.retainedSource encodes.symbol
            (PiRLCFirst54DirectSchedule.candidateIndex candidate))
    _ = PiRLCFirst54DirectPlan.symbolValue program raw.base candidate := by
      simpa [PiRLCFirst54DirectPlan.symbolForm,
        PiRLCRetainedInputs.first54Inputs] using preserves.symbol candidate

/-- Challenge ring selected by the recipe's First54 value block. -/
def challengeRing (recipe : Phi81GroupRecipe) (program : Program)
    (base : BaseValues program) (descriptor : PiRLCProductSchedule.Descriptor) :
    RingF :=
  fun lane =>
    baseBlockValue program base recipe.challengeBlock
        (recipe.challengeSlotBase +
          descriptor.source.val * recipe.challengeSourceStride + lane.val) -
      Poseidon2.ofNat recipe.challengeShift

/-- Value ring selected by the recipe's family-major product-input block. -/
def valueRing (recipe : Phi81GroupRecipe) (program : Program)
    (base : BaseValues program) (descriptor : PiRLCProductSchedule.Descriptor) :
    RingF :=
  fun lane =>
    SourceCompiler.sourceEnv base <|
      AffineRuns.sourceAt recipe.valueSources
        (invocationIndex recipe (descriptor.withLane lane))

/-- Read one quotient coefficient from the complete physical base. The
single retained slot uses the original lane index. -/
def phi81GroupValue (recipe : Phi81GroupRecipe) (program : Program)
    (base : BaseValues program)
    (invocation : Fin PiRLCProductSchedule.invocationCount)
    (_group : Nat) : F :=
  let ring := PiRLCProductRingSchedule.ringInvocation invocation
  let representative := PiRLCProductRingSchedule.laneInvocation ring PiRLCProductRingSchedule.zeroLane
  let descriptor := PiRLCProductSchedule.descriptor representative
  Phi81Relation.QuotientProduct.quotientCoeff
    (challengeRing recipe program base descriptor)
    (valueRing recipe program base descriptor)
    (PiRLCProductSchedule.descriptor invocation).lane

/-- Complete base-only First54 accepted-symbol product evaluator. -/
def first54ProductValue (recipe : First54ProductRecipe) (program : Program)
    (base : BaseValues program) (candidate : Nat) : F :=
  (1 - baseBlockValue program base recipe.rejectBlock candidate) *
    baseBlockValue program base recipe.symbolBlock candidate

/-- The base-only Phi81 executor computes the exact quotient coefficient
used by the product-plan assignment. -/
theorem canonical_phi81GroupValue_eq_honestGroupValue
    {program : Program}
    (raw : PerApplicationCanonicalAssignment.RawValues program)
    (invocation : Fin PiRLCProductSchedule.invocationCount)
    (group : Fin 1) :
    phi81GroupValue (phi81GroupRecipe program) program raw.base invocation group.val =
      PiRLCProductPlan.honestGroupValue
        (PiRLCProductMatrixProgram.inputs
          (PerApplicationCanonicalEncodes.piCcsOrdinaryGeometry program))
        raw.assignment invocation group := by
  let geometry := PerApplicationCanonicalEncodes.retainedGeometry program
  let values := PiRLCValueWiring.form
    (PerApplicationCanonicalEncodes.piCcsOrdinaryGeometry program)
  let inputs := PiRLCRetainedInputs.productInputs values geometry
  let ring := PiRLCProductRingSchedule.ringInvocation invocation
  let representative := PiRLCProductRingSchedule.laneInvocation ring PiRLCProductRingSchedule.zeroLane
  let descriptor := PiRLCProductSchedule.descriptor representative
  have one : raw.assignment inputs.oneColumn = 1 := by
    exact PerApplicationCanonicalAssignment.assignment_one raw
  have encodes := PerApplicationCanonicalEncodes.retainedEncodes raw
  have preserves := PiRLCRetainedPreservation.productInputs_preserves
    values geometry raw.assignment raw.base raw.groupValue raw.products
      (PerApplicationCanonicalEncodes.productValuesPreserve raw) encodes
  have challengeRead :
      challengeRing (phi81GroupRecipe program) program raw.base descriptor =
        PiRLCProductPlan.challengeRing program raw.base descriptor := by
    funext lane
    unfold challengeRing PiRLCProductPlan.challengeRing
    dsimp only [phi81GroupRecipe]
    have slotEq := canonical_challengeSlot (program := program) descriptor.source lane
    dsimp only [phi81GroupRecipe] at slotEq
    rw [slotEq]
    rw [canonical_first54Value_read raw
      (PiRLCProductSourceBlocks.challengeValueDescriptor
        descriptor.source lane)]
    unfold PiRLCFirst54DirectPlan.outputValue
      PiRLCFirst54DirectPlan.baseEnv
    rw [← PiRLCProductSourceBlocks.challengeColumn_eq_first54Value
      descriptor lane]
    rfl
  have valueRead :
      valueRing (phi81GroupRecipe program) program raw.base descriptor =
        PiRLCProductPlan.valueRing program raw.base descriptor := by
    funext lane
    unfold valueRing PiRLCProductPlan.valueRing
    rw [canonical_invocationIndex]
    conv_lhs =>
      arg 2
      arg 1
      dsimp only [phi81GroupRecipe]
    rw [phi81ValueSources_at,
      PiRLCProductSchedule.descriptor_invocation]
    simp only [PiRLCProductPlan.valueColumn,
      PiRLCProductSchedule.Descriptor.withLane_valueColumn]
    rw [PiRLCProductPlan.baseEnv_valueColumn]
    exact SourceCompiler.sourceEnv_at raw.base _
  have challengeStateEval :
      Phi81ProductPlan.evalState raw.assignment
          (PiRLCProductPlan.challengeState inputs representative) =
        PiRLCProductPlan.challengeRing program raw.base descriptor := by
    funext lane
    have challengePreserves :
        (PiRLCProductPlan.challengeForm inputs representative lane).eval
            raw.assignment =
          PiRLCProductPlan.baseEnv program raw.base
            (descriptor.challengeColumn lane) := by
      simpa only [descriptor] using preserves.challenge representative lane
    simp [Phi81ProductPlan.evalState, PiRLCProductPlan.challengeState,
      PiRLCProductPlan.challengeRing, challengePreserves, one,
      sub_eq_add_neg]
  have valueStateEval :
      Phi81ProductPlan.evalState raw.assignment
          (PiRLCProductPlan.valueState inputs representative) =
        PiRLCProductPlan.valueRing program raw.base descriptor := by
    funext lane
    exact preserves.value representative lane
  have challengeEq :
      challengeRing (phi81GroupRecipe program) program raw.base descriptor =
        Phi81ProductPlan.evalState raw.assignment
          (PiRLCProductPlan.challengeState inputs representative) :=
    challengeRead.trans challengeStateEval.symm
  have valueEq :
      valueRing (phi81GroupRecipe program) program raw.base descriptor =
        Phi81ProductPlan.evalState raw.assignment
          (PiRLCProductPlan.valueState inputs representative) :=
    valueRead.trans valueStateEval.symm
  unfold phi81GroupValue
  change Phi81Relation.QuotientProduct.quotientCoeff
      (challengeRing (phi81GroupRecipe program) program raw.base descriptor)
      (valueRing (phi81GroupRecipe program) program raw.base descriptor)
      (PiRLCProductSchedule.descriptor invocation).lane = _
  rw [challengeEq, valueEq]
  rfl

/-- The canonical base-only First54 executor computes the exact honest
accepted-symbol product used by the existing First54 plan. -/
theorem canonical_first54ProductValue_eq_honestProducts
    {program : Program}
    (raw : PerApplicationCanonicalAssignment.RawValues program)
    (candidate : Fin PiRLCFirst54DirectSchedule.candidateCount) :
    first54ProductValue first54ProductRecipe program raw.base candidate.val =
      PiRLCFirst54DirectPlan.honestProducts program raw.base candidate := by
  let descriptor := PiRLCFirst54DirectSchedule.candidate candidate
  have indexEq :
      (PiRLCFirst54DirectSchedule.candidateIndex descriptor).val =
        candidate.val := by
    exact congrArg Fin.val
      (PiRLCFirst54DirectSchedule.candidateIndex_candidate candidate)
  unfold first54ProductValue
  change
    (1 - baseBlockValue program raw.base .first54Reject candidate.val) *
        baseBlockValue program raw.base .first54Symbol candidate.val =
      (1 - PiRLCFirst54DirectPlan.rejectValue program raw.base descriptor) *
        PiRLCFirst54DirectPlan.symbolValue program raw.base descriptor
  rw [← indexEq, canonical_first54Reject_read raw descriptor,
    canonical_first54Symbol_read raw descriptor]

end NightstreamFPrime.Export.Stage1.PerApplicationAssignmentTransportProducts
