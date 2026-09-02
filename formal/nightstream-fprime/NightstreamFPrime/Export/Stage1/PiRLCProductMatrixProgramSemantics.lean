import NightstreamFPrime.Export.Stage1.PiRLCProductMatrixProgram

/-!
Proves that the compact PiRLC product matrix block selects the exact
invocation-major SuperNeo family schedule and retained forms used by
`PiRLCProductPlan`.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCProductMatrixProgram

open NightstreamFPrime.Export.MatrixProgram
open NightstreamFPrime.Export.MatrixProgram.Phi81Product
open NightstreamFPrime.Gadgets.Sampling
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

private theorem coordinates_indexOf {blockCount cellCount : Nat}
    (block : Fin blockCount) (lane : Fin ringDegree)
    (cell : Fin cellCount) :
    CombinationStep.coordinates (CombinationStep.indexOf block lane cell) =
      (block, lane, cell) := by
  simp [CombinationStep.coordinates, CombinationStep.indexOf]

def commitmentOffset : Nat := 0

def publicInputOffset : Nat := commitmentFamily.invocationCount

def evalKOffset : Nat :=
  publicInputOffset + publicInputFamily.invocationCount

def evalAOffset : Nat := evalKOffset + evalKFamily.invocationCount

/-- The proof-relevant wire descriptor corresponding to one authoritative
PiRLC product descriptor. -/
def wireDescriptor : PiRLCProductSchedule.Descriptor →
    MatrixProgram.Phi81Product.Descriptor
  | ⟨.commitment, source, block, lane, cell⟩ =>
      { family := commitmentFamily
        familyOffset := commitmentOffset
        source
        coordinate := CombinationStep.indexOf block lane cell }
  | ⟨.publicInput, source, block, lane, cell⟩ =>
      { family := publicInputFamily
        familyOffset := publicInputOffset
        source
        coordinate := CombinationStep.indexOf block lane cell }
  | ⟨.evalK, source, block, lane, cell⟩ =>
      { family := evalKFamily
        familyOffset := evalKOffset
        source
        coordinate := CombinationStep.indexOf block lane cell }
  | ⟨.evalA, source, block, lane, cell⟩ =>
      { family := evalAFamily
        familyOffset := evalAOffset
        source
        coordinate := CombinationStep.indexOf block lane cell }

@[simp] theorem wireDescriptor_source
    (descriptor : PiRLCProductSchedule.Descriptor) :
    (wireDescriptor descriptor).source.val = descriptor.source.val := by
  rcases descriptor with ⟨family, source, block, lane, cell⟩
  cases family <;> rfl

@[simp] theorem wireDescriptor_block
    (descriptor : PiRLCProductSchedule.Descriptor) :
    (wireDescriptor descriptor).block.val = descriptor.block.val := by
  rcases descriptor with ⟨family, source, block, lane, cell⟩
  cases family
  all_goals
    simp [wireDescriptor, MatrixProgram.Phi81Product.Descriptor.block,
      MatrixProgram.Phi81Product.Descriptor.coordinates,
      coordinates_indexOf]

@[simp] theorem wireDescriptor_lane
    (descriptor : PiRLCProductSchedule.Descriptor) :
    (wireDescriptor descriptor).lane.val = descriptor.lane.val := by
  rcases descriptor with ⟨family, source, block, lane, cell⟩
  cases family
  all_goals
    simp [wireDescriptor, MatrixProgram.Phi81Product.Descriptor.lane,
      MatrixProgram.Phi81Product.Descriptor.coordinates,
      coordinates_indexOf]

@[simp] theorem wireDescriptor_cell
    (descriptor : PiRLCProductSchedule.Descriptor) :
    (wireDescriptor descriptor).cell.val = descriptor.cell.val := by
  rcases descriptor with ⟨family, source, block, lane, cell⟩
  cases family
  all_goals
    simp [wireDescriptor, MatrixProgram.Phi81Product.Descriptor.cell,
      MatrixProgram.Phi81Product.Descriptor.coordinates,
      coordinates_indexOf]

@[simp] theorem wireDescriptor_invocation
    (descriptor : PiRLCProductSchedule.Descriptor) :
    (wireDescriptor descriptor).invocation = descriptor.invocation.val := by
  rcases descriptor with ⟨family, source, block, lane, cell⟩
  cases family <;>
    simp [wireDescriptor, MatrixProgram.Phi81Product.Descriptor.invocation,
      MatrixProgram.Phi81Product.Descriptor.localInvocation,
      PiRLCProductSchedule.Descriptor.invocation,
      PiRLCProductSchedule.Descriptor.familyIndex,
      PiRLCProductSchedule.Family.invocationCount,
      PiRLCProductSchedule.Family.privateCount,
      PiRLCProductSchedule.Family.blockCount,
      PiRLCProductSchedule.Family.cellCount,
      PiRLCCombinationInvocations.sourceCount,
      commitmentOffset, publicInputOffset, evalKOffset, evalAOffset,
      commitmentFamily, publicInputFamily, evalKFamily, evalAFamily,
      MatrixProgram.Phi81Product.Family.invocationCount,
      MatrixProgram.Phi81Product.Family.privateCount,
      CombinationStep.privateCount, Fin.encodeProd, ringDegree] <;> omega

/-- The encoded four-family selector chooses the exact authoritative
descriptor and no Rust-selected schedule. -/
theorem descriptor?_wireDescriptor
    (descriptor : PiRLCProductSchedule.Descriptor) :
    MatrixProgram.Phi81Product.descriptor? families descriptor.invocation.val =
      some (wireDescriptor descriptor) := by
  rcases descriptor with ⟨family, source, block, lane, cell⟩
  cases family
  · change descriptorFrom?
        [commitmentFamily, publicInputFamily, evalKFamily, evalAFamily] 0
        (Fin.encodeProd (source,
          CombinationStep.indexOf block lane cell)).val =
      some {
        family := commitmentFamily
        familyOffset := commitmentOffset
        source
        coordinate := CombinationStep.indexOf block lane cell }
    exact descriptorFrom?_head commitmentFamily
      [publicInputFamily, evalKFamily, evalAFamily] 0 source
        (CombinationStep.indexOf block lane cell)
  · change descriptorFrom?
        [commitmentFamily, publicInputFamily, evalKFamily, evalAFamily] 0
        (commitmentFamily.invocationCount +
          (Fin.encodeProd (source,
            CombinationStep.indexOf block lane cell)).val) =
      some {
        family := publicInputFamily
        familyOffset := publicInputOffset
        source
        coordinate := CombinationStep.indexOf block lane cell }
    rw [descriptorFrom?_tail]
    exact descriptorFrom?_head publicInputFamily [evalKFamily, evalAFamily]
      (0 + commitmentFamily.invocationCount) source
        (CombinationStep.indexOf block lane cell)
  · change descriptorFrom?
        [commitmentFamily, publicInputFamily, evalKFamily, evalAFamily] 0
        (commitmentFamily.invocationCount +
          (publicInputFamily.invocationCount +
            (Fin.encodeProd (source,
              CombinationStep.indexOf block lane cell)).val)) =
      some {
        family := evalKFamily
        familyOffset := evalKOffset
        source
        coordinate := CombinationStep.indexOf block lane cell }
    rw [descriptorFrom?_tail, descriptorFrom?_tail]
    exact descriptorFrom?_head evalKFamily [evalAFamily]
      ((0 + commitmentFamily.invocationCount) +
        publicInputFamily.invocationCount) source
          (CombinationStep.indexOf block lane cell)
  · change descriptorFrom?
        [commitmentFamily, publicInputFamily, evalKFamily, evalAFamily] 0
        (commitmentFamily.invocationCount +
          (publicInputFamily.invocationCount +
            (evalKFamily.invocationCount +
              (Fin.encodeProd (source,
                CombinationStep.indexOf block lane cell)).val))) =
      some {
        family := evalAFamily
        familyOffset := evalAOffset
        source
        coordinate := CombinationStep.indexOf block lane cell }
    rw [descriptorFrom?_tail, descriptorFrom?_tail, descriptorFrom?_tail]
    exact descriptorFrom?_head evalAFamily []
      (((0 + commitmentFamily.invocationCount) +
        publicInputFamily.invocationCount) + evalKFamily.invocationCount)
          source (CombinationStep.indexOf block lane cell)

theorem challengeSlot_eq
    (source : Fin PiRLCCombinationInvocations.sourceCount)
    (lane : Fin ringDegree) :
    challengeSlotStart + source.val * challengeSourceStride + lane.val =
      (PiRLCFirst54DirectSchedule.valueIndex
        (PiRLCProductSourceBlocks.challengeValueDescriptor source lane)).val := by
  simp [challengeSlotStart, challengeSourceStride,
    PiRLCProductSourceBlocks.challengeValueDescriptor,
    PiRLCFirst54DirectSchedule.valueIndex,
    PiRLCFirst54DirectSchedule.candidateIndex,
    PiRLCFirst54DirectSchedule.candidateCount,
    PiRLCFirst54DirectSchedule.sourceCount,
    PiRLCFirst54DirectSchedule.roundCount,
    PiRLCFirst54Invocations.sourceCount,
    PiRLCFirst54Invocations.roundCount,
    First54ValueStep.outputCount, First54.candidateCount,
    Fin.encodeProd]
  ring

@[simp] theorem block_oneColumn?
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth) :
    (block geometry).oneColumn? logicalWidth =
      some (PiRLCRetainedGeometry.oneColumn geometry) := by
  unfold MatrixProgram.Phi81Product.Block.oneColumn? block
  rw [dif_pos (PiRLCRetainedGeometry.oneColumn geometry).isLt]

theorem challenge_form?
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth)
    (descriptor : PiRLCProductSchedule.Descriptor) (lane : Fin ringDegree) :
    (block geometry).challenge.form? logicalWidth
        (challengeSlotStart + descriptor.source.val *
          challengeSourceStride + lane.val) =
      some (PiRLCProductPlan.challengeForm
        (PiRLCRetainedInputs.productInputs geometry)
          descriptor.invocation lane) := by
  have direct := MatrixProgram.RetainedBlock.form?_ofSemantic
    (PiRLCFirst54RetainedBlocks.valueBlock program)
    (PiRLCRetainedGeometry.valueStart program)
    (PiRLCRetainedGeometry.valueFits geometry)
    (PiRLCFirst54DirectSchedule.valueIndex
      (PiRLCProductSourceBlocks.challengeValueDescriptor
        descriptor.source lane))
  rw [← challengeSlot_eq descriptor.source lane] at direct
  simpa [block, PiRLCProductPlan.challengeForm,
    PiRLCRetainedInputs.productInputs] using direct

theorem challengeState?
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth)
    (descriptor : PiRLCProductSchedule.Descriptor) :
    (block geometry).challengeState? logicalWidth (wireDescriptor descriptor) =
      some (PiRLCProductPlan.challengeForm
        (PiRLCRetainedInputs.productInputs geometry) descriptor.invocation) := by
  unfold MatrixProgram.Phi81Product.Block.challengeState?
  apply loadFin?_of_some
  intro lane
  simpa [block, wireDescriptor_source] using
    challenge_form? geometry descriptor lane

@[simp] theorem wireDescriptor_invocationAtLane
    (descriptor : PiRLCProductSchedule.Descriptor) (lane : Fin ringDegree) :
    (wireDescriptor descriptor).invocationAtLane lane =
      (descriptor.withLane lane).invocation.val := by
  rcases descriptor with ⟨family, source, block, productLane, cell⟩
  cases family <;>
    simp [wireDescriptor,
      MatrixProgram.Phi81Product.Descriptor.invocationAtLane,
      PiRLCProductSchedule.Descriptor.withLane,
      PiRLCProductSchedule.Descriptor.invocation,
      PiRLCProductSchedule.Descriptor.familyIndex,
      PiRLCProductSchedule.Family.invocationCount,
      PiRLCProductSchedule.Family.privateCount,
      PiRLCProductSchedule.Family.blockCount,
      PiRLCProductSchedule.Family.cellCount,
      PiRLCCombinationInvocations.sourceCount,
      commitmentOffset, publicInputOffset, evalKOffset, evalAOffset,
      commitmentFamily, publicInputFamily, evalKFamily, evalAFamily,
      MatrixProgram.Phi81Product.Family.invocationCount,
      MatrixProgram.Phi81Product.Family.privateCount,
      MatrixProgram.Phi81Product.Descriptor.block,
      MatrixProgram.Phi81Product.Descriptor.cell,
      MatrixProgram.Phi81Product.Descriptor.coordinates,
      coordinates_indexOf,
      CombinationStep.privateCount, Fin.encodeProd, ringDegree] <;> omega

theorem input_form?
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth)
    (descriptor : PiRLCProductSchedule.Descriptor) (lane : Fin ringDegree) :
    (block geometry).input.form? logicalWidth
        ((wireDescriptor descriptor).invocationAtLane lane) =
      some (PiRLCProductPlan.valueForm
        (PiRLCRetainedInputs.productInputs geometry)
          descriptor.invocation lane) := by
  have direct := MatrixProgram.RetainedBlock.form?_ofSemantic
    (PiRLCRetainedGeometry.productInputBlock program)
    (PiRLCRetainedGeometry.productInputStart program)
    (PiRLCRetainedGeometry.productInputFits geometry)
    (descriptor.withLane lane).invocation
  rw [wireDescriptor_invocationAtLane]
  simpa [block, PiRLCProductPlan.valueForm,
    PiRLCRetainedInputs.productInputs,
    PiRLCProductSchedule.descriptor_invocation] using direct

theorem inputState?
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth)
    (descriptor : PiRLCProductSchedule.Descriptor) :
    (block geometry).inputState? logicalWidth (wireDescriptor descriptor) =
      some (PiRLCProductPlan.valueState
        (PiRLCRetainedInputs.productInputs geometry) descriptor.invocation) := by
  unfold MatrixProgram.Phi81Product.Block.inputState?
    PiRLCProductPlan.valueState
  apply loadFin?_of_some
  intro lane
  exact input_form? geometry descriptor lane

theorem group_form?
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth)
    (descriptor : PiRLCProductSchedule.Descriptor) (group : Fin 33) :
    (block geometry).group.form? logicalWidth
        ((wireDescriptor descriptor).invocation * 33 + group.val) =
      some (PiRLCProductPlan.groupForm
        (PiRLCRetainedInputs.productInputs geometry)
          descriptor.invocation group) := by
  have direct := MatrixProgram.RetainedBlock.form?_ofSemantic
    (PiRLCRetainedGeometry.productGroupBlock program)
    (PiRLCRetainedGeometry.productGroupStart program)
    (PiRLCRetainedGeometry.productGroupFits geometry)
    (Fin.encodeProd (descriptor.invocation, group))
  simpa [block, PiRLCProductPlan.groupForm,
    PiRLCRetainedInputs.productInputs, wireDescriptor_invocation,
    Fin.encodeProd, Nat.mul_comm] using direct

theorem groupOutput?
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth)
    (descriptor : PiRLCProductSchedule.Descriptor) :
    (block geometry).groupOutput? logicalWidth (wireDescriptor descriptor) =
      some (PiRLCProductPlan.groupForm
        (PiRLCRetainedInputs.productInputs geometry) descriptor.invocation) := by
  unfold MatrixProgram.Phi81Product.Block.groupOutput?
  apply loadFin?_of_some
  intro group
  exact group_form? geometry descriptor group

theorem output_form?
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth)
    (descriptor : PiRLCProductSchedule.Descriptor) :
    (block geometry).output.form? logicalWidth
        (wireDescriptor descriptor).invocation =
      some (PiRLCProductPlan.outputForm
        (PiRLCRetainedInputs.productInputs geometry)
          descriptor.invocation) := by
  have direct := MatrixProgram.RetainedBlock.form?_ofSemantic
    (PiRLCRetainedGeometry.productOutputBlock program)
    (PiRLCRetainedGeometry.productOutputStart program)
    (PiRLCRetainedGeometry.productOutputFits geometry)
    descriptor.invocation
  simpa [block, PiRLCProductPlan.outputForm,
    PiRLCRetainedInputs.productInputs, wireDescriptor_invocation] using direct

@[simp] theorem wireDescriptor_lane_eq
    (descriptor : PiRLCProductSchedule.Descriptor) :
    (wireDescriptor descriptor).lane = descriptor.lane := by
  apply Fin.ext
  exact wireDescriptor_lane descriptor

@[simp] theorem wireDescriptor_privateCount
    (descriptor : PiRLCProductSchedule.Descriptor) :
    (wireDescriptor descriptor).family.privateCount =
      descriptor.family.privateCount := by
  rcases descriptor with ⟨family, source, block, lane, cell⟩
  cases family <;>
    rfl

theorem previousInvocation_eq
    (descriptor : PiRLCProductSchedule.Descriptor)
    (notFirst : descriptor.source.val ≠ 0) :
    (wireDescriptor descriptor).invocation -
        (wireDescriptor descriptor).family.privateCount =
      (descriptor.previousSource notFirst).invocation.val := by
  rcases descriptor with ⟨family, source, block, lane, cell⟩
  have sourceNonzero : source.val ≠ 0 := notFirst
  cases family
  · rw [wireDescriptor_invocation, wireDescriptor_privateCount]
    change (1188 * source.val +
        (CombinationStep.indexOf block lane cell).val) - 1188 =
      1188 * (source.val - 1) +
        (CombinationStep.indexOf block lane cell).val
    omega
  · rw [wireDescriptor_invocation, wireDescriptor_privateCount]
    change 20196 + (270 * source.val +
        (CombinationStep.indexOf block lane cell).val) - 270 =
      20196 + (270 * (source.val - 1) +
        (CombinationStep.indexOf block lane cell).val)
    omega
  · rw [wireDescriptor_invocation, wireDescriptor_privateCount]
    change 20196 + (4590 + (108 * source.val +
        (CombinationStep.indexOf block lane cell).val)) - 108 =
      20196 + (4590 + (108 * (source.val - 1) +
        (CombinationStep.indexOf block lane cell).val))
    omega
  · rw [wireDescriptor_invocation, wireDescriptor_privateCount]
    change 20196 + (4590 + (1836 + (1512 * source.val +
        (CombinationStep.indexOf block lane cell).val))) - 1512 =
      20196 + (4590 + (1836 + (1512 * (source.val - 1) +
        (CombinationStep.indexOf block lane cell).val)))
    omega

theorem prior_form?_of_ne
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth)
    (descriptor : PiRLCProductSchedule.Descriptor)
    (notFirst : descriptor.source.val ≠ 0) :
    (block geometry).output.form? logicalWidth
        ((wireDescriptor descriptor).invocation -
          (wireDescriptor descriptor).family.privateCount) =
      some (PiRLCProductPlan.priorForm
        (PiRLCRetainedInputs.productInputs geometry)
          descriptor.invocation) := by
  have direct := MatrixProgram.RetainedBlock.form?_ofSemantic
    (PiRLCRetainedGeometry.productOutputBlock program)
    (PiRLCRetainedGeometry.productOutputStart program)
    (PiRLCRetainedGeometry.productOutputFits geometry)
    (descriptor.previousSource notFirst).invocation
  rw [previousInvocation_eq descriptor notFirst]
  simpa [block, PiRLCProductPlan.priorForm,
    PiRLCRetainedInputs.productInputs,
    PiRLCProductSchedule.descriptor_invocation, notFirst] using direct

theorem prior_form_at_invocation?_of_ne
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth)
    (descriptor : PiRLCProductSchedule.Descriptor)
    (notFirst : descriptor.source.val ≠ 0) :
    (block geometry).output.form? logicalWidth
        (descriptor.invocation.val -
          (wireDescriptor descriptor).family.privateCount) =
      some (PiRLCProductPlan.priorForm
        (PiRLCRetainedInputs.productInputs geometry)
          descriptor.invocation) := by
  simpa using prior_form?_of_ne geometry descriptor notFirst

theorem output_form_at_invocation?
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth)
    (descriptor : PiRLCProductSchedule.Descriptor) :
    (block geometry).output.form? logicalWidth descriptor.invocation.val =
      some (PiRLCProductPlan.outputForm
        (PiRLCRetainedInputs.productInputs geometry)
          descriptor.invocation) := by
  simpa using output_form? geometry descriptor

def semanticInterface
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth)
    (descriptor : PiRLCProductSchedule.Descriptor) :
    ProductSumPlan.Interface logicalWidth :=
  Phi81ProductFamilyPlan.laneInterface
    (PiRLCProductPlan.interface
      (PiRLCRetainedInputs.productInputs geometry)) descriptor.invocation

/-- The complete fail-closed wire interface is exactly the direct semantic
interface for the same PiRLC product invocation. -/
theorem block_interface?
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth)
    (descriptor : PiRLCProductSchedule.Descriptor) :
    (block geometry).interface? logicalWidth (wireDescriptor descriptor) =
      some (semanticInterface geometry descriptor) := by
  unfold MatrixProgram.Phi81Product.Block.interface?
  rw [block_oneColumn?, challengeState?, inputState?, groupOutput?]
  by_cases first : descriptor.source.val = 0
  · have wireFirst : (wireDescriptor descriptor).source.val = 0 := by
      simpa using first
    rw [wireDescriptor_invocation]
    simp only [wireFirst, if_pos]
    rw [output_form_at_invocation?]
    apply congrArg some
    unfold semanticInterface Phi81ProductFamilyPlan.laneInterface
      Phi81ProductFamilyPlan.groupOutputAt PiRLCProductPlan.interface
      PiRLCProductPlan.challengeState PiRLCProductPlan.priorForm
      PiRLCRetainedInputs.productInputs
    simp [first, wireDescriptor_lane_eq]
    apply (Fin.heq_fun_iff (Phi81ProductPlan.groups_length _ _ _).symm).2
    intro group
    rfl
  · have wireNotFirst : (wireDescriptor descriptor).source.val ≠ 0 := by
      simpa using first
    rw [wireDescriptor_invocation]
    simp only [wireNotFirst, if_false]
    rw [prior_form_at_invocation?_of_ne geometry descriptor first,
      output_form_at_invocation?]
    apply congrArg some
    unfold semanticInterface Phi81ProductFamilyPlan.laneInterface
      Phi81ProductFamilyPlan.groupOutputAt PiRLCProductPlan.interface
      PiRLCProductPlan.challengeState PiRLCRetainedInputs.productInputs
    simp [wireDescriptor_lane_eq]
    apply (Fin.heq_fun_iff (Phi81ProductPlan.groups_length _ _ _).symm).2
    intro group
    rfl

/-- One compact physical product row is the exact semantic row for the same
decoded descriptor and local row. -/
theorem block_row?
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth)
    (descriptor : PiRLCProductSchedule.Descriptor) (localRow : Fin 34) :
    (block geometry).row? logicalWidth
        (descriptor.invocation.val * 34 + localRow.val) =
      some (Phi81ProductFamilyPlan.rowForms
        (PiRLCProductPlan.interface
          (PiRLCRetainedInputs.productInputs geometry))
        descriptor.invocation localRow) := by
  let ordinal := descriptor.invocation.val * 34 + localRow.val
  have bound : ordinal < (block geometry).rowCount := by
    rw [block_rowCount]
    dsimp only [ordinal]
    have invocationBound : descriptor.invocation.val < 52326 := by
      simpa using descriptor.invocation.isLt
    omega
  have quotient : ordinal / 34 = descriptor.invocation.val := by
    dsimp only [ordinal]
    omega
  have remainder : ordinal % 34 = localRow.val := by
    dsimp only [ordinal]
    omega
  have selected : MatrixProgram.Phi81Product.descriptor? families
      (ordinal / 34) = some (wireDescriptor descriptor) := by
    rw [quotient]
    exact descriptor?_wireDescriptor descriptor
  let semanticRow := Phi81ProductFamilyPlan.rowAt
    (PiRLCProductPlan.interface
      (PiRLCRetainedInputs.productInputs geometry))
    descriptor.invocation localRow
  have rowBound : localRow.val <
      (ProductSumPlan.rows (semanticInterface geometry descriptor)).length := by
    change localRow.val <
      (ProductSumPlan.rows (Phi81ProductFamilyPlan.laneInterface
        (PiRLCProductPlan.interface
          (PiRLCRetainedInputs.productInputs geometry))
        descriptor.invocation)).length
    rw [Phi81ProductFamilyPlan.laneRows_length]
    exact localRow.isLt
  have rowSelected :
      (ProductSumPlan.rows
        (semanticInterface geometry descriptor))[ordinal % 34]? =
          some semanticRow := by
    rw [remainder, List.getElem?_eq_getElem rowBound]
    apply congrArg some
    rfl
  have loaded := MatrixProgram.Phi81Product.Block.row?_of_loaded
    (block geometry) logicalWidth ordinal bound (wireDescriptor descriptor)
      selected (semanticInterface geometry descriptor)
      (block_interface? geometry descriptor) semanticRow rowSelected
  simpa only [ordinal, semanticRow, Phi81ProductFamilyPlan.rowForms] using loaded

/-- The singleton matrix program returns the exact semantic row for every
authoritative invocation and local row. -/
theorem matrixProgram_invocation_row?
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth)
    (sourceRow : Nat → Option R1CS.Row)
    (invocation : Fin PiRLCProductSchedule.invocationCount)
    (localRow : Fin 34) :
    (matrixProgram geometry).row? logicalWidth sourceRow
        (invocation.val * 34 + localRow.val) =
      some (Phi81ProductFamilyPlan.rowForms
        (PiRLCProductPlan.interface
          (PiRLCRetainedInputs.productInputs geometry)) invocation localRow) := by
  let descriptor := PiRLCProductSchedule.descriptor invocation
  have exactRow := block_row? geometry descriptor localRow
  rw [PiRLCProductSchedule.invocation_descriptor] at exactRow
  have bound : invocation.val * 34 + localRow.val <
      (MatrixProgram.Block.phi81Product (block geometry)).rowCount := by
    change invocation.val * 34 + localRow.val < (block geometry).rowCount
    rw [block_rowCount]
    have invocationBound : invocation.val < 52326 := by
      simpa using invocation.isLt
    omega
  rw [show matrixProgram geometry = MatrixProgram.Program.mk
      [.phi81Product (block geometry)] by rfl]
  rw [MatrixProgram.Program.singleton_row?, if_pos bound]
  exact exactRow

/-- Every physical row in the complete compact product block is the exact
row selected by the canonical Lean invocation-major family plan. -/
theorem matrixProgram_row?
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth)
    (sourceRow : Nat → Option R1CS.Row)
    (global : Fin (PiRLCProductSchedule.invocationCount * 34)) :
    (matrixProgram geometry).row? logicalWidth sourceRow global.val =
      let decoded : Fin PiRLCProductSchedule.invocationCount × Fin 34 :=
        Fin.decodeProd global
      some (Phi81ProductFamilyPlan.rowForms
        (PiRLCProductPlan.interface
          (PiRLCRetainedInputs.productInputs geometry))
        decoded.1 decoded.2) := by
  let decoded : Fin PiRLCProductSchedule.invocationCount × Fin 34 :=
    Fin.decodeProd global
  have exactRow := matrixProgram_invocation_row? geometry sourceRow
    decoded.1 decoded.2
  have encoded : decoded.1.val * 34 + decoded.2.val = global.val := by
    have inverse := congrArg Fin.val (Fin.encodeProd_decodeProd global)
    simpa [decoded, Fin.encodeProd, Nat.mul_comm] using inverse
  rw [encoded] at exactRow
  exact exactRow

end NightstreamFPrime.Export.Stage1.PiRLCProductMatrixProgram
