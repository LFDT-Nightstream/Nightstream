import NightstreamFPrime.Export.Stage1.Wide.ProductInputMap
import NightstreamFPrime.Export.Stage1.PiRLCProductMatrixProgramSemantics

/-! Compact product operands for the candidate: checked challenge bits,
relocated input blocks, and the original output/quotient coefficient order. -/

namespace NightstreamFPrime.Export.Stage1.Wide.ProductMatrix

open NightstreamFPrime.Spec NightstreamFPrime.Layout
open ProductionRelation MatrixProgram
open Spec.Folding.PiCCS.PaperJoint
open PiRlcWideSampler

abbrev Program := RetainedLayout.Program
abbrev Descriptor := PiRLCProductRingSchedule.Descriptor

def interface (program : Program) := Stage1Plan.piRlcInterface program

def left (program : Program) (index : Fin (17 * 54)) : SparseForm (RetainedLayout.logicalWidth program) :=
  let pair : Fin 17 × Fin 54 := Fin.decodeProd index
  Challenges.form (PiRLCGeometry.sampler (interface program)) pair.1 pair.2

def fields : LowNormBlock.Block 52326 where
  kind := .field
  slotCount := 52326
  source := id

private theorem outputFits (program : Program) :
    PiRLCGeometry.fieldStart (interface program) + fields.coordinateCount ≤ RetainedLayout.logicalWidth program := by
  have fits := PiRLCGeometry.fieldFits (interface program)
  change _ + 4290732 ≤ _ at fits
  change _ + 2145366 ≤ _
  omega

private theorem quotientFits (program : Program) :
    PiRLCGeometry.fieldStart (interface program) + 2145366 + fields.coordinateCount ≤ RetainedLayout.logicalWidth program := by
  have fits := PiRLCGeometry.fieldFits (interface program)
  change _ + 4290732 ≤ _ at fits
  change _ + 2145366 + 2145366 ≤ _
  omega

def block (program : Program) : Phi81Product.Block where
  families := PiRLCProductMatrixProgram.families
  oneColumn := (interface program).oneColumn.val
  challenge := .direct (Array.ofFn fun index => WireForm.ofSemantic (left program index)) 54
  input := ProductInputMap.substitution program
  output := RetainedBlock.ofSemantic fields (PiRLCGeometry.fieldStart (interface program))
  group := RetainedBlock.ofSemantic fields (PiRLCGeometry.fieldStart (interface program) + 2145366)

def matrixProgram (program : Program) : MatrixProgram.Program := ⟨[.phi81Product (block program)]⟩

private theorem output_form (program : Program) (index : Fin 52326) :
    (block program).output.form? (RetainedLayout.logicalWidth program) index.val =
      some (PiRLCGeometry.fieldBlock.form (PiRLCGeometry.fieldStart (interface program))
        (PiRLCGeometry.fieldFits (interface program)) ⟨index.val, by have bound := index.isLt; change _ < 104652; omega⟩) := by
  have loaded := RetainedBlock.form?_ofSemantic fields _ (outputFits program) index
  refine loaded.trans (congrArg some ?_)
  exact LowNormBlock.Block.form_eq_of_coordinates fields PiRLCGeometry.fieldBlock _ _ _ _ _ _ rfl rfl

private theorem quotient_form (program : Program) (index : Fin 52326) :
    (block program).group.form? (RetainedLayout.logicalWidth program) index.val =
      some (PiRLCGeometry.fieldBlock.form (PiRLCGeometry.fieldStart (interface program))
        (PiRLCGeometry.fieldFits (interface program)) ⟨52326 + index.val, by have bound := index.isLt; change _ < 104652; omega⟩) := by
  have loaded := RetainedBlock.form?_ofSemantic fields _ (quotientFits program) index
  refine loaded.trans (congrArg some ?_)
  apply LowNormBlock.Block.form_eq_of_coordinates fields PiRLCGeometry.fieldBlock _ _ _ _ _ _ rfl
  change _ + 2145366 + index.val * 41 = _ + (52326 + index.val) * 41
  omega

private theorem wire_lane (descriptor : Descriptor) (lane : Fin ringDegree) :
    (PiRLCProductMatrixProgram.wireRingDescriptor descriptor).invocationAtLane lane =
      (descriptor.withLane lane).invocation.val := by
  rw [PiRLCProductMatrixProgram.wireRingDescriptor, PiRLCProductMatrixProgram.wireDescriptor_invocationAtLane]
  cases descriptor
  rfl

private theorem wire_source (descriptor : Descriptor) :
    (PiRLCProductMatrixProgram.wireRingDescriptor descriptor).source.val = descriptor.source.val := by
  rw [PiRLCProductMatrixProgram.wireRingDescriptor, PiRLCProductMatrixProgram.wireDescriptor_source]
  rfl

private theorem challenge (program : Program) (descriptor : Descriptor) :
    (block program).challengeState? (interface program).oneColumn
        (PiRLCProductMatrixProgram.wireRingDescriptor descriptor) =
      some (Challenges.form (PiRLCGeometry.sampler (interface program)) descriptor.source) := by
  apply Phi81Product.loadFin?_of_some
  intro lane
  change (block program).challenge.form? (interface program).oneColumn _ _ = _
  rw [wire_source]
  have loaded := Phi81Product.Challenge.direct_form (left program) (interface program).oneColumn
    54 descriptor.source.val lane.val (Fin.encodeProd (descriptor.source, lane)) (by
      simp [Fin.encodeProd, Nat.mul_comm])
  simpa only [left, Fin.decodeProd_encodeProd] using loaded

private theorem input (program : Program) (descriptor : Descriptor) :
    (block program).inputState? (RetainedLayout.logicalWidth program)
        (PiRLCProductMatrixProgram.wireRingDescriptor descriptor) =
      some ((interface program).value descriptor.invocation) := by
  apply Phi81Product.loadFin?_of_some
  intro lane
  rw [wire_lane]
  simpa only [PiRLCProductRingSchedule.laneInvocation, PiRLCProductRingSchedule.descriptor_invocation]
    using ProductInputMap.form program descriptor.invocation lane

private theorem quotient (program : Program) (descriptor : Descriptor) :
    (block program).quotientState? (RetainedLayout.logicalWidth program)
        (PiRLCProductMatrixProgram.wireRingDescriptor descriptor) =
      some (PiRLCGeometry.quotient (interface program) descriptor.invocation) := by
  apply Phi81Product.loadFin?_of_some
  intro lane
  rw [wire_lane]
  simpa only [PiRLCGeometry.quotient, PiRLCGeometry.quotientSlot,
    PiRLCGeometry.outputCount_eq, PiRLCProductRingSchedule.laneInvocation,
    PiRLCProductRingSchedule.descriptor_invocation] using quotient_form program (descriptor.withLane lane).invocation

private theorem output (program : Program) (descriptor : Descriptor) :
    (block program).outputState? (RetainedLayout.logicalWidth program)
        (PiRLCProductMatrixProgram.wireRingDescriptor descriptor) =
      some (PiRLCGeometry.output (interface program) descriptor.invocation) := by
  apply Phi81Product.loadFin?_of_some
  intro lane
  rw [wire_lane]
  simpa only [PiRLCGeometry.output, PiRLCGeometry.outputSlot,
    PiRLCProductRingSchedule.laneInvocation, PiRLCProductRingSchedule.descriptor_invocation]
    using output_form program (descriptor.withLane lane).invocation

private theorem prior (program : Program) (descriptor : Descriptor) :
    (block program).priorState? (RetainedLayout.logicalWidth program)
        (PiRLCProductMatrixProgram.wireRingDescriptor descriptor) =
      some (PiRLCGeometry.prior (interface program) descriptor.invocation) := by
  unfold Phi81Product.Block.priorState? PiRLCGeometry.prior
  rw [wire_source, PiRLCProductRingSchedule.descriptor_invocation]
  by_cases first : descriptor.source.val = 0
  · rw [if_pos first, dif_pos first]
  · rw [if_neg first, dif_neg first]
    apply Phi81Product.loadFin?_of_some
    intro lane
    have source : (descriptor.withLane lane).source.val ≠ 0 := first
    have family : (PiRLCProductMatrixProgram.wireRingDescriptor descriptor).family.privateCount =
        (PiRLCProductMatrixProgram.wireDescriptor (descriptor.withLane lane)).family.privateCount := by
      simp only [PiRLCProductMatrixProgram.wireRingDescriptor, PiRLCProductMatrixProgram.wireDescriptor_privateCount]
      cases descriptor
      rfl
    rw [wire_lane, family, ← PiRLCProductMatrixProgram.wireDescriptor_invocation (descriptor.withLane lane),
      PiRLCProductMatrixProgram.previousInvocation_eq _ source]
    have same : (descriptor.withLane lane).previousSource source = (PiRLCGeometry.previous descriptor first).withLane lane := by
      cases descriptor
      rfl
    rw [same]
    simpa only [PiRLCGeometry.output, PiRLCGeometry.outputSlot,
      PiRLCProductRingSchedule.laneInvocation, PiRLCProductRingSchedule.descriptor_invocation]
      using output_form program ((PiRLCGeometry.previous descriptor first).withLane lane).invocation

/-- The loaded product interface uses the checked bits and exact moved inputs. -/
theorem interface_exact (program : Program) (descriptor : Descriptor) :
    (block program).interface? (RetainedLayout.logicalWidth program)
        (PiRLCProductMatrixProgram.wireRingDescriptor descriptor) =
      some (Phi81ProductFamilyPlan.ringInterface (PiRLCPlan.products (PiRLCGeometry.planInterface (interface program)))
        descriptor.invocation) := by
  unfold Phi81Product.Block.interface?
  have one : (block program).oneColumn? (RetainedLayout.logicalWidth program) = some (interface program).oneColumn := by
    simp only [Phi81Product.Block.oneColumn?, block, dif_pos (interface program).oneColumn.isLt]
  rw [one]
  simp only [bind, Option.bind]
  rw [challenge, input, quotient, prior, output]
  apply congrArg some
  simp only [Phi81ProductFamilyPlan.ringInterface, PiRLCPlan.products, PiRLCGeometry.planInterface,
    PiRLCProductRingSchedule.descriptor_invocation]
  rfl

theorem rowCount (program : Program) : (block program).rowCount = 104652 := by
  change Phi81Product.ringCount PiRLCProductMatrixProgram.families * 108 = _
  rw [PiRLCProductMatrixProgram.families_ringCount]

private theorem block_row (program : Program) (descriptor : Descriptor) (localRow : Fin 108) :
    (block program).row? (RetainedLayout.logicalWidth program) (descriptor.invocation.val * 108 + localRow.val) =
      some (Phi81ProductFamilyPlan.rowForms
        (PiRLCPlan.products (PiRLCGeometry.planInterface (interface program))) descriptor.invocation localRow) := by
  let ordinal := descriptor.invocation.val * 108 + localRow.val
  have bounded : ordinal < (block program).rowCount := by
    rw [rowCount]
    have bound : descriptor.invocation.val < 969 := descriptor.invocation.isLt
    dsimp only [ordinal]
    omega
  have quotient : ordinal / 108 = descriptor.invocation.val := by dsimp only [ordinal]; omega
  have remainder : ordinal % 108 = localRow.val := by dsimp only [ordinal]; omega
  have selected : Phi81Product.ringDescriptor? (block program).families (ordinal / 108) =
      some (PiRLCProductMatrixProgram.wireRingDescriptor descriptor) := by
    rw [quotient]
    exact PiRLCProductMatrixProgram.ringDescriptor?_wireRingDescriptor descriptor
  let semantic := Phi81ProductFamilyPlan.ringInterface
    (PiRLCPlan.products (PiRLCGeometry.planInterface (interface program))) descriptor.invocation
  have rowSelected : (Phi81ProductPlan.rows semantic)[ordinal % 108]? =
      some (Phi81ProductPlan.rowAt semantic localRow) := by
    rw [remainder]
    simp only [Phi81ProductPlan.rows, List.getElem?_ofFn, localRow.isLt, dif_pos]
  exact Phi81Product.Block.row?_of_loaded (block program) _ ordinal bounded
    (PiRLCProductMatrixProgram.wireRingDescriptor descriptor) selected semantic
    (interface_exact program descriptor) (Phi81ProductPlan.rowAt semantic localRow) rowSelected

/-- Every compact product row is the exact candidate row, in the proved order. -/
theorem exact (program : Program) (sourceRow : Nat → Option R1CS.Row) :
    MatrixProgram.Exact (matrixProgram program)
      (PiRLCPlan.productPlan (PiRLCGeometry.planInterface (interface program))) sourceRow := by
  refine ⟨?_, ?_⟩
  · change (block program).rowCount + 0 = 104652
    rw [rowCount]
  · intro global
    let pair : Fin PiRLCProductRingSchedule.invocationCount × Fin 108 := Fin.decodeProd global
    have loaded := block_row program (PiRLCProductRingSchedule.descriptor pair.1) pair.2
    rw [PiRLCProductRingSchedule.invocation_descriptor] at loaded
    have position : pair.1.val * 108 + pair.2.val = global.val := by
      have same := congrArg Fin.val (Fin.encodeProd_decodeProd global)
      simpa only [Fin.encodeProd, Fin.coe_mkDivMod, Nat.mul_comm] using same
    rw [position] at loaded
    have bounded : global.val < (MatrixProgram.Block.phi81Product (block program)).rowCount := by
      change global.val < (block program).rowCount
      rw [rowCount]
      exact global.isLt
    rw [show matrixProgram program = ⟨[.phi81Product (block program)]⟩ by rfl,
      MatrixProgram.Program.singleton_row?, if_pos bounded]
    exact loaded

end NightstreamFPrime.Export.Stage1.Wide.ProductMatrix
