import NightstreamFPrime.Export.Stage1.PiDECDirectPlan

/-!
Owns the shared forms between final PiRLC accumulators and PiDEC parents.
The four PiDEC parent views select existing PiRLC coordinates. They allocate
no slots and need no copy rows.
-/

namespace NightstreamFPrime.Export.Stage1.PiDECValueWiring

open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open PiRLCProductSchedule

def finalDescriptor (family : Family) (index : Fin family.privateCount) :
    Descriptor where
  family := family
  source := ⟨16, by decide⟩
  block := (CombinationStep.coordinates index).1
  lane := (CombinationStep.coordinates index).2.1
  cell := (CombinationStep.coordinates index).2.2

private theorem final_logicalIndex (family : Family)
    (index : Fin family.privateCount) :
    (finalDescriptor family index).logicalIndex = index.val := by
  dsimp only [Descriptor.logicalIndex, finalDescriptor]
  rw [← PiRLCCombinationInvocations.indexOf_val,
    PiRLCCombinationInvocations.indexOf_coordinates]

private def finalSlot : Family → Nat
  | .commitment => PiDECRetainedGeometry.parentCommitmentSlot
  | .publicInput => PiDECRetainedGeometry.parentPublicInputSlot
  | .evalK => PiDECRetainedGeometry.parentEvalKSlot
  | .evalA => PiDECRetainedGeometry.parentEvalASlot

private def parentSource : Family → Nat
  | .commitment => PiDECSourceSupport.parentCommitmentStart
  | .publicInput => PiDECSourceSupport.parentPublicInputStart
  | .evalK => PiDECSourceSupport.parentEvalKStart
  | .evalA => PiDECSourceSupport.parentEvalAStart

private theorem final_invocation (family : Family)
    (index : Fin family.privateCount) :
    (finalDescriptor family index).invocation.val = finalSlot family + index.val := by
  cases family <;>
    simp only [finalDescriptor, Descriptor.invocation, Descriptor.familyIndex,
      PiRLCCombinationInvocations.indexOf_coordinates,
      Fin.val_castAdd, Fin.val_natAdd, Fin.encodeProd, Fin.mkDivMod]
  all_goals
    norm_num only [Family.commitment_invocationCount, Family.publicInput_invocationCount,
      Family.evalK_invocationCount, Family.privateCount, Family.blockCount,
      Family.cellCount, CombinationStep.privateCount, ringDegree, finalSlot,
      PiDECRetainedGeometry.parentCommitmentSlot, PiDECRetainedGeometry.parentPublicInputSlot,
      PiDECRetainedGeometry.parentEvalKSlot, PiDECRetainedGeometry.parentEvalASlot]
  all_goals omega

private theorem final_outputColumn (family : Family)
    (index : Fin family.privateCount) :
    (finalDescriptor family index).outputColumn = parentSource family + index.val := by
  rw [Descriptor.outputColumn, final_logicalIndex]
  cases family <;> rfl

private def parentCommitmentSlot (index : Fin 1188) : Fin invocationCount :=
  ⟨PiDECRetainedGeometry.parentCommitmentSlot + index.val, by
    change 19008 + index.val < 52326
    have bound := index.isLt
    omega⟩

private theorem parentCommitment_source (program : Lifecycle.Stage1.Application.Program)
    (index : Fin 1188) :
    (PiDECRetainedBlocks.parentCommitmentBlock program).source index =
      (PiRLCRetainedGeometry.productOutputBlock program).source
        (parentCommitmentSlot index) := by
  have indexEq : (finalDescriptor .commitment index).invocation = parentCommitmentSlot index := by
    apply Fin.ext
    exact final_invocation .commitment index
  rw [← indexEq, PiRLCRetainedPreservation.productOutputBlock_source,
    descriptor_invocation]
  apply Fin.ext
  change PerApplicationPackage.shiftColumn program
      (Spartan.sourceToSpartan (parentSource .commitment + index.val)) =
    PerApplicationPackage.shiftColumn program
      (Spartan.sourceToSpartan (finalDescriptor .commitment index).outputColumn)
  rw [final_outputColumn]

/-- PiDEC reads the final PiRLC commitment value for every assignment. -/
theorem parentCommitment_form_eq_output
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiDECRetainedGeometry.Geometry program logicalWidth)
    (index : Fin 1188) :
    (PiDECDirectPlan.Location.parentCommitment index).form geometry =
      (PiRLCRetainedGeometry.productOutputBlock program).form
        (PiRLCRetainedGeometry.productOutputStart program)
        (PiRLCRetainedGeometry.productOutputFits
          (PiDECRetainedGeometry.piRlcGeometry geometry)) (finalDescriptor .commitment index).invocation := by
  apply LowNormBlock.Block.form_eq_of_coordinates
  · rfl
  · change PiRLCRetainedGeometry.productOutputStart program + 19008 * 41 +
      index.val * 41 = PiRLCRetainedGeometry.productOutputStart program +
        (finalDescriptor .commitment index).invocation.val * 41
    rw [final_invocation]
    change _ = PiRLCRetainedGeometry.productOutputStart program +
      (19008 + index.val) * 41
    ring

/-- The honest constructor reuses the PiRLC encoding at the shared columns. -/
theorem parentCommitmentEncodes
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiDECRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (source : Fin (PiRLCRetainedGeometry.sourceWidth program) → F)
    (output : (PiRLCRetainedGeometry.productOutputBlock program).EncodesAt
      (PiRLCRetainedGeometry.productOutputStart program)
      (PiRLCRetainedGeometry.productOutputFits
        (PiDECRetainedGeometry.piRlcGeometry geometry)) assignment source) :
    (PiDECRetainedBlocks.parentCommitmentBlock program).EncodesAt
      (PiDECRetainedGeometry.parentCommitmentStart program)
      (PiDECRetainedGeometry.parentCommitmentFits geometry) assignment source := by
  intro index coordinate
  have columns :
      (PiDECRetainedBlocks.parentCommitmentBlock program).column
        (PiDECRetainedGeometry.parentCommitmentStart program)
        (PiDECRetainedGeometry.parentCommitmentFits geometry) index coordinate =
      (PiRLCRetainedGeometry.productOutputBlock program).column
        (PiRLCRetainedGeometry.productOutputStart program)
        (PiRLCRetainedGeometry.productOutputFits
          (PiDECRetainedGeometry.piRlcGeometry geometry)) (parentCommitmentSlot index) coordinate := by
    apply Fin.ext
    change PiRLCRetainedGeometry.productOutputStart program + 19008 * 41 +
      (index.val * 41 + coordinate.val) =
      PiRLCRetainedGeometry.productOutputStart program +
        ((19008 + index.val) * 41 + coordinate.val)
    ring
  rw [columns, parentCommitment_source]
  exact output (parentCommitmentSlot index) coordinate

private def parentPublicInputSlot (index : Fin 270) : Fin invocationCount :=
  ⟨PiDECRetainedGeometry.parentPublicInputSlot + index.val, by
    change 24516 + index.val < 52326
    have bound := index.isLt
    omega⟩

private theorem parentPublicInput_source (program : Lifecycle.Stage1.Application.Program)
    (index : Fin 270) :
    (PiDECRetainedBlocks.parentPublicInputBlock program).source index =
      (PiRLCRetainedGeometry.productOutputBlock program).source
        (parentPublicInputSlot index) := by
  have indexEq : (finalDescriptor .publicInput index).invocation = parentPublicInputSlot index := by
    apply Fin.ext
    exact final_invocation .publicInput index
  rw [← indexEq, PiRLCRetainedPreservation.productOutputBlock_source,
    descriptor_invocation]
  apply Fin.ext
  change PerApplicationPackage.shiftColumn program
      (Spartan.sourceToSpartan (parentSource .publicInput + index.val)) =
    PerApplicationPackage.shiftColumn program
      (Spartan.sourceToSpartan (finalDescriptor .publicInput index).outputColumn)
  rw [final_outputColumn]

/-- PiDEC reads the final PiRLC publicInput value for every assignment. -/
theorem parentPublicInput_form_eq_output
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiDECRetainedGeometry.Geometry program logicalWidth)
    (index : Fin 270) :
    (PiDECDirectPlan.Location.parentPublicInput index).form geometry =
      (PiRLCRetainedGeometry.productOutputBlock program).form
        (PiRLCRetainedGeometry.productOutputStart program)
        (PiRLCRetainedGeometry.productOutputFits
          (PiDECRetainedGeometry.piRlcGeometry geometry)) (finalDescriptor .publicInput index).invocation := by
  apply LowNormBlock.Block.form_eq_of_coordinates
  · rfl
  · change PiRLCRetainedGeometry.productOutputStart program + 24516 * 41 +
      index.val * 41 = PiRLCRetainedGeometry.productOutputStart program +
        (finalDescriptor .publicInput index).invocation.val * 41
    rw [final_invocation]
    change _ = PiRLCRetainedGeometry.productOutputStart program +
      (24516 + index.val) * 41
    ring

/-- The honest constructor reuses the PiRLC encoding at the shared columns. -/
theorem parentPublicInputEncodes
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiDECRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (source : Fin (PiRLCRetainedGeometry.sourceWidth program) → F)
    (output : (PiRLCRetainedGeometry.productOutputBlock program).EncodesAt
      (PiRLCRetainedGeometry.productOutputStart program)
      (PiRLCRetainedGeometry.productOutputFits
        (PiDECRetainedGeometry.piRlcGeometry geometry)) assignment source) :
    (PiDECRetainedBlocks.parentPublicInputBlock program).EncodesAt
      (PiDECRetainedGeometry.parentPublicInputStart program)
      (PiDECRetainedGeometry.parentPublicInputFits geometry) assignment source := by
  intro index coordinate
  have columns :
      (PiDECRetainedBlocks.parentPublicInputBlock program).column
        (PiDECRetainedGeometry.parentPublicInputStart program)
        (PiDECRetainedGeometry.parentPublicInputFits geometry) index coordinate =
      (PiRLCRetainedGeometry.productOutputBlock program).column
        (PiRLCRetainedGeometry.productOutputStart program)
        (PiRLCRetainedGeometry.productOutputFits
          (PiDECRetainedGeometry.piRlcGeometry geometry)) (parentPublicInputSlot index) coordinate := by
    apply Fin.ext
    change PiRLCRetainedGeometry.productOutputStart program + 24516 * 41 +
      (index.val * 41 + coordinate.val) =
      PiRLCRetainedGeometry.productOutputStart program +
        ((24516 + index.val) * 41 + coordinate.val)
    ring
  rw [columns, parentPublicInput_source]
  exact output (parentPublicInputSlot index) coordinate

private def parentEvalKSlot (index : Fin 108) : Fin invocationCount :=
  ⟨PiDECRetainedGeometry.parentEvalKSlot + index.val, by
    change 26514 + index.val < 52326
    have bound := index.isLt
    omega⟩

private theorem parentEvalK_source (program : Lifecycle.Stage1.Application.Program)
    (index : Fin 108) :
    (PiDECRetainedBlocks.parentEvalKBlock program).source index =
      (PiRLCRetainedGeometry.productOutputBlock program).source
        (parentEvalKSlot index) := by
  have indexEq : (finalDescriptor .evalK index).invocation = parentEvalKSlot index := by
    apply Fin.ext
    exact final_invocation .evalK index
  rw [← indexEq, PiRLCRetainedPreservation.productOutputBlock_source,
    descriptor_invocation]
  apply Fin.ext
  change PerApplicationPackage.shiftColumn program
      (Spartan.sourceToSpartan (parentSource .evalK + index.val)) =
    PerApplicationPackage.shiftColumn program
      (Spartan.sourceToSpartan (finalDescriptor .evalK index).outputColumn)
  rw [final_outputColumn]

/-- PiDEC reads the final PiRLC evalK value for every assignment. -/
theorem parentEvalK_form_eq_output
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiDECRetainedGeometry.Geometry program logicalWidth)
    (index : Fin 108) :
    (PiDECDirectPlan.Location.parentEvalK index).form geometry =
      (PiRLCRetainedGeometry.productOutputBlock program).form
        (PiRLCRetainedGeometry.productOutputStart program)
        (PiRLCRetainedGeometry.productOutputFits
          (PiDECRetainedGeometry.piRlcGeometry geometry)) (finalDescriptor .evalK index).invocation := by
  apply LowNormBlock.Block.form_eq_of_coordinates
  · rfl
  · change PiRLCRetainedGeometry.productOutputStart program + 26514 * 41 +
      index.val * 41 = PiRLCRetainedGeometry.productOutputStart program +
        (finalDescriptor .evalK index).invocation.val * 41
    rw [final_invocation]
    change _ = PiRLCRetainedGeometry.productOutputStart program +
      (26514 + index.val) * 41
    ring

/-- The honest constructor reuses the PiRLC encoding at the shared columns. -/
theorem parentEvalKEncodes
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiDECRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (source : Fin (PiRLCRetainedGeometry.sourceWidth program) → F)
    (output : (PiRLCRetainedGeometry.productOutputBlock program).EncodesAt
      (PiRLCRetainedGeometry.productOutputStart program)
      (PiRLCRetainedGeometry.productOutputFits
        (PiDECRetainedGeometry.piRlcGeometry geometry)) assignment source) :
    (PiDECRetainedBlocks.parentEvalKBlock program).EncodesAt
      (PiDECRetainedGeometry.parentEvalKStart program)
      (PiDECRetainedGeometry.parentEvalKFits geometry) assignment source := by
  intro index coordinate
  have columns :
      (PiDECRetainedBlocks.parentEvalKBlock program).column
        (PiDECRetainedGeometry.parentEvalKStart program)
        (PiDECRetainedGeometry.parentEvalKFits geometry) index coordinate =
      (PiRLCRetainedGeometry.productOutputBlock program).column
        (PiRLCRetainedGeometry.productOutputStart program)
        (PiRLCRetainedGeometry.productOutputFits
          (PiDECRetainedGeometry.piRlcGeometry geometry)) (parentEvalKSlot index) coordinate := by
    apply Fin.ext
    change PiRLCRetainedGeometry.productOutputStart program + 26514 * 41 +
      (index.val * 41 + coordinate.val) =
      PiRLCRetainedGeometry.productOutputStart program +
        ((26514 + index.val) * 41 + coordinate.val)
    ring
  rw [columns, parentEvalK_source]
  exact output (parentEvalKSlot index) coordinate

private def parentEvalASlot (index : Fin 1512) : Fin invocationCount :=
  ⟨PiDECRetainedGeometry.parentEvalASlot + index.val, by
    change 50814 + index.val < 52326
    have bound := index.isLt
    omega⟩

private theorem parentEvalA_source (program : Lifecycle.Stage1.Application.Program)
    (index : Fin 1512) :
    (PiDECRetainedBlocks.parentEvalABlock program).source index =
      (PiRLCRetainedGeometry.productOutputBlock program).source
        (parentEvalASlot index) := by
  have indexEq : (finalDescriptor .evalA index).invocation = parentEvalASlot index := by
    apply Fin.ext
    exact final_invocation .evalA index
  rw [← indexEq, PiRLCRetainedPreservation.productOutputBlock_source,
    descriptor_invocation]
  apply Fin.ext
  change PerApplicationPackage.shiftColumn program
      (Spartan.sourceToSpartan (parentSource .evalA + index.val)) =
    PerApplicationPackage.shiftColumn program
      (Spartan.sourceToSpartan (finalDescriptor .evalA index).outputColumn)
  rw [final_outputColumn]

/-- PiDEC reads the final PiRLC evalA value for every assignment. -/
theorem parentEvalA_form_eq_output
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiDECRetainedGeometry.Geometry program logicalWidth)
    (index : Fin 1512) :
    (PiDECDirectPlan.Location.parentEvalA index).form geometry =
      (PiRLCRetainedGeometry.productOutputBlock program).form
        (PiRLCRetainedGeometry.productOutputStart program)
        (PiRLCRetainedGeometry.productOutputFits
          (PiDECRetainedGeometry.piRlcGeometry geometry)) (finalDescriptor .evalA index).invocation := by
  apply LowNormBlock.Block.form_eq_of_coordinates
  · rfl
  · change PiRLCRetainedGeometry.productOutputStart program + 50814 * 41 +
      index.val * 41 = PiRLCRetainedGeometry.productOutputStart program +
        (finalDescriptor .evalA index).invocation.val * 41
    rw [final_invocation]
    change _ = PiRLCRetainedGeometry.productOutputStart program +
      (50814 + index.val) * 41
    ring

/-- The honest constructor reuses the PiRLC encoding at the shared columns. -/
theorem parentEvalAEncodes
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiDECRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (source : Fin (PiRLCRetainedGeometry.sourceWidth program) → F)
    (output : (PiRLCRetainedGeometry.productOutputBlock program).EncodesAt
      (PiRLCRetainedGeometry.productOutputStart program)
      (PiRLCRetainedGeometry.productOutputFits
        (PiDECRetainedGeometry.piRlcGeometry geometry)) assignment source) :
    (PiDECRetainedBlocks.parentEvalABlock program).EncodesAt
      (PiDECRetainedGeometry.parentEvalAStart program)
      (PiDECRetainedGeometry.parentEvalAFits geometry) assignment source := by
  intro index coordinate
  have columns :
      (PiDECRetainedBlocks.parentEvalABlock program).column
        (PiDECRetainedGeometry.parentEvalAStart program)
        (PiDECRetainedGeometry.parentEvalAFits geometry) index coordinate =
      (PiRLCRetainedGeometry.productOutputBlock program).column
        (PiRLCRetainedGeometry.productOutputStart program)
        (PiRLCRetainedGeometry.productOutputFits
          (PiDECRetainedGeometry.piRlcGeometry geometry)) (parentEvalASlot index) coordinate := by
    apply Fin.ext
    change PiRLCRetainedGeometry.productOutputStart program + 50814 * 41 +
      (index.val * 41 + coordinate.val) =
      PiRLCRetainedGeometry.productOutputStart program +
        ((50814 + index.val) * 41 + coordinate.val)
    ring
  rw [columns, parentEvalA_source]
  exact output (parentEvalASlot index) coordinate

end NightstreamFPrime.Export.Stage1.PiDECValueWiring
