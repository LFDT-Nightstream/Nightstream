import NightstreamFPrime.Export.Stage1.PiDECValueWiring
import NightstreamFPrime.Export.Stage1.PiRLCProductMatrixProgram
import NightstreamFPrime.Export.Stage1.PerApplicationFixedPoint
import Mathlib.Algebra.BigOperators.Group.Finset.Basic

/-!
Owns the value connection from arbitrary accepted product rows to PiCCS
operands and PiDEC parents. Every source is read from the final assignment.
No honest-encoding or caller-supplied representation premise is used.

Sampler transcript validity and the full augmented step are separate claims.
-/

namespace NightstreamFPrime.Export.Stage1.ActualPiRLCValues

open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open PiRLCProductSchedule

variable {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}

def inputs (geometry : PiDECRetainedGeometry.Geometry program logicalWidth) :=
  PiRLCProductMatrixProgram.inputs (PiDECRetainedGeometry.prefixGeometry geometry)

def outputValue (geometry : PiDECRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) (descriptor : Descriptor) : F :=
  ((inputs geometry).output descriptor.invocation).eval assignment

def challenge (geometry : PiDECRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) (descriptor : Descriptor) : RingF :=
  fun lane => ((inputs geometry).challenge descriptor.invocation lane).eval assignment - 2

def piCcsValue (geometry : PiDECRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) (descriptor : Descriptor) : RingF :=
  fun lane => PiCCSAssignmentSoundness.decodedEnv
    (PiDECRetainedGeometry.prefixGeometry geometry) assignment
    (Spartan.sourceToSpartan (descriptor.valueColumn lane))

def contribution (geometry : PiDECRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) (descriptor : Descriptor) : F :=
  ringFMul (challenge geometry assignment descriptor)
    (piCcsValue geometry assignment descriptor) descriptor.lane

def withSource (descriptor : Descriptor)
    (source : Fin PiRLCCombinationInvocations.sourceCount) : Descriptor :=
  { descriptor with source }

@[simp] private theorem withSource_self (descriptor : Descriptor) :
    withSource descriptor descriptor.source = descriptor := by
  cases descriptor
  rfl

def contributionAt (geometry : PiDECRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) (descriptor : Descriptor) (source : Nat) : F :=
  if bound : source < PiRLCCombinationInvocations.sourceCount then
    contribution geometry assignment (withSource descriptor ⟨source, bound⟩)
  else 0

private theorem valueState_eval
    (geometry : PiDECRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) (descriptor : Descriptor) :
    Phi81ProductPlan.evalState assignment
        (PiRLCProductPlan.valueState (inputs geometry) descriptor.invocation) =
      piCcsValue geometry assignment descriptor := by
  funext lane
  dsimp only [Phi81ProductPlan.evalState, PiRLCProductPlan.valueState,
    PiRLCProductPlan.valueForm, inputs, PiRLCProductMatrixProgram.inputs,
    PiRLCRetainedInputs.productInputs]
  rw [descriptor_invocation, PiRLCValueWiring.form_eval_eq_decodedEnv]
  simp only [descriptor_invocation, Descriptor.withLane_valueColumn, piCcsValue]

private theorem challengeState_eval
    (geometry : PiDECRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) (descriptor : Descriptor)
    (one : assignment (inputs geometry).oneColumn = 1) :
    Phi81ProductPlan.evalState assignment
        (PiRLCProductPlan.challengeState (inputs geometry) descriptor.invocation) =
      challenge geometry assignment descriptor := by
  funext lane
  simp [Phi81ProductPlan.evalState, PiRLCProductPlan.challengeState,
    PiRLCProductPlan.challengeForm, challenge, one, sub_eq_add_neg]

private theorem priorForm_eval
    (geometry : PiDECRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) (descriptor : Descriptor) :
    (PiRLCProductPlan.priorForm (inputs geometry) descriptor.invocation).eval assignment =
      if first : descriptor.source.val = 0 then 0
      else outputValue geometry assignment (descriptor.previousSource first) := by
  unfold PiRLCProductPlan.priorForm
  rw [descriptor_invocation]
  split
  · simp_all
  · rename_i notFirst
    simp only [inputs, PiRLCProductMatrixProgram.inputs, PiRLCRetainedInputs.productInputs,
      descriptor_invocation, notFirst, outputValue]
    rfl

/-- All product rows force the recurrence on the actual shared input values. -/
theorem rowsZero_implies_equation
    (geometry : PiDECRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (inputs geometry).oneColumn = 1)
    (rows : (PiRLCProductPlan.plan (inputs geometry)).RowsZero assignment)
    (descriptor : Descriptor) :
    outputValue geometry assignment descriptor =
      (if first : descriptor.source.val = 0 then 0
       else outputValue geometry assignment (descriptor.previousSource first)) +
        contribution geometry assignment descriptor := by
  have equation := Phi81ProductFamilyPlan.planRowsZero_implies_ringProduct
    (PiRLCProductPlan.interface (inputs geometry)) PiRLCProductPlan.rowCount_le
    assignment one rows descriptor.invocation
  change outputValue geometry assignment descriptor =
    (PiRLCProductPlan.priorForm (inputs geometry) descriptor.invocation).eval assignment +
      ringFMul
        (Phi81ProductPlan.evalState assignment
          (PiRLCProductPlan.challengeState (inputs geometry) descriptor.invocation))
        (Phi81ProductPlan.evalState assignment
          (PiRLCProductPlan.valueState (inputs geometry) descriptor.invocation))
        (PiRLCProductSchedule.descriptor descriptor.invocation).lane at equation
  rw [priorForm_eval, challengeState_eval geometry assignment descriptor one,
    valueState_eval, descriptor_invocation] at equation
  exact equation

private theorem prefix_sum
    (geometry : PiDECRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (inputs geometry).oneColumn = 1)
    (rows : (PiRLCProductPlan.plan (inputs geometry)).RowsZero assignment)
    (descriptor : Descriptor) (source : Nat)
    (bound : source < PiRLCCombinationInvocations.sourceCount) :
    outputValue geometry assignment (withSource descriptor ⟨source, bound⟩) =
      (Finset.range (source + 1)).sum (contributionAt geometry assignment descriptor) := by
  induction source with
  | zero =>
      have equation := rowsZero_implies_equation geometry assignment one rows
        (withSource descriptor ⟨0, bound⟩)
      have first : (withSource descriptor ⟨0, bound⟩).source.val = 0 := rfl
      rw [dif_pos first] at equation
      simpa only [contributionAt, dif_pos bound,
        Nat.zero_add, Finset.range_one, Finset.sum_singleton, zero_add] using equation
  | succ source inductionHypothesis =>
      have smaller : source < PiRLCCombinationInvocations.sourceCount := by omega
      have equation := rowsZero_implies_equation geometry assignment one rows
        (withSource descriptor ⟨source + 1, bound⟩)
      have notFirst : (withSource descriptor ⟨source + 1, bound⟩).source.val ≠ 0 := by
        change source + 1 ≠ 0
        omega
      rw [dif_neg notFirst] at equation
      have previous : (withSource descriptor ⟨source + 1, bound⟩).previousSource notFirst =
          withSource descriptor ⟨source, smaller⟩ := by
        cases descriptor
        rfl
      rw [previous, inductionHypothesis smaller] at equation
      rw [Finset.sum_range_succ]
      simpa only [contributionAt, dif_pos bound] using equation

/-- The complete recurrence uniquely fixes each accumulated coefficient. -/
theorem rowsZero_implies_sum
    (geometry : PiDECRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (inputs geometry).oneColumn = 1)
    (rows : (PiRLCProductPlan.plan (inputs geometry)).RowsZero assignment)
    (descriptor : Descriptor) :
    outputValue geometry assignment descriptor =
      (Finset.range (descriptor.source.val + 1)).sum
        (contributionAt geometry assignment descriptor) := by
  simpa only [withSource_self] using prefix_sum geometry assignment one rows
    descriptor descriptor.source.val descriptor.source.isLt

/-- The PiDEC commitment parent is the complete sum of the shared PiCCS values. -/
theorem rowsZero_implies_parentCommitment_sum
    (geometry : PiDECRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (inputs geometry).oneColumn = 1)
    (rows : (PiRLCProductPlan.plan (inputs geometry)).RowsZero assignment)
    (index : Fin 1188) :
    ((PiDECDirectPlan.Location.parentCommitment index).form geometry).eval assignment =
      (Finset.range PiRLCCombinationInvocations.sourceCount).sum
        (contributionAt geometry assignment (PiDECValueWiring.finalDescriptor .commitment index)) := by
  rw [PiDECValueWiring.parentCommitment_form_eq_output]
  change outputValue geometry assignment (PiDECValueWiring.finalDescriptor .commitment index) = _
  exact rowsZero_implies_sum geometry assignment one rows
    (PiDECValueWiring.finalDescriptor .commitment index)

/-- The PiDEC publicInput parent is the complete sum of the shared PiCCS values. -/
theorem rowsZero_implies_parentPublicInput_sum
    (geometry : PiDECRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (inputs geometry).oneColumn = 1)
    (rows : (PiRLCProductPlan.plan (inputs geometry)).RowsZero assignment)
    (index : Fin 270) :
    ((PiDECDirectPlan.Location.parentPublicInput index).form geometry).eval assignment =
      (Finset.range PiRLCCombinationInvocations.sourceCount).sum
        (contributionAt geometry assignment (PiDECValueWiring.finalDescriptor .publicInput index)) := by
  rw [PiDECValueWiring.parentPublicInput_form_eq_output]
  change outputValue geometry assignment (PiDECValueWiring.finalDescriptor .publicInput index) = _
  exact rowsZero_implies_sum geometry assignment one rows
    (PiDECValueWiring.finalDescriptor .publicInput index)

/-- The PiDEC evalK parent is the complete sum of the shared PiCCS values. -/
theorem rowsZero_implies_parentEvalK_sum
    (geometry : PiDECRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (inputs geometry).oneColumn = 1)
    (rows : (PiRLCProductPlan.plan (inputs geometry)).RowsZero assignment)
    (index : Fin 108) :
    ((PiDECDirectPlan.Location.parentEvalK index).form geometry).eval assignment =
      (Finset.range PiRLCCombinationInvocations.sourceCount).sum
        (contributionAt geometry assignment (PiDECValueWiring.finalDescriptor .evalK index)) := by
  rw [PiDECValueWiring.parentEvalK_form_eq_output]
  change outputValue geometry assignment (PiDECValueWiring.finalDescriptor .evalK index) = _
  exact rowsZero_implies_sum geometry assignment one rows
    (PiDECValueWiring.finalDescriptor .evalK index)

/-- The PiDEC evalA parent is the complete sum of the shared PiCCS values. -/
theorem rowsZero_implies_parentEvalA_sum
    (geometry : PiDECRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (inputs geometry).oneColumn = 1)
    (rows : (PiRLCProductPlan.plan (inputs geometry)).RowsZero assignment)
    (index : Fin 1512) :
    ((PiDECDirectPlan.Location.parentEvalA index).form geometry).eval assignment =
      (Finset.range PiRLCCombinationInvocations.sourceCount).sum
        (contributionAt geometry assignment (PiDECValueWiring.finalDescriptor .evalA index)) := by
  rw [PiDECValueWiring.parentEvalA_form_eq_output]
  change outputValue geometry assignment (PiDECValueWiring.finalDescriptor .evalA index) = _
  exact rowsZero_implies_sum geometry assignment one rows
    (PiDECValueWiring.finalDescriptor .evalA index)

/-- The selected complete rows and actual public marker supply every premise
of the shared-value sum theorem. In particular, no product-row or one-cell
premise is supplied by a witness constructor. The four parent form identities
select the final source of this same recurrence. -/
theorem selectedRowsAndPublic_imply_sums
    (application : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (assignment : Assignment F (PerApplicationFixedPoint.logicalWidth application))
    (digest : Digest)
    (publicEqual : Phi81Relation.projectPublicInput
      (shape := FullShape (PerApplicationFixedPoint.logicalWidth application)
        (PerApplicationFixedPoint.publicFits application))
      (Phi81CarrierLayout.extendAssignment 0 assignment) =
        encHash (publicFits := PerApplicationFixedPoint.publicFits application) digest)
    (accepted : (PerApplicationFixedPoint.structuralPlan application fits).RowsZero
      assignment) :
    let geometry := DirectApplicationPrefixPlan.piDecGeometry
      (PerApplicationFixedPoint.geometry application)
    ∀ descriptor, outputValue geometry assignment descriptor =
      (Finset.range (descriptor.source.val + 1)).sum
        (contributionAt geometry assignment descriptor) := by
  let geometry := PerApplicationFixedPoint.geometry application
  let relation := PerApplicationFixedPoint.relation application fits
  have publicBound : RecursivePublicOutputPlan.publicInput geometry assignment =
      encHash (publicFits := RecursivePublicOutputPlan.carrierPublicFits geometry) digest := by
    rw [RecursivePublicOutputPlan.publicInput_eq_projectPublicInput]
    exact publicEqual
  have one := RecursivePublicOutputPlan.publicEqual_implies_one
    geometry assignment digest publicBound
  have selected : (DirectApplicationPrefixPlan.plan relation fits.package geometry
      ).RowsZero assignment := by
    rw [PerApplicationFixedPoint.plan_fixedPoint]
    exact accepted
  have applicationRows := (DirectApplicationPrefixPlan.rowsZero_iff relation
    fits.package geometry assignment).mp selected
  have prefixRows := (DirectPiRLCSamplerCompletePrefixPlan.rowsZero_iff relation
    (DirectApplicationPrefixPlan.prefixGeometry geometry) assignment).mp
      applicationRows.1.1.1
  have piRlcRows := prefixRows.2.2.1
  change (PiRLCRetainedPlan.plan _ _).RowsZero assignment at piRlcRows
  have productRows := (PiRLCRetainedPlan.rowsZero_iff _ _ assignment).mp piRlcRows
  dsimp only
  intro descriptor
  exact rowsZero_implies_sum (DirectApplicationPrefixPlan.piDecGeometry geometry)
    assignment one productRows.1 descriptor

end NightstreamFPrime.Export.Stage1.ActualPiRLCValues
