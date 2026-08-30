import NightstreamFPrime.Export.Stage1.PiRLCCombinationConformance
import NightstreamFPrime.Export.Stage1.PiRLCProductSemanticCustody

/-!
Owns the direct semantic bridge from the retained PiRLC product plan to the
four canonical combination-family relations. It does not use compact package
row acceptance and does not close PiRLC status.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCProductDirectSemantics

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

/-- Direct product constraints force the canonical commitment-combination
family in the complete retained sampler environment. -/
theorem productSemantics_imply_commitmentCanonical
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    {program : Lifecycle.Stage1.Application.Program} {assignmentWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      assignmentWidth) (assignment : Assignment F assignmentWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (product : ∀ invocation,
      (PiRLCProductSchedule.descriptor invocation).sourceConstraint.eval
        (PiRLCProductPlan.baseEnv program base) = 0) :
    CombinationFamily.CanonicalHolds
      (PiRLCCombinationInvocations.productionCommitmentFamilyInterface
        (logicalWidth := logicalWidth) (publicFits := publicFits))
      PiRLCStarts.commitmentLogicalStart
      (Spartan.pullback
        (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base)) := by
  apply CombinationFamily.relation_implies_canonical
  apply CombinationFamily.parentCoverage
  apply PiRLCCombinationConformance.familyConstraintZeros_imply_prefix
    (PiRLCCombinationInvocations.productionCommitmentFamilyInterface
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    PiRLCStarts.commitmentLogicalStart 1
      PiRLCCombinationInvocations.commitmentValueSourceStart
      (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base)
  · intro source index
    let coordinates := CombinationStep.coordinates index
    let descriptor : PiRLCProductSchedule.Descriptor :=
      { family := .commitment
        source := source
        block := coordinates.1
        lane := coordinates.2.1
        cell := coordinates.2.2 }
    have zero :=
      PiRLCProductSemanticCustody.sourceConstraint_zero_of_productSemantics
        geometry assignment base product descriptor
    simpa only [descriptor,
      PiRLCProductSchedule.Descriptor.sourceConstraint] using zero
  · intro source block lane cell
    exact PiRLCCombinationInvocations.commitmentSourceConstraint_eq_stepAssertion
      (logicalWidth := logicalWidth) (publicFits := publicFits)
      source block lane cell

/-- Direct product constraints force the canonical public-input combination
family in the complete retained sampler environment. -/
theorem productSemantics_imply_publicInputCanonical
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    {program : Lifecycle.Stage1.Application.Program} {assignmentWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      assignmentWidth) (assignment : Assignment F assignmentWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (product : ∀ invocation,
      (PiRLCProductSchedule.descriptor invocation).sourceConstraint.eval
        (PiRLCProductPlan.baseEnv program base) = 0) :
    CombinationFamily.CanonicalHolds
      (PiRLCCombinationInvocations.productionPublicInputFamilyInterface
        (logicalWidth := logicalWidth) (publicFits := publicFits))
      PiRLCStarts.publicInputLogicalStart
      (Spartan.pullback
        (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base)) := by
  apply CombinationFamily.relation_implies_canonical
  apply CombinationFamily.parentCoverage
  apply PiRLCCombinationConformance.familyConstraintZeros_imply_prefix
    (PiRLCCombinationInvocations.productionPublicInputFamilyInterface
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    PiRLCStarts.publicInputLogicalStart 1
      PiRLCCombinationInvocations.publicInputValueSourceStart
      (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base)
  · intro source index
    let coordinates := CombinationStep.coordinates index
    let descriptor : PiRLCProductSchedule.Descriptor :=
      { family := .publicInput
        source := source
        block := coordinates.1
        lane := coordinates.2.1
        cell := coordinates.2.2 }
    have zero :=
      PiRLCProductSemanticCustody.sourceConstraint_zero_of_productSemantics
        geometry assignment base product descriptor
    simpa only [descriptor,
      PiRLCProductSchedule.Descriptor.sourceConstraint] using zero
  · intro source block lane cell
    exact PiRLCCombinationInvocations.publicInputSourceConstraint_eq_stepAssertion
      (logicalWidth := logicalWidth) (publicFits := publicFits)
      source block lane cell

/-- Direct product constraints force the canonical Eval-K combination family
in the complete retained sampler environment. -/
theorem productSemantics_imply_evalKCanonical
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    {program : Lifecycle.Stage1.Application.Program} {assignmentWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      assignmentWidth) (assignment : Assignment F assignmentWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (product : ∀ invocation,
      (PiRLCProductSchedule.descriptor invocation).sourceConstraint.eval
        (PiRLCProductPlan.baseEnv program base) = 0) :
    CombinationFamily.CanonicalHolds
      (PiRLCCombinationInvocations.productionEvalKFamilyInterface
        (logicalWidth := logicalWidth) (publicFits := publicFits))
      PiRLCStarts.evalKLogicalStart
      (Spartan.pullback
        (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base)) := by
  apply CombinationFamily.relation_implies_canonical
  apply CombinationFamily.parentCoverage
  apply PiRLCCombinationConformance.familyConstraintZeros_imply_prefix
    (PiRLCCombinationInvocations.productionEvalKFamilyInterface
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    PiRLCStarts.evalKLogicalStart 2
      PiRLCCombinationInvocations.evalKValueSourceStart
      (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base)
  · intro source index
    let coordinates := CombinationStep.coordinates index
    let descriptor : PiRLCProductSchedule.Descriptor :=
      { family := .evalK
        source := source
        block := coordinates.1
        lane := coordinates.2.1
        cell := coordinates.2.2 }
    have zero :=
      PiRLCProductSemanticCustody.sourceConstraint_zero_of_productSemantics
        geometry assignment base product descriptor
    simpa only [descriptor,
      PiRLCProductSchedule.Descriptor.sourceConstraint] using zero
  · intro source block lane cell
    exact PiRLCCombinationInvocations.evalKSourceConstraint_eq_stepAssertion
      (logicalWidth := logicalWidth) (publicFits := publicFits)
      source block lane cell

/-- Direct product constraints force the canonical Eval-A combination family
in the complete retained sampler environment. -/
theorem productSemantics_imply_evalACanonical
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    {program : Lifecycle.Stage1.Application.Program} {assignmentWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      assignmentWidth) (assignment : Assignment F assignmentWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (product : ∀ invocation,
      (PiRLCProductSchedule.descriptor invocation).sourceConstraint.eval
        (PiRLCProductPlan.baseEnv program base) = 0) :
    CombinationFamily.CanonicalHolds
      (PiRLCCombinationInvocations.productionEvalAFamilyInterface
        (logicalWidth := logicalWidth) (publicFits := publicFits))
      PiRLCStarts.evalALogicalStart
      (Spartan.pullback
        (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base)) := by
  apply CombinationFamily.relation_implies_canonical
  apply CombinationFamily.parentCoverage
  apply PiRLCCombinationConformance.familyConstraintZeros_imply_prefix
    (PiRLCCombinationInvocations.productionEvalAFamilyInterface
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    PiRLCStarts.evalALogicalStart 2
      PiRLCCombinationInvocations.evalAValueSourceStart
      (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base)
  · intro source index
    let coordinates := CombinationStep.coordinates index
    let descriptor : PiRLCProductSchedule.Descriptor :=
      { family := .evalA
        source := source
        block := coordinates.1
        lane := coordinates.2.1
        cell := coordinates.2.2 }
    have zero :=
      PiRLCProductSemanticCustody.sourceConstraint_zero_of_productSemantics
        geometry assignment base product descriptor
    simpa only [descriptor,
      PiRLCProductSchedule.Descriptor.sourceConstraint] using zero
  · intro source block lane cell
    exact PiRLCCombinationInvocations.evalASourceConstraint_eq_stepAssertion
      (logicalWidth := logicalWidth) (publicFits := publicFits)
      source block lane cell

end NightstreamFPrime.Export.Stage1.PiRLCProductDirectSemantics
