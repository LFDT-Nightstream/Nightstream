import NightstreamFPrime.Export.Stage1.PiRLCSamplerPoseidonValues
import NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryDirectPlanSemantics

/-!
Owns canonical ordinary sampler row completeness from actual cumulative
physical rows. Agreement is restricted to the existing ordinary source set:
Poseidon entry outputs, checked core/fresh values, and coefficient words.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryCompleteness

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open NightstreamFPrime.Gadgets.Sampling
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open PiRLCSamplerOrdinaryRetainedBlocks
open PerApplicationAssignmentTransportExecution

private theorem poseidonOutputColumn (source : Fin sourceCount) (lane : Fin 4) :
    (PiRLCSamplerPoseidonValues.physicalInvocation
      (PiRLCSamplerOrdinaryDirectPlan.Location.poseidonInvocation source)).witnessStart +
        584 + (Sampler.rateLane lane).val =
      Spartan.sourceToSpartan (PiRLCSamplerOrdinaryDirectSource.poseidonSource source.val lane) := by
  rw [PiRLCSamplerPoseidonValues.physicalInvocation_witnessStart_sampler]
  have decoded : PermutationPlan.samplerWitnessStartAt
      (PiRLCSamplerOrdinaryDirectPlan.Location.poseidonInvocation source) =
      PermutationPlan.samplerSourceWitnessStartAt source.val ⟨0, by decide⟩ := by
    exact congrArg (fun pair : Fin PiRLCSamplerInvocations.sourceCount ×
        Fin PermutationPlan.samplerStepsPerSource =>
      PermutationPlan.samplerSourceWitnessStartAt pair.1.val pair.2)
      (Fin.decodeProd_encodeProd (source, (⟨0, by decide⟩ : Fin PermutationPlan.samplerStepsPerSource)))
  rw [decoded]
  simp only [PermutationPlan.samplerSourceWitnessStartAt, if_true,
    PiRLCSamplerInvocations.sourceLogicalStart, Sampler.rateLane]
  have localStart : Spartan.piCcsPhaseOffset ≤ PiRLCStarts.samplerSourceLogicalStart source.val := by
    simp only [PiRLCStarts.samplerSourceLogicalStart, SamplerChain.sourceOffset,
      PiRLCStarts.samplerLogicalStart, Formal.samplerOffset, PiRLCStarts.phaseLogicalStart_eq]
    norm_num [Spartan.piCcsPhaseOffset]
    omega
  simpa only [PiRLCSamplerOrdinaryDirectSource.poseidonSource, Nat.add_assoc] using
    (Spartan.sourceToSpartan_add_of_piCcsLocal
      (PiRLCStarts.samplerSourceLogicalStart source.val) (584 + lane.val) localStart).symm

section Sources

variable {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
  (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry application logicalWidth)
  (assignment : Assignment F logicalWidth)
  (base : Fin (PiRLCProductPlan.baseSourceWidth application) → F)
  (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 1 → F)
  (retained : PiRLCRetainedPreservation.Encodes
    (PiRLCSamplerOrdinaryDirectPlan.piRlcGeometry geometry) assignment base groupValue)
  (ordinary : PiRLCSamplerOrdinaryRetainedGeometry.Encodes geometry assignment
    (PiRLCRetainedPreservation.sourceAssignment application base groupValue))
  (packets : PiRLCPackageCompleteness.RemappedPacketRowsHold
    (RunningTransitionDirectPlan.packageEnv application base))

include groupValue retained ordinary packets

private theorem form_source (location : PiRLCSamplerOrdinaryDirectPlan.Location) :
    (location.form geometry).eval assignment =
      RunningTransitionDirectPlan.packageEnv application base
        (Spartan.sourceToSpartan location.sourceColumn) := by
  cases location with
  | poseidon source lane =>
      have encoding := PiRLCSamplerPoseidonPreservation.encodingOfRetained
        (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry geometry) assignment
        (PiRLCRetainedPreservation.sourceAssignment application base groupValue)
        _ retained.laterPoseidon
      have output := congrFun (PiRLCSamplerPoseidonValues.outputValue_of_packets
        (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry geometry) assignment
        base groupValue encoding packets
        (PiRLCSamplerOrdinaryDirectPlan.Location.poseidonInvocation source))
        (Sampler.rateLane lane)
      change ((PiRLCSamplerPoseidonPlan.interface
        (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry geometry)).output
        (PiRLCSamplerOrdinaryDirectPlan.Location.poseidonInvocation source)
        (Sampler.rateLane lane)).eval assignment = _
      exact output.trans (congrArg (RunningTransitionDirectPlan.packageEnv application base)
        (poseidonOutputColumn source lane))
  | logical source position =>
      change ((logicalBlock application).form
        (PiRLCSamplerOrdinaryRetainedGeometry.logicalStart application)
        (PiRLCSamplerOrdinaryRetainedGeometry.logicalFits geometry)
        (logicalSlot source position)).eval assignment = _
      rw [LowNormBlock.Block.form_eval _ _ _ assignment _ ordinary.logical, logicalBlock_source]
      exact RunningTransitionDirectPlan.sourceAssignment_packageSource application base
        groupValue _ (logicalSource_lt source position)
  | fresh source position =>
      change ((freshBlock application).form
        (PiRLCSamplerOrdinaryRetainedGeometry.freshStart application)
        (PiRLCSamplerOrdinaryRetainedGeometry.freshFits geometry)
        (freshSlot source position)).eval assignment = _
      rw [LowNormBlock.Block.form_eval _ _ _ assignment _ ordinary.fresh,
        freshBlock_source]
      exact RunningTransitionDirectPlan.sourceAssignment_packageSource application base
        groupValue _ (freshSource_lt source position)
  | word source position =>
      change ((PiRLCRetainedGeometry.challengeBlock application).form
        (PiRLCRetainedGeometry.challengeStart application)
        (PiRLCRetainedGeometry.challengeFits (PiRLCSamplerOrdinaryDirectPlan.piRlcGeometry geometry))
        (PiRLCProductSourceBlocks.challengeIndex source position)).eval assignment = _
      rw [LowNormBlock.Block.form_eval _ _ _ assignment _ retained.challenge]
      simp only [PiRLCRetainedGeometry.challengeBlock, PiRLCProductSourceBlocks.challengeBlock,
        PiRLCProductSourceBlocks.challengeIndex, Fin.decodeProd_encodeProd]
      rw [PiRLCRetainedPreservation.sourceAssignment_package]
      rfl

private theorem resolvedEnv_of_target (column : Nat)
    (support : PiRLCSamplerOrdinaryDirectSource.Target column) :
    PiRLCSamplerOrdinaryDirectPlan.resolvedEnv geometry assignment column =
      RunningTransitionDirectPlan.packageEnv application base column := by
  rcases support with ⟨source, supported, rfl⟩
  have complete := PiRLCSamplerOrdinaryDirectPlan.classifySource_complete supported
  cases found : PiRLCSamplerOrdinaryDirectPlan.classifySource source with
  | none => exact False.elim (complete found)
  | some location =>
      have same := PiRLCSamplerOrdinaryDirectPlan.classifySource_sound found
      have bounded : source < Spartan.SourceColumnCount := by
        rw [← same]
        exact location.sourceColumn_lt
      change (PiRLCSamplerOrdinaryDirectPlan.resolvedForm geometry
        (Spartan.sourceToSpartan source)).eval assignment = _
      rw [PiRLCSamplerRetainedCustody.resolvedForm_of_source geometry bounded found, ← same]
      exact form_source geometry assignment base groupValue retained ordinary packets location

end Sources

variable {relationLogicalWidth : Nat}
  {relationPublicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth relationLogicalWidth}

private theorem sourceRows_eq_data :
    PiRLCSamplerOrdinaryDirectSource.sourceRows
      (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits) =
      PiRLCSamplerOrdinaryDirectSource.sourceRows
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits) := by
  unfold PiRLCSamplerOrdinaryDirectSource.sourceRows
  apply congrArg (List.map Rows.CompiledRow.toR1CS)
  unfold PiRLCSamplerOrdinaryRows.rows
  apply congrArg (fun packets : Nat → List Rows.CompiledRow =>
    (List.range PiRLCSamplerInvocations.sourceCount).flatMap packets)
  funext source
  have interfaceEq : PiRLCSamplerOrdinaryRows.rangeInterface
      (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits) source =
      PiRLCSamplerOrdinaryRows.rangeInterface
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits) source := by
    apply congrArg WideReduction.Interface.mk
    funext lane offset
    simp only [PiRLCSamplerInvocations.fastAdvanceState,
      PiRLCSamplerProjection.fastProductionEntryOutput_eq_scheduleOutput]
  simp only [PiRLCSamplerOrdinaryRows.sourceRows, PiRLCSamplerOrdinaryRows.rangeRows,
    PiRLCSamplerOrdinaryRows.rangeConstraints, interfaceEq]

/-- Actual cumulative physical rows make every ordinary sampler row vanish
on the canonical retained assignment. The exact source set supplies every
readback; all Poseidon output equalities are derived from its physical rows. -/
theorem rowsZero_of_completed
    (application : Lifecycle.Stage1.Application.Program)
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (ajtai : AjtaiKey
      (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits))
    (target : Env)
    (applicationPrivate : Fin (PerApplicationPackage.addedPrivateColumnCount application) → F)
    (physical : R1CS.RowsHold target (Spartan.remappedRows relation)) :
    (PiRLCSamplerOrdinaryDirectPlan.plan relation
      (PerApplicationCanonicalEncodes.samplerGeometry application)).RowsZero
      (canonicalRawValues application
        (PerApplicationSourceAssignment.ofCompleted application target applicationPrivate)).assignment := by
  let base := PerApplicationSourceAssignment.ofCompleted application target applicationPrivate
  let raw := canonicalRawValues application base
  let geometry := PerApplicationCanonicalEncodes.samplerGeometry application
  have packets := PiRLCRetainedCompleteness.packets_of_completed
    application relation ajtai target applicationPrivate physical
  have rows := PiRLCSamplerCompleteness.remappedPacket_implies_ordinaryRows _ packets
  have sourceRows : R1CS.RowsHold (RunningTransitionDirectPlan.packageEnv application base)
      (PiRLCSamplerOrdinaryDirectSource.sourceRows
        (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits)) := by
    rw [sourceRows_eq_data]
    exact rows
  apply (PiRLCSamplerOrdinaryDirectPlan.rowsZero_iff_rowsHold relation geometry
    raw.assignment (PerApplicationCanonicalAssignment.assignment_one raw)).mpr
  apply R1CS.rowsHold_of_agree _ PiRLCSamplerOrdinaryDirectSource.Target
    (RunningTransitionDirectPlan.packageEnv application base)
    (PiRLCSamplerOrdinaryDirectPlan.resolvedEnv geometry raw.assignment)
    PiRLCSamplerOrdinaryDirectSource.sourceRows_varsSatisfy _ sourceRows
  intro column support
  exact resolvedEnv_of_target geometry raw.assignment base raw.groupValue
    (PerApplicationCanonicalEncodes.retainedEncodes raw)
    (PerApplicationCanonicalEncodes.samplerPrefixEncodes raw).samplerOrdinary
    packets column support

end NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryCompleteness
