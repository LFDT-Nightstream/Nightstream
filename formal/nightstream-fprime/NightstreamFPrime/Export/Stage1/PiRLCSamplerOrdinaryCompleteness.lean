import NightstreamFPrime.Export.Stage1.PiRLCSamplerPoseidonValues
import NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryDirectPlanSemantics

/-!
Owns canonical ordinary sampler row completeness from actual cumulative
physical rows. Agreement is restricted to the existing ordinary source set:
Poseidon outputs, digest-lane logical/fresh cells, and final selector cells.
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

private theorem poseidonOutputColumn (descriptor : Lane) (lane : Fin 4) :
    (PiRLCSamplerPoseidonValues.physicalInvocation
      (PiRLCSamplerOrdinaryDirectPlan.Location.poseidonInvocation descriptor)).witnessStart +
        584 + (DigestWindow.rateLane lane).val =
      Spartan.sourceToSpartan
        (PiRLCSamplerOrdinaryDirectSource.poseidonSource descriptor.source.val descriptor.round.val lane) := by
  rw [PiRLCSamplerPoseidonValues.physicalInvocation_witnessStart_sampler]
  have decoded : PermutationPlan.samplerWitnessStartAt
      (PiRLCSamplerOrdinaryDirectPlan.Location.poseidonInvocation descriptor) =
      PermutationPlan.samplerSourceWitnessStartAt descriptor.source.val
        ⟨descriptor.round.val, Nat.lt_trans descriptor.round.isLt (by decide)⟩ := by
    exact congrArg
      (fun pair : Fin PiRLCSamplerInvocations.sourceCount ×
          Fin PermutationPlan.samplerStepsPerSource =>
        PermutationPlan.samplerSourceWitnessStartAt pair.1.val pair.2)
      (Fin.decodeProd_encodeProd
        (descriptor.source, (⟨descriptor.round.val,
          Nat.lt_trans descriptor.round.isLt (by decide)⟩ :
          Fin PermutationPlan.samplerStepsPerSource)))
  rw [decoded]
  dsimp only [PermutationPlan.samplerSourceWitnessStartAt]
  have mapped (start : Nat) (localStart : Spartan.piCcsPhaseOffset ≤ start) :
      Spartan.sourceToSpartan start + 584 + lane.val =
        Spartan.sourceToSpartan (start + 584 + lane.val) := by
    simpa only [Nat.add_assoc] using
      (Spartan.sourceToSpartan_add_of_piCcsLocal start (584 + lane.val) localStart).symm
  cases roundEq : descriptor.round.val with
  | zero =>
      have localStart : Spartan.piCcsPhaseOffset ≤
          PiRLCStarts.samplerSourceLogicalStart descriptor.source.val := by
        unfold PiRLCStarts.samplerSourceLogicalStart
        have initial : Spartan.piCcsPhaseOffset ≤ PiRLCStarts.samplerLogicalStart := by
          norm_num [Spartan.piCcsPhaseOffset, PiRLCStarts.samplerLogicalStart,
            PiRLCStarts.phaseLogicalStart, PiRLCInputs.phaseOffset, Formal.samplerOffset]
        omega
      simpa only [PermutationPlan.samplerSourceWitnessStartAt, roundEq,
        if_pos, PiRLCSamplerInvocations.sourceLogicalStart, DigestWindow.rateLane,
        PiRLCSamplerOrdinaryDirectSource.poseidonSource] using
          mapped (PiRLCStarts.samplerSourceLogicalStart descriptor.source.val) localStart
  | succ previous =>
      have localStart : Spartan.piCcsPhaseOffset ≤
          PiRLCStarts.digestPermutationLogicalStart descriptor.source.val previous := by
        unfold PiRLCStarts.digestPermutationLogicalStart PiRLCStarts.windowLogicalStart
          PiRLCStarts.samplerSourceLogicalStart
        have initial : Spartan.piCcsPhaseOffset ≤ PiRLCStarts.samplerLogicalStart := by
          norm_num [Spartan.piCcsPhaseOffset, PiRLCStarts.samplerLogicalStart,
            PiRLCStarts.phaseLogicalStart, PiRLCInputs.phaseOffset, Formal.samplerOffset]
        omega
      simpa only [PermutationPlan.samplerSourceWitnessStartAt, roundEq,
        Nat.succ_ne_zero, if_false, Nat.succ_sub_one, DigestWindow.rateLane,
        PiRLCSamplerOrdinaryDirectSource.poseidonSource,
        PiRLCStarts.digestPermutationLogicalStart, PiRLCStarts.windowLogicalStart,
        PiRLCStarts.samplerSourceLogicalStart, DigestWindow.permutationOffset,
        Sampler.windowOffset, Sampler.windowBase, SamplerChain.sourceOffset] using
          mapped (PiRLCStarts.digestPermutationLogicalStart descriptor.source.val previous) localStart

section Sources

variable {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
  (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry application logicalWidth)
  (assignment : Assignment F logicalWidth)
  (base : Fin (PiRLCProductPlan.baseSourceWidth application) → F)
  (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
  (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
  (retained : PiRLCRetainedPreservation.Encodes
    (PiRLCSamplerOrdinaryDirectPlan.piRlcGeometry geometry) assignment base groupValue products)
  (ordinary : PiRLCSamplerOrdinaryRetainedGeometry.Encodes geometry assignment
    (PiRLCRetainedPreservation.sourceAssignment application base groupValue products))
  (packets : PiRLCPackageCompleteness.RemappedPacketRowsHold
    (RunningTransitionDirectPlan.packageEnv application base))

include groupValue products retained ordinary packets

private theorem form_source (location : PiRLCSamplerOrdinaryDirectPlan.Location) :
    (location.form geometry).eval assignment =
      RunningTransitionDirectPlan.packageEnv application base
        (Spartan.sourceToSpartan location.sourceColumn) := by
  cases location with
  | poseidon descriptor lane =>
      have encoding := PiRLCSamplerPoseidonPreservation.encodingOfRetained
        (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry geometry) assignment
        (PiRLCRetainedPreservation.sourceAssignment application base groupValue products)
        _ retained.laterPoseidon
      have output := congrFun (PiRLCSamplerPoseidonValues.outputValue_of_packets
        (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry geometry) assignment
        base groupValue products encoding packets
        (PiRLCSamplerOrdinaryDirectPlan.Location.poseidonInvocation descriptor))
        (DigestWindow.rateLane lane)
      change ((PiRLCSamplerPoseidonPlan.interface
        (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry geometry)).output
        (PiRLCSamplerOrdinaryDirectPlan.Location.poseidonInvocation descriptor)
        (DigestWindow.rateLane lane)).eval assignment = _
      exact output.trans (congrArg (RunningTransitionDirectPlan.packageEnv application base)
        (poseidonOutputColumn descriptor lane))
  | logical descriptor position =>
      change (PiRLCSamplerCandidateWiring.logicalForm geometry descriptor position).eval assignment = _
      rw [PiRLCSamplerCandidateWiring.logicalForm_eval geometry assignment _ ordinary,
        logicalBlock_source]
      exact RunningTransitionDirectPlan.sourceAssignment_packageSource application base
        groupValue products _ (logicalSource_lt descriptor position)
  | fresh descriptor position =>
      change ((freshBlock application).form
        (PiRLCSamplerOrdinaryRetainedGeometry.freshStart application)
        (PiRLCSamplerOrdinaryRetainedGeometry.freshFits geometry)
        (freshSlot descriptor position)).eval assignment = _
      rw [LowNormBlock.Block.form_eval _ _ _ assignment _ ordinary.fresh,
        freshBlock_source]
      exact RunningTransitionDirectPlan.sourceAssignment_packageSource application base
        groupValue products _ (freshSource_lt descriptor position)
  | selector source =>
      change ((PiRLCFirst54RetainedBlocks.positionBlock application).form
        (PiRLCRetainedGeometry.positionStart application)
        (PiRLCRetainedGeometry.positionFits
          (PiRLCSamplerOrdinaryDirectPlan.piRlcGeometry geometry))
        (PiRLCFirst54DirectSchedule.positionIndex
          (PiRLCFirst54DirectPlan.finalPositionDescriptor source))).eval assignment = _
      rw [LowNormBlock.Block.form_eval _ _ _ assignment _ retained.position,
        PiRLCFirst54RetainedBlocks.positionBlock_source,
        PiRLCFirst54DirectSchedule.position_positionIndex]
      unfold PiRLCFirst54DirectPlan.retainedPositionColumn
      rw [PiRLCRetainedPreservation.sourceAssignment_package,
        PiRLCFirst54DirectPlan.finalPositionDescriptor_positionColumn]
      rfl

private theorem resolvedEnv_of_target (column : Nat)
    (support : PiRLCSamplerOrdinaryDirectSource.Target column) :
    PiRLCSamplerOrdinaryDirectPlan.resolvedEnv geometry assignment column =
      RunningTransitionDirectPlan.packageEnv application base column := by
  rcases support with ⟨source, supported, rfl⟩
  obtain ⟨location, found⟩ := Option.isSome_iff_exists.mp
    (PiRLCSamplerOrdinaryDirectPlan.classifySource_complete supported)
  have same := PiRLCSamplerOrdinaryDirectPlan.classifySource_sound found
  have bounded : source < Spartan.SourceColumnCount := by
    rw [← same]
    exact location.sourceColumn_lt
  change (PiRLCSamplerOrdinaryDirectPlan.resolvedForm geometry
    (Spartan.sourceToSpartan source)).eval assignment = _
  rw [PiRLCSamplerRetainedCustody.resolvedForm_of_source geometry bounded found, ← same]
  exact form_source geometry assignment base groupValue products retained ordinary packets location

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
    (List.range PiRLCSamplerOrdinaryRows.sourceCount).flatMap packets)
  funext source
  unfold PiRLCSamplerOrdinaryRows.sourceRows
  apply congrArg (fun rows : List Rows.CompiledRow =>
    rows ++ PiRLCSamplerOrdinaryRows.selectorFinalRows source)
  apply congrArg (fun packets : Nat → List Rows.CompiledRow =>
    (List.range PiRLCSamplerOrdinaryRows.digestRoundCount).flatMap packets)
  funext round
  unfold PiRLCSamplerOrdinaryRows.windowRows
  apply congrArg (fun packets : Fin 4 → List Rows.CompiledRow =>
    (List.finRange 4).flatMap packets)
  funext lane
  unfold PiRLCSamplerOrdinaryRows.laneRows
  apply congrArg (PiCCSArithmetic.compilePacket _ _)
  unfold PiRLCSamplerOrdinaryRows.laneConstraints
  simp only [PiRLCSamplerOrdinaryDirectSource.fastLaneSource_eq_var]

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
  exact resolvedEnv_of_target geometry raw.assignment base raw.groupValue raw.products
    (PerApplicationCanonicalEncodes.retainedEncodes raw)
    (PerApplicationCanonicalEncodes.samplerPrefixEncodes raw).samplerOrdinary
    packets column support

end NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryCompleteness
