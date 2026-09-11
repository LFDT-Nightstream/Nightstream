import NightstreamFPrime.Export.Stage1.PerApplicationSourceAssignment
import NightstreamFPrime.Export.Stage1.PiRLCCombinationCompleteness
import NightstreamFPrime.Export.Stage1.PiRLCFirst54Completeness
import NightstreamFPrime.Export.Stage1.PiRLCSamplerCompleteness
import NightstreamFPrime.Export.Stage1.PiRLCRetainedPlan

/-!
Owns the adjacent conversion from completed Spartan PiRLC rows to the product
and First54 plans on the canonical retained assignment. The physical source
copy supplies the packet proofs; the assignment transport supplies the honest
intermediate values. No caller supplies product equations or selector bits.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiRLCRetainedCompleteness

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open PiRLCCombinationInvocations
open PiRLCCombinationConformance
open PerApplicationAssignmentTransportExecution

private theorem familyRows_at
    (logicalStart rowStart freshStart blockCount cellCount valueStride : Nat)
    [NeZero cellCount] (valueSourceStart : Nat → Nat → Nat → Nat)
    (env : Env)
    (rows : FamilyInvocationRowsHold logicalStart rowStart freshStart blockCount
      cellCount valueStride valueSourceStart env)
    (source : Fin sourceCount) (block : Fin blockCount)
    (lane : Fin ringDegree) (cell : Fin cellCount) :
    R1CS.RowsHold env
      (CompactRows.instantiateRows
        (CompactRows.inputColumnOfRanges
          (invocation logicalStart rowStart freshStart blockCount cellCount
            valueStride source.val block.val lane.val cell.val valueSourceStart).inputRanges)
        (invocation logicalStart rowStart freshStart blockCount cellCount
          valueStride source.val block.val lane.val cell.val valueSourceStart).localStart
        (PiRLCCombinationTemplates.template (firstSource source.val) lane)) := by
  have coordinates := PiRLCCombinationProjection.coordinates_eq_of_val
    (CombinationStep.indexOf block lane cell) block lane cell
    (indexOf_val block lane cell)
  simpa only [coordinates] using rows source (CombinationStep.indexOf block lane cell)

private theorem descriptor_sourceConstraint
    (env : Env)
    (rows : PiRLCCombinationCompleteness.ProductionFamilyInvocationRowsHold env)
    (descriptor : PiRLCProductSchedule.Descriptor) :
    descriptor.sourceConstraint.eval (Spartan.pullback env) = 0 := by
  rcases descriptor with ⟨family, source, block, lane, cell⟩
  cases family with
  | commitment =>
      exact commitmentInvocationRows_imply_sourceConstraint source.val block.val
        cell.val lane source.isLt block.isLt env
        (familyRows_at _ _ _ _ _ _ _ env rows.commitment source block lane cell)
  | publicInput =>
      exact publicInputInvocationRows_imply_sourceConstraint source.val block.val
        cell.val lane source.isLt block.isLt env
        (familyRows_at _ _ _ _ _ _ _ env rows.publicInput source block lane cell)
  | evalK =>
      exact evalKInvocationRows_imply_sourceConstraint source.val block.val
        cell.val lane source.isLt cell.isLt env
        (familyRows_at _ _ _ _ _ _ _ env rows.evalK source block lane cell)
  | evalA =>
      exact evalAInvocationRows_imply_sourceConstraint source.val block.val
        cell.val lane source.isLt block.isLt cell.isLt env
        (familyRows_at _ _ _ _ _ _ _ env rows.evalA source block lane cell)

private theorem semantics_of_packets
    (application : Lifecycle.Stage1.Application.Program)
    (base : BaseValues application)
    (packets : PiRLCPackageCompleteness.RemappedPacketRowsHold
      (RunningTransitionDirectPlan.packageEnv application base)) :
    PiRLCRetainedPlan.Semantics application base := by
  let env := RunningTransitionDirectPlan.packageEnv application base
  have families :=
    PiRLCCombinationCompleteness.remappedPackets_imply_familyInvocationRows env packets
  have selectors :=
    PiRLCFirst54Completeness.remappedPacket_implies_first54Invocations env packets
  have ordinary := PiRLCSamplerCompleteness.remappedPacket_implies_ordinaryRows env packets
  have templates : (Data.circuitPackage ()).compactRowTemplates =
      PiRLCFirst54Invocations.packageTemplates :=
    Data.circuitPackage_compactRowTemplates
  refine ⟨?_, ?_⟩
  · intro index
    exact descriptor_sourceConstraint env families (PiRLCProductSchedule.descriptor index)
  · intro source
    have specification := PiRLCFirst54Conformance.packageInvocations_imply_spec
      (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)
      (Data.circuitPackage ()) templates source.val source.isLt env selectors ordinary
    exact PiRLCFirst54DirectBridge.specHolds_implies_sourceHolds application base
      (PiRLCFirst54Conformance.sourceInterface
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits) source.val)
      source.val source specification

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

private theorem copied_packets
    (application : Lifecycle.Stage1.Application.Program)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (target : Env)
    (applicationPrivate : Fin (PerApplicationPackage.addedPrivateColumnCount application) → F)
    (physical : R1CS.RowsHold target (Spartan.remappedRows relation)) :
    PiRLCPackageCompleteness.RemappedPacketRowsHold
      (RunningTransitionDirectPlan.packageEnv application
        (PerApplicationSourceAssignment.ofCompleted application target applicationPrivate)) := by
  let base := PerApplicationSourceAssignment.ofCompleted application target applicationPrivate
  let env := RunningTransitionDirectPlan.packageEnv application base
  have allRows := (Spartan.remappedRows_hold relation target).mp physical
  have throughD := ((PilotPiCCSPiRLCPiDECRunningTransition.physicalHolds_iff
    relation (Spartan.pullback target)).mp allRows).1
  have throughR := ((PilotPiCCSPiRLCPiDEC.physicalHolds_iff relation
    (Spartan.pullback target)).mp throughD).1
  have rRows := ((PilotPiCCSPiRLC.physicalHolds_iff relation
    (Spartan.pullback target)).mp throughR).2
  have assumptions := PiRLCInputBounds.assumptions relation (Spartan.pullback target)
  have phase := NightstreamFPrime.Layout.PiRLC.v1_1.physical_implies_phaseHolds
    relation ajtai PiRLCInputs.interface PiRLCInputs.phaseOffset
    (Spartan.pullback target) assumptions rRows
  have scope := NightstreamFPrime.Layout.PiRLC.v1_1.physicalRows_varsBelow_of_phase
    relation ajtai PiRLCInputs.interface PiRLCInputs.phaseOffset
    (Spartan.pullback target) assumptions phase
  have endpoint : NightstreamFPrime.Layout.PiRLC.v1_1.physicalColumnCount relation
      PiRLCInputs.interface PiRLCInputs.phaseOffset ≤ Spartan.SourceColumnCount := by
    calc
      _ ≤ PilotPiCCSPiRLC.physicalColumnCount relation := Nat.le_max_right _ _
      _ = PiDECInputs.proofInputStart := (PiDECInputs.proofInputStart_matches_piRlc relation).symm
      _ ≤ PiDECInputs.phaseOffset := Nat.le_add_right _ _
      _ ≤ Spartan.SourceColumnCount := Spartan.sourceColumnCount_ge_piDecPhaseOffset
  have copiedRows : R1CS.RowsHold (Spartan.pullback env)
      (NightstreamFPrime.Layout.PiRLC.v1_1.physicalRows relation
        PiRLCInputs.interface PiRLCInputs.phaseOffset) := by
    apply R1CS.rowsHold_of_agree_below _ _ (Spartan.pullback target)
      (Spartan.pullback env) scope _ rRows
    intro index bound
    exact PerApplicationSourceAssignment.source_ofCompleted application target
      applicationPrivate index (Nat.lt_of_lt_of_le bound endpoint)
  exact PiRLCPackageCompleteness.remappedPhysicalRows_imply_packets relation env
    ((Spartan.remapRows_hold env _).mpr copiedRows)

/-- The completed physical prefix makes the exact canonical product and
First54 plan vanish. The actual PiRLC phase supplies its own scope and packet
constraints. The canonical assignment transport supplies the honest Phi81
and First54 intermediate values; no source equation, selector-bit condition,
or equality between caller environments is a premise. -/
theorem rowsZero_of_completed
    (application : Lifecycle.Stage1.Application.Program)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (target : Env)
    (applicationPrivate : Fin (PerApplicationPackage.addedPrivateColumnCount application) → F)
    (physical : R1CS.RowsHold target (Spartan.remappedRows relation)) :
    (PiRLCRetainedPlan.plan
      (PiRLCValueWiring.form (PerApplicationCanonicalEncodes.piCcsOrdinaryGeometry application))
      (PerApplicationCanonicalEncodes.retainedGeometry application)).RowsZero
        (canonicalRawValues application
          (PerApplicationSourceAssignment.ofCompleted application target applicationPrivate)).assignment := by
  let base := PerApplicationSourceAssignment.ofCompleted application target applicationPrivate
  let raw := canonicalRawValues application base
  have packets := copied_packets application relation ajtai target applicationPrivate physical
  have semantics := semantics_of_packets application base packets
  have groups : raw.groupValue = PiRLCProductPlan.honestGroupValue
      (PiRLCRetainedInputs.productInputs
        (PiRLCValueWiring.form (PerApplicationCanonicalEncodes.piCcsOrdinaryGeometry application))
        (PerApplicationCanonicalEncodes.retainedGeometry application)) raw.assignment := by
    funext invocation group
    exact canonicalRawValues_groupValue_eq_honestGroupValue application base invocation group
  have products : raw.products = PiRLCFirst54DirectPlan.honestProducts application base := by
    funext candidate
    exact canonicalRawValues_products_eq_honestProducts application base candidate
  have encodes := PerApplicationCanonicalEncodes.retainedEncodes raw
  rw [groups, products] at encodes
  exact PiRLCRetainedPlan.semantics_implies_rowsZero _ _ raw.assignment base
    (PerApplicationCanonicalAssignment.assignment_one raw)
    (PerApplicationCanonicalEncodes.productValuesPreserve raw) encodes semantics

end NightstreamFPrime.Export.Stage1.PiRLCRetainedCompleteness
