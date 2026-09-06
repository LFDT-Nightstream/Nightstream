import NightstreamFPrime.Export.Stage1.PerApplicationFixedPoint
import NightstreamFPrime.Layout.ProductionRelation.CanonicalBlockAssignment

/-!
Owns the canonical low-norm assignment constructor for one Lean-authored
application. One raw value packet supplies the package source values and the
two derived PiRLC value families. The assignment uses the exact retained-block
order already owned by the direct Stage 1 geometry.

This module constructs values only. It does not assume row acceptance or
select a production application.
-/

namespace NightstreamFPrime.Export.Stage1.PerApplicationCanonicalAssignment

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

namespace Canonical

abbrev Schedule := CanonicalBlockAssignment.Schedule

def ofBlock {sourceWidth : Nat} (block : LowNormBlock.Block sourceWidth)
    (source : Fin sourceWidth → F) : CanonicalBlockAssignment.BlockValue :=
  CanonicalBlockAssignment.ofBlock block source

def coordinateCount (value : Schedule) : Nat :=
  CanonicalBlockAssignment.coordinateCount value

def assignment {logicalWidth : Nat}
    (publicInput : Fin ProductionAssignment.publicWidth → F)
    (value : Schedule) : Assignment F logicalWidth :=
  CanonicalBlockAssignment.assignment publicInput value

namespace BlockValue

def coordinateCount (entry : CanonicalBlockAssignment.BlockValue) : Nat :=
  entry.coordinateCount

end BlockValue

theorem assignment_encHashMarker {logicalWidth : Nat} (digest : Digest)
    (value : Schedule)
    (fits : ProductionAssignment.publicWidth ≤ logicalWidth) :
    assignment (logicalWidth := logicalWidth) (encodedHashCells digest) value
        (CanonicalBlockAssignment.publicColumn fits encHashMarkerIndex) = 1 :=
  CanonicalBlockAssignment.assignment_encHashMarker digest value fits

end Canonical

abbrev Program := Lifecycle.Stage1.Application.Program

/-- Raw prover values before canonical low-norm coordinate encoding. -/
structure RawValues (application : Program) where
  base : Fin (PiRLCProductPlan.baseSourceWidth application) → F
  groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F
  products : Fin PiRLCFirst54DirectSchedule.candidateCount → F

namespace RawValues

/-- The recursive public digest is read from the same pilot output columns
that the Stage 1 rows constrain. It is not caller-supplied authority. -/
def outputDigest {application : Program} (raw : RawValues application) : Digest :=
  List.ofFn fun lane : Fin PilotProduction.digestWords =>
    (PilotProduction.outputInterface.digest
      (Lifecycle.Pilot.outputOffset PilotProduction.interface
        PilotProduction.witnessOffset) lane).eval
      (PilotSpartan.pullback
        (PilotOrdinaryDirectPlan.pilotEnv application raw.base))

def retainedSource {application : Program} (raw : RawValues application) :
    Fin (PiRLCRetainedGeometry.sourceWidth application) → F :=
  PiRLCRetainedPreservation.sourceAssignment application raw.base
    raw.groupValue raw.products

def payloadSource {application : Program} (raw : RawValues application) :
    Fin (PiCCSActionPayloadBlock.sourceWidth application) → F :=
  PiCCSPoseidonPreservation.sourceAssignment application raw.retainedSource

def applicationSource {application : Program} (raw : RawValues application) :
    Fin (ApplicationRetainedBlocks.sourceWidth application) → F :=
  DirectApplicationPrefixPlan.applicationSource application raw.base

/-- Exact retained block order from the public prefix through the selected
application. The list is small; no slot or coordinate list is materialized. -/
def schedule {application : Program} (raw : RawValues application) :
    Canonical.Schedule :=
  [ Canonical.ofBlock (PiRLCRetainedGeometry.priorPoseidonBlock application)
      raw.retainedSource
  , Canonical.ofBlock (PiRLCRetainedGeometry.outputPoseidonBlock application)
      raw.retainedSource
  , Canonical.ofBlock (PiRLCRetainedGeometry.laterPoseidonBlock application)
      raw.retainedSource
  , Canonical.ofBlock (PiRLCRetainedGeometry.productGroupBlock application)
      raw.retainedSource
  , Canonical.ofBlock (PiRLCFirst54RetainedBlocks.rejectBlock application)
      raw.retainedSource
  , Canonical.ofBlock (PiRLCFirst54RetainedBlocks.symbolBlock application)
      raw.retainedSource
  , Canonical.ofBlock (PiRLCFirst54RetainedBlocks.positionBlock application)
      raw.retainedSource
  , Canonical.ofBlock (PiRLCFirst54RetainedBlocks.valueBlock application)
      raw.retainedSource
  , Canonical.ofBlock (PiRLCFirst54RetainedBlocks.productBlock application)
      raw.retainedSource
  , Canonical.ofBlock (PiRLCRetainedGeometry.productInputBlock application)
      raw.retainedSource
  , Canonical.ofBlock (PiRLCRetainedGeometry.productOutputBlock application)
      raw.retainedSource
  , Canonical.ofBlock (PiRLCPoseidonGeometry.priorInputBlock application)
      raw.retainedSource
  , Canonical.ofBlock (PiRLCPoseidonGeometry.outputInputBlock application)
      raw.retainedSource
  , Canonical.ofBlock (PiCCSActionPayloadBlock.block application)
      raw.payloadSource
  , Canonical.ofBlock (RunningTransitionRetainedBlocks.roundC0Block application)
      raw.retainedSource
  , Canonical.ofBlock (RunningTransitionRetainedBlocks.roundC1Block application)
      raw.retainedSource
  , Canonical.ofBlock (RunningTransitionRetainedBlocks.piDecBlock application)
      raw.retainedSource
  , Canonical.ofBlock (RunningTransitionRetainedBlocks.freshBlock application)
      raw.retainedSource
  , Canonical.ofBlock
      (PiCCSOrdinaryRetainedBlocks.freshPublicInputBlock application)
      raw.retainedSource
  , Canonical.ofBlock (PiCCSOrdinaryRetainedBlocks.priorLastBlock application)
      raw.retainedSource
  , Canonical.ofBlock (PiCCSOrdinaryRetainedBlocks.outputLastBlock application)
      raw.retainedSource
  , Canonical.ofBlock
      (PiCCSOrdinaryRetainedBlocks.expectedContextBlock application)
      raw.retainedSource
  , Canonical.ofBlock
      (PiCCSOrdinaryRetainedBlocks.proofLogicalBlock application)
      raw.retainedSource
  , Canonical.ofBlock
      (PiCCSOrdinaryRetainedBlocks.outputEndpointBlock application)
      raw.retainedSource
  , Canonical.ofBlock (PiCCSOrdinaryRetainedBlocks.freshBlock application)
      raw.retainedSource
  , Canonical.ofBlock
      (PilotOrdinaryRetainedBlocks.canonicalLocalBlock application)
      raw.retainedSource
  , Canonical.ofBlock
      (PilotOrdinaryRetainedBlocks.canonicalFreshBlock application)
      raw.retainedSource
  , Canonical.ofBlock (PilotOrdinaryRetainedBlocks.outputDigestBlock application)
      raw.retainedSource
  , Canonical.ofBlock (PiDECRetainedBlocks.parentCommitmentBlock application)
      raw.retainedSource
  , Canonical.ofBlock (PiDECRetainedBlocks.parentPublicInputBlock application)
      raw.retainedSource
  , Canonical.ofBlock (PiDECRetainedBlocks.parentEvalKBlock application)
      raw.retainedSource
  , Canonical.ofBlock (PiDECRetainedBlocks.parentEvalABlock application)
      raw.retainedSource
  , Canonical.ofBlock (PiDECRetainedBlocks.logicalBlock application)
      raw.retainedSource
  , Canonical.ofBlock (PiDECRetainedBlocks.freshBlock application)
      raw.retainedSource
  , Canonical.ofBlock
      (PiRLCSamplerOrdinaryRetainedBlocks.logicalBlock application)
      raw.retainedSource
  , Canonical.ofBlock
      (PiRLCSamplerOrdinaryRetainedBlocks.freshBlock application)
      raw.retainedSource
  , Canonical.ofBlock (ApplicationRetainedBlocks.witnessBlock application)
      raw.applicationSource
  , Canonical.ofBlock (ApplicationRetainedBlocks.localBlock application)
      raw.applicationSource ]

end RawValues

@[simp] theorem schedule_length {application : Program}
    (raw : RawValues application) : raw.schedule.length = 38 := by
  rfl

/-- The block schedule has exactly the final logical width after its 270-word
public prefix. -/
theorem schedule_width {application : Program} (raw : RawValues application) :
    ProductionAssignment.publicWidth +
        Canonical.coordinateCount raw.schedule =
      PerApplicationFixedPoint.logicalWidth application := by
  simp only [RawValues.schedule, Canonical.coordinateCount, Canonical.ofBlock,
    CanonicalBlockAssignment.coordinateCount,
    CanonicalBlockAssignment.BlockValue.coordinateCount,
    CanonicalBlockAssignment.ofBlock]
  unfold PerApplicationFixedPoint.logicalWidth
    ApplicationRetainedGeometry.completeLogicalWidth
    ApplicationRetainedGeometry.localStart
    ApplicationRetainedGeometry.witnessStart
    PiRLCSamplerOrdinaryRetainedGeometry.completeLogicalWidth
    PiRLCSamplerOrdinaryRetainedGeometry.freshStart
    PiRLCSamplerOrdinaryRetainedGeometry.logicalStart
    PiRLCSamplerOrdinaryRetainedGeometry.prefixLogicalWidth
    PiDECRetainedGeometry.completeLogicalWidth PiDECRetainedGeometry.freshStart
    PiDECRetainedGeometry.logicalStart
    PiDECRetainedGeometry.parentEvalAStart
    PiDECRetainedGeometry.parentEvalKStart
    PiDECRetainedGeometry.parentPublicInputStart
    PiDECRetainedGeometry.parentCommitmentStart
    PiDECRetainedGeometry.prefixLogicalWidth
    PilotOrdinaryRetainedGeometry.completeLogicalWidth
    PilotOrdinaryRetainedGeometry.outputDigestStart
    PilotOrdinaryRetainedGeometry.canonicalFreshStart
    PilotOrdinaryRetainedGeometry.canonicalLocalStart
    PilotOrdinaryRetainedGeometry.prefixLogicalWidth
    PiCCSOrdinaryRetainedGeometry.completeLogicalWidth
    PiCCSOrdinaryRetainedGeometry.freshStart
    PiCCSOrdinaryRetainedGeometry.outputEndpointStart
    PiCCSOrdinaryRetainedGeometry.proofLogicalStart
    PiCCSOrdinaryRetainedGeometry.expectedContextStart
    PiCCSOrdinaryRetainedGeometry.outputLastStart
    PiCCSOrdinaryRetainedGeometry.priorLastStart
    PiCCSOrdinaryRetainedGeometry.freshPublicInputStart
    PiCCSOrdinaryRetainedGeometry.prefixLogicalWidth
    RunningTransitionRetainedGeometry.completeLogicalWidth
    RunningTransitionRetainedGeometry.freshStart
    RunningTransitionRetainedGeometry.piDecStart
    RunningTransitionRetainedGeometry.roundC1Start
    RunningTransitionRetainedGeometry.roundC0Start
    PiCCSActionPayloadBlock.logicalWidth PiCCSActionPayloadBlock.payloadStart
    PiRLCPoseidonGeometry.pilotLogicalWidth
    PiRLCPoseidonGeometry.outputInputStart
    PiRLCPoseidonGeometry.priorInputStart
    PiRLCRetainedGeometry.prefixLogicalWidth
    PiRLCRetainedGeometry.productOutputStart
    PiRLCRetainedGeometry.productInputStart
    PiRLCRetainedGeometry.first54ProductStart
    PiRLCRetainedGeometry.valueStart PiRLCRetainedGeometry.positionStart
    PiRLCRetainedGeometry.symbolStart PiRLCRetainedGeometry.rejectStart
    PiRLCRetainedGeometry.productGroupStart
    PiRLCRetainedGeometry.laterPoseidonStart
    PiRLCRetainedGeometry.outputPoseidonStart
    PiRLCRetainedGeometry.priorPoseidonStart
  omega

namespace RawValues

def assignment {application : Program} (raw : RawValues application) :
    Assignment F (PerApplicationFixedPoint.logicalWidth application) :=
  Canonical.assignment (encodedHashCells raw.outputDigest) raw.schedule

end RawValues

theorem scheduleFits {application : Program} (raw : RawValues application) :
    ProductionAssignment.publicWidth +
        Canonical.coordinateCount raw.schedule ≤
      PerApplicationFixedPoint.logicalWidth application := by
  rw [schedule_width raw]

theorem publicFits {application : Program} :
    ProductionAssignment.publicWidth ≤
      PerApplicationFixedPoint.logicalWidth application := by
  unfold PerApplicationFixedPoint.logicalWidth
  rw [ApplicationRetainedGeometry.completeLogicalWidth_eq,
    ProductionAssignment.publicWidth_eq]
  omega

namespace RawValues

/-- Canonical zero-padding from the logical assignment to the complete Phi81
carrier used by the key-facing SuperNeo relation. -/
def completeAssignment {application : Program} (raw : RawValues application) :
    Lifecycle.PaperAlgebra.Assignment
      (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application) :=
  fun column =>
    if logicalRegion :
        column.val < PerApplicationFixedPoint.logicalWidth application then
      raw.assignment ⟨column.val, logicalRegion⟩
    else
      0

end RawValues

/-- The final paper-carrier assignment exposes exactly the encoding of the
digest read from the constrained step output. -/
theorem projectPublicInput_completeAssignment {application : Program}
    (raw : RawValues application) :
    Spec.Phi81Relation.projectPublicInput raw.completeAssignment =
      encHash (publicFits := PerApplicationFixedPoint.publicFits application)
        raw.outputDigest := by
  funext column
  unfold Spec.Phi81Relation.projectPublicInput RawValues.completeAssignment
    Spec.Phi81Relation.Shape.publicColumn
  have logicalRegion :
      column.val < PerApplicationFixedPoint.logicalWidth application := by
    have bound := column.isLt
    change column.val < ProductionAssignment.publicWidth at bound
    exact Nat.lt_of_lt_of_le bound (publicFits (application := application))
  rw [dif_pos logicalRegion]
  change raw.assignment
      (CanonicalBlockAssignment.publicColumn
        (publicFits (application := application)) column) =
    encodedHashCells raw.outputDigest column
  exact CanonicalBlockAssignment.assignment_publicColumn
    (encodedHashCells raw.outputDigest) raw.schedule
    (publicFits (application := application)) column

/-- The relation's distinguished one-coordinate is fixed by the verifier-owned
encoded-hash public input. -/
theorem assignment_one {application : Program} (raw : RawValues application) :
    raw.assignment
        (ApplicationRetainedGeometry.oneColumn
          (PerApplicationFixedPoint.geometry application)) = 1 := by
  have marker := Canonical.assignment_encHashMarker raw.outputDigest raw.schedule
    (publicFits (application := application))
  exact marker

end NightstreamFPrime.Export.Stage1.PerApplicationCanonicalAssignment
