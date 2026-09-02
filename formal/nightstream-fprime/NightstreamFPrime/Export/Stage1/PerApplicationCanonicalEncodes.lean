import NightstreamFPrime.Export.Stage1.PerApplicationCanonicalAssignment

/-!
Owns the proof that the compact per-application assignment canonically encodes
every retained Stage 1 block. A structural cursor consumes one block at a time.
No proof normalizes the complete 45-block schedule or an expanded slot list.
-/

namespace NightstreamFPrime.Export.Stage1.PerApplicationCanonicalEncodes

open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open PerApplicationCanonicalAssignment

abbrev Program := Lifecycle.Stage1.Application.Program
abbrev RawValues := PerApplicationCanonicalAssignment.RawValues

namespace Cursor

structure State {application : Program} (raw : RawValues application)
    (start : Nat) where
  before : Canonical.Schedule
  after : Canonical.Schedule
  split : raw.schedule = before ++ after
  startEq : start = ProductionAssignment.publicWidth +
    CanonicalBlockAssignment.coordinateCount before

def initial {application : Program} (raw : RawValues application) :
    State raw (PiRLCRetainedGeometry.priorPoseidonStart application) where
  before := []
  after := raw.schedule
  split := by simp
  startEq := rfl

theorem headEncodes {application : Program} {raw : RawValues application}
    {start sourceWidth : Nat} (cursor : State raw start)
    (block : LowNormBlock.Block sourceWidth) (source : Fin sourceWidth → F)
    (after : Canonical.Schedule)
    (head : cursor.after =
      CanonicalBlockAssignment.ofBlock block source :: after)
    (fits : start + block.coordinateCount ≤
      PerApplicationFixedPoint.logicalWidth application) :
    block.EncodesAt start fits raw.assignment source := by
  unfold PerApplicationCanonicalAssignment.RawValues.assignment
    PerApplicationCanonicalAssignment.Canonical.assignment
  rw [cursor.split, head]
  exact CanonicalBlockAssignment.assignment_encodesAt
    (encodedHashCells raw.outputDigest) cursor.before after block source start
      fits cursor.startEq

def next {application : Program} {raw : RawValues application}
    {start sourceWidth : Nat} (cursor : State raw start)
    (block : LowNormBlock.Block sourceWidth) (source : Fin sourceWidth → F)
    (after : Canonical.Schedule)
    (head : cursor.after =
      CanonicalBlockAssignment.ofBlock block source :: after) :
    State raw (start + block.coordinateCount) where
  before := cursor.before ++
    [CanonicalBlockAssignment.ofBlock block source]
  after := after
  split := by
    calc
      raw.schedule = cursor.before ++ cursor.after := cursor.split
      _ = cursor.before ++
          (CanonicalBlockAssignment.ofBlock block source :: after) := by
        rw [head]
      _ = (cursor.before ++
          [CanonicalBlockAssignment.ofBlock block source]) ++ after := by
        simp [List.append_assoc]
  startEq := by
    rw [CanonicalBlockAssignment.coordinateCount_append]
    simp only [CanonicalBlockAssignment.coordinateCount,
      CanonicalBlockAssignment.BlockValue.coordinateCount,
      CanonicalBlockAssignment.ofBlock]
    have previous := cursor.startEq
    omega

end Cursor

def tail {application : Program} (raw : RawValues application)
    (index : Nat) : Canonical.Schedule :=
  raw.schedule.drop index

structure Position {application : Program} (raw : RawValues application)
    (start sourceWidth : Nat) (block : LowNormBlock.Block sourceWidth)
    (source : Fin sourceWidth → F) where
  cursor : Cursor.State raw start
  after : Canonical.Schedule
  head : cursor.after =
    CanonicalBlockAssignment.ofBlock block source :: after

namespace Position

theorem encodes {application : Program} {raw : RawValues application}
    {start sourceWidth : Nat} {block : LowNormBlock.Block sourceWidth}
    {source : Fin sourceWidth → F}
    (position : Position raw start sourceWidth block source)
    (fits : start + block.coordinateCount ≤
      PerApplicationFixedPoint.logicalWidth application) :
    block.EncodesAt start fits raw.assignment source :=
  Cursor.headEncodes position.cursor block source position.after position.head fits

def next {application : Program} {raw : RawValues application}
    {start sourceWidth : Nat} {block : LowNormBlock.Block sourceWidth}
    {source : Fin sourceWidth → F}
    (position : Position raw start sourceWidth block source) :
    Cursor.State raw (start + block.coordinateCount) :=
  Cursor.next position.cursor block source position.after position.head

end Position

def retainedGeometry (application : Program) :=
  DirectPrefixPlan.prefixGeometry <|
    DirectRunningPrefixPlan.prefixGeometry <|
      DirectPiDECPrefixPlan.runningGeometry <|
        DirectPiRLCSamplerCompletePrefixPlan.piDecGeometry <|
          DirectApplicationPrefixPlan.prefixGeometry <|
            PerApplicationFixedPoint.geometry application

attribute [local simp] PiRLCFirst54RetainedBlocks.sourceWidth
  PiRLCRetainedGeometry.sourceWidth

private def position0 {application : Program} (raw : RawValues application) :
    Position raw (PiRLCRetainedGeometry.priorPoseidonStart application)
      (PiRLCRetainedGeometry.sourceWidth application)
      (PiRLCRetainedGeometry.priorPoseidonBlock application)
      raw.retainedSource where
  cursor := Cursor.initial raw
  after := tail raw 1
  head := by
    simp [Cursor.initial, tail, RawValues.schedule, Canonical.ofBlock,
      CanonicalBlockAssignment.ofBlock]

private def position1 {application : Program} (raw : RawValues application) :
    Position raw (PiRLCRetainedGeometry.outputPoseidonStart application)
      (PiRLCRetainedGeometry.sourceWidth application)
      (PiRLCRetainedGeometry.outputPoseidonBlock application)
      raw.retainedSource where
  cursor := (position0 raw).next
  after := tail raw 2
  head := by
    simp [Position.next, Cursor.next, position0, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position2 {application : Program} (raw : RawValues application) :
    Position raw (PiRLCRetainedGeometry.laterPoseidonStart application)
      (PiRLCRetainedGeometry.sourceWidth application)
      (PiRLCRetainedGeometry.laterPoseidonBlock application)
      raw.retainedSource where
  cursor := (position1 raw).next
  after := tail raw 3
  head := by
    simp [Position.next, Cursor.next, position1, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position3 {application : Program} (raw : RawValues application) :
    Position raw (PiRLCRetainedGeometry.productGroupStart application)
      (PiRLCRetainedGeometry.sourceWidth application)
      (PiRLCRetainedGeometry.productGroupBlock application)
      raw.retainedSource where
  cursor := (position2 raw).next
  after := tail raw 4
  head := by
    simp [Position.next, Cursor.next, position2, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position4 {application : Program} (raw : RawValues application) :
    Position raw (PiRLCRetainedGeometry.rejectStart application)
      (PiRLCRetainedGeometry.sourceWidth application)
      (PiRLCFirst54RetainedBlocks.rejectBlock application)
      raw.retainedSource where
  cursor := (position3 raw).next
  after := tail raw 5
  head := by
    simp [Position.next, Cursor.next, position3, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position5 {application : Program} (raw : RawValues application) :
    Position raw (PiRLCRetainedGeometry.symbolStart application)
      (PiRLCRetainedGeometry.sourceWidth application)
      (PiRLCFirst54RetainedBlocks.symbolBlock application)
      raw.retainedSource where
  cursor := (position4 raw).next
  after := tail raw 6
  head := by
    simp [Position.next, Cursor.next, position4, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position6 {application : Program} (raw : RawValues application) :
    Position raw (PiRLCRetainedGeometry.positionStart application)
      (PiRLCRetainedGeometry.sourceWidth application)
      (PiRLCFirst54RetainedBlocks.positionBlock application)
      raw.retainedSource where
  cursor := (position5 raw).next
  after := tail raw 7
  head := by
    simp [Position.next, Cursor.next, position5, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position7 {application : Program} (raw : RawValues application) :
    Position raw (PiRLCRetainedGeometry.valueStart application)
      (PiRLCRetainedGeometry.sourceWidth application)
      (PiRLCFirst54RetainedBlocks.valueBlock application)
      raw.retainedSource where
  cursor := (position6 raw).next
  after := tail raw 8
  head := by
    simp [Position.next, Cursor.next, position6, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position8 {application : Program} (raw : RawValues application) :
    Position raw (PiRLCRetainedGeometry.first54ProductStart application)
      (PiRLCRetainedGeometry.sourceWidth application)
      (PiRLCFirst54RetainedBlocks.productBlock application)
      raw.retainedSource where
  cursor := (position7 raw).next
  after := tail raw 9
  head := by
    simp [Position.next, Cursor.next, position7, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position9 {application : Program} (raw : RawValues application) :
    Position raw (PiRLCRetainedGeometry.productInputStart application)
      (PiRLCRetainedGeometry.sourceWidth application)
      (PiRLCRetainedGeometry.productInputBlock application)
      raw.retainedSource where
  cursor := (position8 raw).next
  after := tail raw 10
  head := by
    simp [Position.next, Cursor.next, position8, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position10 {application : Program} (raw : RawValues application) :
    Position raw (PiRLCRetainedGeometry.productOutputStart application)
      (PiRLCRetainedGeometry.sourceWidth application)
      (PiRLCRetainedGeometry.productOutputBlock application)
      raw.retainedSource where
  cursor := (position9 raw).next
  after := tail raw 11
  head := by
    simp [Position.next, Cursor.next, position9, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

/-- The first retained prefix is encoded by structural schedule succession. -/
theorem retainedEncodes {application : Program} (raw : RawValues application) :
    PiRLCRetainedPreservation.Encodes (retainedGeometry application)
      raw.assignment raw.base raw.groupValue raw.products where
  priorPoseidon := (position0 raw).encodes
    (PiRLCRetainedGeometry.priorPoseidonFits (retainedGeometry application))
  outputPoseidon := (position1 raw).encodes
    (PiRLCRetainedGeometry.outputPoseidonFits (retainedGeometry application))
  laterPoseidon := (position2 raw).encodes
    (PiRLCRetainedGeometry.laterPoseidonFits (retainedGeometry application))
  productGroup := (position3 raw).encodes
    (PiRLCRetainedGeometry.productGroupFits (retainedGeometry application))
  reject := (position4 raw).encodes
    (PiRLCRetainedGeometry.rejectFits (retainedGeometry application))
  symbol := (position5 raw).encodes
    (PiRLCRetainedGeometry.symbolFits (retainedGeometry application))
  position := (position6 raw).encodes
    (PiRLCRetainedGeometry.positionFits (retainedGeometry application))
  value := (position7 raw).encodes
    (PiRLCRetainedGeometry.valueFits (retainedGeometry application))
  first54Product := (position8 raw).encodes
    (PiRLCRetainedGeometry.first54ProductFits (retainedGeometry application))
  productInput := (position9 raw).encodes
    (PiRLCRetainedGeometry.productInputFits (retainedGeometry application))
  productOutput := (position10 raw).encodes
    (PiRLCRetainedGeometry.productOutputFits (retainedGeometry application))

attribute [local simp] PiRLCPoseidonGeometry.sourceWidth
  RunningTransitionRetainedBlocks.sourceWidth

private def position11 {application : Program} (raw : RawValues application) :
    Position raw (PiRLCPoseidonGeometry.priorInputStart application)
      (PiRLCPoseidonGeometry.sourceWidth application)
      (PiRLCPoseidonGeometry.priorInputBlock application)
      raw.retainedSource where
  cursor := (position10 raw).next
  after := tail raw 12
  head := by
    simp [Position.next, Cursor.next, position10, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position12 {application : Program} (raw : RawValues application) :
    Position raw (PiRLCPoseidonGeometry.outputInputStart application)
      (PiRLCPoseidonGeometry.sourceWidth application)
      (PiRLCPoseidonGeometry.outputInputBlock application)
      raw.retainedSource where
  cursor := (position11 raw).next
  after := tail raw 13
  head := by
    simp [Position.next, Cursor.next, position11, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position13 {application : Program} (raw : RawValues application) :
    Position raw (PiCCSActionPayloadBlock.payloadStart application)
      (PiCCSActionPayloadBlock.sourceWidth application)
      (PiCCSActionPayloadBlock.block application) raw.payloadSource where
  cursor := (position12 raw).next
  after := tail raw 14
  head := by
    simp [Position.next, Cursor.next, position12, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position14 {application : Program} (raw : RawValues application) :
    Position raw (RunningTransitionRetainedGeometry.stateStart application)
      (RunningTransitionRetainedBlocks.sourceWidth application)
      (RunningTransitionRetainedBlocks.stateBlock application)
      raw.retainedSource where
  cursor := (position13 raw).next
  after := tail raw 15
  head := by
    simp [Position.next, Cursor.next, position13, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position15 {application : Program} (raw : RawValues application) :
    Position raw (RunningTransitionRetainedGeometry.outputStart application)
      (RunningTransitionRetainedBlocks.sourceWidth application)
      (RunningTransitionRetainedBlocks.outputBlock application)
      raw.retainedSource where
  cursor := (position14 raw).next
  after := tail raw 16
  head := by
    simp [Position.next, Cursor.next, position14, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position16 {application : Program} (raw : RawValues application) :
    Position raw (RunningTransitionRetainedGeometry.roundC0Start application)
      (RunningTransitionRetainedBlocks.sourceWidth application)
      (RunningTransitionRetainedBlocks.roundC0Block application)
      raw.retainedSource where
  cursor := (position15 raw).next
  after := tail raw 17
  head := by
    simp [Position.next, Cursor.next, position15, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position17 {application : Program} (raw : RawValues application) :
    Position raw (RunningTransitionRetainedGeometry.roundC1Start application)
      (RunningTransitionRetainedBlocks.sourceWidth application)
      (RunningTransitionRetainedBlocks.roundC1Block application)
      raw.retainedSource where
  cursor := (position16 raw).next
  after := tail raw 18
  head := by
    simp [Position.next, Cursor.next, position16, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position18 {application : Program} (raw : RawValues application) :
    Position raw (RunningTransitionRetainedGeometry.piDecStart application)
      (RunningTransitionRetainedBlocks.sourceWidth application)
      (RunningTransitionRetainedBlocks.piDecBlock application)
      raw.retainedSource where
  cursor := (position17 raw).next
  after := tail raw 19
  head := by
    simp [Position.next, Cursor.next, position17, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position19 {application : Program} (raw : RawValues application) :
    Position raw (RunningTransitionRetainedGeometry.freshStart application)
      (RunningTransitionRetainedBlocks.sourceWidth application)
      (RunningTransitionRetainedBlocks.freshBlock application)
      raw.retainedSource where
  cursor := (position18 raw).next
  after := tail raw 20
  head := by
    simp [Position.next, Cursor.next, position18, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

def runningGeometry (application : Program) :=
  DirectPiDECPrefixPlan.runningGeometry <|
    DirectPiRLCSamplerCompletePrefixPlan.piDecGeometry <|
      DirectApplicationPrefixPlan.prefixGeometry <|
        PerApplicationFixedPoint.geometry application

def poseidonGeometry (application : Program) :=
  DirectRunningPrefixPlan.prefixGeometry (runningGeometry application)

private theorem directPrefixEncodes {application : Program}
    (raw : RawValues application) :
    DirectPrefixPlan.Encodes (poseidonGeometry application) raw.assignment
      raw.base raw.groupValue raw.products where
  retained := retainedEncodes raw
  pilotPriorInput := (position11 raw).encodes
    (PiRLCPoseidonGeometry.priorInputFits
      (DirectPrefixPlan.pilotGeometry (poseidonGeometry application)))
  pilotOutputInput := (position12 raw).encodes
    (PiRLCPoseidonGeometry.outputInputFits
      (DirectPrefixPlan.pilotGeometry (poseidonGeometry application)))
  payload := (position13 raw).encodes
    (PiCCSPoseidonPlan.payloadFits (poseidonGeometry application))

private theorem transitionEncodes {application : Program}
    (raw : RawValues application) :
    RunningTransitionRetainedGeometry.Encodes (runningGeometry application)
      raw.assignment raw.retainedSource where
  state := (position14 raw).encodes
    (RunningTransitionRetainedGeometry.stateFits (runningGeometry application))
  output := (position15 raw).encodes
    (RunningTransitionRetainedGeometry.outputFits (runningGeometry application))
  roundC0 := (position16 raw).encodes
    (RunningTransitionRetainedGeometry.roundC0Fits (runningGeometry application))
  roundC1 := (position17 raw).encodes
    (RunningTransitionRetainedGeometry.roundC1Fits (runningGeometry application))
  piDec := (position18 raw).encodes
    (RunningTransitionRetainedGeometry.piDecFits (runningGeometry application))
  fresh := (position19 raw).encodes
    (RunningTransitionRetainedGeometry.freshFits (runningGeometry application))

/-- Canonical assignment evidence through the running-instance transition. -/
theorem runningPrefixEncodes {application : Program} (raw : RawValues application) :
    DirectRunningPrefixPlan.Encodes (runningGeometry application)
      raw.assignment raw.base raw.groupValue raw.products :=
  ⟨directPrefixEncodes raw, transitionEncodes raw⟩

attribute [local simp] PiCCSOrdinaryRetainedBlocks.sourceWidth
  PilotOrdinaryRetainedBlocks.sourceWidth PiDECRetainedBlocks.sourceWidth
  PiRLCSamplerOrdinaryRetainedBlocks.sourceWidth

private def position20 {application : Program} (raw : RawValues application) :
    Position raw (PiCCSOrdinaryRetainedGeometry.priorInputStart application)
      (PiCCSOrdinaryRetainedBlocks.sourceWidth application)
      (PiCCSOrdinaryRetainedBlocks.priorInputBlock application)
      raw.retainedSource where
  cursor := (position19 raw).next
  after := tail raw 21
  head := by
    simp [Position.next, Cursor.next, position19, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position21 {application : Program} (raw : RawValues application) :
    Position raw (PiCCSOrdinaryRetainedGeometry.outputInputStart application)
      (PiCCSOrdinaryRetainedBlocks.sourceWidth application)
      (PiCCSOrdinaryRetainedBlocks.outputInputBlock application)
      raw.retainedSource where
  cursor := (position20 raw).next
  after := tail raw 22
  head := by
    simp [Position.next, Cursor.next, position20, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position22 {application : Program} (raw : RawValues application) :
    Position raw
      (PiCCSOrdinaryRetainedGeometry.freshPublicInputStart application)
      (PiCCSOrdinaryRetainedBlocks.sourceWidth application)
      (PiCCSOrdinaryRetainedBlocks.freshPublicInputBlock application)
      raw.retainedSource where
  cursor := (position21 raw).next
  after := tail raw 23
  head := by
    simp [Position.next, Cursor.next, position21, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position23 {application : Program} (raw : RawValues application) :
    Position raw (PiCCSOrdinaryRetainedGeometry.priorLastStart application)
      (PiCCSOrdinaryRetainedBlocks.sourceWidth application)
      (PiCCSOrdinaryRetainedBlocks.priorLastBlock application)
      raw.retainedSource where
  cursor := (position22 raw).next
  after := tail raw 24
  head := by
    simp [Position.next, Cursor.next, position22, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position24 {application : Program} (raw : RawValues application) :
    Position raw (PiCCSOrdinaryRetainedGeometry.outputLastStart application)
      (PiCCSOrdinaryRetainedBlocks.sourceWidth application)
      (PiCCSOrdinaryRetainedBlocks.outputLastBlock application)
      raw.retainedSource where
  cursor := (position23 raw).next
  after := tail raw 25
  head := by
    simp [Position.next, Cursor.next, position23, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position25 {application : Program} (raw : RawValues application) :
    Position raw (PiCCSOrdinaryRetainedGeometry.expectedContextStart application)
      (PiCCSOrdinaryRetainedBlocks.sourceWidth application)
      (PiCCSOrdinaryRetainedBlocks.expectedContextBlock application)
      raw.retainedSource where
  cursor := (position24 raw).next
  after := tail raw 26
  head := by
    simp [Position.next, Cursor.next, position24, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position26 {application : Program} (raw : RawValues application) :
    Position raw (PiCCSOrdinaryRetainedGeometry.proofLogicalStart application)
      (PiCCSOrdinaryRetainedBlocks.sourceWidth application)
      (PiCCSOrdinaryRetainedBlocks.proofLogicalBlock application)
      raw.retainedSource where
  cursor := (position25 raw).next
  after := tail raw 27
  head := by
    simp [Position.next, Cursor.next, position25, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position27 {application : Program} (raw : RawValues application) :
    Position raw (PiCCSOrdinaryRetainedGeometry.outputEndpointStart application)
      (PiCCSOrdinaryRetainedBlocks.sourceWidth application)
      (PiCCSOrdinaryRetainedBlocks.outputEndpointBlock application)
      raw.retainedSource where
  cursor := (position26 raw).next
  after := tail raw 28
  head := by
    simp [Position.next, Cursor.next, position26, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position28 {application : Program} (raw : RawValues application) :
    Position raw (PiCCSOrdinaryRetainedGeometry.freshStart application)
      (PiCCSOrdinaryRetainedBlocks.sourceWidth application)
      (PiCCSOrdinaryRetainedBlocks.freshBlock application)
      raw.retainedSource where
  cursor := (position27 raw).next
  after := tail raw 29
  head := by
    simp [Position.next, Cursor.next, position27, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position29 {application : Program} (raw : RawValues application) :
    Position raw (PilotOrdinaryRetainedGeometry.canonicalLocalStart application)
      (PilotOrdinaryRetainedBlocks.sourceWidth application)
      (PilotOrdinaryRetainedBlocks.canonicalLocalBlock application)
      raw.retainedSource where
  cursor := (position28 raw).next
  after := tail raw 30
  head := by
    simp [Position.next, Cursor.next, position28, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position30 {application : Program} (raw : RawValues application) :
    Position raw (PilotOrdinaryRetainedGeometry.canonicalFreshStart application)
      (PilotOrdinaryRetainedBlocks.sourceWidth application)
      (PilotOrdinaryRetainedBlocks.canonicalFreshBlock application)
      raw.retainedSource where
  cursor := (position29 raw).next
  after := tail raw 31
  head := by
    simp [Position.next, Cursor.next, position29, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position31 {application : Program} (raw : RawValues application) :
    Position raw (PilotOrdinaryRetainedGeometry.outputDigestStart application)
      (PilotOrdinaryRetainedBlocks.sourceWidth application)
      (PilotOrdinaryRetainedBlocks.outputDigestBlock application)
      raw.retainedSource where
  cursor := (position30 raw).next
  after := tail raw 32
  head := by
    simp [Position.next, Cursor.next, position30, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position32 {application : Program} (raw : RawValues application) :
    Position raw (PiDECRetainedGeometry.parentCommitmentStart application)
      (PiDECRetainedBlocks.sourceWidth application)
      (PiDECRetainedBlocks.parentCommitmentBlock application)
      raw.retainedSource where
  cursor := (position31 raw).next
  after := tail raw 33
  head := by
    simp [Position.next, Cursor.next, position31, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position33 {application : Program} (raw : RawValues application) :
    Position raw (PiDECRetainedGeometry.parentPublicInputStart application)
      (PiDECRetainedBlocks.sourceWidth application)
      (PiDECRetainedBlocks.parentPublicInputBlock application)
      raw.retainedSource where
  cursor := (position32 raw).next
  after := tail raw 34
  head := by
    simp [Position.next, Cursor.next, position32, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position34 {application : Program} (raw : RawValues application) :
    Position raw (PiDECRetainedGeometry.parentEvalKStart application)
      (PiDECRetainedBlocks.sourceWidth application)
      (PiDECRetainedBlocks.parentEvalKBlock application)
      raw.retainedSource where
  cursor := (position33 raw).next
  after := tail raw 35
  head := by
    simp [Position.next, Cursor.next, position33, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position35 {application : Program} (raw : RawValues application) :
    Position raw (PiDECRetainedGeometry.parentEvalAStart application)
      (PiDECRetainedBlocks.sourceWidth application)
      (PiDECRetainedBlocks.parentEvalABlock application)
      raw.retainedSource where
  cursor := (position34 raw).next
  after := tail raw 36
  head := by
    simp [Position.next, Cursor.next, position34, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position36 {application : Program} (raw : RawValues application) :
    Position raw (PiDECRetainedGeometry.proofStart application)
      (PiDECRetainedBlocks.sourceWidth application)
      (PiDECRetainedBlocks.proofBlock application) raw.retainedSource where
  cursor := (position35 raw).next
  after := tail raw 37
  head := by
    simp [Position.next, Cursor.next, position35, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position37 {application : Program} (raw : RawValues application) :
    Position raw (PiDECRetainedGeometry.logicalStart application)
      (PiDECRetainedBlocks.sourceWidth application)
      (PiDECRetainedBlocks.logicalBlock application) raw.retainedSource where
  cursor := (position36 raw).next
  after := tail raw 38
  head := by
    simp [Position.next, Cursor.next, position36, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position38 {application : Program} (raw : RawValues application) :
    Position raw (PiDECRetainedGeometry.freshStart application)
      (PiDECRetainedBlocks.sourceWidth application)
      (PiDECRetainedBlocks.freshBlock application) raw.retainedSource where
  cursor := (position37 raw).next
  after := tail raw 39
  head := by
    simp [Position.next, Cursor.next, position37, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position39 {application : Program} (raw : RawValues application) :
    Position raw (PiRLCSamplerOrdinaryRetainedGeometry.logicalStart application)
      (PiRLCSamplerOrdinaryRetainedBlocks.sourceWidth application)
      (PiRLCSamplerOrdinaryRetainedBlocks.logicalBlock application)
      raw.retainedSource where
  cursor := (position38 raw).next
  after := tail raw 40
  head := by
    simp [Position.next, Cursor.next, position38, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position40 {application : Program} (raw : RawValues application) :
    Position raw (PiRLCSamplerOrdinaryRetainedGeometry.freshStart application)
      (PiRLCSamplerOrdinaryRetainedBlocks.sourceWidth application)
      (PiRLCSamplerOrdinaryRetainedBlocks.freshBlock application)
      raw.retainedSource where
  cursor := (position39 raw).next
  after := tail raw 41
  head := by
    simp [Position.next, Cursor.next, position39, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

def samplerGeometry (application : Program) :=
  DirectApplicationPrefixPlan.prefixGeometry <|
    PerApplicationFixedPoint.geometry application

def piDecGeometry (application : Program) :=
  DirectPiRLCSamplerCompletePrefixPlan.piDecGeometry
    (samplerGeometry application)

def pilotOrdinaryGeometry (application : Program) :=
  DirectPiDECPrefixPlan.pilotOrdinaryGeometry (piDecGeometry application)

def piCcsOrdinaryGeometry (application : Program) :=
  PilotOrdinaryDirectPlan.piCcsGeometry (pilotOrdinaryGeometry application)

private theorem piCcsOrdinaryEncodes {application : Program}
    (raw : RawValues application) :
    PiCCSOrdinaryRetainedGeometry.Encodes
      (piCcsOrdinaryGeometry application) raw.assignment raw.retainedSource where
  priorInput := (position20 raw).encodes
    (PiCCSOrdinaryRetainedGeometry.priorInputFits
      (piCcsOrdinaryGeometry application))
  outputInput := (position21 raw).encodes
    (PiCCSOrdinaryRetainedGeometry.outputInputFits
      (piCcsOrdinaryGeometry application))
  freshPublicInput := (position22 raw).encodes
    (PiCCSOrdinaryRetainedGeometry.freshPublicInputFits
      (piCcsOrdinaryGeometry application))
  priorLast := (position23 raw).encodes
    (PiCCSOrdinaryRetainedGeometry.priorLastFits
      (piCcsOrdinaryGeometry application))
  outputLast := (position24 raw).encodes
    (PiCCSOrdinaryRetainedGeometry.outputLastFits
      (piCcsOrdinaryGeometry application))
  expectedContext := (position25 raw).encodes
    (PiCCSOrdinaryRetainedGeometry.expectedContextFits
      (piCcsOrdinaryGeometry application))
  proofLogical := (position26 raw).encodes
    (PiCCSOrdinaryRetainedGeometry.proofLogicalFits
      (piCcsOrdinaryGeometry application))
  outputEndpoint := (position27 raw).encodes
    (PiCCSOrdinaryRetainedGeometry.outputEndpointFits
      (piCcsOrdinaryGeometry application))
  fresh := (position28 raw).encodes
    (PiCCSOrdinaryRetainedGeometry.freshFits
      (piCcsOrdinaryGeometry application))

private theorem pilotAddedEncodes {application : Program}
    (raw : RawValues application) :
    PilotOrdinaryRetainedGeometry.Encodes (pilotOrdinaryGeometry application)
      raw.assignment raw.retainedSource where
  canonicalLocal := (position29 raw).encodes
    (PilotOrdinaryRetainedGeometry.canonicalLocalFits
      (pilotOrdinaryGeometry application))
  canonicalFresh := (position30 raw).encodes
    (PilotOrdinaryRetainedGeometry.canonicalFreshFits
      (pilotOrdinaryGeometry application))
  outputDigest := (position31 raw).encodes
    (PilotOrdinaryRetainedGeometry.outputDigestFits
      (pilotOrdinaryGeometry application))

private theorem pilotOrdinaryEncodes {application : Program}
    (raw : RawValues application) :
    PilotOrdinaryDirectPlan.Encodes (pilotOrdinaryGeometry application)
      raw.assignment raw.base raw.groupValue raw.products :=
  ⟨piCcsOrdinaryEncodes raw, pilotAddedEncodes raw⟩

private theorem piDecRetainedEncodes {application : Program}
    (raw : RawValues application) :
    PiDECRetainedGeometry.Encodes (piDecGeometry application) raw.assignment
      raw.retainedSource where
  parentCommitment := (position32 raw).encodes
    (PiDECRetainedGeometry.parentCommitmentFits (piDecGeometry application))
  parentPublicInput := (position33 raw).encodes
    (PiDECRetainedGeometry.parentPublicInputFits (piDecGeometry application))
  parentEvalK := (position34 raw).encodes
    (PiDECRetainedGeometry.parentEvalKFits (piDecGeometry application))
  parentEvalA := (position35 raw).encodes
    (PiDECRetainedGeometry.parentEvalAFits (piDecGeometry application))
  proof := (position36 raw).encodes
    (PiDECRetainedGeometry.proofFits (piDecGeometry application))
  logical := (position37 raw).encodes
    (PiDECRetainedGeometry.logicalFits (piDecGeometry application))
  fresh := (position38 raw).encodes
    (PiDECRetainedGeometry.freshFits (piDecGeometry application))

private theorem piDecPrefixEncodes {application : Program}
    (raw : RawValues application) :
    DirectPiDECPrefixPlan.Encodes (piDecGeometry application) raw.assignment
      raw.base raw.groupValue raw.products :=
  ⟨runningPrefixEncodes raw, pilotOrdinaryEncodes raw,
    piDecRetainedEncodes raw⟩

private theorem samplerOrdinaryEncodes {application : Program}
    (raw : RawValues application) :
    PiRLCSamplerOrdinaryRetainedGeometry.Encodes (samplerGeometry application)
      raw.assignment raw.retainedSource where
  logical := (position39 raw).encodes
    (PiRLCSamplerOrdinaryRetainedGeometry.logicalFits
      (samplerGeometry application))
  fresh := (position40 raw).encodes
    (PiRLCSamplerOrdinaryRetainedGeometry.freshFits
      (samplerGeometry application))

/-- Canonical assignment evidence through the complete sampler prefix. -/
theorem samplerPrefixEncodes {application : Program} (raw : RawValues application) :
    DirectPiRLCSamplerCompletePrefixPlan.Encodes (samplerGeometry application)
      raw.assignment raw.base raw.groupValue raw.products :=
  ⟨piDecPrefixEncodes raw, samplerOrdinaryEncodes raw⟩

private def position41 {application : Program} (raw : RawValues application) :
    Position raw (ApplicationRetainedGeometry.inputStart application)
      (ApplicationRetainedBlocks.sourceWidth application)
      (ApplicationRetainedBlocks.inputBlock application)
      raw.applicationSource where
  cursor := (position40 raw).next
  after := tail raw 42
  head := by
    simp [Position.next, Cursor.next, position40, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position42 {application : Program} (raw : RawValues application) :
    Position raw (ApplicationRetainedGeometry.witnessStart application)
      (ApplicationRetainedBlocks.sourceWidth application)
      (ApplicationRetainedBlocks.witnessBlock application)
      raw.applicationSource where
  cursor := (position41 raw).next
  after := tail raw 43
  head := by
    simp [Position.next, Cursor.next, position41, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position43 {application : Program} (raw : RawValues application) :
    Position raw (ApplicationRetainedGeometry.outputStart application)
      (ApplicationRetainedBlocks.sourceWidth application)
      (ApplicationRetainedBlocks.outputBlock application)
      raw.applicationSource where
  cursor := (position42 raw).next
  after := tail raw 44
  head := by
    simp [Position.next, Cursor.next, position42, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position44 {application : Program} (raw : RawValues application) :
    Position raw (ApplicationRetainedGeometry.localStart application)
      (ApplicationRetainedBlocks.sourceWidth application)
      (ApplicationRetainedBlocks.localBlock application)
      raw.applicationSource where
  cursor := (position43 raw).next
  after := tail raw 45
  head := by
    simp [Position.next, Cursor.next, position43, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

def applicationGeometry (application : Program) :=
  PerApplicationFixedPoint.geometry application

private theorem applicationRetainedEncodes {application : Program}
    (raw : RawValues application) :
    ApplicationRetainedGeometry.Encodes (applicationGeometry application)
      raw.assignment raw.applicationSource where
  input := (position41 raw).encodes
    (ApplicationRetainedGeometry.inputFits (applicationGeometry application))
  witness := (position42 raw).encodes
    (ApplicationRetainedGeometry.witnessFits (applicationGeometry application))
  output := (position43 raw).encodes
    (ApplicationRetainedGeometry.outputFits (applicationGeometry application))
  localValues := (position44 raw).encodes
    (ApplicationRetainedGeometry.localFits (applicationGeometry application))

/-- One raw packet constructs the complete final assignment-encoding contract.
No caller supplies `DirectApplicationPrefixPlan.Encodes`. -/
theorem encodes {application : Program} (raw : RawValues application) :
    DirectApplicationPrefixPlan.Encodes (applicationGeometry application)
      raw.assignment raw.base raw.groupValue raw.products :=
  ⟨samplerPrefixEncodes raw, applicationRetainedEncodes raw⟩

end NightstreamFPrime.Export.Stage1.PerApplicationCanonicalEncodes
