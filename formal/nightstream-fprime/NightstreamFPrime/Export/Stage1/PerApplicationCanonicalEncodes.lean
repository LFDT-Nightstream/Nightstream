import NightstreamFPrime.Export.Stage1.PiDECValueWiring
import NightstreamFPrime.Export.Stage1.PerApplicationCanonicalAssignment

/-!
Owns the proof that the compact per-application assignment canonically encodes
every retained Stage 1 block. A structural cursor consumes one block at a time.
No proof normalizes the complete 33-block schedule or an expanded slot list.
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
    Position raw (PiRLCRetainedGeometry.productOutputStart application)
      (PiRLCRetainedGeometry.sourceWidth application)
      (PiRLCRetainedGeometry.productOutputBlock application)
      raw.retainedSource where
  cursor := (position8 raw).next
  after := tail raw 10
  head := by
    simp [Position.next, Cursor.next, position8, tail, RawValues.schedule,
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
  productOutput := (position9 raw).encodes
    (PiRLCRetainedGeometry.productOutputFits (retainedGeometry application))

attribute [local simp] PiRLCPoseidonGeometry.sourceWidth
  RunningTransitionRetainedBlocks.sourceWidth

private def position10 {application : Program} (raw : RawValues application) :
    Position raw (PiRLCPoseidonGeometry.priorInputStart application)
      (PiRLCPoseidonGeometry.sourceWidth application)
      (PiRLCPoseidonGeometry.priorInputBlock application)
      raw.retainedSource where
  cursor := (position9 raw).next
  after := tail raw 11
  head := by
    simp [Position.next, Cursor.next, position9, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position11 {application : Program} (raw : RawValues application) :
    Position raw (PiRLCPoseidonGeometry.outputInputStart application)
      (PiRLCPoseidonGeometry.sourceWidth application)
      (PiRLCPoseidonGeometry.outputInputBlock application)
      raw.retainedSource where
  cursor := (position10 raw).next
  after := tail raw 12
  head := by
    simp [Position.next, Cursor.next, position10, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position12 {application : Program} (raw : RawValues application) :
    Position raw (PiCCSActionPayloadBlock.payloadStart application)
      (PiCCSActionPayloadBlock.sourceWidth application)
      (PiCCSActionPayloadBlock.block application) raw.payloadSource where
  cursor := (position11 raw).next
  after := tail raw 13
  head := by
    simp [Position.next, Cursor.next, position11, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position13 {application : Program} (raw : RawValues application) :
    Position raw (RunningTransitionRetainedGeometry.roundC0Start application)
      (RunningTransitionRetainedBlocks.sourceWidth application)
      (RunningTransitionRetainedBlocks.roundC0Block application)
      raw.retainedSource where
  cursor := (position12 raw).next
  after := tail raw 14
  head := by
    simp [Position.next, Cursor.next, position12, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position14 {application : Program} (raw : RawValues application) :
    Position raw (RunningTransitionRetainedGeometry.roundC1Start application)
      (RunningTransitionRetainedBlocks.sourceWidth application)
      (RunningTransitionRetainedBlocks.roundC1Block application)
      raw.retainedSource where
  cursor := (position13 raw).next
  after := tail raw 15
  head := by
    simp [Position.next, Cursor.next, position13, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position15 {application : Program} (raw : RawValues application) :
    Position raw (RunningTransitionRetainedGeometry.piDecStart application)
      (RunningTransitionRetainedBlocks.sourceWidth application)
      (RunningTransitionRetainedBlocks.piDecBlock application)
      raw.retainedSource where
  cursor := (position14 raw).next
  after := tail raw 16
  head := by
    simp [Position.next, Cursor.next, position14, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position16 {application : Program} (raw : RawValues application) :
    Position raw (RunningTransitionRetainedGeometry.freshStart application)
      (RunningTransitionRetainedBlocks.sourceWidth application)
      (RunningTransitionRetainedBlocks.freshBlock application)
      raw.retainedSource where
  cursor := (position15 raw).next
  after := tail raw 17
  head := by
    simp [Position.next, Cursor.next, position15, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

def runningGeometry (application : Program) :=
  DirectPiDECPrefixPlan.runningGeometry <|
    DirectPiRLCSamplerCompletePrefixPlan.piDecGeometry <|
      DirectApplicationPrefixPlan.prefixGeometry <|
        PerApplicationFixedPoint.geometry application

def poseidonGeometry (application : Program) :=
  DirectRunningPrefixPlan.prefixGeometry (runningGeometry application)

/-- The prior preimage source view selects the same physical package word as
its existing pilot block. The package shift leaves these private words fixed. -/
private theorem priorInputSource_eq (application : Program)
    (slot : Fin Layout.PilotProduction.stateHashWords) :
    (PiRLCPoseidonGeometry.priorInputBlock application).source slot =
      (PiCCSOrdinaryRetainedBlocks.priorInputBlock application).source slot := by
  have slotBound := slot.isLt
  change slot.val < 49393 at slotBound
  have mapped : Layout.Stage1.Spartan.sourceToSpartan (0 + slot.val) =
      0 + slot.val := by
    have zero : Layout.Stage1.Spartan.sourceToSpartan 0 = 0 := by rfl
    simpa only [zero] using
      Layout.Stage1.Spartan.sourceToSpartan_add_of_pilotPriorPrivate
        0 slot.val (by change 0 + slot.val < 49393; omega)
  have constant : PerApplicationPackage.basePackage.layout.constantColumn =
      29336446 :=
    NightstreamFPrime.Export.Stage1.Package.circuitPackage_layout_values.2.2.1
  apply Fin.ext
  change 0 + slot.val = PerApplicationPackage.shiftColumn application
    (Layout.Stage1.Spartan.sourceToSpartan (0 + slot.val))
  rw [mapped, PerApplicationPackage.shiftColumn_private application _ (by
    rw [constant]
    omega)]

/-- The output source view applies the public-column permutation before the
same private package word is selected by the pilot block. -/
private theorem outputInputSource_eq (application : Program)
    (slot : Fin Layout.PilotProduction.stateHashWords) :
    (PiRLCPoseidonGeometry.outputInputBlock application).source slot =
      (PiCCSOrdinaryRetainedBlocks.outputInputBlock application).source slot := by
  have slotBound := slot.isLt
  change slot.val < 49393 at slotBound
  have mapped : Layout.Stage1.Spartan.sourceToSpartan (49663 + slot.val) =
      49393 + slot.val := by
    unfold Layout.Stage1.Spartan.sourceToSpartan
    rw [if_pos (by change 49663 + slot.val < 14722512; omega)]
    unfold Layout.PilotSpartan.sourceToSpartan
    rw [if_neg (by change ¬ (49663 + slot.val < 49393); omega),
      if_neg (by change ¬ (49663 + slot.val < 49663); omega),
      if_pos (by change 49663 + slot.val < 99056; omega)]
    have offset : 49663 + slot.val - Layout.PilotSpartan.outputPreimageStart =
        slot.val := by
      change 49663 + slot.val - 49663 = slot.val
      omega
    rw [offset]
    unfold Layout.Stage1.Spartan.liftPilotColumn
    rw [if_pos (by change 49393 + slot.val < 98786; omega)]
    rw [Layout.PilotSpartan.secondPrivateStart_value]
  have constant : PerApplicationPackage.basePackage.layout.constantColumn =
      29336446 :=
    NightstreamFPrime.Export.Stage1.Package.circuitPackage_layout_values.2.2.1
  apply Fin.ext
  change 49393 + slot.val = PerApplicationPackage.shiftColumn application
    (Layout.Stage1.Spartan.sourceToSpartan (49663 + slot.val))
  rw [mapped, PerApplicationPackage.shiftColumn_private application _ (by
    rw [constant]
    omega)]

private theorem transitionStateEncodes {application : Program}
    (raw : RawValues application) :
    (RunningTransitionRetainedBlocks.stateBlock application).EncodesAt
      (RunningTransitionRetainedGeometry.stateStart application)
      (RunningTransitionRetainedGeometry.stateFits (runningGeometry application))
      raw.assignment raw.retainedSource := by
  let parent := PiRLCPoseidonGeometry.priorInputBlock application
  have slots : 28 + 11 ≤ parent.slotCount := by
    simp [parent, PiRLCPoseidonGeometry.priorInputBlock]
  have fits : PiRLCPoseidonGeometry.priorInputStart application +
      28 * parent.kind.width + (parent.slice 28 11 slots).coordinateCount ≤
        PerApplicationFixedPoint.logicalWidth application :=
    RunningTransitionRetainedGeometry.stateFits (runningGeometry application)
  have view := parent.encodesAt_slice 28 11 slots
    (PiRLCPoseidonGeometry.priorInputStart application)
    (PiRLCPoseidonGeometry.priorInputFits
      (RunningTransitionRetainedGeometry.pilotGeometry (runningGeometry application)))
    fits raw.assignment raw.retainedSource
    ((position10 raw).encodes (PiRLCPoseidonGeometry.priorInputFits
      (RunningTransitionRetainedGeometry.pilotGeometry (runningGeometry application))))
  intro slot coordinate
  have selected : parent.source
      (RunningTransitionDirectPlan.Location.statePreimageWord slot) =
      (RunningTransitionRetainedBlocks.stateBlock application).source slot := by
    apply Fin.ext
    have selected := congrArg Fin.val (priorInputSource_eq application
      (RunningTransitionDirectPlan.Location.statePreimageWord slot))
    change 0 + (28 + slot.val) = PerApplicationPackage.shiftColumn application
      (Layout.Stage1.Spartan.sourceToSpartan (0 + (28 + slot.val))) at selected
    change 0 + (28 + slot.val) = PerApplicationPackage.shiftColumn application
      (Layout.Stage1.Spartan.sourceToSpartan
        (Layout.Stage1.RunningTransitionSourceSupport.stateStart + slot.val))
    rw [Layout.Stage1.RunningTransitionSourceSupport.stateStart_eq]
    simpa only [Nat.zero_add] using selected
  have value := view slot coordinate
  change raw.assignment
      ((RunningTransitionRetainedBlocks.stateBlock application).column
        (RunningTransitionRetainedGeometry.stateStart application)
        (RunningTransitionRetainedGeometry.stateFits (runningGeometry application))
        slot coordinate) =
    LowNormSlot.coordinate .field
      (raw.retainedSource (parent.source
        (RunningTransitionDirectPlan.Location.statePreimageWord slot))) coordinate at value
  rw [selected] at value
  exact value

private theorem transitionOutputEncodes {application : Program}
    (raw : RawValues application) :
    (RunningTransitionRetainedBlocks.outputBlock application).EncodesAt
      (RunningTransitionRetainedGeometry.outputStart application)
      (RunningTransitionRetainedGeometry.outputFits (runningGeometry application))
      raw.assignment raw.retainedSource := by
  intro slot coordinate
  have value := (position11 raw).encodes
    (PiRLCPoseidonGeometry.outputInputFits
      (RunningTransitionRetainedGeometry.pilotGeometry (runningGeometry application)))
    slot coordinate
  change raw.assignment
      ((RunningTransitionRetainedBlocks.outputBlock application).column
        (RunningTransitionRetainedGeometry.outputStart application)
        (RunningTransitionRetainedGeometry.outputFits (runningGeometry application))
        slot coordinate) =
    LowNormSlot.coordinate .field
      (raw.retainedSource
        ((PiRLCPoseidonGeometry.outputInputBlock application).source slot)) coordinate at value
  rw [outputInputSource_eq application slot] at value
  exact value

private theorem transitionEncodes {application : Program}
    (raw : RawValues application) :
    RunningTransitionRetainedGeometry.Encodes (runningGeometry application)
      raw.assignment raw.retainedSource where
  state := transitionStateEncodes raw
  output := transitionOutputEncodes raw
  roundC0 := (position13 raw).encodes
    (RunningTransitionRetainedGeometry.roundC0Fits (runningGeometry application))
  roundC1 := (position14 raw).encodes
    (RunningTransitionRetainedGeometry.roundC1Fits (runningGeometry application))
  piDec := (position15 raw).encodes
    (RunningTransitionRetainedGeometry.piDecFits (runningGeometry application))
  fresh := (position16 raw).encodes
    (RunningTransitionRetainedGeometry.freshFits (runningGeometry application))
  sboxes := PiCCSPoseidonPlan.retainedBlock_encodesAt
    (RunningTransitionRetainedGeometry.poseidonGeometry (runningGeometry application))
    raw.assignment raw.retainedSource
    (PiRLCRetainedGeometry.laterPoseidonFits (retainedGeometry application))
    (retainedEncodes raw).laterPoseidon

attribute [local simp] PiCCSOrdinaryRetainedBlocks.sourceWidth
  PilotOrdinaryRetainedBlocks.sourceWidth PiDECRetainedBlocks.sourceWidth
  PiRLCSamplerOrdinaryRetainedBlocks.sourceWidth

private def position17 {application : Program} (raw : RawValues application) :
    Position raw
      (PiCCSOrdinaryRetainedGeometry.freshPublicInputStart application)
      (PiCCSOrdinaryRetainedBlocks.sourceWidth application)
      (PiCCSOrdinaryRetainedBlocks.freshPublicInputBlock application)
      raw.retainedSource where
  cursor := (position16 raw).next
  after := tail raw 18
  head := by
    simp [Position.next, Cursor.next, position16, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position18 {application : Program} (raw : RawValues application) :
    Position raw (PiCCSOrdinaryRetainedGeometry.priorLastStart application)
      (PiCCSOrdinaryRetainedBlocks.sourceWidth application)
      (PiCCSOrdinaryRetainedBlocks.priorLastBlock application)
      raw.retainedSource where
  cursor := (position17 raw).next
  after := tail raw 19
  head := by
    simp [Position.next, Cursor.next, position17, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position19 {application : Program} (raw : RawValues application) :
    Position raw (PiCCSOrdinaryRetainedGeometry.outputLastStart application)
      (PiCCSOrdinaryRetainedBlocks.sourceWidth application)
      (PiCCSOrdinaryRetainedBlocks.outputLastBlock application)
      raw.retainedSource where
  cursor := (position18 raw).next
  after := tail raw 20
  head := by
    simp [Position.next, Cursor.next, position18, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position20 {application : Program} (raw : RawValues application) :
    Position raw (PiCCSOrdinaryRetainedGeometry.expectedContextStart application)
      (PiCCSOrdinaryRetainedBlocks.sourceWidth application)
      (PiCCSOrdinaryRetainedBlocks.expectedContextBlock application)
      raw.retainedSource where
  cursor := (position19 raw).next
  after := tail raw 21
  head := by
    simp [Position.next, Cursor.next, position19, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position21 {application : Program} (raw : RawValues application) :
    Position raw (PiCCSOrdinaryRetainedGeometry.proofLogicalStart application)
      (PiCCSOrdinaryRetainedBlocks.sourceWidth application)
      (PiCCSOrdinaryRetainedBlocks.proofLogicalBlock application)
      raw.retainedSource where
  cursor := (position20 raw).next
  after := tail raw 22
  head := by
    simp [Position.next, Cursor.next, position20, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position22 {application : Program} (raw : RawValues application) :
    Position raw (PiCCSOrdinaryRetainedGeometry.outputEndpointStart application)
      (PiCCSOrdinaryRetainedBlocks.sourceWidth application)
      (PiCCSOrdinaryRetainedBlocks.outputEndpointBlock application)
      raw.retainedSource where
  cursor := (position21 raw).next
  after := tail raw 23
  head := by
    simp [Position.next, Cursor.next, position21, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position23 {application : Program} (raw : RawValues application) :
    Position raw (PiCCSOrdinaryRetainedGeometry.freshStart application)
      (PiCCSOrdinaryRetainedBlocks.sourceWidth application)
      (PiCCSOrdinaryRetainedBlocks.freshBlock application)
      raw.retainedSource where
  cursor := (position22 raw).next
  after := tail raw 24
  head := by
    simp [Position.next, Cursor.next, position22, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position24 {application : Program} (raw : RawValues application) :
    Position raw (PilotOrdinaryRetainedGeometry.canonicalLocalStart application)
      (PilotOrdinaryRetainedBlocks.sourceWidth application)
      (PilotOrdinaryRetainedBlocks.canonicalLocalBlock application)
      raw.retainedSource where
  cursor := (position23 raw).next
  after := tail raw 25
  head := by
    simp [Position.next, Cursor.next, position23, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position25 {application : Program} (raw : RawValues application) :
    Position raw (PilotOrdinaryRetainedGeometry.canonicalFreshStart application)
      (PilotOrdinaryRetainedBlocks.sourceWidth application)
      (PilotOrdinaryRetainedBlocks.canonicalFreshBlock application)
      raw.retainedSource where
  cursor := (position24 raw).next
  after := tail raw 26
  head := by
    simp [Position.next, Cursor.next, position24, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position26 {application : Program} (raw : RawValues application) :
    Position raw (PilotOrdinaryRetainedGeometry.outputDigestStart application)
      (PilotOrdinaryRetainedBlocks.sourceWidth application)
      (PilotOrdinaryRetainedBlocks.outputDigestBlock application)
      raw.retainedSource where
  cursor := (position25 raw).next
  after := tail raw 27
  head := by
    simp [Position.next, Cursor.next, position25, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position27 {application : Program} (raw : RawValues application) :
    Position raw (PiDECRetainedGeometry.logicalStart application)
      (PiDECRetainedBlocks.sourceWidth application)
      (PiDECRetainedBlocks.logicalBlock application) raw.retainedSource where
  cursor := (position26 raw).next
  after := tail raw 28
  head := by
    simp [Position.next, Cursor.next, position26, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position28 {application : Program} (raw : RawValues application) :
    Position raw (PiDECRetainedGeometry.freshStart application)
      (PiDECRetainedBlocks.sourceWidth application)
      (PiDECRetainedBlocks.freshBlock application) raw.retainedSource where
  cursor := (position27 raw).next
  after := tail raw 29
  head := by
    simp [Position.next, Cursor.next, position27, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position29 {application : Program} (raw : RawValues application) :
    Position raw (PiRLCSamplerOrdinaryRetainedGeometry.logicalStart application)
      (PiRLCSamplerOrdinaryRetainedBlocks.sourceWidth application)
      (PiRLCSamplerOrdinaryRetainedBlocks.logicalBlock application)
      raw.retainedSource where
  cursor := (position28 raw).next
  after := tail raw 30
  head := by
    simp [Position.next, Cursor.next, position28, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position30 {application : Program} (raw : RawValues application) :
    Position raw (PiRLCSamplerOrdinaryRetainedGeometry.freshStart application)
      (PiRLCSamplerOrdinaryRetainedBlocks.sourceWidth application)
      (PiRLCSamplerOrdinaryRetainedBlocks.freshBlock application)
      raw.retainedSource where
  cursor := (position29 raw).next
  after := tail raw 31
  head := by
    simp [Position.next, Cursor.next, position29, tail, RawValues.schedule,
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
  priorInput := by
    intro slot coordinate
    have value := (position10 raw).encodes
      (PiRLCPoseidonGeometry.priorInputFits
        (PiCCSOrdinaryRetainedGeometry.pilotGeometry
          (piCcsOrdinaryGeometry application))) slot coordinate
    change raw.assignment
        ((PiCCSOrdinaryRetainedBlocks.priorInputBlock application).column
          (PiCCSOrdinaryRetainedGeometry.priorInputStart application)
          (PiCCSOrdinaryRetainedGeometry.priorInputFits
            (piCcsOrdinaryGeometry application)) slot coordinate) =
      LowNormSlot.coordinate .field
        (raw.retainedSource
          ((PiRLCPoseidonGeometry.priorInputBlock application).source slot)) coordinate at value
    rw [priorInputSource_eq application slot] at value
    exact value
  outputInput := transitionOutputEncodes raw
  freshPublicInput := (position17 raw).encodes
    (PiCCSOrdinaryRetainedGeometry.freshPublicInputFits
      (piCcsOrdinaryGeometry application))
  priorLast := (position18 raw).encodes
    (PiCCSOrdinaryRetainedGeometry.priorLastFits
      (piCcsOrdinaryGeometry application))
  outputLast := (position19 raw).encodes
    (PiCCSOrdinaryRetainedGeometry.outputLastFits
      (piCcsOrdinaryGeometry application))
  expectedContext := (position20 raw).encodes
    (PiCCSOrdinaryRetainedGeometry.expectedContextFits
      (piCcsOrdinaryGeometry application))
  proofLogical := (position21 raw).encodes
    (PiCCSOrdinaryRetainedGeometry.proofLogicalFits
      (piCcsOrdinaryGeometry application))
  outputEndpoint := (position22 raw).encodes
    (PiCCSOrdinaryRetainedGeometry.outputEndpointFits
      (piCcsOrdinaryGeometry application))
  fresh := (position23 raw).encodes
    (PiCCSOrdinaryRetainedGeometry.freshFits
      (piCcsOrdinaryGeometry application))
  sboxes := PiCCSPoseidonPlan.retainedBlock_encodesAt
    (PiCCSOrdinaryRetainedGeometry.poseidonGeometry (piCcsOrdinaryGeometry application))
    raw.assignment raw.retainedSource
    (PiRLCRetainedGeometry.laterPoseidonFits (retainedGeometry application))
    (retainedEncodes raw).laterPoseidon

/-- The honest constructor supplies PiRLC values through the PiCCS owner forms. -/
theorem productValuesPreserve {application : Program} (raw : RawValues application)
    (invocation : Fin PiRLCProductSchedule.invocationCount) :
    (PiRLCValueWiring.form (piCcsOrdinaryGeometry application) invocation).eval
        raw.assignment =
      PiRLCProductPlan.baseEnv application raw.base
        ((PiRLCProductSchedule.descriptor invocation).valueColumn
          (PiRLCProductSchedule.descriptor invocation).lane) :=
  PiRLCValueWiring.form_eval_source (piCcsOrdinaryGeometry application)
    invocation raw.assignment raw.base raw.groupValue raw.products
    (piCcsOrdinaryEncodes raw)

private theorem directPrefixEncodes {application : Program}
    (raw : RawValues application) :
    DirectPrefixPlan.Encodes
      (DirectPiDECPrefixPlan.piCcsPayload (piDecGeometry application))
      (DirectPiDECPrefixPlan.piRlcValues (piDecGeometry application))
      (poseidonGeometry application) raw.assignment
      raw.base raw.groupValue raw.products where
  retained := retainedEncodes raw
  piRlcValues := productValuesPreserve raw
  pilotPriorInput := (position10 raw).encodes
    (PiRLCPoseidonGeometry.priorInputFits
      (DirectPrefixPlan.pilotGeometry (poseidonGeometry application)))
  pilotOutputInput := (position11 raw).encodes
    (PiRLCPoseidonGeometry.outputInputFits
      (DirectPrefixPlan.pilotGeometry (poseidonGeometry application)))
  payload := by
    intro index
    exact PiCCSPayloadWiring.form_eval_source (piCcsOrdinaryGeometry application)
      index raw.assignment raw.base raw.groupValue raw.products
      (piCcsOrdinaryEncodes raw) (PerApplicationCanonicalAssignment.assignment_one raw)

/-- Canonical assignment evidence through the running-instance transition. -/
theorem runningPrefixEncodes {application : Program} (raw : RawValues application) :
    DirectRunningPrefixPlan.Encodes
      (DirectPiDECPrefixPlan.piCcsPayload (piDecGeometry application))
      (DirectPiDECPrefixPlan.piRlcValues (piDecGeometry application))
      (runningGeometry application)
      raw.assignment raw.base raw.groupValue raw.products :=
  ⟨directPrefixEncodes raw, transitionEncodes raw⟩

private theorem pilotAddedEncodes {application : Program}
    (raw : RawValues application) :
    PilotOrdinaryRetainedGeometry.Encodes (pilotOrdinaryGeometry application)
      raw.assignment raw.retainedSource where
  canonicalLocal := (position24 raw).encodes
    (PilotOrdinaryRetainedGeometry.canonicalLocalFits
      (pilotOrdinaryGeometry application))
  canonicalFresh := (position25 raw).encodes
    (PilotOrdinaryRetainedGeometry.canonicalFreshFits
      (pilotOrdinaryGeometry application))
  outputDigest := (position26 raw).encodes
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
  parentCommitment := PiDECValueWiring.parentCommitmentEncodes
    (piDecGeometry application) raw.assignment raw.retainedSource
    (retainedEncodes raw).productOutput
  parentPublicInput := PiDECValueWiring.parentPublicInputEncodes
    (piDecGeometry application) raw.assignment raw.retainedSource
    (retainedEncodes raw).productOutput
  parentEvalK := PiDECValueWiring.parentEvalKEncodes
    (piDecGeometry application) raw.assignment raw.retainedSource
    (retainedEncodes raw).productOutput
  parentEvalA := PiDECValueWiring.parentEvalAEncodes
    (piDecGeometry application) raw.assignment raw.retainedSource
    (retainedEncodes raw).productOutput
  proof := (position15 raw).encodes
    (PiDECRetainedGeometry.proofFits (piDecGeometry application))
  logical := (position27 raw).encodes
    (PiDECRetainedGeometry.logicalFits (piDecGeometry application))
  fresh := (position28 raw).encodes
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
  logical := (position29 raw).encodes
    (PiRLCSamplerOrdinaryRetainedGeometry.logicalFits
      (samplerGeometry application))
  fresh := (position30 raw).encodes
    (PiRLCSamplerOrdinaryRetainedGeometry.freshFits
      (samplerGeometry application))

/-- Canonical assignment evidence through the complete sampler prefix. -/
theorem samplerPrefixEncodes {application : Program} (raw : RawValues application) :
    DirectPiRLCSamplerCompletePrefixPlan.Encodes (samplerGeometry application)
      raw.assignment raw.base raw.groupValue raw.products :=
  ⟨piDecPrefixEncodes raw, samplerOrdinaryEncodes raw⟩

private def position31 {application : Program} (raw : RawValues application) :
    Position raw (ApplicationRetainedGeometry.witnessStart application)
      (ApplicationRetainedBlocks.sourceWidth application)
      (ApplicationRetainedBlocks.witnessBlock application)
      raw.applicationSource where
  cursor := (position30 raw).next
  after := tail raw 32
  head := by
    simp [Position.next, Cursor.next, position30, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

private def position32 {application : Program} (raw : RawValues application) :
    Position raw (ApplicationRetainedGeometry.localStart application)
      (ApplicationRetainedBlocks.sourceWidth application)
      (ApplicationRetainedBlocks.localBlock application)
      raw.applicationSource where
  cursor := (position31 raw).next
  after := tail raw 33
  head := by
    simp [Position.next, Cursor.next, position31, tail, RawValues.schedule,
      Canonical.ofBlock, CanonicalBlockAssignment.ofBlock]

def applicationGeometry (application : Program) :=
  PerApplicationFixedPoint.geometry application

private def applicationBaseColumn {application : Program}
    (column : Fin (ApplicationRetainedBlocks.sourceWidth application)) :
    Fin (PiRLCProductPlan.baseSourceWidth application) :=
  ⟨column.val, Nat.lt_of_lt_of_le column.isLt
    (DirectApplicationPrefixPlan.applicationSourceWidth_le_baseSourceWidth application)⟩

private theorem retainedSource_applicationColumn {application : Program}
    (raw : RawValues application)
    (column : Fin (ApplicationRetainedBlocks.sourceWidth application)) :
    raw.retainedSource
      (PiRLCRetainedPreservation.baseSourceColumn application
        (applicationBaseColumn column)) = raw.applicationSource column := by
  exact PiRLCRetainedPreservation.sourceAssignment_base application raw.base
    raw.groupValue raw.products (applicationBaseColumn column)

private theorem applicationInputEncodes {application : Program}
    (raw : RawValues application) :
    (ApplicationRetainedBlocks.inputBlock application).EncodesAt
      (ApplicationRetainedGeometry.inputStart application)
      (ApplicationRetainedGeometry.inputFits (applicationGeometry application))
      raw.assignment raw.applicationSource := by
  let parent := PiRLCPoseidonGeometry.priorInputBlock application
  have slots : 35 + 4 ≤ parent.slotCount := by
    simp [parent, PiRLCPoseidonGeometry.priorInputBlock]
  have fits : PiRLCPoseidonGeometry.priorInputStart application +
      35 * parent.kind.width + (parent.slice 35 4 slots).coordinateCount ≤
        PerApplicationFixedPoint.logicalWidth application :=
    ApplicationRetainedGeometry.inputFits (applicationGeometry application)
  have view := parent.encodesAt_slice 35 4 slots
    (PiRLCPoseidonGeometry.priorInputStart application)
    (PiRLCPoseidonGeometry.priorInputFits
      (ApplicationRetainedGeometry.pilotGeometry (applicationGeometry application)))
    fits raw.assignment raw.retainedSource
    ((position10 raw).encodes (PiRLCPoseidonGeometry.priorInputFits
      (ApplicationRetainedGeometry.pilotGeometry (applicationGeometry application))))
  intro slot coordinate
  have selected : parent.source (ApplicationDirectPlan.Location.preimageWord slot) =
      PiRLCRetainedPreservation.baseSourceColumn application
        (applicationBaseColumn ((ApplicationRetainedBlocks.inputBlock application).source
          slot)) := by
    apply Fin.ext
    change 0 + (35 + slot.val) = Layout.Stage1.ApplicationInputs.inputColumn slot
    rw [Nat.zero_add]
    exact (Layout.Stage1.ApplicationInputs.inputColumn_value slot).symm
  have value := view slot coordinate
  change raw.assignment
      ((ApplicationRetainedBlocks.inputBlock application).column
        (ApplicationRetainedGeometry.inputStart application)
        (ApplicationRetainedGeometry.inputFits (applicationGeometry application))
        slot coordinate) =
    LowNormSlot.coordinate .field
      (raw.retainedSource (parent.source (ApplicationDirectPlan.Location.preimageWord slot)))
      coordinate at value
  rw [selected, retainedSource_applicationColumn raw] at value
  exact value

private theorem applicationOutputEncodes {application : Program}
    (raw : RawValues application) :
    (ApplicationRetainedBlocks.outputBlock application).EncodesAt
      (ApplicationRetainedGeometry.outputStart application)
      (ApplicationRetainedGeometry.outputFits (applicationGeometry application))
      raw.assignment raw.applicationSource := by
  let parent := PiRLCPoseidonGeometry.outputInputBlock application
  have slots : 35 + 4 ≤ parent.slotCount := by
    simp [parent, PiRLCPoseidonGeometry.outputInputBlock]
  have fits : PiRLCPoseidonGeometry.outputInputStart application +
      35 * parent.kind.width + (parent.slice 35 4 slots).coordinateCount ≤
        PerApplicationFixedPoint.logicalWidth application :=
    ApplicationRetainedGeometry.outputFits (applicationGeometry application)
  have view := parent.encodesAt_slice 35 4 slots
    (PiRLCPoseidonGeometry.outputInputStart application)
    (PiRLCPoseidonGeometry.outputInputFits
      (ApplicationRetainedGeometry.pilotGeometry (applicationGeometry application)))
    fits raw.assignment raw.retainedSource
    ((position11 raw).encodes (PiRLCPoseidonGeometry.outputInputFits
      (ApplicationRetainedGeometry.pilotGeometry (applicationGeometry application))))
  intro slot coordinate
  have selected : parent.source (ApplicationDirectPlan.Location.preimageWord slot) =
      PiRLCRetainedPreservation.baseSourceColumn application
        (applicationBaseColumn ((ApplicationRetainedBlocks.outputBlock application).source
          slot)) := by
    apply Fin.ext
    change 49393 + (35 + slot.val) = Layout.Stage1.ApplicationInputs.outputColumn slot
    rw [Layout.Stage1.ApplicationInputs.outputColumn_value]
    omega
  have value := view slot coordinate
  change raw.assignment
      ((ApplicationRetainedBlocks.outputBlock application).column
        (ApplicationRetainedGeometry.outputStart application)
        (ApplicationRetainedGeometry.outputFits (applicationGeometry application))
        slot coordinate) =
    LowNormSlot.coordinate .field
      (raw.retainedSource (parent.source (ApplicationDirectPlan.Location.preimageWord slot)))
      coordinate at value
  rw [selected, retainedSource_applicationColumn raw] at value
  exact value

private theorem applicationRetainedEncodes {application : Program}
    (raw : RawValues application) :
    ApplicationRetainedGeometry.Encodes (applicationGeometry application)
      raw.assignment raw.applicationSource where
  input := applicationInputEncodes raw
  witness := (position31 raw).encodes
    (ApplicationRetainedGeometry.witnessFits (applicationGeometry application))
  output := applicationOutputEncodes raw
  localValues := (position32 raw).encodes
    (ApplicationRetainedGeometry.localFits (applicationGeometry application))

/-- One raw packet constructs the complete final assignment-encoding contract.
No caller supplies `DirectApplicationPrefixPlan.Encodes`. -/
theorem encodes {application : Program} (raw : RawValues application) :
    DirectApplicationPrefixPlan.Encodes (applicationGeometry application)
      raw.assignment raw.base raw.groupValue raw.products :=
  ⟨samplerPrefixEncodes raw, applicationRetainedEncodes raw⟩

end NightstreamFPrime.Export.Stage1.PerApplicationCanonicalEncodes
