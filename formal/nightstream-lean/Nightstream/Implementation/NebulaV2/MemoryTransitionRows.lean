import Nightstream.Implementation.NebulaV2.ConditionalEqualityOneRows
import Nightstream.Implementation.NebulaV2.ConditionalEqualityRows
import Nightstream.Implementation.NebulaV2.MemoryCarryPublicRows
import Nightstream.Implementation.NebulaV2.MemoryClaimRows
import Nightstream.Implementation.NebulaV2.UnsignedAdditionRows
import Nightstream.Implementation.NebulaV2.UnsignedLessOrEqualRows
import Nightstream.Implementation.R1CS.Core.ConstantPins
import Nightstream.Implementation.R1CS.Core.EqualityPins

/-!
Contract: exact local row program for one delayed Nebula V2 memory-carry
transition.

Assurance tier: implementation model.

Owns all prior-claim agreement pairs, the active prior-phase pin, the
64-segment bound, exact integer timestamp and counter arithmetic, both branch
pair lists, and the outgoing-phase branch gate.

Does not own the three parser blocks, product-balance multiplication rows,
incoming state-hash authority, full-claim/NIFS authority, product-update rows,
or absolute generated columns.

Emits constraints: yes.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.NebulaV2.MemoryTransitionRows

open Nightstream.Implementation.NebulaV2.MemoryCarryCodec
open Nightstream.Implementation.NebulaV2.MemoryClaimCodec
open Nightstream.Implementation.R1CS
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.Lifecycle
open Nightstream.Protocol.NebulaV2.MemoryWireGeometry

structure Layout where
  before : MemoryCarryPublicRows.Layout
  claim : MemoryClaimRows.Layout
  after : MemoryCarryPublicRows.Layout
  claimsPerSegmentColumn : Nat
  nextStepColumn : Nat
  nextSegmentColumn : Nat
  segmentLimitSlackColumn : Nat
  segmentLimitSlackBitStart : Nat
  startGlobalSlackColumn : Nat
  startGlobalSlackBitStart : Nat
  globalEndSlackColumn : Nat
  globalEndSlackBitStart : Nat
  outputEndSlackColumn : Nat
  outputEndSlackBitStart : Nat

def Layout.beforeColumn (layout : Layout) (tag : MemoryCarryCodec.FieldTag) :
    Nat :=
  layout.before.carry.fieldColumn tag

def Layout.afterColumn (layout : Layout) (tag : MemoryCarryCodec.FieldTag) :
    Nat :=
  layout.after.carry.fieldColumn tag

def Layout.claimCounterColumn (layout : Layout)
    (counter : MemoryClaimCounterRows.Counter) : Nat :=
  layout.claim.counterValueColumn counter

def Layout.claimFieldColumn (layout : Layout)
    (slot : MemoryClaimFieldRows.Slot) : Nat :=
  Relabel.column (layout.claim.fieldColumnMap slot) CanonicalU64.varCol

def Layout.pins (layout : Layout) : List (Nat × Nat) :=
  [(layout.beforeColumn .phase, 1),
    (layout.claimsPerSegmentColumn, claimsPerSegment)]

def Layout.segmentLimit (layout : Layout) :
    LessThanConstantLinkedRows.Layout :=
  { width := segmentIndexBits
    limit := maximumSegments
    valueColumn := layout.beforeColumn .segmentIndex
    slackColumn := layout.segmentLimitSlackColumn
    slackBitStart := layout.segmentLimitSlackBitStart }

theorem Layout.segmentLimit_valid (layout : Layout) :
    layout.segmentLimit.Valid where
  limitPositive := by simp [Layout.segmentLimit]; decide
  limitFits := by simp [Layout.segmentLimit]; decide
  sumFits := by simp [Layout.segmentLimit]; decide

def Layout.segmentEndAddition (layout : Layout) :
    UnsignedAdditionRows.Layout :=
  { leftWidth := MemoryWireGeometry.timestampBits
    rightWidth := segmentActiveAccessCountBits
    leftColumn := layout.beforeColumn .segmentStartTimestamp
    rightColumn := layout.beforeColumn .segmentActiveAccessCount
    outputColumn := layout.beforeColumn .segmentEndTimestamp }

def Layout.timestampAddition (layout : Layout) :
    UnsignedAdditionRows.Layout :=
  { leftWidth := MemoryWireGeometry.timestampBits
    rightWidth := stepActiveAccessCountBits
    leftColumn := layout.claimCounterColumn .timestampIn
    rightColumn := layout.claimCounterColumn .activeAccessCount
    outputColumn := layout.claimCounterColumn .timestampOut }

def Layout.stepAddition (layout : Layout) : UnsignedAdditionRows.Layout :=
  { leftWidth := stepIndexBits
    rightWidth := 1
    leftColumn := layout.beforeColumn .stepIndex
    rightColumn := 0
    outputColumn := layout.nextStepColumn }

def Layout.segmentAddition (layout : Layout) : UnsignedAdditionRows.Layout :=
  { leftWidth := segmentIndexBits
    rightWidth := 1
    leftColumn := layout.beforeColumn .segmentIndex
    rightColumn := 0
    outputColumn := layout.nextSegmentColumn }

theorem Layout.segmentEndAddition_valid (layout : Layout) :
    layout.segmentEndAddition.Valid where
  sumFits := by simp [Layout.segmentEndAddition]; decide

theorem Layout.timestampAddition_valid (layout : Layout) :
    layout.timestampAddition.Valid where
  sumFits := by simp [Layout.timestampAddition]; decide

theorem Layout.stepAddition_valid (layout : Layout) :
    layout.stepAddition.Valid where
  sumFits := by simp [Layout.stepAddition]; decide

theorem Layout.segmentAddition_valid (layout : Layout) :
    layout.segmentAddition.Valid where
  sumFits := by simp [Layout.segmentAddition]; decide

def Layout.startLeGlobal (layout : Layout) :
    UnsignedLessOrEqualRows.Layout :=
  { width := MemoryWireGeometry.timestampBits
    leftColumn := layout.beforeColumn .segmentStartTimestamp
    rightColumn := layout.beforeColumn .globalTimestamp
    slackColumn := layout.startGlobalSlackColumn
    slackBitStart := layout.startGlobalSlackBitStart }

def Layout.globalLeEnd (layout : Layout) :
    UnsignedLessOrEqualRows.Layout :=
  { width := MemoryWireGeometry.timestampBits
    leftColumn := layout.beforeColumn .globalTimestamp
    rightColumn := layout.beforeColumn .segmentEndTimestamp
    slackColumn := layout.globalEndSlackColumn
    slackBitStart := layout.globalEndSlackBitStart }

def Layout.outputLeEnd (layout : Layout) :
    UnsignedLessOrEqualRows.Layout :=
  { width := MemoryWireGeometry.timestampBits
    leftColumn := layout.claimCounterColumn .timestampOut
    rightColumn := layout.beforeColumn .segmentEndTimestamp
    slackColumn := layout.outputEndSlackColumn
    slackBitStart := layout.outputEndSlackBitStart }

theorem Layout.startLeGlobal_valid (layout : Layout) :
    layout.startLeGlobal.Valid where
  sumFits := by simp [Layout.startLeGlobal]; decide

theorem Layout.globalLeEnd_valid (layout : Layout) :
    layout.globalLeEnd.Valid where
  sumFits := by simp [Layout.globalLeEnd]; decide

theorem Layout.outputLeEnd_valid (layout : Layout) :
    layout.outputLeEnd.Valid where
  sumFits := by simp [Layout.outputLeEnd]; decide

def challengePairs (layout : Layout) : List (Nat × Nat) :=
  (List.ofFn fun repetition : Fin 2 =>
    (List.ofFn fun coordinate : Fin 2 =>
      List.ofFn fun limb : Fin 2 =>
        (layout.claimFieldColumn (.challenge repetition coordinate limb),
          layout.beforeColumn (.challenge repetition coordinate limb))).flatten
    ).flatten

def productBeforePairs (layout : Layout) : List (Nat × Nat) :=
  (List.ofFn fun repetition : Fin 2 =>
    (productRoles.map fun role =>
      List.ofFn fun limb : Fin 2 =>
        (layout.claimFieldColumn (.product 0 repetition role limb),
          layout.beforeColumn (.product repetition role limb))).flatten
    ).flatten

def rootMatchPairs (layout : Layout) (stage : RootStage)
    (source : RootRole → RootSource) : List (Nat × Nat) :=
  (rootRoles.map fun role =>
    List.ofFn fun lane : Fin 4 =>
      (layout.claimFieldColumn (.root stage role lane),
        layout.beforeColumn (.root (source role) lane))).flatten

def matchingPairs (layout : Layout) : List (Nat × Nat) :=
  [ (layout.claimCounterColumn .segmentIndex,
      layout.beforeColumn .segmentIndex)
  , (layout.claimCounterColumn .stepIndex,
      layout.beforeColumn .stepIndex)
  , (layout.claimCounterColumn .timestampIn,
      layout.beforeColumn .globalTimestamp)
  , (layout.claimCounterColumn .segmentStartTimestamp,
      layout.beforeColumn .segmentStartTimestamp)
  , (layout.claimCounterColumn .segmentEndTimestamp,
      layout.beforeColumn .segmentEndTimestamp)
  ] ++ challengePairs layout ++
    rootMatchPairs layout .precommit RootSource.precommit ++
    rootMatchPairs layout .seenBefore RootSource.seen ++
    productBeforePairs layout

def productAfterPairs (layout : Layout) : List (Nat × Nat) :=
  (List.ofFn fun repetition : Fin 2 =>
    (productRoles.map fun role =>
      List.ofFn fun limb : Fin 2 =>
        (layout.afterColumn (.product repetition role limb),
          layout.claimFieldColumn (.product 1 repetition role limb))).flatten
    ).flatten

def afterClaimRootPairs (layout : Layout)
    (source : RootRole → RootSource)
    (stage : RootStage) : List (Nat × Nat) :=
  (rootRoles.map fun role =>
    List.ofFn fun lane : Fin 4 =>
      (layout.afterColumn (.root (source role) lane),
        layout.claimFieldColumn (.root stage role lane))).flatten

def afterBeforeRootPairs (layout : Layout)
    (source : RootRole → RootSource) :
    List (Nat × Nat) :=
  (rootRoles.map fun role =>
    List.ofFn fun lane : Fin 4 =>
      (layout.afterColumn (.root (source role) lane),
        layout.beforeColumn (.root (source role) lane))).flatten

def afterBeforeChallengePairs (layout : Layout) : List (Nat × Nat) :=
  (List.ofFn fun repetition : Fin 2 =>
    (List.ofFn fun coordinate : Fin 2 =>
      List.ofFn fun limb : Fin 2 =>
        (layout.afterColumn (.challenge repetition coordinate limb),
          layout.beforeColumn (.challenge repetition coordinate limb))).flatten
    ).flatten

def memoryRootPair (layout : Layout) : List (Nat × Nat) :=
  List.ofFn fun lane : Fin 4 =>
    (layout.afterColumn (.root .memory lane),
      layout.beforeColumn (.root .memory lane))

def interiorPairs (layout : Layout) : List (Nat × Nat) :=
  [ (layout.afterColumn .segmentIndex,
      layout.beforeColumn .segmentIndex)
  , (layout.afterColumn .stepIndex, layout.nextStepColumn)
  , (layout.afterColumn .globalTimestamp,
      layout.claimCounterColumn .timestampOut)
  , (layout.afterColumn .segmentStartTimestamp,
      layout.beforeColumn .segmentStartTimestamp)
  , (layout.afterColumn .segmentActiveAccessCount,
      layout.beforeColumn .segmentActiveAccessCount)
  , (layout.afterColumn .segmentEndTimestamp,
      layout.beforeColumn .segmentEndTimestamp)
  ] ++ afterBeforeChallengePairs layout ++ productAfterPairs layout ++
    afterBeforeRootPairs layout RootSource.precommit ++
    afterClaimRootPairs layout RootSource.seen .seenAfter ++
    memoryRootPair layout

def seenEqualsPrecommitPairs (layout : Layout) : List (Nat × Nat) :=
  (rootRoles.map fun role =>
    List.ofFn fun lane : Fin 4 =>
      (layout.claimFieldColumn (.root .seenAfter role lane),
        layout.claimFieldColumn (.root .precommit role lane))).flatten

def initialEqualsMemoryPairs (layout : Layout) : List (Nat × Nat) :=
  List.ofFn fun lane : Fin 4 =>
    (layout.claimFieldColumn (.root .seenAfter .initialSnapshot lane),
      layout.beforeColumn (.root .memory lane))

def closeMemoryRootPairs (layout : Layout) : List (Nat × Nat) :=
  List.ofFn fun lane : Fin 4 =>
    (layout.afterColumn (.root .memory lane),
      layout.claimFieldColumn (.root .seenAfter .finalSnapshot lane))

def closePairs (layout : Layout) : List (Nat × Nat) :=
  [ (layout.nextStepColumn, layout.claimsPerSegmentColumn)
  , (layout.afterColumn .segmentIndex, layout.nextSegmentColumn)
  , (layout.afterColumn .globalTimestamp,
      layout.claimCounterColumn .timestampOut)
  , (layout.claimCounterColumn .timestampOut,
      layout.beforeColumn .segmentEndTimestamp)
  ] ++ seenEqualsPrecommitPairs layout ++
    initialEqualsMemoryPairs layout ++ closeMemoryRootPairs layout

def rows (layout : Layout) : List Row :=
  ConstantPins.rows layout.pins ++
    EqualityPins.rows (matchingPairs layout) ++
    LessThanConstantLinkedRows.rows layout.segmentLimit ++
    UnsignedAdditionRows.rows layout.segmentEndAddition ++
    UnsignedAdditionRows.rows layout.timestampAddition ++
    UnsignedAdditionRows.rows layout.stepAddition ++
    UnsignedAdditionRows.rows layout.segmentAddition ++
    UnsignedLessOrEqualRows.rows layout.startLeGlobal ++
    UnsignedLessOrEqualRows.rows layout.globalLeEnd ++
    UnsignedLessOrEqualRows.rows layout.outputLeEnd ++
    ConditionalEqualityOneRows.rows (layout.afterColumn .phase)
      (interiorPairs layout) ++
    ConditionalEqualityRows.rows (layout.afterColumn .phase)
      (closePairs layout)

theorem challengePairs_length (layout : Layout) :
    (challengePairs layout).length = 8 := by simp [challengePairs]

theorem productBeforePairs_length (layout : Layout) :
    (productBeforePairs layout).length = 16 := by
  simp [productBeforePairs, productRoles]

theorem rootMatchPairs_length (layout : Layout) (stage : RootStage)
    (source : RootRole → RootSource) :
    (rootMatchPairs layout stage source).length = 12 := by
  simp [rootMatchPairs, rootRoles]

theorem matchingPairs_length (layout : Layout) :
    (matchingPairs layout).length = 53 := by
  simp [matchingPairs, challengePairs_length, productBeforePairs_length,
    rootMatchPairs_length]

theorem productAfterPairs_length (layout : Layout) :
    (productAfterPairs layout).length = 16 := by
  simp [productAfterPairs, productRoles]

theorem afterClaimRootPairs_length (layout : Layout)
    (source : RootRole → RootSource) (stage : RootStage) :
    (afterClaimRootPairs layout source stage).length = 12 := by
  simp [afterClaimRootPairs, rootRoles]

theorem afterBeforeRootPairs_length (layout : Layout)
    (source : RootRole → RootSource) :
    (afterBeforeRootPairs layout source).length = 12 := by
  simp [afterBeforeRootPairs, rootRoles]

theorem afterBeforeChallengePairs_length (layout : Layout) :
    (afterBeforeChallengePairs layout).length = 8 := by
  simp [afterBeforeChallengePairs]

theorem interiorPairs_length (layout : Layout) :
    (interiorPairs layout).length = 58 := by
  simp [interiorPairs, afterBeforeChallengePairs_length,
    productAfterPairs_length, afterBeforeRootPairs_length,
    afterClaimRootPairs_length, memoryRootPair]

theorem seenEqualsPrecommitPairs_length (layout : Layout) :
    (seenEqualsPrecommitPairs layout).length = 12 := by
  simp [seenEqualsPrecommitPairs, rootRoles]

theorem closePairs_length (layout : Layout) :
    (closePairs layout).length = 24 := by
  simp [closePairs, seenEqualsPrecommitPairs_length,
    initialEqualsMemoryPairs, closeMemoryRootPairs]

theorem rows_length_exact (layout : Layout) :
    (rows layout).length = 225 := by
  simp [rows, Layout.pins, ConstantPins.rows, EqualityPins.rows,
    matchingPairs_length, LessThanConstantLinkedRows.rows_length,
    UnsignedAdditionRows.rows_length, UnsignedLessOrEqualRows.rows_length,
    ConditionalEqualityOneRows.rows_length,
    ConditionalEqualityRows.rows_length, interiorPairs_length,
    closePairs_length, Layout.segmentLimit, Layout.startLeGlobal,
    Layout.globalLeEnd, Layout.outputLeEnd,
    MemoryWireGeometry.timestampBits, segmentIndexBits]

end Nightstream.Implementation.NebulaV2.MemoryTransitionRows
