import Nightstream.Implementation.NebulaV2.Core.LessThanConstantLinkedRows
import Nightstream.Implementation.NebulaV2.Memory.Carry.PublicRows
import Nightstream.Implementation.NebulaV2.Memory.Transition.OpenSegment
import Nightstream.Implementation.NebulaV2.Core.UnsignedAdditionRows
import Nightstream.Implementation.R1CS.Core.EqualityPins

/-!
Contract: exact local rows for the V2 closed-to-active memory transition.

Assurance tier: implementation model.

Owns phase pins, the 64-segment bound, segment-end integer addition,
all deterministic carry copies, all-one product initialization, transcript
input links, transcript challenge-output links, and canonical header links.

Does not own either carry parser, transcript permutation rows, authority
digest placement, precommit extraction, absolute generated columns, or Rust
conformance.

Emits constraints: yes.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.MemoryOpenSegmentRows

open Nightstream.Implementation.NebulaV2.MemoryCarryCodec
open Nightstream.Implementation.NebulaV2.MemoryClaimCodec
open Nightstream.Implementation.R1CS
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.MemoryWireGeometry

structure Layout where
  before : MemoryCarryPublicRows.Layout
  after : MemoryCarryPublicRows.Layout
  transcript : MemoryTranscriptPoseidonRows.Layout
  segmentLimitSlackColumn : Nat
  segmentLimitSlackBitStart : Nat

def Layout.beforeColumn (layout : Layout) (tag : MemoryCarryCodec.FieldTag) :
    Nat :=
  layout.before.carry.fieldColumn tag

def Layout.afterColumn (layout : Layout) (tag : MemoryCarryCodec.FieldTag) :
    Nat :=
  layout.after.carry.fieldColumn tag

def rootPosition (role : RootRole) (lane : Fin 4) : Fin 12 :=
  match role with
  | .operations => ⟨lane.val, lane.isLt.trans_le (by decide)⟩
  | .initialSnapshot => ⟨4 + lane.val, by omega⟩
  | .finalSnapshot => ⟨8 + lane.val, by omega⟩

theorem rootColumns_typed (layout : Layout) :
    layout.transcript.frame.rootColumns =
      (rootRoles.map fun role =>
        List.ofFn fun lane : Fin 4 =>
          layout.transcript.frame.rootColumn (rootPosition role lane)).flatten := by
  apply List.ext_getElem
  · simp [MemoryTranscriptHashFrameRows.Layout.rootColumns, rootRoles]
  · intro index leftBound rightBound
    have indexBound : index < 12 := by
      simpa [MemoryTranscriptHashFrameRows.Layout.rootColumns] using leftBound
    interval_cases index <;>
      simp [MemoryTranscriptHashFrameRows.Layout.rootColumns, rootRoles,
        rootPosition]

def Layout.segmentLimit (layout : Layout) :
    LessThanConstantLinkedRows.Layout :=
  { width := segmentIndexBits
    limit := Lifecycle.maximumSegments
    valueColumn := layout.beforeColumn .segmentIndex
    slackColumn := layout.segmentLimitSlackColumn
    slackBitStart := layout.segmentLimitSlackBitStart }

theorem Layout.segmentLimit_valid (layout : Layout) :
    layout.segmentLimit.Valid where
  limitPositive := by simp [Layout.segmentLimit]; decide
  limitFits := by simp [Layout.segmentLimit]; decide
  sumFits := by simp [Layout.segmentLimit]; decide

def Layout.endAddition (layout : Layout) : UnsignedAdditionRows.Layout :=
  { leftWidth := MemoryWireGeometry.timestampBits
    rightWidth := segmentActiveAccessCountBits
    leftColumn := layout.beforeColumn .globalTimestamp
    rightColumn := layout.afterColumn .segmentActiveAccessCount
    outputColumn := layout.afterColumn .segmentEndTimestamp }

theorem Layout.endAddition_valid (layout : Layout) :
    layout.endAddition.Valid where
  sumFits := by simp [Layout.endAddition]; decide

def productPins (layout : Layout) : List (Nat × Nat) :=
  (List.ofFn fun repetition : Fin 2 =>
    (productRoles.map fun role =>
      List.ofFn fun limb : Fin 2 =>
        (layout.afterColumn (.product repetition role limb),
          if limb = 0 then 1 else 0)).flatten).flatten

def pins (layout : Layout) : List (Nat × Nat) :=
  [ (layout.beforeColumn .phase, 0)
  , (layout.afterColumn .phase, 1)
  , (layout.afterColumn .stepIndex, 0)
  ] ++ productPins layout

def carryPairs (layout : Layout) : List (Nat × Nat) :=
  [ (layout.afterColumn .segmentIndex,
      layout.beforeColumn .segmentIndex)
  , (layout.afterColumn .globalTimestamp,
      layout.beforeColumn .globalTimestamp)
  , (layout.afterColumn .segmentStartTimestamp,
      layout.beforeColumn .globalTimestamp)
  ] ++
    (List.ofFn fun lane : Fin 4 =>
      (layout.afterColumn (.root .memory lane),
        layout.beforeColumn (.root .memory lane)))

def challengePairs (layout : Layout) : List (Nat × Nat) :=
  (List.ofFn fun repetition : Fin 2 =>
    (List.ofFn fun coordinate : Fin 2 =>
      List.ofFn fun limb : Fin 2 =>
        (layout.afterColumn (.challenge repetition coordinate limb),
          layout.transcript.challengeColumn repetition coordinate limb)
      ).flatten).flatten

def transcriptCounterPairs (layout : Layout) : List (Nat × Nat) :=
  [ (layout.transcript.frame.counterColumn 0,
      layout.beforeColumn .segmentIndex)
  , (layout.transcript.frame.counterColumn 1,
      layout.beforeColumn .globalTimestamp)
  , (layout.transcript.frame.counterColumn 2,
      layout.afterColumn .segmentActiveAccessCount)
  , (layout.transcript.frame.counterColumn 3,
      layout.afterColumn .segmentEndTimestamp)
  ]

def transcriptRootPairs (layout : Layout) : List (Nat × Nat) :=
  (rootRoles.map fun role =>
    List.ofFn fun lane : Fin 4 =>
      (layout.transcript.frame.rootColumn (rootPosition role lane),
        layout.afterColumn (.root (.precommit role) lane))).flatten

def seenHeaderPairs (layout : Layout) : List (Nat × Nat) :=
  (rootRoles.map fun role =>
    List.ofFn fun lane : Fin 4 =>
      (layout.afterColumn (.root (.seen role) lane),
        layout.after.carry.headerColumn role lane)).flatten

def equalityPairs (layout : Layout) : List (Nat × Nat) :=
  carryPairs layout ++ challengePairs layout ++
    transcriptCounterPairs layout ++ transcriptRootPairs layout ++
      seenHeaderPairs layout

def rows (layout : Layout) : List Row :=
  ConstantPins.rows (pins layout) ++
    EqualityPins.rows (equalityPairs layout) ++
    LessThanConstantLinkedRows.rows layout.segmentLimit ++
    UnsignedAdditionRows.rows layout.endAddition

theorem productPins_length (layout : Layout) :
    (productPins layout).length = 16 := by
  simp [productPins, productRoles]

theorem pins_length (layout : Layout) :
    (pins layout).length = 19 := by
  simp [pins, productPins_length]

theorem carryPairs_length (layout : Layout) :
    (carryPairs layout).length = 7 := by
  simp [carryPairs]

theorem challengePairs_length (layout : Layout) :
    (challengePairs layout).length = 8 := by
  simp [challengePairs]

theorem transcriptRootPairs_length (layout : Layout) :
    (transcriptRootPairs layout).length = 12 := by
  simp [transcriptRootPairs, rootRoles]

theorem seenHeaderPairs_length (layout : Layout) :
    (seenHeaderPairs layout).length = 12 := by
  simp [seenHeaderPairs, rootRoles]

theorem equalityPairs_length (layout : Layout) :
    (equalityPairs layout).length = 43 := by
  simp [equalityPairs, carryPairs_length, challengePairs_length,
    transcriptCounterPairs, transcriptRootPairs_length,
    seenHeaderPairs_length]

theorem rows_length_exact (layout : Layout) :
    (rows layout).length = 72 := by
  simp [rows, ConstantPins.rows, EqualityPins.rows, pins_length,
    equalityPairs_length, LessThanConstantLinkedRows.rows_length,
    UnsignedAdditionRows.rows_length, Layout.segmentLimit,
    segmentIndexBits]

private theorem pin_values_canonical (layout : Layout) :
    ConstantPins.ValuesCanonical (pins layout) := by
  intro pin member
  simp only [pins, List.mem_append] at member
  rcases member with fixed | product
  · simp only [List.mem_cons, List.not_mem_nil, or_false] at fixed
    rcases fixed with rfl | rfl | rfl <;> norm_num [goldilocksP]
  · simp only [productPins, List.mem_flatten] at product
    rcases product with ⟨repetitionBlock, repetitionMember, product⟩
    rcases List.mem_ofFn.mp repetitionMember with ⟨repetition, rfl⟩
    rcases List.mem_flatten.mp product with
      ⟨roleBlock, roleMember, product⟩
    rcases List.mem_map.mp roleMember with ⟨role, _roleMember, rfl⟩
    rcases List.mem_ofFn.mp product with ⟨limb, rfl⟩
    fin_cases limb <;> norm_num [goldilocksP]

/-- Honest local data for one closed-to-active transition. These fields state
the intended values of the four independent row gadgets. They do not assume
row satisfaction or any transcript challenge output. -/
structure Honest (layout : Layout) (assignment : Nat → Nat) : Prop where
  pins : ∀ pin ∈ MemoryOpenSegmentRows.pins layout,
    assignment pin.1 = pin.2
  equalities : ∀ pair ∈ equalityPairs layout,
    assignment pair.1 = assignment pair.2
  segmentLimit : LessThanConstantLinkedRows.Honest layout.segmentLimit
    assignment (assignment layout.segmentLimit.valueColumn)
  endAddition : UnsignedAdditionRows.Honest layout.endAddition assignment
    (assignment layout.endAddition.leftColumn)
    (assignment layout.endAddition.rightColumn)

/-- Every honest closed-to-active assignment satisfies the exact 72 rows. -/
theorem rows_complete
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (honest : Honest layout assignment) :
    Satisfies (rows layout) assignment := by
  have pinsHold := ConstantPins.complete (pin_values_canonical layout)
    one honest.pins
  have equalitiesHold := EqualityPins.rows_complete canonical one
    honest.equalities
  have limitHolds := LessThanConstantLinkedRows.rows_complete
    (Layout.segmentLimit_valid layout) one honest.segmentLimit
  have additionHolds := UnsignedAdditionRows.rows_complete
    (Layout.endAddition_valid layout) one honest.endAddition
  intro row member
  rw [rows] at member
  rcases List.mem_append.mp member with remaining | additionMember
  · rcases List.mem_append.mp remaining with remaining | limitMember
    · rcases List.mem_append.mp remaining with pinMember | equalityMember
      · exact pinsHold row pinMember
      · exact equalitiesHold row equalityMember
    · exact limitHolds row limitMember
  · exact additionHolds row additionMember

end Nightstream.Implementation.NebulaV2.MemoryOpenSegmentRows
