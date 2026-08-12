import Nightstream.Implementation.NebulaV2.MemoryOpenSegmentRows

/-!
Contract: semantic soundness of the exact V2 segment-open row program.

Assurance tier: implementation model.

Owns extraction of the closed prior carry, construction of the exact
transcript input, row-derived fixed Poseidon2 challenges, all-one products,
canonical headers, range conditions, and equality with the specialized
`MemoryOpenSegment.openCarry` transition.

Does not own incoming state-hash authority, precommit sequence extraction,
Fiat--Shamir unpredictability, absolute generated columns, or Rust
conformance.

Emits constraints: no new rows. It interprets `MemoryOpenSegmentRows.rows`.
-/

set_option autoImplicit false
set_option maxRecDepth 10000
set_option maxHeartbeats 5000000

namespace Nightstream.Implementation.NebulaV2.MemoryOpenSegmentSound

open Nightstream.Implementation.NebulaV2.MemoryCarryCodec
open Nightstream.Implementation.NebulaV2.MemoryClaimCodec
open Nightstream.Implementation.NebulaV2.MemoryOpenSegmentRows
open Nightstream.Implementation.R1CS
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.Protocol.NebulaV2.MemoryWireGeometry
open Nightstream.Protocol.NebulaV2.ProductState
open Nightstream.SuperNeo.Concrete

local instance concreteKOne : One K := ⟨K.one⟩

def closedOfWire (value : MemoryCarryCodec.Value) :
    ClosedCarry Digest.Value :=
  { segmentIndex := value.segmentIndex
    globalTimestamp := value.globalTimestamp
    memoryRoot := value.memoryRoot }

def activeOfWire
    (value : MemoryCarryCodec.Value)
    (stepBound : value.stepIndex < Lifecycle.claimsPerSegment) :
    ActiveCarry Digest.Value (Challenges K) (State K) :=
  { segmentIndex := value.segmentIndex
    stepIndex := ⟨value.stepIndex, stepBound⟩
    globalTimestamp := value.globalTimestamp
    segmentStartTimestamp := value.segmentStartTimestamp
    segmentActiveAccessCount := value.segmentActiveAccessCount
    segmentEndTimestamp := value.segmentEndTimestamp
    challenge := value.challenges
    products := value.products
    dPre := value.dPre
    dSeen := value.dSeen
    memoryRoot := value.memoryRoot }

/-- The enclosing relation must derive these 28 fields from verifier-owned
statement and recursive-state sources. This predicate contains no challenge,
root, counter, or row-satisfaction conclusion. -/
structure AuthorityPlaced
    (layout : MemoryOpenSegmentRows.Layout) (assignment : Nat → Nat)
    (authority : MemoryOpenSegment.Authority) : Prop where
  fields : layout.transcript.frame.authorityColumns.map assignment =
    authority.digestFields

/-- One assignment and one ordered authority-column frame can place at most
one typed authority. -/
theorem AuthorityPlaced.unique
    {layout : MemoryOpenSegmentRows.Layout} {assignment : Nat → Nat}
    {left right : MemoryOpenSegment.Authority}
    (leftPlaced : AuthorityPlaced layout assignment left)
    (rightPlaced : AuthorityPlaced layout assignment right) :
    left = right := by
  apply MemoryOpenSegment.Authority.digestFields_injective
  exact leftPlaced.fields.symm.trans rightPlaced.fields

def authorityPosition (digest : Fin 7) (lane : Fin 4) : Fin 28 :=
  ⟨digest.val * 4 + lane.val, by omega⟩

/-- Pointwise lane authority is sufficient for the exact ordered 28-field
frame. This theorem prevents callers from hiding a reordered digest list in
one aggregate equality. -/
theorem authorityPlaced_of_lanes
    {layout : MemoryOpenSegmentRows.Layout} {assignment : Nat -> Nat}
    {authority : MemoryOpenSegment.Authority}
    (lanes : forall digest : Fin 7, forall lane : Fin 4,
      assignment
          (layout.transcript.frame.authorityColumn
            (authorityPosition digest lane)) =
        ((authority.digestAt digest).lanes lane).val) :
    AuthorityPlaced layout assignment authority := by
  constructor
  rw [MemoryOpenSegment.Authority.digestFields_eq_indexed]
  apply List.ext_getElem
  · simp [MemoryTranscriptHashFrameRows.Layout.authorityColumns,
      MemoryTranscriptHashFrame.digestFields, Digest.laneCount]
  · intro index leftBound rightBound
    have indexBound : index < 28 := by
      simpa [MemoryTranscriptHashFrameRows.Layout.authorityColumns]
        using leftBound
    interval_cases index
    all_goals
      first
      | simpa [MemoryTranscriptHashFrameRows.Layout.authorityColumns,
          MemoryTranscriptHashFrame.digestFields,
          MemoryOpenSegment.Authority.digestAt, authorityPosition] using
          lanes (0 : Fin 7) (0 : Fin 4)
      | simpa [MemoryTranscriptHashFrameRows.Layout.authorityColumns,
          MemoryTranscriptHashFrame.digestFields,
          MemoryOpenSegment.Authority.digestAt, authorityPosition] using
          lanes (0 : Fin 7) (1 : Fin 4)
      | simpa [MemoryTranscriptHashFrameRows.Layout.authorityColumns,
          MemoryTranscriptHashFrame.digestFields,
          MemoryOpenSegment.Authority.digestAt, authorityPosition] using
          lanes (0 : Fin 7) (2 : Fin 4)
      | simpa [MemoryTranscriptHashFrameRows.Layout.authorityColumns,
          MemoryTranscriptHashFrame.digestFields,
          MemoryOpenSegment.Authority.digestAt, authorityPosition] using
          lanes (0 : Fin 7) (3 : Fin 4)
      | simpa [MemoryTranscriptHashFrameRows.Layout.authorityColumns,
          MemoryTranscriptHashFrame.digestFields,
          MemoryOpenSegment.Authority.digestAt, authorityPosition] using
          lanes (1 : Fin 7) (0 : Fin 4)
      | simpa [MemoryTranscriptHashFrameRows.Layout.authorityColumns,
          MemoryTranscriptHashFrame.digestFields,
          MemoryOpenSegment.Authority.digestAt, authorityPosition] using
          lanes (1 : Fin 7) (1 : Fin 4)
      | simpa [MemoryTranscriptHashFrameRows.Layout.authorityColumns,
          MemoryTranscriptHashFrame.digestFields,
          MemoryOpenSegment.Authority.digestAt, authorityPosition] using
          lanes (1 : Fin 7) (2 : Fin 4)
      | simpa [MemoryTranscriptHashFrameRows.Layout.authorityColumns,
          MemoryTranscriptHashFrame.digestFields,
          MemoryOpenSegment.Authority.digestAt, authorityPosition] using
          lanes (1 : Fin 7) (3 : Fin 4)
      | simpa [MemoryTranscriptHashFrameRows.Layout.authorityColumns,
          MemoryTranscriptHashFrame.digestFields,
          MemoryOpenSegment.Authority.digestAt, authorityPosition] using
          lanes (2 : Fin 7) (0 : Fin 4)
      | simpa [MemoryTranscriptHashFrameRows.Layout.authorityColumns,
          MemoryTranscriptHashFrame.digestFields,
          MemoryOpenSegment.Authority.digestAt, authorityPosition] using
          lanes (2 : Fin 7) (1 : Fin 4)
      | simpa [MemoryTranscriptHashFrameRows.Layout.authorityColumns,
          MemoryTranscriptHashFrame.digestFields,
          MemoryOpenSegment.Authority.digestAt, authorityPosition] using
          lanes (2 : Fin 7) (2 : Fin 4)
      | simpa [MemoryTranscriptHashFrameRows.Layout.authorityColumns,
          MemoryTranscriptHashFrame.digestFields,
          MemoryOpenSegment.Authority.digestAt, authorityPosition] using
          lanes (2 : Fin 7) (3 : Fin 4)
      | simpa [MemoryTranscriptHashFrameRows.Layout.authorityColumns,
          MemoryTranscriptHashFrame.digestFields,
          MemoryOpenSegment.Authority.digestAt, authorityPosition] using
          lanes (3 : Fin 7) (0 : Fin 4)
      | simpa [MemoryTranscriptHashFrameRows.Layout.authorityColumns,
          MemoryTranscriptHashFrame.digestFields,
          MemoryOpenSegment.Authority.digestAt, authorityPosition] using
          lanes (3 : Fin 7) (1 : Fin 4)
      | simpa [MemoryTranscriptHashFrameRows.Layout.authorityColumns,
          MemoryTranscriptHashFrame.digestFields,
          MemoryOpenSegment.Authority.digestAt, authorityPosition] using
          lanes (3 : Fin 7) (2 : Fin 4)
      | simpa [MemoryTranscriptHashFrameRows.Layout.authorityColumns,
          MemoryTranscriptHashFrame.digestFields,
          MemoryOpenSegment.Authority.digestAt, authorityPosition] using
          lanes (3 : Fin 7) (3 : Fin 4)
      | simpa [MemoryTranscriptHashFrameRows.Layout.authorityColumns,
          MemoryTranscriptHashFrame.digestFields,
          MemoryOpenSegment.Authority.digestAt, authorityPosition] using
          lanes (4 : Fin 7) (0 : Fin 4)
      | simpa [MemoryTranscriptHashFrameRows.Layout.authorityColumns,
          MemoryTranscriptHashFrame.digestFields,
          MemoryOpenSegment.Authority.digestAt, authorityPosition] using
          lanes (4 : Fin 7) (1 : Fin 4)
      | simpa [MemoryTranscriptHashFrameRows.Layout.authorityColumns,
          MemoryTranscriptHashFrame.digestFields,
          MemoryOpenSegment.Authority.digestAt, authorityPosition] using
          lanes (4 : Fin 7) (2 : Fin 4)
      | simpa [MemoryTranscriptHashFrameRows.Layout.authorityColumns,
          MemoryTranscriptHashFrame.digestFields,
          MemoryOpenSegment.Authority.digestAt, authorityPosition] using
          lanes (4 : Fin 7) (3 : Fin 4)
      | simpa [MemoryTranscriptHashFrameRows.Layout.authorityColumns,
          MemoryTranscriptHashFrame.digestFields,
          MemoryOpenSegment.Authority.digestAt, authorityPosition] using
          lanes (5 : Fin 7) (0 : Fin 4)
      | simpa [MemoryTranscriptHashFrameRows.Layout.authorityColumns,
          MemoryTranscriptHashFrame.digestFields,
          MemoryOpenSegment.Authority.digestAt, authorityPosition] using
          lanes (5 : Fin 7) (1 : Fin 4)
      | simpa [MemoryTranscriptHashFrameRows.Layout.authorityColumns,
          MemoryTranscriptHashFrame.digestFields,
          MemoryOpenSegment.Authority.digestAt, authorityPosition] using
          lanes (5 : Fin 7) (2 : Fin 4)
      | simpa [MemoryTranscriptHashFrameRows.Layout.authorityColumns,
          MemoryTranscriptHashFrame.digestFields,
          MemoryOpenSegment.Authority.digestAt, authorityPosition] using
          lanes (5 : Fin 7) (3 : Fin 4)
      | simpa [MemoryTranscriptHashFrameRows.Layout.authorityColumns,
          MemoryTranscriptHashFrame.digestFields,
          MemoryOpenSegment.Authority.digestAt, authorityPosition] using
          lanes (6 : Fin 7) (0 : Fin 4)
      | simpa [MemoryTranscriptHashFrameRows.Layout.authorityColumns,
          MemoryTranscriptHashFrame.digestFields,
          MemoryOpenSegment.Authority.digestAt, authorityPosition] using
          lanes (6 : Fin 7) (1 : Fin 4)
      | simpa [MemoryTranscriptHashFrameRows.Layout.authorityColumns,
          MemoryTranscriptHashFrame.digestFields,
          MemoryOpenSegment.Authority.digestAt, authorityPosition] using
          lanes (6 : Fin 7) (2 : Fin 4)
      | simpa [MemoryTranscriptHashFrameRows.Layout.authorityColumns,
          MemoryTranscriptHashFrame.digestFields,
          MemoryOpenSegment.Authority.digestAt, authorityPosition] using
          lanes (6 : Fin 7) (3 : Fin 4)

private theorem subrows_hold
    {layout : MemoryOpenSegmentRows.Layout} {assignment : Nat → Nat}
    (holds : Satisfies (MemoryOpenSegmentRows.rows layout) assignment)
    (subrows : List Row)
    (included : ∀ row ∈ subrows,
      row ∈ MemoryOpenSegmentRows.rows layout) :
    Satisfies subrows assignment := by
  intro row member
  exact holds row (included row member)

private theorem pins_hold
    {layout : MemoryOpenSegmentRows.Layout} {assignment : Nat → Nat}
    (holds : Satisfies (MemoryOpenSegmentRows.rows layout) assignment) :
    Satisfies (ConstantPins.rows (pins layout)) assignment :=
  subrows_hold holds _ (by
    intro row member
    simp [MemoryOpenSegmentRows.rows, member])

private theorem equalities_hold
    {layout : MemoryOpenSegmentRows.Layout} {assignment : Nat → Nat}
    (holds : Satisfies (MemoryOpenSegmentRows.rows layout) assignment) :
    Satisfies (EqualityPins.rows (equalityPairs layout)) assignment :=
  subrows_hold holds _ (by
    intro row member
    simp [MemoryOpenSegmentRows.rows, member])

private theorem segment_limit_hold
    {layout : MemoryOpenSegmentRows.Layout} {assignment : Nat → Nat}
    (holds : Satisfies (MemoryOpenSegmentRows.rows layout) assignment) :
    Satisfies (LessThanConstantLinkedRows.rows layout.segmentLimit)
      assignment :=
  subrows_hold holds _ (by
    intro row member
    simp [MemoryOpenSegmentRows.rows, member])

private theorem end_addition_hold
    {layout : MemoryOpenSegmentRows.Layout} {assignment : Nat → Nat}
    (holds : Satisfies (MemoryOpenSegmentRows.rows layout) assignment) :
    Satisfies (UnsignedAdditionRows.rows layout.endAddition) assignment :=
  subrows_hold holds _ (by
    intro row member
    simp [MemoryOpenSegmentRows.rows, member])

private theorem pins_values_canonical
    (layout : MemoryOpenSegmentRows.Layout) :
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

private theorem rowsIncluded_self (program : List Row) :
    rowsIncluded program program = true := by
  rw [rowsIncluded, List.all_eq_true]
  intro row member
  exact decide_eq_true member

theorem pin_facts
    {layout : MemoryOpenSegmentRows.Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (MemoryOpenSegmentRows.rows layout) assignment) :
    ∀ pin ∈ pins layout, assignment pin.1 = pin.2 := by
  exact ConstantPins.sound (pins_values_canonical layout)
    (rowsIncluded_self (ConstantPins.rows (pins layout))) canonical one
    (pins_hold holds)

theorem equality_facts
    {layout : MemoryOpenSegmentRows.Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (MemoryOpenSegmentRows.rows layout) assignment) :
    ∀ pair ∈ equalityPairs layout,
      assignment pair.1 = assignment pair.2 :=
  EqualityPins.rows_sound canonical one (equalities_hold holds)

private theorem phase_before_closed
    {layout : MemoryOpenSegmentRows.Layout} {assignment : Nat → Nat}
    {headers : ChainHeaders Digest.Value}
    {before : MemoryCarryCodec.Value}
    (beforeParsed : MemoryCarryPublicRows.ParsedColumnsMatch layout.before
      assignment headers before)
    (facts : ∀ pin ∈ pins layout, assignment pin.1 = pin.2) :
    before.phase = .closed := by
  have phaseValueZero : before.fieldValue .phase = 0 :=
    (beforeParsed.placed .phase).symm.trans
      (facts (layout.beforeColumn .phase, 0) (by simp [pins]))
  cases phaseExact : before.phase with
  | closed => rfl
  | active =>
      simp [Value.fieldValue, phaseValue, phaseExact] at phaseValueZero

private theorem phase_after_active
    {layout : MemoryOpenSegmentRows.Layout} {assignment : Nat → Nat}
    {headers : ChainHeaders Digest.Value}
    {after : MemoryCarryCodec.Value}
    (afterParsed : MemoryCarryPublicRows.ParsedColumnsMatch layout.after
      assignment headers after)
    (facts : ∀ pin ∈ pins layout, assignment pin.1 = pin.2) :
    after.phase = .active := by
  have phaseValueOne : after.fieldValue .phase = 1 :=
    (afterParsed.placed .phase).symm.trans
      (facts (layout.afterColumn .phase, 1) (by simp [pins]))
  cases phaseExact : after.phase with
  | closed =>
      simp [Value.fieldValue, phaseValue, phaseExact] at phaseValueOne
  | active => rfl

private theorem transcript_counter_value
    {layout : MemoryOpenSegmentRows.Layout} {assignment : Nat → Nat}
    {headers : ChainHeaders Digest.Value}
    {before after : MemoryCarryCodec.Value}
    (beforeParsed : MemoryCarryPublicRows.ParsedColumnsMatch layout.before
      assignment headers before)
    (afterParsed : MemoryCarryPublicRows.ParsedColumnsMatch layout.after
      assignment headers after)
    (equalities : ∀ pair ∈ equalityPairs layout,
      assignment pair.1 = assignment pair.2)
    (index : Fin 4) :
    assignment (layout.transcript.frame.counterColumn index) =
      [before.segmentIndex, before.globalTimestamp,
        after.segmentActiveAccessCount, after.segmentEndTimestamp].getD
          index.val 0 := by
  fin_cases index
  · exact (equalities
      (layout.transcript.frame.counterColumn 0,
        layout.beforeColumn .segmentIndex) (by
      simp [equalityPairs, transcriptCounterPairs])).trans
      (beforeParsed.placed .segmentIndex)
  · exact (equalities
      (layout.transcript.frame.counterColumn 1,
        layout.beforeColumn .globalTimestamp) (by
      simp [equalityPairs, transcriptCounterPairs])).trans
      (beforeParsed.placed .globalTimestamp)
  · exact (equalities
      (layout.transcript.frame.counterColumn 2,
        layout.afterColumn .segmentActiveAccessCount) (by
      simp [equalityPairs, transcriptCounterPairs])).trans
      (afterParsed.placed .segmentActiveAccessCount)
  · exact (equalities
      (layout.transcript.frame.counterColumn 3,
        layout.afterColumn .segmentEndTimestamp) (by
      simp [equalityPairs, transcriptCounterPairs])).trans
      (afterParsed.placed .segmentEndTimestamp)

private theorem transcript_root_value
    {layout : MemoryOpenSegmentRows.Layout} {assignment : Nat → Nat}
    {headers : ChainHeaders Digest.Value}
    {after : MemoryCarryCodec.Value}
    (afterParsed : MemoryCarryPublicRows.ParsedColumnsMatch layout.after
      assignment headers after)
    (equalities : ∀ pair ∈ equalityPairs layout,
      assignment pair.1 = assignment pair.2)
    (role : RootRole) (lane : Fin 4) :
    assignment
        (layout.transcript.frame.rootColumn (rootPosition role lane)) =
      MemoryClaimCodec.rootValue after.dPre role lane := by
  have roleMember : role ∈ rootRoles := by
    cases role <;> simp [rootRoles]
  have rootMember :
      (layout.transcript.frame.rootColumn (rootPosition role lane),
        layout.afterColumn (.root (.precommit role) lane)) ∈
        transcriptRootPairs layout := by
    rw [transcriptRootPairs]
    apply List.mem_flatten.mpr
    refine ⟨List.ofFn fun selectedLane : Fin 4 =>
      (layout.transcript.frame.rootColumn
          (rootPosition role selectedLane),
        layout.afterColumn (.root (.precommit role) selectedLane)), ?_, ?_⟩
    · exact List.mem_map.mpr ⟨role, roleMember, rfl⟩
    · exact List.mem_ofFn.mpr ⟨lane, rfl⟩
  exact (equalities
      (layout.transcript.frame.rootColumn (rootPosition role lane),
        layout.afterColumn (.root (.precommit role) lane)) (by
          simp [equalityPairs, rootMember])).trans
    (afterParsed.placed (.root (.precommit role) lane))

private theorem transcript_variable_placed
    {layout : MemoryOpenSegmentRows.Layout} {assignment : Nat → Nat}
    {headers : ChainHeaders Digest.Value}
    {before after : MemoryCarryCodec.Value}
    {authority : MemoryOpenSegment.Authority}
    (beforeParsed : MemoryCarryPublicRows.ParsedColumnsMatch layout.before
      assignment headers before)
    (afterParsed : MemoryCarryPublicRows.ParsedColumnsMatch layout.after
      assignment headers after)
    (authorityPlaced : AuthorityPlaced layout assignment authority)
    (equalities : ∀ pair ∈ equalityPairs layout,
      assignment pair.1 = assignment pair.2)
    (endExact : after.segmentEndTimestamp =
      before.globalTimestamp + after.segmentActiveAccessCount) :
    MemoryTranscriptHashFrameRows.VariablePlaced layout.transcript.frame
      assignment
      (MemoryOpenSegment.transcriptInput authority (closedOfWire before)
        after.dPre after.segmentActiveAccessCount) := by
  refine
    { authority := ?_
      counters := ?_
      roots := ?_ }
  · rw [MemoryOpenSegment.transcriptInput_authority_fields]
    exact authorityPlaced.fields
  · apply List.ext_getElem
    · simp [MemoryTranscriptHashFrameRows.Layout.counterColumns]
    · intro index leftBound rightBound
      have indexBound : index < 4 := by
        simpa [MemoryTranscriptHashFrameRows.Layout.counterColumns]
          using leftBound
      interval_cases index
      · simpa [MemoryTranscriptHashFrameRows.Layout.counterColumns,
          MemoryOpenSegment.transcriptInput] using
          transcript_counter_value beforeParsed afterParsed equalities
            (0 : Fin 4)
      · simpa [MemoryTranscriptHashFrameRows.Layout.counterColumns,
          MemoryOpenSegment.transcriptInput] using
          transcript_counter_value beforeParsed afterParsed equalities
            (1 : Fin 4)
      · simpa [MemoryTranscriptHashFrameRows.Layout.counterColumns,
          MemoryOpenSegment.transcriptInput] using
          transcript_counter_value beforeParsed afterParsed equalities
            (2 : Fin 4)
      · simpa [MemoryTranscriptHashFrameRows.Layout.counterColumns,
          MemoryOpenSegment.transcriptInput] using
          (transcript_counter_value beforeParsed afterParsed equalities
            (3 : Fin 4)).trans endExact
  · rw [rootColumns_typed]
    simp only [List.map_flatten, List.map_map]
    simp only [MemoryTranscriptHashFrame.rootFields,
      MemoryTranscriptHashFrame.rootDigests,
      MemoryTranscriptHashFrame.encodeDigests,
      MemoryTranscriptHashFrame.digestFields,
      MemoryOpenSegment.transcriptInput, rootRoles, List.flatMap_cons,
      List.flatMap_nil, List.map_cons, List.map_nil, List.flatten_cons,
      List.flatten_nil, List.map_ofFn, Function.comp_def, List.append_nil]
    congr 1
    · apply congrArg List.ofFn
      funext lane
      exact transcript_root_value afterParsed equalities .operations lane
    · congr 1
      · apply congrArg List.ofFn
        funext lane
        exact transcript_root_value afterParsed equalities .initialSnapshot lane
      · apply congrArg List.ofFn
        funext lane
        exact transcript_root_value afterParsed equalities .finalSnapshot lane

private theorem end_timestamp_exact
    {layout : MemoryOpenSegmentRows.Layout} {assignment : Nat → Nat}
    {headers : ChainHeaders Digest.Value}
    {before after : MemoryCarryCodec.Value}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (beforeParsed : MemoryCarryPublicRows.ParsedColumnsMatch layout.before
      assignment headers before)
    (afterParsed : MemoryCarryPublicRows.ParsedColumnsMatch layout.after
      assignment headers after)
    (holds : Satisfies (MemoryOpenSegmentRows.rows layout) assignment) :
    after.segmentEndTimestamp =
      before.globalTimestamp + after.segmentActiveAccessCount := by
  have leftBound :
      assignment (layout.beforeColumn .globalTimestamp) <
        2 ^ MemoryWireGeometry.timestampBits := by
    change assignment
      (layout.before.carry.fieldColumn .globalTimestamp) < _
    rw [beforeParsed.placed .globalTimestamp]
    exact beforeParsed.rowCanonical.globalTimestamp
  have rightBound :
      assignment (layout.afterColumn .segmentActiveAccessCount) <
        2 ^ segmentActiveAccessCountBits := by
    change assignment
      (layout.after.carry.fieldColumn .segmentActiveAccessCount) < _
    rw [afterParsed.placed .segmentActiveAccessCount]
    exact afterParsed.rowCanonical.segmentActiveAccessCount
  have exactSum := UnsignedAdditionRows.output_eq_add
    (Layout.endAddition_valid layout) leftBound rightBound canonical one
    (end_addition_hold holds)
  simp only [Layout.endAddition] at exactSum
  calc
    after.segmentEndTimestamp =
        assignment (layout.afterColumn .segmentEndTimestamp) := by
      exact (afterParsed.placed .segmentEndTimestamp).symm
    _ = assignment (layout.beforeColumn .globalTimestamp) +
          assignment (layout.afterColumn .segmentActiveAccessCount) :=
      exactSum
    _ = before.globalTimestamp + after.segmentActiveAccessCount := by
      have beforePlaced :
          assignment (layout.beforeColumn .globalTimestamp) =
            before.globalTimestamp := beforeParsed.placed .globalTimestamp
      have afterPlaced :
          assignment (layout.afterColumn .segmentActiveAccessCount) =
            after.segmentActiveAccessCount :=
        afterParsed.placed .segmentActiveAccessCount
      rw [beforePlaced, afterPlaced]

private theorem segment_index_in_range
    {layout : MemoryOpenSegmentRows.Layout} {assignment : Nat → Nat}
    {headers : ChainHeaders Digest.Value}
    {before : MemoryCarryCodec.Value}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (beforeParsed : MemoryCarryPublicRows.ParsedColumnsMatch layout.before
      assignment headers before)
    (holds : Satisfies (MemoryOpenSegmentRows.rows layout) assignment) :
    before.segmentIndex < Lifecycle.maximumSegments := by
  have valueBound :
      assignment (layout.beforeColumn .segmentIndex) <
        2 ^ segmentIndexBits := by
    change assignment (layout.before.carry.fieldColumn .segmentIndex) < _
    rw [beforeParsed.placed .segmentIndex]
    exact beforeParsed.rowCanonical.segmentIndex
  have bound := LessThanConstantLinkedRows.value_lt_limit
    (Layout.segmentLimit_valid layout) valueBound canonical one
    (segment_limit_hold holds)
  simp only [Layout.segmentLimit] at bound
  calc
    before.segmentIndex =
        assignment (layout.beforeColumn .segmentIndex) := by
      exact (beforeParsed.placed .segmentIndex).symm
    _ < Lifecycle.maximumSegments := bound

private theorem product_pin_exact
    {layout : MemoryOpenSegmentRows.Layout} {assignment : Nat → Nat}
    (facts : ∀ pin ∈ pins layout, assignment pin.1 = pin.2)
    (repetition : Fin 2) (role : ProductRole) (limb : Fin 2) :
    assignment (layout.afterColumn (.product repetition role limb)) =
      if limb = 0 then 1 else 0 := by
  exact facts
    (layout.afterColumn (.product repetition role limb),
      if limb = 0 then 1 else 0) (by
        fin_cases repetition <;> cases role <;> fin_cases limb <;>
          simp [pins, productPins, productRoles])

private theorem products_are_one
    {layout : MemoryOpenSegmentRows.Layout} {assignment : Nat → Nat}
    {headers : ChainHeaders Digest.Value}
    {after : MemoryCarryCodec.Value}
    (afterParsed : MemoryCarryPublicRows.ParsedColumnsMatch layout.after
      assignment headers after)
    (facts : ∀ pin ∈ pins layout, assignment pin.1 = pin.2) :
    after.products = MemoryCarryCodec.oneProductsK := by
  funext repetition
  apply MemoryClaimCodec.product_eq_of_values
  intro role limb
  calc
    MemoryClaimCodec.productValue (after.products repetition) role limb =
        assignment (layout.afterColumn (.product repetition role limb)) :=
      (afterParsed.placed (.product repetition role limb)).symm
    _ = if limb = 0 then 1 else 0 :=
      product_pin_exact facts repetition role limb
    _ = MemoryClaimCodec.productValue
          (MemoryCarryCodec.oneProductsK repetition) role limb := by
      cases role <;> fin_cases limb <;> rfl

private theorem challenge_link_exact
    {layout : MemoryOpenSegmentRows.Layout} {assignment : Nat → Nat}
    (equalities : ∀ pair ∈ equalityPairs layout,
      assignment pair.1 = assignment pair.2)
    (repetition coordinate limb : Fin 2) :
    assignment (layout.afterColumn
        (.challenge repetition coordinate limb)) =
      assignment
        (layout.transcript.challengeColumn repetition coordinate limb) := by
  exact equalities
    (layout.afterColumn (.challenge repetition coordinate limb),
      layout.transcript.challengeColumn repetition coordinate limb) (by
        fin_cases repetition <;> fin_cases coordinate <;> fin_cases limb <;>
          simp [equalityPairs, challengePairs])

private theorem challenges_are_derived
    {layout : MemoryOpenSegmentRows.Layout} {assignment : Nat → Nat}
    {headers : ChainHeaders Digest.Value}
    {before after : MemoryCarryCodec.Value}
    {authority : MemoryOpenSegment.Authority}
    (valid : layout.transcript.Valid)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (beforeParsed : MemoryCarryPublicRows.ParsedColumnsMatch layout.before
      assignment headers before)
    (afterParsed : MemoryCarryPublicRows.ParsedColumnsMatch layout.after
      assignment headers after)
    (authorityPlaced : AuthorityPlaced layout assignment authority)
    (equalities : ∀ pair ∈ equalityPairs layout,
      assignment pair.1 = assignment pair.2)
    (endExact : after.segmentEndTimestamp =
      before.globalTimestamp + after.segmentActiveAccessCount)
    (transcriptHolds : Satisfies
      (MemoryTranscriptPoseidonRows.rows layout.transcript) assignment) :
    after.challenges =
      MemoryOpenSegment.derive authority (closedOfWire before) after.dPre
        after.segmentActiveAccessCount := by
  have variablePlaced := transcript_variable_placed beforeParsed afterParsed
    authorityPlaced equalities endExact
  have outputExact := MemoryTranscriptPoseidonRows.challenges_exact valid
    canonical one variablePlaced transcriptHolds
  funext repetition
  apply MemoryClaimCodec.challenge_eq_of_values
  intro coordinate limb
  calc
    MemoryClaimCodec.challengeValue (after.challenges repetition)
        coordinate limb =
        assignment (layout.afterColumn
          (.challenge repetition coordinate limb)) :=
      (afterParsed.placed (.challenge repetition coordinate limb)).symm
    _ = assignment
          (layout.transcript.challengeColumn repetition coordinate limb) :=
      challenge_link_exact equalities repetition coordinate limb
    _ = MemoryTranscriptPoseidonRows.pureCoordinate
          (MemoryOpenSegment.transcriptInput authority
            (closedOfWire before) after.dPre
            after.segmentActiveAccessCount)
          (Transcript.coordinateIndex repetition coordinate) limb :=
      outputExact repetition coordinate limb
    _ = MemoryClaimCodec.challengeValue
          (MemoryOpenSegment.derive authority (closedOfWire before)
            after.dPre after.segmentActiveAccessCount repetition)
          coordinate limb := by
      fin_cases coordinate <;> fin_cases limb <;> rfl

/-- The profile-indexed transcript rows derive the challenge for the exact
verifier-selected profile. The selected profile is an input to both the frame
rows and the pure challenge function. -/
private theorem challenges_are_derived_for
    {profile : Profile.Identity}
    {layout : MemoryOpenSegmentRows.Layout} {assignment : Nat → Nat}
    {headers : ChainHeaders Digest.Value}
    {before after : MemoryCarryCodec.Value}
    {authority : MemoryOpenSegment.Authority}
    (profileCanonical : MemoryTranscriptHashFrame.ProfileCanonical profile)
    (transcriptValid :
      MemoryTranscriptPoseidonRows.ProfileIndexed.Valid profile
        layout.transcript)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (beforeParsed : MemoryCarryPublicRows.ParsedColumnsMatch layout.before
      assignment headers before)
    (afterParsed : MemoryCarryPublicRows.ParsedColumnsMatch layout.after
      assignment headers after)
    (authorityPlaced : AuthorityPlaced layout assignment authority)
    (equalities : ∀ pair ∈ equalityPairs layout,
      assignment pair.1 = assignment pair.2)
    (endExact : after.segmentEndTimestamp =
      before.globalTimestamp + after.segmentActiveAccessCount)
    (transcriptHolds : Satisfies
      (MemoryTranscriptPoseidonRows.ProfileIndexed.rows profile
        layout.transcript) assignment) :
    after.challenges =
      MemoryOpenSegment.deriveFor profile authority (closedOfWire before)
        after.dPre after.segmentActiveAccessCount := by
  have variablePlaced := transcript_variable_placed beforeParsed afterParsed
    authorityPlaced equalities endExact
  have outputExact :=
    MemoryTranscriptPoseidonRows.ProfileIndexed.challenges_exact
      transcriptValid profileCanonical canonical one variablePlaced
      transcriptHolds
  funext repetition
  apply MemoryClaimCodec.challenge_eq_of_values
  intro coordinate limb
  calc
    MemoryClaimCodec.challengeValue (after.challenges repetition)
        coordinate limb =
        assignment (layout.afterColumn
          (.challenge repetition coordinate limb)) :=
      (afterParsed.placed (.challenge repetition coordinate limb)).symm
    _ = assignment
          (layout.transcript.challengeColumn repetition coordinate limb) :=
      challenge_link_exact equalities repetition coordinate limb
    _ = MemoryTranscriptPoseidonRows.ProfileIndexed.pureCoordinate profile
          (MemoryOpenSegment.transcriptInput authority
            (closedOfWire before) after.dPre
            after.segmentActiveAccessCount)
          (Transcript.coordinateIndex repetition coordinate) limb :=
      outputExact repetition coordinate limb
    _ = MemoryClaimCodec.challengeValue
          (MemoryOpenSegment.deriveFor profile authority
            (closedOfWire before) after.dPre
            after.segmentActiveAccessCount repetition)
          coordinate limb := by
      fin_cases coordinate <;> fin_cases limb <;> rfl

private theorem carry_segment_index_exact
    {layout : MemoryOpenSegmentRows.Layout} {assignment : Nat → Nat}
    {headers : ChainHeaders Digest.Value}
    {before after : MemoryCarryCodec.Value}
    (beforeParsed : MemoryCarryPublicRows.ParsedColumnsMatch layout.before
      assignment headers before)
    (afterParsed : MemoryCarryPublicRows.ParsedColumnsMatch layout.after
      assignment headers after)
    (equalities : ∀ pair ∈ equalityPairs layout,
      assignment pair.1 = assignment pair.2) :
    after.segmentIndex = before.segmentIndex := by
  calc
    after.segmentIndex = assignment (layout.afterColumn .segmentIndex) :=
      (afterParsed.placed .segmentIndex).symm
    _ = assignment (layout.beforeColumn .segmentIndex) :=
      equalities
        (layout.afterColumn .segmentIndex,
          layout.beforeColumn .segmentIndex) (by
            simp [equalityPairs, carryPairs])
    _ = before.segmentIndex := beforeParsed.placed .segmentIndex

private theorem carry_global_timestamp_exact
    {layout : MemoryOpenSegmentRows.Layout} {assignment : Nat → Nat}
    {headers : ChainHeaders Digest.Value}
    {before after : MemoryCarryCodec.Value}
    (beforeParsed : MemoryCarryPublicRows.ParsedColumnsMatch layout.before
      assignment headers before)
    (afterParsed : MemoryCarryPublicRows.ParsedColumnsMatch layout.after
      assignment headers after)
    (equalities : ∀ pair ∈ equalityPairs layout,
      assignment pair.1 = assignment pair.2) :
    after.globalTimestamp = before.globalTimestamp := by
  calc
    after.globalTimestamp =
        assignment (layout.afterColumn .globalTimestamp) :=
      (afterParsed.placed .globalTimestamp).symm
    _ = assignment (layout.beforeColumn .globalTimestamp) :=
      equalities
        (layout.afterColumn .globalTimestamp,
          layout.beforeColumn .globalTimestamp) (by
            simp [equalityPairs, carryPairs])
    _ = before.globalTimestamp := beforeParsed.placed .globalTimestamp

private theorem carry_segment_start_exact
    {layout : MemoryOpenSegmentRows.Layout} {assignment : Nat → Nat}
    {headers : ChainHeaders Digest.Value}
    {before after : MemoryCarryCodec.Value}
    (beforeParsed : MemoryCarryPublicRows.ParsedColumnsMatch layout.before
      assignment headers before)
    (afterParsed : MemoryCarryPublicRows.ParsedColumnsMatch layout.after
      assignment headers after)
    (equalities : ∀ pair ∈ equalityPairs layout,
      assignment pair.1 = assignment pair.2) :
    after.segmentStartTimestamp = before.globalTimestamp := by
  calc
    after.segmentStartTimestamp =
        assignment (layout.afterColumn .segmentStartTimestamp) :=
      (afterParsed.placed .segmentStartTimestamp).symm
    _ = assignment (layout.beforeColumn .globalTimestamp) :=
      equalities
        (layout.afterColumn .segmentStartTimestamp,
          layout.beforeColumn .globalTimestamp) (by
            simp [equalityPairs, carryPairs])
    _ = before.globalTimestamp := beforeParsed.placed .globalTimestamp

private theorem carry_memory_lane_exact
    {layout : MemoryOpenSegmentRows.Layout} {assignment : Nat → Nat}
    {headers : ChainHeaders Digest.Value}
    {before after : MemoryCarryCodec.Value}
    (beforeParsed : MemoryCarryPublicRows.ParsedColumnsMatch layout.before
      assignment headers before)
    (afterParsed : MemoryCarryPublicRows.ParsedColumnsMatch layout.after
      assignment headers after)
    (equalities : ∀ pair ∈ equalityPairs layout,
      assignment pair.1 = assignment pair.2)
    (lane : Fin 4) :
    (after.memoryRoot.lanes lane).val =
      (before.memoryRoot.lanes lane).val := by
  have linked :
      assignment (layout.afterColumn (.root .memory lane)) =
        assignment (layout.beforeColumn (.root .memory lane)) :=
    equalities
      (layout.afterColumn (.root .memory lane),
        layout.beforeColumn (.root .memory lane)) (by
          fin_cases lane <;> simp [equalityPairs, carryPairs])
  calc
    (after.memoryRoot.lanes lane).val =
        assignment (layout.afterColumn (.root .memory lane)) :=
      (afterParsed.placed (.root .memory lane)).symm
    _ = assignment (layout.beforeColumn (.root .memory lane)) := linked
    _ = (before.memoryRoot.lanes lane).val :=
      beforeParsed.placed (.root .memory lane)

private theorem carry_memory_root_exact
    {layout : MemoryOpenSegmentRows.Layout} {assignment : Nat → Nat}
    {headers : ChainHeaders Digest.Value}
    {before after : MemoryCarryCodec.Value}
    (beforeParsed : MemoryCarryPublicRows.ParsedColumnsMatch layout.before
      assignment headers before)
    (afterParsed : MemoryCarryPublicRows.ParsedColumnsMatch layout.after
      assignment headers after)
    (equalities : ∀ pair ∈ equalityPairs layout,
      assignment pair.1 = assignment pair.2) :
    after.memoryRoot = before.memoryRoot := by
  exact MemoryClaimCodec.digest_eq_of_lane_values
    (carry_memory_lane_exact beforeParsed afterParsed equalities)

private theorem seen_header_link_exact
    {layout : MemoryOpenSegmentRows.Layout} {assignment : Nat → Nat}
    (equalities : ∀ pair ∈ equalityPairs layout,
      assignment pair.1 = assignment pair.2)
    (role : RootRole) (lane : Fin 4) :
    assignment (layout.afterColumn (.root (.seen role) lane)) =
      assignment (layout.after.carry.headerColumn role lane) := by
  exact equalities
    (layout.afterColumn (.root (.seen role) lane),
      layout.after.carry.headerColumn role lane) (by
        cases role <;> fin_cases lane <;>
          simp [equalityPairs, seenHeaderPairs, rootRoles])

private theorem seen_headers_exact
    {layout : MemoryOpenSegmentRows.Layout} {assignment : Nat → Nat}
    {headers : ChainHeaders Digest.Value}
    {after : MemoryCarryCodec.Value}
    (afterParsed : MemoryCarryPublicRows.ParsedColumnsMatch layout.after
      assignment headers after)
    (equalities : ∀ pair ∈ equalityPairs layout,
      assignment pair.1 = assignment pair.2) :
    after.dSeen = headers.roots := by
  apply MemoryClaimCodec.roots_eq_of_values
  intro role lane
  calc
    MemoryClaimCodec.rootValue after.dSeen role lane =
        assignment (layout.afterColumn (.root (.seen role) lane)) :=
      (afterParsed.placed (.root (.seen role) lane)).symm
    _ = assignment (layout.after.carry.headerColumn role lane) :=
      seen_header_link_exact equalities role lane
    _ = MemoryClaimCodec.rootValue headers.roots role lane :=
      afterParsed.headersPlaced role lane

private theorem step_index_zero
    {layout : MemoryOpenSegmentRows.Layout} {assignment : Nat → Nat}
    {headers : ChainHeaders Digest.Value}
    {after : MemoryCarryCodec.Value}
    (afterParsed : MemoryCarryPublicRows.ParsedColumnsMatch layout.after
      assignment headers after)
    (facts : ∀ pin ∈ pins layout, assignment pin.1 = pin.2) :
    after.stepIndex = 0 := by
  calc
    after.stepIndex = assignment (layout.afterColumn .stepIndex) :=
      (afterParsed.placed .stepIndex).symm
    _ = 0 := facts (layout.afterColumn .stepIndex, 0) (by simp [pins])

private theorem activeCarry_eq_of_fields
    {left right : ActiveCarry Digest.Value (Challenges K) (State K)}
    (segmentIndex : left.segmentIndex = right.segmentIndex)
    (stepIndex : left.stepIndex = right.stepIndex)
    (globalTimestamp : left.globalTimestamp = right.globalTimestamp)
    (segmentStartTimestamp :
      left.segmentStartTimestamp = right.segmentStartTimestamp)
    (segmentActiveAccessCount :
      left.segmentActiveAccessCount = right.segmentActiveAccessCount)
    (segmentEndTimestamp :
      left.segmentEndTimestamp = right.segmentEndTimestamp)
    (challenge : left.challenge = right.challenge)
    (products : left.products = right.products)
    (dPre : left.dPre = right.dPre)
    (dSeen : left.dSeen = right.dSeen)
    (memoryRoot : left.memoryRoot = right.memoryRoot) :
    left = right := by
  cases left
  cases right
  simp_all

private theorem rows_sound_of_challenge_exact
    {profile : Profile.Identity}
    {layout : MemoryOpenSegmentRows.Layout} {assignment : Nat → Nat}
    {headers : ChainHeaders Digest.Value}
    {before after : MemoryCarryCodec.Value}
    {authority : MemoryOpenSegment.Authority}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (beforeParsed : MemoryCarryPublicRows.ParsedColumnsMatch layout.before
      assignment headers before)
    (afterParsed : MemoryCarryPublicRows.ParsedColumnsMatch layout.after
      assignment headers after)
    (challengeExact : after.challenges =
      MemoryOpenSegment.deriveFor profile authority (closedOfWire before)
        after.dPre after.segmentActiveAccessCount)
    (localHolds : Satisfies (MemoryOpenSegmentRows.rows layout) assignment) :
    ∃ (canOpen : (closedOfWire before).CanOpen)
      (activeCountInRange :
        after.segmentActiveAccessCount < operationCountLimit)
      (endTimestampInRange :
        (closedOfWire before).globalTimestamp +
            after.segmentActiveAccessCount < timestampLimit)
      (stepBound : after.stepIndex < Lifecycle.claimsPerSegment),
      before.phase = .closed ∧ after.phase = .active ∧
        Carry.active (activeOfWire after stepBound) =
          MemoryOpenSegment.openCarryFor profile authority headers after.dPre
            after.segmentActiveAccessCount (closedOfWire before) canOpen
            activeCountInRange endTimestampInRange := by
  have pinsExact := pin_facts canonical one localHolds
  have equalities := equality_facts canonical one localHolds
  have beforeClosed := phase_before_closed beforeParsed pinsExact
  have afterActive := phase_after_active afterParsed pinsExact
  have endExact := end_timestamp_exact canonical one beforeParsed afterParsed
    localHolds
  have segmentRange := segment_index_in_range canonical one beforeParsed
    localHolds
  have timestampRange : before.globalTimestamp < timestampLimit := by
    simpa [timestampLimit, Nightstream.Protocol.NebulaV2.timestampBits,
      MemoryWireGeometry.timestampBits] using
      beforeParsed.rowCanonical.globalTimestamp
  have canOpen : (closedOfWire before).CanOpen :=
    ⟨segmentRange, timestampRange⟩
  have activeCountInRange :
      after.segmentActiveAccessCount < operationCountLimit := by
    simpa [operationCountLimit, operationCountBits,
      segmentActiveAccessCountBits] using
      afterParsed.rowCanonical.segmentActiveAccessCount
  have afterEndRange : after.segmentEndTimestamp < timestampLimit := by
    simpa [timestampLimit, Nightstream.Protocol.NebulaV2.timestampBits,
      MemoryWireGeometry.timestampBits] using
      afterParsed.rowCanonical.segmentEndTimestamp
  have endTimestampInRange :
      (closedOfWire before).globalTimestamp +
          after.segmentActiveAccessCount < timestampLimit := by
    simpa [closedOfWire, endExact] using afterEndRange
  have stepBound : after.stepIndex < Lifecycle.claimsPerSegment :=
    afterParsed.rowCanonical.stepIndex
  have stepZero := step_index_zero afterParsed pinsExact
  have segmentExact := carry_segment_index_exact beforeParsed afterParsed
    equalities
  have globalExact := carry_global_timestamp_exact beforeParsed afterParsed
    equalities
  have startExact := carry_segment_start_exact beforeParsed afterParsed
    equalities
  have productsExact :
      after.products = (ProductState.one : State K) := by
    simpa [MemoryCarryCodec.oneProductsK, ProductState.one] using
      products_are_one afterParsed pinsExact
  have seenExact := seen_headers_exact afterParsed equalities
  have memoryExact := carry_memory_root_exact beforeParsed afterParsed
    equalities
  let expected : ActiveCarry Digest.Value (Challenges K) (State K) :=
    { segmentIndex := before.segmentIndex
      stepIndex := ⟨0, by decide⟩
      globalTimestamp := before.globalTimestamp
      segmentStartTimestamp := before.globalTimestamp
      segmentActiveAccessCount := after.segmentActiveAccessCount
      segmentEndTimestamp :=
        before.globalTimestamp + after.segmentActiveAccessCount
      challenge := MemoryOpenSegment.deriveFor profile authority
        (closedOfWire before) after.dPre after.segmentActiveAccessCount
      products := ProductState.one
      dPre := after.dPre
      dSeen := headers.roots
      memoryRoot := before.memoryRoot }
  have activeExact : activeOfWire after stepBound = expected := by
    apply activeCarry_eq_of_fields
    · simpa [activeOfWire, expected] using segmentExact
    · apply Fin.ext
      simpa [activeOfWire, expected] using stepZero
    · simpa [activeOfWire, expected] using globalExact
    · simpa [activeOfWire, expected] using startExact
    · rfl
    · simpa [activeOfWire, expected] using endExact
    · simpa [activeOfWire, expected] using challengeExact
    · simpa [activeOfWire, expected] using productsExact
    · rfl
    · simpa [activeOfWire, expected] using seenExact
    · simpa [activeOfWire, expected] using memoryExact
  refine ⟨canOpen, activeCountInRange, endTimestampInRange, stepBound,
    beforeClosed, afterActive, ?_⟩
  calc
    Carry.active (activeOfWire after stepBound) = Carry.active expected :=
      congrArg Carry.active activeExact
    _ = MemoryOpenSegment.openCarryFor profile authority headers after.dPre
          after.segmentActiveAccessCount (closedOfWire before) canOpen
          activeCountInRange endTimestampInRange := by
      simpa [expected] using
        (MemoryOpenSegment.open_exact_for profile authority headers after.dPre
          after.segmentActiveAccessCount (closedOfWire before) canOpen
          activeCountInRange endTimestampInRange).symm

/-- Exact non-circular row soundness of the production V2 segment opening.
The assumptions contain only parsed public carries, verifier-owned authority
placement, structural transcript validity, and satisfaction of the emitted
rows. The conclusion derives the complete specialized transition. -/
theorem rows_sound
    {layout : MemoryOpenSegmentRows.Layout} {assignment : Nat → Nat}
    {headers : ChainHeaders Digest.Value}
    {before after : MemoryCarryCodec.Value}
    {authority : MemoryOpenSegment.Authority}
    (transcriptValid : layout.transcript.Valid)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (beforeParsed : MemoryCarryPublicRows.ParsedColumnsMatch layout.before
      assignment headers before)
    (afterParsed : MemoryCarryPublicRows.ParsedColumnsMatch layout.after
      assignment headers after)
    (authorityPlaced : AuthorityPlaced layout assignment authority)
    (localHolds : Satisfies (MemoryOpenSegmentRows.rows layout) assignment)
    (transcriptHolds : Satisfies
      (MemoryTranscriptPoseidonRows.rows layout.transcript) assignment) :
    ∃ (canOpen : (closedOfWire before).CanOpen)
      (activeCountInRange :
        after.segmentActiveAccessCount < operationCountLimit)
      (endTimestampInRange :
        (closedOfWire before).globalTimestamp +
            after.segmentActiveAccessCount < timestampLimit)
      (stepBound : after.stepIndex < Lifecycle.claimsPerSegment),
      before.phase = .closed ∧ after.phase = .active ∧
        Carry.active (activeOfWire after stepBound) =
          MemoryOpenSegment.openCarry authority headers after.dPre
            after.segmentActiveAccessCount (closedOfWire before) canOpen
            activeCountInRange endTimestampInRange := by
  have equalities := equality_facts canonical one localHolds
  have endExact := end_timestamp_exact canonical one beforeParsed afterParsed
    localHolds
  have challengeExact := challenges_are_derived transcriptValid canonical one
    beforeParsed afterParsed authorityPlaced equalities endExact
    transcriptHolds
  exact rows_sound_of_challenge_exact (profile := Profile.v2) canonical one
    beforeParsed afterParsed challengeExact localHolds

/-- Exact row soundness for a verifier-selected successor profile. Unlike the
reference wrapper, the candidate identity is inside the constrained transcript
frame and inside the derived challenge function. -/
theorem rows_sound_for
    {profile : Profile.Identity}
    {layout : MemoryOpenSegmentRows.Layout} {assignment : Nat → Nat}
    {headers : ChainHeaders Digest.Value}
    {before after : MemoryCarryCodec.Value}
    {authority : MemoryOpenSegment.Authority}
    (profileCanonical : MemoryTranscriptHashFrame.ProfileCanonical profile)
    (transcriptValid :
      MemoryTranscriptPoseidonRows.ProfileIndexed.Valid profile
        layout.transcript)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (beforeParsed : MemoryCarryPublicRows.ParsedColumnsMatch layout.before
      assignment headers before)
    (afterParsed : MemoryCarryPublicRows.ParsedColumnsMatch layout.after
      assignment headers after)
    (authorityPlaced : AuthorityPlaced layout assignment authority)
    (localHolds : Satisfies (MemoryOpenSegmentRows.rows layout) assignment)
    (transcriptHolds : Satisfies
      (MemoryTranscriptPoseidonRows.ProfileIndexed.rows profile
        layout.transcript) assignment) :
    ∃ (canOpen : (closedOfWire before).CanOpen)
      (activeCountInRange :
        after.segmentActiveAccessCount < operationCountLimit)
      (endTimestampInRange :
        (closedOfWire before).globalTimestamp +
            after.segmentActiveAccessCount < timestampLimit)
      (stepBound : after.stepIndex < Lifecycle.claimsPerSegment),
      before.phase = .closed ∧ after.phase = .active ∧
        Carry.active (activeOfWire after stepBound) =
          MemoryOpenSegment.openCarryFor profile authority headers after.dPre
            after.segmentActiveAccessCount (closedOfWire before) canOpen
            activeCountInRange endTimestampInRange := by
  have equalities := equality_facts canonical one localHolds
  have endExact := end_timestamp_exact canonical one beforeParsed afterParsed
    localHolds
  have challengeExact := challenges_are_derived_for profileCanonical
    transcriptValid canonical one beforeParsed afterParsed authorityPlaced
    equalities endExact transcriptHolds
  exact rows_sound_of_challenge_exact canonical one beforeParsed afterParsed
    challengeExact localHolds

end Nightstream.Implementation.NebulaV2.MemoryOpenSegmentSound
