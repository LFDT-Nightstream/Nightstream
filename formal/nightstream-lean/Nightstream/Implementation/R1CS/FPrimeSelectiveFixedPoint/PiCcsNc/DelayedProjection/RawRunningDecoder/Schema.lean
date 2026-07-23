import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsDomain
import Nightstream.SuperNeo.Concrete.Parameters

/-!
Shared exact coordinate schema for the production fixed-profile raw running-child
decoder.

Assurance tier: model-level.

Owns: the fixed `14 × 270` logical source domain; Rust's exact packed-table
addressing rule `lane = logicalColumn % 54`,
`block = logicalColumn / 54`; the inverse rule
`logicalColumn = block * 54 + lane`; the five-live-block/eight-virtual-block
and 54-live-lane/64-virtual-lane domains; the exact centered-scalar,
balanced-ternary, and binary decoding rules; and a compact proof-free
`{ child, logicalColumn, allocation }` record schema for a later generated
artifact.

Does not own: any concrete allocation, assignment value, R1CS row,
decoder-emitter provenance, matrix equality, commitment binding, transcript,
cost, or row-removal authority. In particular, the arbitrary
`SourceAllocationMap.allocation` field below is data, not evidence.

Emits constraints: none; typed coordinate schema only.

The Rust/R1CS exporter must eventually instantiate `SourceAllocationMap` from
actual allocation intervals and prove that its generated records are exactly
`SourceAllocationMap.records`. This file deliberately contains no generated
data and no closed computation over the 3,780-record domain.

| Stage path | Mathematical obligation | Authority class | Open boundary |
|---|---|---|---|
| `nifs.pi_ccs.nc.delayed.raw_decoder.logical` | fourteen children, each with exactly 270 logical scalar coordinates | computed | production profile provenance |
| `nifs.pi_ccs.nc.delayed.raw_decoder.packed` | `c ↔ (c % 54, c / 54)` is a bijection with the 54-by-5 live table | derived | Rust allocation conformance |
| `nifs.pi_ccs.nc.delayed.raw_decoder.records` | compact child-major encoded-scalar schema has one coordinate owner | derived | generated allocation values |
| `nifs.pi_ccs.nc.delayed.raw_decoder.virtual` | live 54-by-5 table embeds into a 64-by-8 virtual table | derived | generated zero-enforcement rows |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder

open Nightstream.SuperNeo.Concrete

/-- Fixed active PiDEC arity. -/
def childCount : Nat := 14

/-- Complete scalar width of five Phi81 blocks. -/
def logicalColumnCount : Nat := 270

/-- Number of coefficients in one Phi81 block. -/
def packedLaneCount : Nat := 54

/-- Number of live packed blocks in one 270-scalar assignment. -/
def liveBlockCount : Nat := 5

/-- Boolean-cube lane capacity used by the block/lane NC arithmetization. -/
def virtualLaneCount : Nat := 64

/-- Boolean-cube block capacity used by the block/lane NC arithmetization. -/
def virtualBlockCount : Nat := 8

abbrev Child := Fin childCount
abbrev LogicalColumn := Fin logicalColumnCount
abbrev PackedLane := Fin packedLaneCount
abbrev LiveBlock := Fin liveBlockCount
abbrev VirtualLane := Fin virtualLaneCount
abbrev VirtualBlock := Fin virtualBlockCount

theorem profile_dimensions :
    childCount = 14 /\
      logicalColumnCount = 270 /\
      packedLaneCount = ringDegree /\
      liveBlockCount * packedLaneCount = logicalColumnCount /\
      virtualLaneCount = 2 ^ 6 /\
      virtualBlockCount = 2 ^ 3 := by
  decide

theorem virtual_padding_counts :
    virtualLaneCount - packedLaneCount = 10 /\
      virtualBlockCount - liveBlockCount = 3 := by
  decide

theorem childCount_eq_productionArity :
    childCount = productionGlobalParams.k := by
  rfl

/-- Exact cast from the compact artifact child index to the active PiDEC
child index. -/
def productionChild (child : Child) : Fin productionGlobalParams.k :=
  Fin.cast childCount_eq_productionArity child

@[simp] theorem productionChild_val (child : Child) :
    (productionChild child).val = child.val := by
  rfl

/-- Exact physical address inside one Rust `54 × 5` packed assignment table.
The table is stored lane-major by `Mat`, while its logical scalar index is
block-major. -/
structure PackedAddress where
  lane : PackedLane
  block : LiveBlock
deriving DecidableEq, Repr

private theorem packedAddress_ext
    {left right : PackedAddress}
    (laneEq : left.lane = right.lane)
    (blockEq : left.block = right.block) :
    left = right := by
  cases left with
  | mk leftLane leftBlock =>
      cases right with
      | mk rightLane rightBlock =>
          cases laneEq
          cases blockEq
          rfl

/-- Rust's logical-scalar-to-packed-table address calculation. -/
def packedAddress (column : LogicalColumn) : PackedAddress where
  lane :=
    ⟨column.val % packedLaneCount,
      Nat.mod_lt _ (by decide : 0 < packedLaneCount)⟩
  block :=
    ⟨column.val / packedLaneCount,
      (Nat.div_lt_iff_lt_mul (by decide : 0 < packedLaneCount)).2 (by
        simpa [logicalColumnCount, liveBlockCount, packedLaneCount,
          Nat.mul_comm] using column.isLt)⟩

/-- Inverse block-major logical index. -/
def logicalColumnAt (address : PackedAddress) : LogicalColumn :=
  ⟨address.block.val * packedLaneCount + address.lane.val, by
    have blockNext : address.block.val + 1 <= liveBlockCount :=
      Nat.succ_le_of_lt address.block.isLt
    have scaled :
        (address.block.val + 1) * packedLaneCount <=
          liveBlockCount * packedLaneCount :=
      Nat.mul_le_mul_right packedLaneCount blockNext
    have belowNext :
        address.block.val * packedLaneCount + address.lane.val <
          (address.block.val + 1) * packedLaneCount := by
      simpa [Nat.add_mul] using
        Nat.add_lt_add_left address.lane.isLt
          (address.block.val * packedLaneCount)
    simpa [logicalColumnCount, liveBlockCount, packedLaneCount] using
      Nat.lt_of_lt_of_le belowNext scaled⟩

@[simp] theorem logicalColumnAt_packedAddress (column : LogicalColumn) :
    logicalColumnAt (packedAddress column) = column := by
  apply Fin.ext
  change column.val / packedLaneCount * packedLaneCount +
      column.val % packedLaneCount = column.val
  simpa [Nat.mul_comm] using
    Nat.div_add_mod column.val packedLaneCount

@[simp] theorem packedAddress_logicalColumnAt (address : PackedAddress) :
    packedAddress (logicalColumnAt address) = address := by
  cases address with
  | mk lane block =>
      have laneEq :
          (packedAddress
            (logicalColumnAt { lane := lane, block := block })).lane =
            lane := by
        apply Fin.ext
        change
          (block.val * packedLaneCount + lane.val) % packedLaneCount =
            lane.val
        simpa [Nat.mod_eq_of_lt lane.isLt] using
          Nat.mul_add_mod_self_right block.val packedLaneCount lane.val
      have blockEq :
          (packedAddress
            (logicalColumnAt { lane := lane, block := block })).block =
            block := by
        apply Fin.ext
        change
          (block.val * packedLaneCount + lane.val) / packedLaneCount =
            block.val
        rw [Nat.mul_comm block.val packedLaneCount,
          Nat.mul_add_div (by decide : 0 < packedLaneCount),
          Nat.div_eq_of_lt lane.isLt, Nat.add_zero]
      exact packedAddress_ext laneEq blockEq

theorem packedAddress_injective : Function.Injective packedAddress := by
  intro left right equal
  calc
    left = logicalColumnAt (packedAddress left) :=
      (logicalColumnAt_packedAddress left).symm
    _ = logicalColumnAt (packedAddress right) := by rw [equal]
    _ = right := logicalColumnAt_packedAddress right

theorem packedAddress_surjective : Function.Surjective packedAddress := by
  intro address
  exact ⟨logicalColumnAt address, packedAddress_logicalColumnAt address⟩

/-- Exact bijection; no observed row count is used to construct it. -/
theorem packedAddress_bijective :
    Function.Injective packedAddress /\
      Function.Surjective packedAddress :=
  ⟨packedAddress_injective, packedAddress_surjective⟩

/-! ## Child-major schema -/

/-- Physical representation selected by the production lowering for one
logical field scalar. These constructors name decoding rules, not semantic
authority for the decoded value. -/
inductive Encoding where
  | centeredScalar
  | balancedTernary
  | binary
deriving DecidableEq, Repr, Inhabited

/-- Production balanced-ternary field width. -/
def balancedTernaryWidth : Nat := 41

namespace Encoding

/-- Shape required by each exact decoding rule. A centered scalar occupies
one field column, this production balanced-ternary representation occupies
41 signed-digit columns, and a binary representation is nonempty. -/
def ValidWidth (encoding : Encoding) (width : Nat) : Prop :=
  match encoding with
  | .centeredScalar => width = 1
  | .balancedTernary => width = balancedTernaryWidth
  | .binary => 0 < width

instance (encoding : Encoding) (width : Nat) :
    Decidable (encoding.ValidWidth width) := by
  cases encoding <;> simp only [ValidWidth] <;> infer_instance

end Encoding

/-- Exact final-assignment interval and its decoding rule for one logical
field scalar. This is proof-free data suitable for bounded generated
certificates. -/
structure EncodedScalar where
  start : Nat
  width : Nat
  encoding : Encoding
deriving DecidableEq, Repr, Inhabited

namespace EncodedScalar

/-- The allocation is nonempty with the exact encoding width and lies inside
the final selectively lowered assignment. -/
def WellFormed (allocation : EncodedScalar) (finalColumnCount : Nat) : Prop :=
  allocation.encoding.ValidWidth allocation.width /\
    allocation.start + allocation.width <= finalColumnCount

instance (allocation : EncodedScalar) (finalColumnCount : Nat) :
    Decidable (allocation.WellFormed finalColumnCount) := by
  unfold WellFormed
  infer_instance

/-- Positional reconstruction for an uncentered radix encoding. The assignment
columns are interpreted in little-endian order. -/
def decodeRadix (radix : F) (allocation : EncodedScalar)
    (assignment : Nat -> F) : F :=
  ((List.range allocation.width).map fun index =>
    assignment (allocation.start + index) * radix ^ index).sum

/-- Exact field reconstruction selected by the allocation encoding.

`centeredScalar` is the one-slot field value itself. `balancedTernary` is the
little-endian signed-digit polynomial `sum digit[i] * 3^i`; negative digits
are already represented as field elements in the assignment. `binary` is the
analogous little-endian base-two polynomial. -/
def decode (allocation : EncodedScalar) (assignment : Nat -> F) : F :=
  match allocation.encoding with
  | .centeredScalar => assignment allocation.start
  | .balancedTernary => decodeRadix 3 allocation assignment
  | .binary => decodeRadix 2 allocation assignment

@[simp] theorem decode_centeredScalar
    (start : Nat) (assignment : Nat -> F) :
    decode { start := start, width := 1, encoding := .centeredScalar }
        assignment = assignment start := by
  rfl

@[simp] theorem decode_balancedTernary
    (start width : Nat) (assignment : Nat -> F) :
    decode { start := start, width := width, encoding := .balancedTernary }
        assignment =
      ((List.range width).map fun index =>
        assignment (start + index) * (3 : F) ^ index).sum := by
  rfl

@[simp] theorem decode_binary
    (start width : Nat) (assignment : Nat -> F) :
    decode { start := start, width := width, encoding := .binary }
        assignment =
      ((List.range width).map fun index =>
        assignment (start + index) * (2 : F) ^ index).sum := by
  rfl

end EncodedScalar

/-- Compact proof-free generated record. Coordinate bounds are checked by the
generic schema, while physical interval bounds belong to the generated
artifact certificate; neither is embedded as proof terms in every datum. -/
structure SourceColumnRecord where
  child : Nat
  logicalColumn : Nat
  allocation : EncodedScalar
deriving DecidableEq, Repr

namespace SourceColumnRecord

/-- The two generated coordinate fields are in the exact production domain.
This deliberately does not certify the generated physical allocation. -/
def CoordinatesValid (record : SourceColumnRecord) : Prop :=
  record.child < childCount /\ record.logicalColumn < logicalColumnCount

/-- Typed child coordinate after a coordinate-domain check. -/
def typedChild (record : SourceColumnRecord)
    (valid : record.CoordinatesValid) : Child :=
  ⟨record.child, valid.1⟩

/-- Typed logical coordinate after a coordinate-domain check. -/
def typedLogicalColumn (record : SourceColumnRecord)
    (valid : record.CoordinatesValid) : LogicalColumn :=
  ⟨record.logicalColumn, valid.2⟩

end SourceColumnRecord

/-- One encoded-scalar allocation for every typed child/logical coordinate.
This structure contains no provenance proof; an arbitrary function inhabits
it. -/
structure SourceAllocationMap where
  allocation : Child -> LogicalColumn -> EncodedScalar

namespace SourceAllocationMap

/-- Decode one typed logical coordinate from the final selectively lowered
assignment according to its complete physical allocation. -/
def decode (columns : SourceAllocationMap) (assignment : Nat -> F)
    (child : Child) (logicalColumn : LogicalColumn) : F :=
  (columns.allocation child logicalColumn).decode assignment

/-- Canonical compact record at one typed coordinate. -/
def recordAt (columns : SourceAllocationMap)
    (child : Child) (logicalColumn : LogicalColumn) : SourceColumnRecord where
  child := child.val
  logicalColumn := logicalColumn.val
  allocation := columns.allocation child logicalColumn

@[simp] theorem recordAt_child (columns : SourceAllocationMap)
    (child : Child) (logicalColumn : LogicalColumn) :
    (recordAt columns child logicalColumn).child = child.val := by
  rfl

@[simp] theorem recordAt_logicalColumn (columns : SourceAllocationMap)
    (child : Child) (logicalColumn : LogicalColumn) :
    (recordAt columns child logicalColumn).logicalColumn =
      logicalColumn.val := by
  rfl

@[simp] theorem recordAt_allocation (columns : SourceAllocationMap)
    (child : Child) (logicalColumn : LogicalColumn) :
    (recordAt columns child logicalColumn).allocation =
      columns.allocation child logicalColumn := by
  rfl

theorem recordAt_coordinatesValid (columns : SourceAllocationMap)
    (child : Child) (logicalColumn : LogicalColumn) :
    (recordAt columns child logicalColumn).CoordinatesValid := by
  exact ⟨child.isLt, logicalColumn.isLt⟩

/-- Canonical child-major list. A generated artifact may emit this list in
bounded shards, but semantic clients should consume `recordAt` rather than
normalize the whole list. -/
def records (columns : SourceAllocationMap) : List SourceColumnRecord :=
  (List.ofFn fun child : Child =>
    List.ofFn fun logicalColumn : LogicalColumn =>
      recordAt columns child logicalColumn).flatten

private theorem sum_ofFn_constant (count value : Nat) :
    (List.ofFn fun _ : Fin count => value).sum = count * value := by
  induction count with
  | zero => simp
  | succ count inductionHypothesis =>
      rw [List.ofFn_succ, List.sum_cons, inductionHypothesis, Nat.succ_mul]
      omega

private theorem flatten_ofFn_length
    {Alpha : Type}
    {count width : Nat}
    (blocks : Fin count -> List Alpha)
    (blockLength : forall index, (blocks index).length = width) :
    (List.ofFn blocks).flatten.length = count * width := by
  rw [List.length_flatten, List.map_ofFn]
  have lengths :
      List.ofFn (List.length ∘ blocks) =
        List.ofFn (fun _ : Fin count => width) := by
    apply congrArg List.ofFn
    funext index
    exact blockLength index
  rw [lengths, sum_ofFn_constant]

@[simp] theorem records_length (columns : SourceAllocationMap) :
    columns.records.length = childCount * logicalColumnCount := by
  unfold records
  apply flatten_ofFn_length
  intro child
  exact List.length_ofFn

theorem recordAt_mem_records (columns : SourceAllocationMap)
    (child : Child) (logicalColumn : LogicalColumn) :
    recordAt columns child logicalColumn ∈ columns.records := by
  unfold records
  apply List.mem_flatten.mpr
  refine ⟨List.ofFn fun current : LogicalColumn =>
      recordAt columns child current, ?_, ?_⟩
  · exact List.mem_ofFn.mpr ⟨child, rfl⟩
  · exact List.mem_ofFn.mpr ⟨logicalColumn, rfl⟩

/-- Every canonical record has exactly one typed coordinate owner. This does
not claim that two coordinates have disjoint physical allocations. -/
theorem recordAt_eq_recordAt_iff (columns : SourceAllocationMap)
    (leftChild rightChild : Child)
    (leftColumn rightColumn : LogicalColumn) :
    recordAt columns leftChild leftColumn =
        recordAt columns rightChild rightColumn <->
      leftChild = rightChild /\ leftColumn = rightColumn := by
  constructor
  · intro equal
    constructor
    · apply Fin.ext
      exact congrArg SourceColumnRecord.child equal
    · apply Fin.ext
      exact congrArg SourceColumnRecord.logicalColumn equal
  · rintro ⟨rfl, rfl⟩
    rfl

theorem recordAt_injective (columns : SourceAllocationMap) :
    Function.Injective fun coordinate : Child × LogicalColumn =>
      recordAt columns coordinate.1 coordinate.2 := by
  intro left right equal
  rcases (recordAt_eq_recordAt_iff columns left.1 right.1 left.2 right.2).1
      equal with ⟨childEq, columnEq⟩
  exact Prod.ext childEq columnEq

/-- Membership exposes the typed coordinate used to construct the record. -/
theorem mem_records_iff (columns : SourceAllocationMap)
    (record : SourceColumnRecord) :
    record ∈ columns.records <->
      ∃ child logicalColumn,
        record = recordAt columns child logicalColumn := by
  constructor
  · intro member
    rcases List.mem_flatten.mp member with ⟨childRecords, childRecordsMem,
      recordMem⟩
    rcases List.mem_ofFn.mp childRecordsMem with ⟨child, rfl⟩
    rcases List.mem_ofFn.mp recordMem with ⟨logicalColumn, rfl⟩
    exact ⟨child, logicalColumn, rfl⟩
  · rintro ⟨child, logicalColumn, rfl⟩
    exact recordAt_mem_records columns child logicalColumn

theorem records_all_coordinatesValid (columns : SourceAllocationMap) :
    ∀ record ∈ columns.records, record.CoordinatesValid := by
  intro record member
  rcases (mem_records_iff columns record).1 member with
    ⟨child, logicalColumn, rfl⟩
  exact recordAt_coordinatesValid columns child logicalColumn

/-- Exact coordinate coverage and ownership, stated without a global
`Nodup` computation. -/
theorem existsUnique_record_with_coordinate (columns : SourceAllocationMap)
    (child : Child) (logicalColumn : LogicalColumn) :
    ∃ record,
      (record ∈ columns.records /\
        record.child = child.val /\
        record.logicalColumn = logicalColumn.val) /\
      ∀ other,
        other ∈ columns.records /\
          other.child = child.val /\
          other.logicalColumn = logicalColumn.val ->
        other = record := by
  refine ⟨recordAt columns child logicalColumn, ?_, ?_⟩
  · exact ⟨recordAt_mem_records columns child logicalColumn, rfl, rfl⟩
  · intro record properties
    rcases (mem_records_iff columns record).1 properties.1 with
      ⟨ownerChild, ownerColumn, rfl⟩
    have childEq : ownerChild = child := by
      apply Fin.ext
      exact properties.2.1
    have columnEq : ownerColumn = logicalColumn := by
      apply Fin.ext
      exact properties.2.2
    subst ownerChild
    subst ownerColumn
    rfl

end SourceAllocationMap

/-! ## Virtual block/lane domain -/

/-- Embed a live packed lane into the six-bit virtual lane domain. -/
def virtualLaneOfLive (lane : PackedLane) : VirtualLane :=
  ⟨lane.val, by
    have live := lane.isLt
    simp only [packedLaneCount, virtualLaneCount] at live ⊢
    omega⟩

/-- Embed a live packed block into the three-bit virtual block domain. -/
def virtualBlockOfLive (block : LiveBlock) : VirtualBlock :=
  ⟨block.val, by
    have live := block.isLt
    simp only [liveBlockCount, virtualBlockCount] at live ⊢
    omega⟩

@[simp] theorem virtualLaneOfLive_val (lane : PackedLane) :
    (virtualLaneOfLive lane).val = lane.val := by
  rfl

@[simp] theorem virtualBlockOfLive_val (block : LiveBlock) :
    (virtualBlockOfLive block).val = block.val := by
  rfl

theorem liveLane_or_padding (lane : VirtualLane) :
    lane.val < packedLaneCount \/ packedLaneCount <= lane.val := by
  omega

theorem liveBlock_or_padding (block : VirtualBlock) :
    block.val < liveBlockCount \/ liveBlockCount <= block.val := by
  omega

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder
