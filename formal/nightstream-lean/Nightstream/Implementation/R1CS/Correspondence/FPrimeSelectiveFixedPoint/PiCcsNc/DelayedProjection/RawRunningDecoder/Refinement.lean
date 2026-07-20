import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Schema
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority.DelayedRawChildren

/-!
Generic semantic refinement contract for a bounded 270-coordinate running
table decoder.

Assurance tier: model-level.

Owns: decoding a physical R1CS assignment through a compact
`SourceColumnMap`; the primitive per-record equation that a future concrete
row proof must establish; transport of those equations to the exact
`DelayedRawChildren.rawRunningAssignments` table; and the live/virtual
packed-table consequences.

Does not own: any concrete source-column number, generated record, sparse
`A/B/C` row, satisfaction theorem for those rows, assignment-allocation
decoder, Rust emitter, commitment binding, transcript, cost, or row-removal
authority. `SourceColumnRowsBind` is intentionally the open leaf: unlike the
conclusion it is a family of primitive physical-column equations, suitable
for discharge from exact generated rows. No theorem takes decoded-table
equality or raw-child authority as a premise.

Emits constraints: none; generic correspondence theorem only.

No output claim, digest, or carried `CeClaim.y_zcol` value occurs in this
contract. A concrete instantiation is authoritative only if its physical
source columns decode the full packed witness; the currently generated
artifact decodes `CeClaim.X` and therefore supplies only a public-prefix
fixture.

| Stage path | Mathematical obligation | Authority class | Open boundary |
|---|---|---|---|
| `nifs.pi_ccs.nc.delayed.raw_decoder.decode` | read each logical scalar from its generated physical source column | direct dataflow | generated source-column map |
| `nifs.pi_ccs.nc.delayed.raw_decoder.rows` | each physical source column is equated to the corresponding raw running-assignment coordinate | checked contract | exact sparse `A/B/C` row proof |
| `nifs.pi_ccs.nc.delayed.raw_decoder.refinement` | checked source-column rows imply coordinatewise equality with `rawRunningAssignments` | derived | none beyond the checked rows |
| `nifs.pi_ccs.nc.delayed.raw_decoder.padding` | the 10 lane and 3 block padding positions evaluate to zero by construction | computed | concrete zero-row emission remains open |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources

universe uState

/-- The only fixed-profile equality not already carried by the active PiDEC
arity: every raw running assignment has exactly 270 logical scalars. -/
structure Profile (shape : SemanticShape) : Prop where
  carrierWidth_eq : shape.carrierWidth = logicalColumnCount

namespace Profile

/-- Cast one compact logical coordinate into the semantic carrier. -/
def semanticColumn {shape : SemanticShape}
    (profile : Profile shape) (column : LogicalColumn) :
    Fin shape.carrierWidth :=
  Fin.cast profile.carrierWidth_eq.symm column

@[simp] theorem semanticColumn_val {shape : SemanticShape}
    (profile : Profile shape) (column : LogicalColumn) :
    (profile.semanticColumn column).val = column.val := by
  rfl

end Profile

/-- Physical field assignment indexed by compiler allocation column. -/
abbrev PhysicalAssignment := Nat -> F

/-- Logical child-major decoder induced by an exact source-column map. -/
def decodedLogical
    (columns : SourceColumnMap)
    (assignment : PhysicalAssignment)
    (child : Child)
    (logicalColumn : LogicalColumn) : F :=
  assignment (columns.sourceColumn child logicalColumn)

section SemanticTarget

variable
  {shape : SemanticShape}
  {State : Type uState}
  {publicRingColumns verifierRows : Nat}
  {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}

/-- Primitive concrete obligation for one generated record. The future row
proof must derive this equation from actual sparse rows and their satisfaction;
it is not semantic acceptance and it is not the desired decoded-table
equality. -/
def SourceColumnEquation
    (profile : Profile shape)
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Sources.Data shape)
    (columns : SourceColumnMap)
    (assignment : PhysicalAssignment)
    (child : Child)
    (logicalColumn : LogicalColumn) : Prop :=
  assignment (columns.sourceColumn child logicalColumn) =
    DelayedRawChildren.rawRunningAssignments context data
      (productionChild child) (profile.semanticColumn logicalColumn)

/-- Exact family of primitive source-column equations. This is the sole open
row-semantic boundary of the handwritten decoder contract. -/
def SourceColumnRowsBind
    (profile : Profile shape)
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Sources.Data shape)
    (columns : SourceColumnMap)
    (assignment : PhysicalAssignment) : Prop :=
  forall child logicalColumn,
    SourceColumnEquation profile context data columns assignment child
      logicalColumn

/-- Record spelling of the same primitive equation. It allows a generated
artifact to prove one bounded shard of compact records at a time. -/
def RecordEquation
    (profile : Profile shape)
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Sources.Data shape)
    (assignment : PhysicalAssignment)
    (record : SourceColumnRecord)
    (wellFormed : record.WellFormed) : Prop :=
  assignment record.sourceColumn =
    DelayedRawChildren.rawRunningAssignments context data
      (productionChild (record.typedChild wellFormed))
      (profile.semanticColumn (record.typedLogicalColumn wellFormed))

/-- If every canonical generated record's primitive equation is proved, the
coordinate-indexed row contract follows. Proof irrelevance discharges the
different bound witnesses; no list normalization is required. -/
theorem recordEquations_imply_sourceColumnRowsBind
    (profile : Profile shape)
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Sources.Data shape)
    (columns : SourceColumnMap)
    (assignment : PhysicalAssignment)
    (recordRows : forall record,
      (member : record ∈ columns.records) ->
      RecordEquation profile context data assignment record
        (SourceColumnMap.records_all_wellFormed columns record member)) :
    SourceColumnRowsBind profile context data columns assignment := by
  intro child logicalColumn
  let record := columns.recordAt child logicalColumn
  have member : record ∈ columns.records :=
    SourceColumnMap.recordAt_mem_records columns child logicalColumn
  have equation := recordRows record member
  simpa [record, RecordEquation, SourceColumnEquation,
    SourceColumnRecord.typedChild,
    SourceColumnRecord.typedLogicalColumn] using equation

/-- Main coordinatewise refinement theorem. Its premise is the primitive
physical-column equation family, not the conclusion. -/
theorem sourceColumnRows_imply_decodedLogical_eq_rawRunningAssignments
    (profile : Profile shape)
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Sources.Data shape)
    (columns : SourceColumnMap)
    (assignment : PhysicalAssignment)
    (rows : SourceColumnRowsBind profile context data columns assignment) :
    forall child logicalColumn,
      decodedLogical columns assignment child logicalColumn =
        DelayedRawChildren.rawRunningAssignments context data
          (productionChild child) (profile.semanticColumn logicalColumn) := by
  intro child logicalColumn
  exact rows child logicalColumn

/-- Generated record-shard interface: exact record equations directly imply
the coordinatewise raw-child result. -/
theorem recordEquations_imply_decodedLogical_eq_rawRunningAssignments
    (profile : Profile shape)
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Sources.Data shape)
    (columns : SourceColumnMap)
    (assignment : PhysicalAssignment)
    (recordRows : forall record,
      (member : record ∈ columns.records) ->
      RecordEquation profile context data assignment record
        (SourceColumnMap.records_all_wellFormed columns record member)) :
    forall child logicalColumn,
      decodedLogical columns assignment child logicalColumn =
        DelayedRawChildren.rawRunningAssignments context data
          (productionChild child) (profile.semanticColumn logicalColumn) := by
  exact sourceColumnRows_imply_decodedLogical_eq_rawRunningAssignments
    profile context data columns assignment
      (recordEquations_imply_sourceColumnRowsBind profile context data
        columns assignment recordRows)

end SemanticTarget

/-! ## Virtual packed-table decoder -/

/-- Decode the physical assignment as a `14 × 64 × 8` virtual table. Only
the `54 × 5` live rectangle reads physical source columns. Every other cell
is definitionally zero. -/
def decodedVirtual
    (columns : SourceColumnMap)
    (assignment : PhysicalAssignment)
    (child : Child)
    (lane : VirtualLane)
    (block : VirtualBlock) : F :=
  if laneLive : lane.val < packedLaneCount then
    if blockLive : block.val < liveBlockCount then
      decodedLogical columns assignment child
        (logicalColumnAt {
          lane := ⟨lane.val, laneLive⟩
          block := ⟨block.val, blockLive⟩ })
    else
      0
  else
    0

@[simp] theorem decodedVirtual_live
    (columns : SourceColumnMap)
    (assignment : PhysicalAssignment)
    (child : Child)
    (lane : PackedLane)
    (block : LiveBlock) :
    decodedVirtual columns assignment child
        (virtualLaneOfLive lane) (virtualBlockOfLive block) =
      decodedLogical columns assignment child
        (logicalColumnAt { lane := lane, block := block }) := by
  simp [decodedVirtual, virtualLaneOfLive, virtualBlockOfLive,
    lane.isLt, block.isLt]

theorem decodedVirtual_lanePadding_zero
    (columns : SourceColumnMap)
    (assignment : PhysicalAssignment)
    (child : Child)
    (lane : VirtualLane)
    (block : VirtualBlock)
    (padding : packedLaneCount <= lane.val) :
    decodedVirtual columns assignment child lane block = 0 := by
  simp [decodedVirtual, Nat.not_lt.mpr padding]

theorem decodedVirtual_blockPadding_zero
    (columns : SourceColumnMap)
    (assignment : PhysicalAssignment)
    (child : Child)
    (lane : VirtualLane)
    (block : VirtualBlock)
    (padding : liveBlockCount <= block.val) :
    decodedVirtual columns assignment child lane block = 0 := by
  by_cases laneLive : lane.val < packedLaneCount
  · simp [decodedVirtual, laneLive, Nat.not_lt.mpr padding]
  · simp [decodedVirtual, laneLive]

section VirtualSemanticTarget

variable
  {shape : SemanticShape}
  {State : Type uState}
  {publicRingColumns verifierRows : Nat}
  {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}

/-- On every live packed cell, checked source-column rows recover the exact
raw next-step running assignment at `block * 54 + lane`. -/
theorem sourceColumnRows_imply_decodedVirtual_live_eq_rawRunningAssignments
    (profile : Profile shape)
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Sources.Data shape)
    (columns : SourceColumnMap)
    (assignment : PhysicalAssignment)
    (rows : SourceColumnRowsBind profile context data columns assignment)
    (child : Child)
    (lane : PackedLane)
    (block : LiveBlock) :
    decodedVirtual columns assignment child
        (virtualLaneOfLive lane) (virtualBlockOfLive block) =
      DelayedRawChildren.rawRunningAssignments context data
        (productionChild child)
        (profile.semanticColumn
          (logicalColumnAt { lane := lane, block := block })) := by
  rw [decodedVirtual_live]
  exact sourceColumnRows_imply_decodedLogical_eq_rawRunningAssignments
    profile context data columns assignment rows child
      (logicalColumnAt { lane := lane, block := block })

end VirtualSemanticTarget

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder
