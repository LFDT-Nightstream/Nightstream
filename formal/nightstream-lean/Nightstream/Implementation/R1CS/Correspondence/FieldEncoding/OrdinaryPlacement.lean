import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.SourceCensus

/-!
Contract: derive ordinary-private encoded word starts from an exact source-role
census and the production source-loop allocation ABI.

Owns: fixed per-role source-loop widths, public-prefix accounting, exact
source-phase end calculation, source-column-to-word-start lookup, and compact
branch metadata validation.

Does not own: source-role generation, witness values, centered-word choice,
deferred allocation details, constraint rows, CE coordinates, lifecycle
authority, NIVC invertibility, or permission to remove rows.

Emits constraints: no.

Authority boundary: starts are derived from checked source-role runs; they are
never prover-supplied metadata. `Metadata.check` binds only the source-phase
end and final encoded-column bound exported by Rust. Exact accepted centered
words remain an open refinement bridge because accepted representations are
not unique.

| Surface | Mathematical obligation | Main result | Assurance tier |
|---|---|---|---|
| role widths | Boolean 1, ordinary/SIS 41, canonical-u64 95, all other source-loop roles 0 | `sourceLoopWidth` lemmas | model-level |
| segment placement | ordinary offset `i` starts at `cursor + 41*i` | `segmentPlacementStart_some_iff` | model-level |
| word geometry | each derived word is exactly 41 consecutive coordinates | `wordRun_length`, `coordinate_mem_wordRun` | model-level |
| local ordering | increasing offsets produce disjoint ordered words | `sameSegment_wordRun_before` | model-level |
| branch summary | derived source-phase end matches generated metadata and fits final width | `Metadata.check_sound` | artifact interface |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFieldLayout
namespace OrdinaryPlacement

/-- Producer/consumer version for the compact placement summary. -/
def currentFormatVersion : Nat := 1

/-- Width of one ordinary-private centered word in the production ABI. -/
def ordinaryWordWidth : Nat := 41

/-- Fresh coordinates allocated when the source loop visits one source
column. Public bits were allocated before this loop. -/
def sourceLoopWidth : SlotRole → Nat
  | .constantOne => 0
  | .ordinaryPrivateField => ordinaryWordWidth
  | .privateBoolean => 1
  | .publicBit => 0
  | .canonicalU64 => 95
  | .sisOpening => 41
  | .linearlyDerived => 0
  | .structuralBalancedAlias => 0
  | .gadgetDerived => 0
  | .productDerived => 0
  | .gadgetTemporary => 0

theorem sourceLoopWidth_ordinary :
    sourceLoopWidth .ordinaryPrivateField = 41 := rfl

theorem sourceLoopWidth_privateBoolean :
    sourceLoopWidth .privateBoolean = 1 := rfl

/-- The source loop advances across both the 64 decoder-visible bits and the
31 canonical-prefix auxiliaries. -/
theorem sourceLoopWidth_canonicalU64 :
    sourceLoopWidth .canonicalU64 = 64 + 31 := rfl

theorem sourceLoopWidth_sisOpening :
    sourceLoopWidth .sisOpening = 41 := rfl

/-- Public prefix is coordinate zero plus all validated public bits. -/
def publicInputLength (artifact : SourceCensusArtifact) : Nat :=
  1 + artifact.declaredRoleCount .publicBit

/-- Allocation consumed by one complete source-role segment. -/
def segmentAllocationWidth (segment : SourceSegment) : Nat :=
  segment.source.length * sourceLoopWidth segment.role

/-- End of source-owned allocation, before deferred sampler and synthetic
coordinates. -/
def sourcePhaseEnd (artifact : SourceCensusArtifact) : Nat :=
  publicInputLength artifact +
    (artifact.sourceSegments.map segmentAllocationWidth).sum

/-- Total ordinary-private coordinates in this fixed 41-per-field ABI. -/
def ordinaryCoordinateCount (artifact : SourceCensusArtifact) : Nat :=
  artifact.eligibleCount * ordinaryWordWidth

/-- Exact start inside one source segment. It succeeds only for a contained
ordinary-private source column. -/
def segmentPlacementStart (cursor sourceColumn : Nat)
    (segment : SourceSegment) : Option Nat :=
  if segment.source.Contains sourceColumn then
    if segment.role = .ordinaryPrivateField then
      some (cursor +
        (sourceColumn - segment.source.start) * ordinaryWordWidth)
    else
      none
  else
    none

theorem segmentPlacementStart_some_iff
    (cursor sourceColumn encodedStart : Nat) (segment : SourceSegment) :
    segmentPlacementStart cursor sourceColumn segment = some encodedStart ↔
      segment.source.Contains sourceColumn ∧
        segment.role = .ordinaryPrivateField ∧
        encodedStart = cursor +
          (sourceColumn - segment.source.start) * ordinaryWordWidth := by
  simp [segmentPlacementStart, eq_comm]

theorem segmentPlacementStart_excluded
    (cursor sourceColumn : Nat) (segment : SourceSegment)
    (excluded : segment.role ≠ .ordinaryPrivateField) :
    segmentPlacementStart cursor sourceColumn segment = none := by
  simp [segmentPlacementStart, excluded]

/-- Search checked source segments in source order while carrying the exact
encoded cursor. Starts are computed, not read from an artifact. -/
def placementStartInSegments (sourceColumn : Nat) :
    Nat → List SourceSegment → Option Nat
  | _, [] => none
  | cursor, segment :: tail =>
      if segment.source.Contains sourceColumn then
        segmentPlacementStart cursor sourceColumn segment
      else
        placementStartInSegments sourceColumn
          (cursor + segmentAllocationWidth segment) tail

/-- Exact production-derived encoded start for one source column. -/
def placementStart? (artifact : SourceCensusArtifact)
    (sourceColumn : Nat) : Option Nat :=
  placementStartInSegments sourceColumn (publicInputLength artifact)
    artifact.sourceSegments

theorem placementStart_unique (artifact : SourceCensusArtifact)
    (sourceColumn first second : Nat)
    (firstPlacement : placementStart? artifact sourceColumn = some first)
    (secondPlacement : placementStart? artifact sourceColumn = some second) :
    first = second := by
  rw [firstPlacement] at secondPlacement
  exact Option.some.inj secondPlacement

/-- The exact 41-coordinate half-open word beginning at `encodedStart`. -/
def wordRun (encodedStart : Nat) : CoordinateRun :=
  { start := encodedStart, length := ordinaryWordWidth }

theorem wordRun_length (encodedStart : Nat) :
    (wordRun encodedStart).length = 41 := rfl

/-- Coordinate selected by one digit offset inside a derived word. -/
def coordinate (encodedStart digit : Nat) : Nat :=
  encodedStart + digit

theorem coordinate_eq (encodedStart digit : Nat) :
    coordinate encodedStart digit = encodedStart + digit := rfl

theorem coordinate_mem_wordRun (encodedStart digit : Nat)
    (digitLt : digit < 41) :
    (wordRun encodedStart).Contains (coordinate encodedStart digit) := by
  simp [wordRun, coordinate, CoordinateRun.Contains,
    CoordinateRun.endExclusive, ordinaryWordWidth]
  omega

/-- Start formula for offset `fieldOffset` within one ordinary source run. -/
def sameSegmentStart (encodedCursor fieldOffset : Nat) : Nat :=
  encodedCursor + fieldOffset * ordinaryWordWidth

theorem sameSegmentStart_eq (encodedCursor fieldOffset : Nat) :
    sameSegmentStart encodedCursor fieldOffset =
      encodedCursor + fieldOffset * 41 := by
  rfl

/-- Two increasing field offsets inside one ordinary run produce ordered,
therefore disjoint, 41-coordinate words. -/
theorem sameSegment_wordRun_before (encodedCursor first second : Nat)
    (ordered : first < second) :
    (wordRun (sameSegmentStart encodedCursor first)).endExclusive ≤
      (wordRun (sameSegmentStart encodedCursor second)).start := by
  simp [wordRun, sameSegmentStart, CoordinateRun.endExclusive,
    ordinaryWordWidth]
  omega

/-- Compact generated branch summary. No per-field start is accepted here. -/
structure Metadata where
  formatVersion : Nat
  sourcePhaseEnd : Nat
  encodedColumnCount : Nat
deriving DecidableEq, Repr, Inhabited

namespace Metadata

/-- The summary is valid only when its source-phase end is recomputed from the
checked source census and lies within the final encoded assignment. -/
def ValidFor (metadata : Metadata) (artifact : SourceCensusArtifact) : Prop :=
  metadata.formatVersion = currentFormatVersion ∧
    metadata.sourcePhaseEnd = OrdinaryPlacement.sourcePhaseEnd artifact ∧
    metadata.sourcePhaseEnd ≤ metadata.encodedColumnCount

instance (metadata : Metadata) (artifact : SourceCensusArtifact) :
    Decidable (metadata.ValidFor artifact) := by
  unfold ValidFor
  infer_instance

def check (metadata : Metadata) (artifact : SourceCensusArtifact) : Bool :=
  decide (metadata.ValidFor artifact)

theorem check_sound (metadata : Metadata) (artifact : SourceCensusArtifact)
    (accepted : metadata.check artifact = true) :
    metadata.ValidFor artifact := by
  exact of_decide_eq_true accepted

theorem ValidFor.sourcePhaseEnd_le_encodedColumnCount
    {metadata : Metadata} {artifact : SourceCensusArtifact}
    (valid : metadata.ValidFor artifact) :
    OrdinaryPlacement.sourcePhaseEnd artifact ≤
      metadata.encodedColumnCount := by
  rcases valid with ⟨_, phaseExact, bound⟩
  simpa [phaseExact] using bound

end Metadata

end OrdinaryPlacement
end Nightstream.Implementation.R1CS.FPrimeFieldLayout
