import Nightstream.Implementation.R1CS.Correspondence.CanonicalU64.CanonicalU64Sound
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.ScheduleRefinement

/-!
Canonical lane decomposition and chunk order for the recursive-profile
`Pi_RLC` challenge sampler.

Owns: the sixteen exact canonical-u64 owner pieces following the four digest
calls; their independently proved integer recomposition and bitness; and the
proof that each consecutive little-endian 16-bit bit window is the same
candidate as the pure transcript machine's quotient/modulo `laneChunk`.

Does not own: rejection, first-accepted selection, cumulative-count rows,
coefficient assembly, the pre-`Pi_RLC` transcript prefix, native Rust
conformance, any other scalar coordinate/profile, or row/cost totals.

Emits constraints: no.

Authority boundary: Poseidon output lanes become sampler candidates only
after an accepted canonical-u64 piece proves their unique little-endian bit
representation. Generated adjacency and column numbers alone do not establish
that interpretation.

| Protocol | Phase | Constraint family | Field columns | Bit starts | Proven obligation |
|---|---|---|---|---|---|
| `Pi_RLC` | digest block 0 | four canonical-u64 leaves | `351846..351849` | `351854`, `352012`, `352170`, `352328` | lanes 0-3 decompose canonically |
| `Pi_RLC` | digest block 1 | four canonical-u64 leaves | `353082..353085` | `353090`, `353248`, `353406`, `353564` | lanes 0-3 decompose canonically |
| `Pi_RLC` | digest block 2 | four canonical-u64 leaves | `354318..354321` | `354326`, `354484`, `354642`, `354800` | lanes 0-3 decompose canonically |
| `Pi_RLC` | digest block 3 | four canonical-u64 leaves | `355554..355557` | `355562`, `355720`, `355878`, `356036` | lanes 0-3 decompose canonically |
| `Pi_RLC` | candidate order | 4 chunks per lane | same lanes | consecutive 16-bit windows | lane-major, little-endian chunks agree with `digestChunks` |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.ChunkOrder

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.OwnerCertificate
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine

set_option maxHeartbeats 1000000

abbrev CanonicalAssignment (assignment : Nat → Nat) :=
  ∀ column, assignment column < goldilocksP

/-! ## Fixed bit-window mathematics -/

/-- Integer represented by `width` little-endian canonical-u64 bits starting
at source-bit index `start`. -/
def bitWindowValue (source : Nat → Nat) (start width : Nat) : Nat :=
  (List.range width).foldl
    (fun value offset =>
      value + 2 ^ offset * source (CanonicalU64.bitCol (start + offset))) 0

private theorem range16 :
    List.range 16 =
      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] := by
  decide

private theorem range64 :
    List.range 64 =
      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
       16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
       32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
       48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] := by
  decide

/-- The canonical 64-bit value is the radix-`2^16` composition of its four
consecutive windows. This is pure integer arithmetic, independent of R1CS. -/
theorem bitsValue_eq_fourWindows (source : Nat → Nat) :
    bitsValue source =
      bitWindowValue source 0 16 +
        2 ^ 16 * bitWindowValue source 16 16 +
        2 ^ 32 * bitWindowValue source 32 16 +
        2 ^ 48 * bitWindowValue source 48 16 := by
  simp [bitsValue, bitWindowValue, range64, range16,
    CanonicalU64.bitCol]
  omega

/-- A little-endian Boolean window fits in its declared bit width. -/
theorem bitWindowValue_lt_pow
    (source : Nat → Nat) (start width : Nat)
    (boolean : ∀ offset, offset < width →
      source (CanonicalU64.bitCol (start + offset)) ≤ 1) :
    bitWindowValue source start width < 2 ^ width := by
  induction width with
  | zero => simp [bitWindowValue]
  | succ width inductionHypothesis =>
      have prefixBoolean : ∀ offset, offset < width →
          source (CanonicalU64.bitCol (start + offset)) ≤ 1 := by
        intro offset offsetLt
        exact boolean offset (by omega)
      have prefixBound := inductionHypothesis prefixBoolean
      have finalBound := boolean width (by omega)
      rw [bitWindowValue, List.range_succ, List.foldl_append]
      simp only [List.foldl_cons, List.foldl_nil]
      rw [← bitWindowValue]
      rw [Nat.pow_succ]
      have finalCases :
          source (CanonicalU64.bitCol (start + width)) = 0 ∨
            source (CanonicalU64.bitCol (start + width)) = 1 := by
        omega
      rcases finalCases with finalZero | finalOne
      · simp [finalZero]
        omega
      · simp [finalOne]
        omega

/-- Every 16-bit window is exactly the quotient/remainder chunk selected by
the independent transcript machine. -/
theorem bitWindowValue_eq_laneChunk
    (source : Nat → Nat)
    (boolean : ∀ bit, bit < 64 →
      source (CanonicalU64.bitCol bit) ≤ 1)
    (part : Fin 4) :
    bitWindowValue source (16 * part.val) 16 =
      (bitsValue source / (2 ^ (16 * part.val))) % 65536 := by
  have window0Bound : bitWindowValue source 0 16 < 2 ^ 16 :=
    bitWindowValue_lt_pow source 0 16 (by
      intro offset offsetLt
      simpa using boolean offset (by omega))
  have window1Bound : bitWindowValue source 16 16 < 2 ^ 16 :=
    bitWindowValue_lt_pow source 16 16 (by
      intro offset offsetLt
      exact boolean (16 + offset) (by omega))
  have window2Bound : bitWindowValue source 32 16 < 2 ^ 16 :=
    bitWindowValue_lt_pow source 32 16 (by
      intro offset offsetLt
      exact boolean (32 + offset) (by omega))
  have window3Bound : bitWindowValue source 48 16 < 2 ^ 16 :=
    bitWindowValue_lt_pow source 48 16 (by
      intro offset offsetLt
      exact boolean (48 + offset) (by omega))
  have decomposition := bitsValue_eq_fourWindows source
  have partLt : part.val < 4 := part.isLt
  have cases : part.val = 0 ∨ part.val = 1 ∨ part.val = 2 ∨ part.val = 3 := by
    omega
  rcases cases with h | h | h | h <;>
    simp [h] at decomposition ⊢ <;> omega

/-! ## Exact generated canonical-u64 pieces -/

namespace Artifact

def block0Lane0 : Piece :=
  { rowStart := 354301, rowEnd := 354370
    payload := .canonicalU64 351846 351854 }
def block0Lane1 : Piece :=
  { rowStart := 354474, rowEnd := 354543
    payload := .canonicalU64 351847 352012 }
def block0Lane2 : Piece :=
  { rowStart := 354647, rowEnd := 354716
    payload := .canonicalU64 351848 352170 }
def block0Lane3 : Piece :=
  { rowStart := 354820, rowEnd := 354889
    payload := .canonicalU64 351849 352328 }

def block1Lane0 : Piece :=
  { rowStart := 355597, rowEnd := 355666
    payload := .canonicalU64 353082 353090 }
def block1Lane1 : Piece :=
  { rowStart := 355770, rowEnd := 355839
    payload := .canonicalU64 353083 353248 }
def block1Lane2 : Piece :=
  { rowStart := 355943, rowEnd := 356012
    payload := .canonicalU64 353084 353406 }
def block1Lane3 : Piece :=
  { rowStart := 356116, rowEnd := 356185
    payload := .canonicalU64 353085 353564 }

def block2Lane0 : Piece :=
  { rowStart := 356893, rowEnd := 356962
    payload := .canonicalU64 354318 354326 }
def block2Lane1 : Piece :=
  { rowStart := 357066, rowEnd := 357135
    payload := .canonicalU64 354319 354484 }
def block2Lane2 : Piece :=
  { rowStart := 357239, rowEnd := 357308
    payload := .canonicalU64 354320 354642 }
def block2Lane3 : Piece :=
  { rowStart := 357412, rowEnd := 357481
    payload := .canonicalU64 354321 354800 }

def block3Lane0 : Piece :=
  { rowStart := 358189, rowEnd := 358258
    payload := .canonicalU64 355554 355562 }
def block3Lane1 : Piece :=
  { rowStart := 358362, rowEnd := 358431
    payload := .canonicalU64 355555 355720 }
def block3Lane2 : Piece :=
  { rowStart := 358535, rowEnd := 358604
    payload := .canonicalU64 355556 355878 }
def block3Lane3 : Piece :=
  { rowStart := 358708, rowEnd := 358777
    payload := .canonicalU64 355557 356036 }

theorem block0Lane0_mem : block0Lane0 ∈
    FPrimeFullHistoryRecursivePiRlcTranscriptRhos.owner.pieces := by decide
theorem block0Lane1_mem : block0Lane1 ∈
    FPrimeFullHistoryRecursivePiRlcTranscriptRhos.owner.pieces := by decide
theorem block0Lane2_mem : block0Lane2 ∈
    FPrimeFullHistoryRecursivePiRlcTranscriptRhos.owner.pieces := by decide
theorem block0Lane3_mem : block0Lane3 ∈
    FPrimeFullHistoryRecursivePiRlcTranscriptRhos.owner.pieces := by decide
theorem block1Lane0_mem : block1Lane0 ∈
    FPrimeFullHistoryRecursivePiRlcTranscriptRhos.owner.pieces := by decide
theorem block1Lane1_mem : block1Lane1 ∈
    FPrimeFullHistoryRecursivePiRlcTranscriptRhos.owner.pieces := by decide
theorem block1Lane2_mem : block1Lane2 ∈
    FPrimeFullHistoryRecursivePiRlcTranscriptRhos.owner.pieces := by decide
theorem block1Lane3_mem : block1Lane3 ∈
    FPrimeFullHistoryRecursivePiRlcTranscriptRhos.owner.pieces := by decide
theorem block2Lane0_mem : block2Lane0 ∈
    FPrimeFullHistoryRecursivePiRlcTranscriptRhos.owner.pieces := by decide
theorem block2Lane1_mem : block2Lane1 ∈
    FPrimeFullHistoryRecursivePiRlcTranscriptRhos.owner.pieces := by decide
theorem block2Lane2_mem : block2Lane2 ∈
    FPrimeFullHistoryRecursivePiRlcTranscriptRhos.owner.pieces := by decide
theorem block2Lane3_mem : block2Lane3 ∈
    FPrimeFullHistoryRecursivePiRlcTranscriptRhos.owner.pieces := by decide
theorem block3Lane0_mem : block3Lane0 ∈
    FPrimeFullHistoryRecursivePiRlcTranscriptRhos.owner.pieces := by decide
theorem block3Lane1_mem : block3Lane1 ∈
    FPrimeFullHistoryRecursivePiRlcTranscriptRhos.owner.pieces := by decide
theorem block3Lane2_mem : block3Lane2 ∈
    FPrimeFullHistoryRecursivePiRlcTranscriptRhos.owner.pieces := by decide
theorem block3Lane3_mem : block3Lane3 ∈
    FPrimeFullHistoryRecursivePiRlcTranscriptRhos.owner.pieces := by decide

end Artifact

/-! ## Canonical-u64 semantic bridge -/

/-- Source-column view of one exact renamed canonical-u64 leaf. -/
def laneSource (assignment : Nat → Nat) (fieldColumn bitStart : Nat) :
    Nat → Nat :=
  Relabel.assignment
    (canonicalU64ColumnMap fieldColumn bitStart) assignment

@[simp] theorem laneSource_var
    (assignment : Nat → Nat) (fieldColumn bitStart : Nat) :
    laneSource assignment fieldColumn bitStart CanonicalU64.varCol =
      assignment fieldColumn := by
  simp [laneSource, canonicalU64ColumnMap, Relabel.assignment,
    Relabel.column, CanonicalU64.varCol]

theorem laneSource_bit
    (assignment : Nat → Nat) (fieldColumn bitStart bit : Nat)
    (bitLt : bit < 64) :
    laneSource assignment fieldColumn bitStart (CanonicalU64.bitCol bit) =
      assignment (bitStart + bit) := by
  simp [laneSource, canonicalU64ColumnMap, Relabel.assignment,
    Relabel.column, CanonicalU64.bitCol]
  rw [List.getElem?_range (by omega : bit < 66)]
  rfl

private theorem canonical_bit_rows_prefix :
    (List.range 64).map
        (fun bit => bitRow (CanonicalU64.bitCol bit)) =
      CanonicalU64.rows.take 64 := by
  decide

private theorem canonical_bitRow_mem
    {bit : Nat} (bitLt : bit < 64) :
    bitRow (CanonicalU64.bitCol bit) ∈ CanonicalU64.rows := by
  have inGenerated : bitRow (CanonicalU64.bitCol bit) ∈
      (List.range 64).map
        (fun index => bitRow (CanonicalU64.bitCol index)) := by
    exact List.mem_map.mpr ⟨bit, List.mem_range.mpr bitLt, rfl⟩
  rw [canonical_bit_rows_prefix] at inGenerated
  exact List.mem_of_mem_take inGenerated

private theorem acceptedLaneRows
    {assignment : Nat → Nat}
    (accepted :
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Accepted assignment)
    (piece : Piece) (fieldColumn bitStart : Nat)
    (payload : piece.payload = .canonicalU64 fieldColumn bitStart)
    (member : piece ∈
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.owner.pieces) :
    Satisfies CanonicalU64.rows
      (laneSource assignment fieldColumn bitStart) := by
  have pieceAccepted := accepted piece member
  rw [Piece.Accepted, payload, Payload.Accepted] at pieceAccepted
  change Satisfies
      (CanonicalU64.rows.map
        (Relabel.row (canonicalU64ColumnMap fieldColumn bitStart)))
      assignment at pieceAccepted
  exact (Relabel.satisfies_mapped_iff CanonicalU64.rows
    (canonicalU64ColumnMap fieldColumn bitStart) assignment).mp pieceAccepted

/-- Semantic result of one exact canonical-u64 owner leaf. The result names
both the integer representation and all four independent-machine chunks. -/
structure LaneRefines
    (assignment : Nat → Nat)
    (canonical : CanonicalAssignment assignment)
    (fieldColumn bitStart : Nat) : Prop where
  fieldRecomposes :
    assignment fieldColumn =
      bitsValue (laneSource assignment fieldColumn bitStart)
  valueCanonical :
    bitsValue (laneSource assignment fieldColumn bitStart) < goldilocksP
  bitsBoolean : ∀ bit, bit < 64 →
    laneSource assignment fieldColumn bitStart
      (CanonicalU64.bitCol bit) ≤ 1
  chunks : ∀ part : Fin 4,
    bitWindowValue (laneSource assignment fieldColumn bitStart)
        (16 * part.val) 16 =
      (laneChunk (DigestRounds.fieldAt assignment canonical fieldColumn)
        part).val

/-- Satisfaction of one readable canonical-u64 leaf is sufficient for its
complete mathematical interpretation. This theorem is deliberately independent
of every generated owner and profile-specific column schedule. -/
theorem satisfyingLane_refines
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (fieldColumn bitStart : Nat)
    (satisfies : Satisfies CanonicalU64.rows
      (laneSource assignment fieldColumn bitStart)) :
    LaneRefines assignment canonical fieldColumn bitStart := by
  let source := laneSource assignment fieldColumn bitStart
  have sourceCanonical : ∀ column, source column < goldilocksP :=
    Relabel.canonical canonical
  have sourceOne : source 0 = 1 := by
    apply Relabel.constantOne
    · rfl
    · exact one
  have sourceRows : Satisfies CanonicalU64.rows source := by
    simpa [source] using satisfies
  have sound := canonicalU64_sound prime sourceCanonical sourceOne sourceRows
  have fieldRecomposes :
      assignment fieldColumn = bitsValue source := by
    simpa [source] using sound.1
  have boolean : ∀ bit, bit < 64 →
      source (CanonicalU64.bitCol bit) ≤ 1 := by
    intro bit bitLt
    apply bitRow_le_one prime (sourceCanonical _) sourceOne
    exact sourceRows _ (canonical_bitRow_mem bitLt)
  refine {
    fieldRecomposes := fieldRecomposes
    valueCanonical := sound.2
    bitsBoolean := boolean
    chunks := ?_
  }
  · intro part
    have chunk := bitWindowValue_eq_laneChunk source boolean part
    rw [← fieldRecomposes] at chunk
    simpa [source, laneChunk, DigestRounds.fieldAt,
      Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.chunkModulus]
      using chunk

/-- Owner acceptance of one named canonical-u64 piece is sufficient for the
complete lane interpretation; no neighboring sampler rows are used. -/
theorem acceptedLane_refines
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Accepted assignment)
    (piece : Piece) (fieldColumn bitStart : Nat)
    (payload : piece.payload = .canonicalU64 fieldColumn bitStart)
    (member : piece ∈
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.owner.pieces) :
    LaneRefines assignment canonical fieldColumn bitStart := by
  exact satisfyingLane_refines prime canonical one fieldColumn bitStart
    (acceptedLaneRows accepted piece fieldColumn bitStart payload member)

/-! ## Protocol → block → lane proof tree -/

/-- All four canonical lanes owned by digest block zero. -/
structure Block0Refines
    (assignment : Nat → Nat)
    (canonical : CanonicalAssignment assignment) : Prop where
  lane0 : LaneRefines assignment canonical 351846 351854
  lane1 : LaneRefines assignment canonical 351847 352012
  lane2 : LaneRefines assignment canonical 351848 352170
  lane3 : LaneRefines assignment canonical 351849 352328

/-- All four canonical lanes owned by digest block one. -/
structure Block1Refines
    (assignment : Nat → Nat)
    (canonical : CanonicalAssignment assignment) : Prop where
  lane0 : LaneRefines assignment canonical 353082 353090
  lane1 : LaneRefines assignment canonical 353083 353248
  lane2 : LaneRefines assignment canonical 353084 353406
  lane3 : LaneRefines assignment canonical 353085 353564

/-- All four canonical lanes owned by digest block two. -/
structure Block2Refines
    (assignment : Nat → Nat)
    (canonical : CanonicalAssignment assignment) : Prop where
  lane0 : LaneRefines assignment canonical 354318 354326
  lane1 : LaneRefines assignment canonical 354319 354484
  lane2 : LaneRefines assignment canonical 354320 354642
  lane3 : LaneRefines assignment canonical 354321 354800

/-- All four canonical lanes owned by digest block three. -/
structure Block3Refines
    (assignment : Nat → Nat)
    (canonical : CanonicalAssignment assignment) : Prop where
  lane0 : LaneRefines assignment canonical 355554 355562
  lane1 : LaneRefines assignment canonical 355555 355720
  lane2 : LaneRefines assignment canonical 355556 355878
  lane3 : LaneRefines assignment canonical 355557 356036

/-- Hierarchical semantic result for all 64 fixed candidates. Each block owns
four lane leaves, and each leaf owns four exact 16-bit chunks. -/
structure RefinesCandidateLanes
    (assignment : Nat → Nat)
    (canonical : CanonicalAssignment assignment) : Prop where
  block0 : Block0Refines assignment canonical
  block1 : Block1Refines assignment canonical
  block2 : Block2Refines assignment canonical
  block3 : Block3Refines assignment canonical

/-- Exact owner acceptance forces canonical decomposition and chunk semantics
for all sixteen digest lanes. This theorem still does not consume any
rejection or selection row. -/
theorem accepted_refines_candidateLanes
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Accepted assignment) :
    RefinesCandidateLanes assignment canonical := by
  refine {
    block0 := {
      lane0 := acceptedLane_refines prime canonical one accepted
        Artifact.block0Lane0 351846 351854 rfl Artifact.block0Lane0_mem
      lane1 := acceptedLane_refines prime canonical one accepted
        Artifact.block0Lane1 351847 352012 rfl Artifact.block0Lane1_mem
      lane2 := acceptedLane_refines prime canonical one accepted
        Artifact.block0Lane2 351848 352170 rfl Artifact.block0Lane2_mem
      lane3 := acceptedLane_refines prime canonical one accepted
        Artifact.block0Lane3 351849 352328 rfl Artifact.block0Lane3_mem
    }
    block1 := {
      lane0 := acceptedLane_refines prime canonical one accepted
        Artifact.block1Lane0 353082 353090 rfl Artifact.block1Lane0_mem
      lane1 := acceptedLane_refines prime canonical one accepted
        Artifact.block1Lane1 353083 353248 rfl Artifact.block1Lane1_mem
      lane2 := acceptedLane_refines prime canonical one accepted
        Artifact.block1Lane2 353084 353406 rfl Artifact.block1Lane2_mem
      lane3 := acceptedLane_refines prime canonical one accepted
        Artifact.block1Lane3 353085 353564 rfl Artifact.block1Lane3_mem
    }
    block2 := {
      lane0 := acceptedLane_refines prime canonical one accepted
        Artifact.block2Lane0 354318 354326 rfl Artifact.block2Lane0_mem
      lane1 := acceptedLane_refines prime canonical one accepted
        Artifact.block2Lane1 354319 354484 rfl Artifact.block2Lane1_mem
      lane2 := acceptedLane_refines prime canonical one accepted
        Artifact.block2Lane2 354320 354642 rfl Artifact.block2Lane2_mem
      lane3 := acceptedLane_refines prime canonical one accepted
        Artifact.block2Lane3 354321 354800 rfl Artifact.block2Lane3_mem
    }
    block3 := {
      lane0 := acceptedLane_refines prime canonical one accepted
        Artifact.block3Lane0 355554 355562 rfl Artifact.block3Lane0_mem
      lane1 := acceptedLane_refines prime canonical one accepted
        Artifact.block3Lane1 355555 355720 rfl Artifact.block3Lane1_mem
      lane2 := acceptedLane_refines prime canonical one accepted
        Artifact.block3Lane2 355556 355878 rfl Artifact.block3Lane2_mem
      lane3 := acceptedLane_refines prime canonical one accepted
        Artifact.block3Lane3 355557 356036 rfl Artifact.block3Lane3_mem
    }
  }

/-! ## Flattened machine-candidate correspondence -/

/-- The four canonical lanes exposed by a digest state. -/
def digestLanes (state : State) : Fin 4 → Field :=
  fun lane => state.lanes ⟨lane.val, by
    have laneLt := lane.isLt
    change lane.val < width
    simp only [width]
    omega⟩

/-- The sixteen lane-major chunks determined by one digest state. -/
def stateChunks (state : State) :
    Fin Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.chunksPerDigest →
      Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.Chunk :=
  digestChunks (digestLanes state)

/-- `digestBlock` candidates and its successor state are two projections of
the same digest execution. -/
theorem digestBlock_candidates_eq_stateChunks
    (state : State) (counter : Nat) :
    (digestBlock state counter).2 =
      stateChunks (digestBlock state counter).1 := by
  rfl

/-- Lane-major position of one `(lane, part)` pair. -/
def chunkPosition (lane part : Fin 4) :
    Fin Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.chunksPerDigest :=
  ⟨lane.val * 4 + part.val, by
    have laneLt := lane.isLt
    have partLt := part.isLt
    change lane.val * 4 + part.val <
      Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.chunksPerDigest
    simp only [Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.chunksPerDigest]
    omega⟩

theorem stateChunks_lane_part
    (state : State) (lane part : Fin 4) :
    stateChunks state (chunkPosition lane part) =
      laneChunk (digestLanes state lane) part := by
  exact digestChunks_lane_part (digestLanes state) lane part

private theorem fin4_value_cases (index : Fin 4) :
    index.val = 0 ∨ index.val = 1 ∨
      index.val = 2 ∨ index.val = 3 := by
  have indexLt := index.isLt
  omega

/-- Total four-way selector. The `Fin 4` index makes the final branch exact,
not a fallback for an out-of-range value. -/
def select4 {α : Type} (index : Fin 4)
    (value0 value1 value2 value3 : α) : α :=
  if index.val = 0 then value0
  else if index.val = 1 then value1
  else if index.val = 2 then value2
  else value3

/-- Exact digest-output field column at `(block, lane)`. -/
def fieldColumn (block lane : Fin 4) : Nat :=
  select4 block
    (select4 lane 351846 351847 351848 351849)
    (select4 lane 353082 353083 353084 353085)
    (select4 lane 354318 354319 354320 354321)
    (select4 lane 355554 355555 355556 355557)

/-- Exact first bit column at `(block, lane)`. -/
def bitStart (block lane : Fin 4) : Nat :=
  select4 block
    (select4 lane 351854 352012 352170 352328)
    (select4 lane 353090 353248 353406 353564)
    (select4 lane 354326 354484 354642 354800)
    (select4 lane 355562 355720 355878 356036)

/-- Indexed view of the hierarchical canonical-lane result. This theorem
keeps the protocol -> block -> lane ownership tree while giving later sampler
refinement one total lookup rather than sixteen implementation-specific
cases. -/
theorem accepted_refines_lane
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Accepted assignment)
    (block lane : Fin 4) :
    LaneRefines assignment canonical
      (fieldColumn block lane) (bitStart block lane) := by
  have lanes := accepted_refines_candidateLanes prime canonical one accepted
  rcases fin4_value_cases block with hb | hb | hb | hb <;>
    rcases fin4_value_cases lane with hl | hl | hl | hl <;>
    simp [fieldColumn, bitStart, select4, hb, hl]
  all_goals first
    | exact lanes.block0.lane0
    | exact lanes.block0.lane1
    | exact lanes.block0.lane2
    | exact lanes.block0.lane3
    | exact lanes.block1.lane0
    | exact lanes.block1.lane1
    | exact lanes.block1.lane2
    | exact lanes.block1.lane3
    | exact lanes.block2.lane0
    | exact lanes.block2.lane1
    | exact lanes.block2.lane2
    | exact lanes.block2.lane3
    | exact lanes.block3.lane0
    | exact lanes.block3.lane1
    | exact lanes.block3.lane2
    | exact lanes.block3.lane3

/-- Artifact-side integer value of one lane-major candidate. -/
def artifactChunkValue
    (assignment : Nat → Nat) (block lane part : Fin 4) : Nat :=
  bitWindowValue
    (laneSource assignment (fieldColumn block lane) (bitStart block lane))
    (16 * part.val) 16

/-- Exact artifact state after one of the four digest blocks. -/
def artifactBlockState
    (assignment : Nat → Nat)
    (canonical : CanonicalAssignment assignment)
    (block : Fin 4) : State :=
  select4 block
    (ScheduleRefinement.block0State assignment canonical)
    (ScheduleRefinement.block1State assignment canonical)
    (ScheduleRefinement.block2State assignment canonical)
    (ScheduleRefinement.block3State assignment canonical)

/-- Exact state from which the independent machine executes each block. -/
def machineBlockInput
    (assignment : Nat → Nat)
    (canonical : CanonicalAssignment assignment)
    (block : Fin 4) : State :=
  select4 block
    (ScheduleRefinement.afterEnterState assignment canonical)
    (ScheduleRefinement.block0State assignment canonical)
    (ScheduleRefinement.block1State assignment canonical)
    (ScheduleRefinement.block2State assignment canonical)

/-- Each artifact block state's first four lanes are exactly the field
columns named by the canonical-u64 schedule. -/
theorem artifactBlockState_lane
    {assignment : Nat → Nat}
    (canonical : CanonicalAssignment assignment)
    (block lane : Fin 4) :
    digestLanes (artifactBlockState assignment canonical block) lane =
      DigestRounds.fieldAt assignment canonical (fieldColumn block lane) := by
  rcases fin4_value_cases block with hb | hb | hb | hb <;>
    rcases fin4_value_cases lane with hl | hl | hl | hl <;>
    simp [artifactBlockState, digestLanes, fieldColumn, select4,
      ScheduleRefinement.block0State, ScheduleRefinement.block1State,
      ScheduleRefinement.block2State, ScheduleRefinement.block3State,
      DigestRounds.callOutputState, Schedule.Artifact.block0DigestCall,
      Schedule.Artifact.block1DigestCall, Schedule.Artifact.block2DigestCall,
      Schedule.Artifact.block3DigestCall, Poseidon2Call.Call.columnMap, hb, hl]

/-- All 64 artifact bit windows equal the chunks determined by the four exact
artifact digest states, with block/lane/part order explicit in the indices. -/
theorem accepted_refines_stateChunks
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Accepted assignment)
    (block lane part : Fin 4) :
    artifactChunkValue assignment block lane part =
      (stateChunks (artifactBlockState assignment canonical block)
        (chunkPosition lane part)).val := by
  have lanes := accepted_refines_candidateLanes prime canonical one accepted
  rw [stateChunks_lane_part, artifactBlockState_lane canonical]
  rcases fin4_value_cases block with hb | hb | hb | hb <;>
    rcases fin4_value_cases lane with hl | hl | hl | hl
  · simpa [artifactChunkValue, fieldColumn, bitStart, select4, hb, hl] using
      lanes.block0.lane0.chunks part
  · simpa [artifactChunkValue, fieldColumn, bitStart, select4, hb, hl] using
      lanes.block0.lane1.chunks part
  · simpa [artifactChunkValue, fieldColumn, bitStart, select4, hb, hl] using
      lanes.block0.lane2.chunks part
  · simpa [artifactChunkValue, fieldColumn, bitStart, select4, hb, hl] using
      lanes.block0.lane3.chunks part
  · simpa [artifactChunkValue, fieldColumn, bitStart, select4, hb, hl] using
      lanes.block1.lane0.chunks part
  · simpa [artifactChunkValue, fieldColumn, bitStart, select4, hb, hl] using
      lanes.block1.lane1.chunks part
  · simpa [artifactChunkValue, fieldColumn, bitStart, select4, hb, hl] using
      lanes.block1.lane2.chunks part
  · simpa [artifactChunkValue, fieldColumn, bitStart, select4, hb, hl] using
      lanes.block1.lane3.chunks part
  · simpa [artifactChunkValue, fieldColumn, bitStart, select4, hb, hl] using
      lanes.block2.lane0.chunks part
  · simpa [artifactChunkValue, fieldColumn, bitStart, select4, hb, hl] using
      lanes.block2.lane1.chunks part
  · simpa [artifactChunkValue, fieldColumn, bitStart, select4, hb, hl] using
      lanes.block2.lane2.chunks part
  · simpa [artifactChunkValue, fieldColumn, bitStart, select4, hb, hl] using
      lanes.block2.lane3.chunks part
  · simpa [artifactChunkValue, fieldColumn, bitStart, select4, hb, hl] using
      lanes.block3.lane0.chunks part
  · simpa [artifactChunkValue, fieldColumn, bitStart, select4, hb, hl] using
      lanes.block3.lane1.chunks part
  · simpa [artifactChunkValue, fieldColumn, bitStart, select4, hb, hl] using
      lanes.block3.lane2.chunks part
  · simpa [artifactChunkValue, fieldColumn, bitStart, select4, hb, hl] using
      lanes.block3.lane3.chunks part

/-- Accepted artifact equations make every artifact candidate equal the
corresponding output of the independent digest-block machine. This closes the
state-to-candidate boundary only; rejection and first-accepted selection are
proved by the sampler refinement layer. -/
theorem accepted_refines_machineCandidates
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Accepted assignment)
    (block lane part : Fin 4) :
    artifactChunkValue assignment block lane part =
      ((digestBlock
          (machineBlockInput assignment canonical block)
          block.val).2 (chunkPosition lane part)).val := by
  have stateSchedule :=
    ScheduleRefinement.accepted_refines_stateSchedule canonical one accepted
  have stateEq :
      (digestBlock
          (machineBlockInput assignment canonical block)
          block.val).1 =
        artifactBlockState assignment canonical block := by
    rcases fin4_value_cases block with hb | hb | hb | hb
    · simpa [machineBlockInput, artifactBlockState, select4, hb] using
        stateSchedule.block0
    · simpa [machineBlockInput, artifactBlockState, select4, hb] using
        stateSchedule.block1
    · simpa [machineBlockInput, artifactBlockState, select4, hb] using
        stateSchedule.block2
    · simpa [machineBlockInput, artifactBlockState, select4, hb] using
        stateSchedule.block3
  have candidatesEq := congrFun
    (digestBlock_candidates_eq_stateChunks
      (machineBlockInput assignment canonical block) block.val)
    (chunkPosition lane part)
  calc
    artifactChunkValue assignment block lane part =
        (stateChunks (artifactBlockState assignment canonical block)
          (chunkPosition lane part)).val :=
      accepted_refines_stateChunks prime canonical one accepted block lane part
    _ = (stateChunks
          (digestBlock
            (machineBlockInput assignment canonical block)
            block.val).1
          (chunkPosition lane part)).val := by
      rw [stateEq]
    _ = ((digestBlock
          (machineBlockInput assignment canonical block)
          block.val).2 (chunkPosition lane part)).val :=
      congrArg Fin.val candidatesEq.symm

/-- One-scalar transcript refinement grouped at the protocol/phase boundary:
the accepted artifact has both the exact four-block state schedule and all 64
machine-derived candidates. -/
structure RefinesOneScalarSchedule
    (assignment : Nat → Nat)
    (canonical : CanonicalAssignment assignment) : Prop where
  state : ScheduleRefinement.RefinesStateSchedule assignment canonical
  candidates : ∀ block lane part : Fin 4,
    artifactChunkValue assignment block lane part =
      ((digestBlock
          (machineBlockInput assignment canonical block)
          block.val).2 (chunkPosition lane part)).val

/-- Aggregate closure theorem for the digest/candidate phase of one scalar.
It deliberately stops before the sampler's rejection and selection phase. -/
theorem accepted_refines_oneScalarSchedule
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Accepted assignment) :
    RefinesOneScalarSchedule assignment canonical :=
  { state :=
      ScheduleRefinement.accepted_refines_stateSchedule canonical one accepted
    candidates :=
      accepted_refines_machineCandidates prime canonical one accepted }

end Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.ChunkOrder
