import Nightstream.Assurance.ConstraintMinimization
import Nightstream.Implementation.R1CS.Core.SeededPhi81

/-!
Chunked string-payload decoding for complete source artifacts.

Assurance tier: executable expansion with structural composition. A
multi-million-row source relation cannot ride Lean list literals, and its
proofs cannot ride whole-artifact `native_decide` evaluation. The wire is
therefore row-chunk-aligned: every payload string covers one fixed-size row
chunk, `sourceArtifactOf` is the concatenation of per-chunk expansions, and
each proof obligation about the artifact decomposes into bounded per-chunk
leaf certificates glued by the universal theorems in this file. No leaf's
cost grows with the artifact; only the leaf count does.

Owns: base64 and little-endian payload decoding; per-chunk row expansion
(CSR terms merged with clipped seeded-block terms in strict column order);
the artifact assembly; the composition theorems from chunk facts to
`WellFormed` and `CoversFullRelation`; the zero-evaluation exact-validation
discharge; and the exclusive-column override transport lemmas used by
generated removal counterexamples.

Does not own: the minimization theorems (`ConstraintMinimization`), the
sampler semantics (`SeededPhi81`), Rust emitter conformance (drift gates),
or any removal authority. The expanded value still faces the consuming
theorems' premises; this file only changes how those premises are proved.

Emits constraints: no.
-/

namespace Nightstream.Assurance.CompactSourceArtifact

open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS

/-- Base64 alphabet value, or 64 for padding/ignored characters. -/
def base64Value (c : UInt8) : UInt32 :=
  if c >= 65 && c <= 90 then c.toUInt32 - 65
  else if c >= 97 && c <= 122 then c.toUInt32 - 71
  else if c >= 48 && c <= 57 then c.toUInt32 + 4
  else if c == 43 then 62
  else if c == 47 then 63
  else 64

/-- Standard base64 decoding. Non-alphabet bytes are skipped, so `=` padding
and newlines are tolerated; the length checks downstream stay the authority
on payload shape. -/
def decodeBase64 (s : String) : ByteArray := Id.run do
  let bytes := s.toUTF8
  let mut out := ByteArray.emptyWithCapacity (bytes.size * 3 / 4 + 3)
  let mut acc : UInt32 := 0
  let mut bits : UInt32 := 0
  for i in [0:bytes.size] do
    let value := base64Value bytes[i]!
    if value < 64 then
      acc := (acc <<< 6) ||| value
      bits := bits + 6
      if bits >= 8 then
        bits := bits - 8
        out := out.push (UInt8.ofNat ((acc >>> bits).toNat &&& 0xFF))
  return out

def readU16s (bytes : ByteArray) : Option (Array Nat) := Id.run do
  if bytes.size % 2 != 0 then
    return none
  let count := bytes.size / 2
  let mut out := Array.emptyWithCapacity count
  for i in [0:count] do
    let base := i * 2
    out := out.push (bytes[base]!.toNat ||| (bytes[base + 1]!.toNat <<< 8))
  return some out

def readU32s (bytes : ByteArray) : Option (Array Nat) := Id.run do
  if bytes.size % 4 != 0 then
    return none
  let count := bytes.size / 4
  let mut out := Array.emptyWithCapacity count
  for i in [0:count] do
    let base := i * 4
    out := out.push (bytes[base]!.toNat ||| (bytes[base + 1]!.toNat <<< 8) |||
      (bytes[base + 2]!.toNat <<< 16) ||| (bytes[base + 3]!.toNat <<< 24))
  return some out

def readU64s (bytes : ByteArray) : Option (Array Nat) := Id.run do
  if bytes.size % 8 != 0 then
    return none
  let count := bytes.size / 8
  let mut out := Array.emptyWithCapacity count
  for i in [0:count] do
    let base := i * 8
    let mut word : Nat := 0
    for j in [0:8] do
      word := word ||| (bytes[base + j]!.toNat <<< (8 * j))
    out := out.push word
  return some out

/-- One family and the half-open source-row ranges it owns. -/
structure FamilyRanges where
  name : String
  ranges : List (Nat × Nat)
deriving DecidableEq, Repr

/-- Family owner of one source row under a range table, or `""`. -/
def familyAt (families : List FamilyRanges) (row : Nat) : String :=
  match families.find? (fun family =>
    family.ranges.any (fun range => range.1 <= row && row < range.2)) with
  | some family => family.name
  | none => ""

/-- One matrix's payload for one row chunk: per-row term counts, absolute
term columns in strict row-major column order, and per-term indices into the
shared value table. -/
structure MatrixChunk where
  rowCounts : String
  columns : String
  valueIndexes : String
deriving Repr

/-- One row chunk of the relation. Seeded blocks are clipped to the chunk's
row window during expansion, so a block may appear in two adjacent chunks. -/
structure ChunkWire where
  a : MatrixChunk
  b : MatrixChunk
  c : MatrixChunk
  seededBlocksA : List SeededPhi81.Block
deriving Repr

/-- Complete chunk-aligned string-payload source artifact. -/
structure Wire where
  schema : String
  profile : String
  scope : String
  diagnosticDigest : String
  fieldModulus : String
  totalRows : Nat
  columnCount : Nat
  constantOneColumn : Nat
  publicInputCount : Nat
  completeFamilies : List String
  valueTable : String
  families : List FamilyRanges
  chunkRows : Nat
  chunks : Array ChunkWire

namespace Wire

def chunkCount (wire : Wire) : Nat := wire.chunks.size

def chunkStart (wire : Wire) (chunk : Nat) : Nat := chunk * wire.chunkRows

def chunkLength (wire : Wire) (chunk : Nat) : Nat :=
  min wire.chunkRows (wire.totalRows - wire.chunkStart chunk)

def valueTableArray (wire : Wire) : Array Nat :=
  (readU64s (decodeBase64 wire.valueTable)).getD #[]

end Wire

/-- Merge two strictly column-increasing term lists. A shared column is a
malformation because the exporter never emits duplicate columns. -/
def mergeSortedTerms : List (Nat × Nat) → List (Nat × Nat) →
    Option (List (Nat × Nat))
  | [], rest => some rest
  | rest, [] => some rest
  | leftHead :: leftTail, rightHead :: rightTail =>
      if leftHead.1 < rightHead.1 then
        (mergeSortedTerms leftTail (rightHead :: rightTail)).map (leftHead :: ·)
      else if rightHead.1 < leftHead.1 then
        (mergeSortedTerms (leftHead :: leftTail) rightTail).map (rightHead :: ·)
      else
        none
termination_by left right => left.length + right.length

private def fieldNeg (value : Nat) : Nat :=
  let value := value % goldilocksP
  if value = 0 then 0 else goldilocksP - value

private def fieldSub (left right : Nat) : Nat :=
  (left % goldilocksP + fieldNeg right) % goldilocksP

/-- Array form of `SeededPhi81.rotatePhi81`: multiplication by `X` modulo
`Phi_81 = X^54 + X^27 + 1`. -/
private def rotateArray (current : Array Nat) : Array Nat := Id.run do
  let dimension := SeededPhi81.dimension
  let last := current[dimension - 1]!
  let mut next := Array.emptyWithCapacity dimension
  next := next.push (fieldNeg last)
  for i in [1:dimension] do
    next := next.push current[i - 1]!
  next := next.set! 27 (fieldSub current[26]! last)
  return next

/-- Efficient expansion of one seeded block into per-row term arrays indexed
from the block start. Semantics are `SeededPhi81.Block.terms` (message column
outer, message row inner, zero elision); the conformance tests pin the two
extensionally on every fixture class. -/
def expandSeededBlock (block : SeededPhi81.Block) :
    Option (Array (Array (Nat × Nat))) := Id.run do
  let dimension := SeededPhi81.dimension
  if !(decide block.Valid) then
    return none
  let rotations := block.baseRotations
  if rotations.length != block.kappa then
    return none
  let mut rows : Array (Array (Nat × Nat)) :=
    Array.replicate (dimension * block.kappa) (Array.emptyWithCapacity 0)
  let wordStarts := block.wordStarts.toArray
  let bitCount := wordStarts.size * block.wordWidth
  let mut output := 0
  for outputRotations in rotations do
    if outputRotations.length != block.messageCols then
      return none
    let mut messageCol := 0
    for base in outputRotations do
      let mut rotation := base.toArray
      if rotation.size != dimension then
        return none
      for messageRow in [0:dimension] do
        let bitIndex := messageRow * block.messageCols + messageCol
        if block.wordWidth > 0 && bitIndex < bitCount then
          let column := wordStarts[bitIndex / block.wordWidth]! +
            bitIndex % block.wordWidth
          for coordinate in [0:dimension] do
            let coefficient := rotation[coordinate]!
            if coefficient != 0 then
              let rowIndex := output * dimension + coordinate
              rows := rows.set! rowIndex (rows[rowIndex]!.push (column, coefficient))
        rotation := rotateArray rotation
      messageCol := messageCol + 1
    output := output + 1
  return some rows

/-- Expand one matrix chunk into per-row strictly column-sorted term lists
for rows `[start, start + length)`. Fails closed on any shape mismatch,
out-of-range column or value index, unsorted payload, invalid seeded block,
or duplicate column. -/
def expandMatrixChunk (matrix : MatrixChunk) (blocks : List SeededPhi81.Block)
    (start length columnCount : Nat) (valueTable : Array Nat) :
    Option (Array (List (Nat × Nat))) := Id.run do
  let some rowCounts := readU16s (decodeBase64 matrix.rowCounts) | return none
  let some columns := readU32s (decodeBase64 matrix.columns) | return none
  let some valueIndexes := readU16s (decodeBase64 matrix.valueIndexes) | return none
  if rowCounts.size != length then
    return none
  if columns.size != valueIndexes.size then
    return none
  let mut seededTerms : Array (Option (List (Nat × Nat))) :=
    Array.replicate length none
  for block in blocks do
    let some blockRows := expandSeededBlock block | return none
    let mut local_ := 0
    for terms in blockRows do
      let row := block.rowStart + local_
      -- Clip to this chunk's window; a block may straddle two chunks.
      if row >= start && row < start + length then
        if (seededTerms[row - start]!).isSome then
          return none
        let sorted := terms.qsort (fun left right => left.1 < right.1)
        seededTerms := seededTerms.set! (row - start) (some sorted.toList)
      local_ := local_ + 1
  let mut rows : Array (List (Nat × Nat)) := Array.emptyWithCapacity length
  let mut cursor := 0
  for offset in [0:length] do
    let count := rowCounts[offset]!
    if cursor + count > columns.size then
      return none
    let mut terms : Array (Nat × Nat) := Array.emptyWithCapacity count
    let mut previous : Option Nat := none
    for k in [cursor:cursor + count] do
      let column := columns[k]!
      let index := valueIndexes[k]!
      if column >= columnCount then
        return none
      if index >= valueTable.size then
        return none
      match previous with
      | some p => if column <= p then return none
      | none => pure ()
      previous := some column
      terms := terms.push (column, valueTable[index]!)
    cursor := cursor + count
    match seededTerms[offset]! with
    | none => rows := rows.push terms.toList
    | some seeded =>
        let some merged := mergeSortedTerms terms.toList seeded | return none
        rows := rows.push merged
  if cursor != columns.size then
    return none
  return some rows

/-- Expand one row chunk into indexed rows. Family names come from the
reviewed range table; an unowned row fails closed. -/
def expandChunk (wire : Wire) (chunk : Nat) : Option (List IndexedRow) := Id.run do
  let some chunkWire := wire.chunks[chunk]? | return none
  let start := wire.chunkStart chunk
  let length := wire.chunkLength chunk
  if length == 0 then
    return none
  let valueTable := wire.valueTableArray
  let some aRows := expandMatrixChunk chunkWire.a chunkWire.seededBlocksA
    start length wire.columnCount valueTable | return none
  let some bRows := expandMatrixChunk chunkWire.b [] start length
    wire.columnCount valueTable | return none
  let some cRows := expandMatrixChunk chunkWire.c [] start length
    wire.columnCount valueTable | return none
  let mut rows : Array IndexedRow := Array.emptyWithCapacity length
  for offset in [0:length] do
    let family := familyAt wire.families (start + offset)
    if family == "" then
      return none
    rows := rows.push
      { sourceIndex := start + offset
        family := family
        row := ⟨aRows[offset]!, bRows[offset]!, cRows[offset]!⟩ }
  return some rows.toList

/-- Per-chunk rows with a fail-closed empty default. A failed chunk yields
`[]`, which the coverage leaves reject because the source-index census no
longer matches `List.range totalRows`. -/
def rowsChunk (wire : Wire) (chunk : Nat) : List IndexedRow :=
  (expandChunk wire chunk).getD []

def artifactRows (wire : Wire) : List IndexedRow :=
  (List.range wire.chunkCount).flatMap (rowsChunk wire)

/-- The artifact the wire denotes. Every property of this value is proved
from bounded per-chunk facts through the composition theorems below. -/
def sourceArtifactOf (wire : Wire) : Artifact :=
  { schema := wire.schema
    profile := wire.profile
    scope := wire.scope
    diagnosticDigest := wire.diagnosticDigest
    fieldModulus := wire.fieldModulus
    totalRows := wire.totalRows
    columnCount := wire.columnCount
    constantOneColumn := wire.constantOneColumn
    publicInputCount := wire.publicInputCount
    completeFamilies := wire.completeFamilies
    rows := artifactRows wire }

-- ── Composition: chunk index censuses to full coverage ──────────────────

theorem flatMap_map {α β γ : Type} (f : α → List β) (g : β → γ) (l : List α) :
    (l.flatMap f).map g = l.flatMap (fun a => (f a).map g) := by
  induction l with
  | nil => rfl
  | cons head tail ih => simp [List.flatMap_cons, ih]

/-- Concatenating consecutive index windows yields one window. -/
theorem range'_flatMap (count width : Nat) (len : Nat → Nat)
    (bound : ∀ k < count, len k = width) :
    ((List.range count).flatMap fun k => List.range' (k * width) (len k)) =
      List.range (count * width) := by
  induction count with
  | zero => simp
  | succ n ih =>
      have step : ∀ k < n, len k = width := fun k hk =>
        bound k (Nat.lt_succ_of_lt hk)
      rw [List.range_succ, List.flatMap_append, ih step]
      simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil,
        bound n (Nat.lt_succ_self n)]
      have append := List.range'_append_1 (s := 0) (m := n * width) (n := width)
      simp only [Nat.zero_add] at append
      rw [List.range_eq_range', List.range_eq_range', append, Nat.succ_mul]

/-- The final, possibly short, window still composes. -/
theorem range'_flatMap_with_tail (count width total : Nat) (len : Nat → Nat)
    (full : ∀ k, k + 1 < count → len k = width)
    (last : count ≠ 0 → (count - 1) * width + len (count - 1) = total)
    (lead : count ≠ 0 → (count - 1) * width ≤ total)
    (empty : count = 0 → total = 0) :
    ((List.range count).flatMap fun k => List.range' (k * width) (len k)) =
      List.range total := by
  cases count with
  | zero => simp [empty rfl]
  | succ n =>
      rw [List.range_succ, List.flatMap_append]
      have leading :
          ((List.range n).flatMap fun k => List.range' (k * width) (len k)) =
            List.range (n * width) := by
        apply range'_flatMap
        intro k hk
        exact full k (Nat.succ_lt_succ hk)
      rw [leading]
      simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil]
      have lastLen : n * width + len n = total := by
        simpa using last (Nat.succ_ne_zero n)
      have append := List.range'_append_1 (s := 0) (m := n * width) (n := len n)
      simp only [Nat.zero_add] at append
      rw [List.range_eq_range', List.range_eq_range', append, lastLen]

/-- Strict source-index increase follows from an exact index census. -/
theorem strictlyIncreasing_of_index_census (rows : List IndexedRow)
    (start count : Nat)
    (census : rows.map (fun row => row.sourceIndex) = List.range' start count) :
    Artifact.strictlyIncreasingSourceRows rows = true := by
  induction rows generalizing start count with
  | nil => rfl
  | cons head tail ih =>
      cases count with
      | zero => simp at census
      | succ n =>
          simp [List.range'] at census
          obtain ⟨headIndex, tailCensus⟩ := census
          cases tail with
          | nil => rfl
          | cons second rest =>
              have tailIncreasing := ih (start + 1) n tailCensus
              have secondIndex : second.sourceIndex = start + 1 := by
                cases n with
                | zero => simp at tailCensus
                | succ m =>
                    simp [List.range'] at tailCensus
                    exact tailCensus.1
              unfold Artifact.strictlyIncreasingSourceRows
              rw [headIndex, secondIndex]
              simp [tailIncreasing]

-- ── Zero-evaluation exact validation ─────────────────────────────────────

/-- Exact validation of an artifact against itself costs nothing once
`WellFormed` is proved structurally: the decidable proposition is discharged
by its proof, never by evaluating the artifact-scale Boolean. -/
theorem exactValidation_self {artifact : Artifact}
    (wellFormed : artifact.WellFormed) :
    Artifact.ExactValidation artifact artifact = true :=
  decide_eq_true ⟨rfl, wellFormed⟩

-- ── Exclusive-column override transport ──────────────────────────────────

/-- Point override of an assignment at one column. -/
def overrideAt (base : Nat → Field) (column : Nat) (value : Field) :
    Nat → Field :=
  fun index => if index = column then value else base index

theorem linearEval_overrideAt (base : Nat → Field) (column : Nat)
    (value : Field) (terms : List (Nat × Nat))
    (absent : ∀ term ∈ terms, term.1 ≠ column) :
    Algebraic.linearEval (overrideAt base column value) terms =
      Algebraic.linearEval base terms := by
  induction terms with
  | nil => rfl
  | cons head tail ih =>
      have headAbsent : head.1 ≠ column := absent head (List.mem_cons_self)
      have tailAbsent : ∀ term ∈ tail, term.1 ≠ column := fun term member =>
        absent term (List.mem_cons_of_mem head member)
      simp [Algebraic.linearEval, overrideAt, headAbsent, ih tailAbsent]

/-- A row that never reads the overridden column keeps its semantics. -/
theorem holds_overrideAt (base : Nat → Field) (column : Nat) (value : Field)
    (row : Numeric.Row)
    (absentA : ∀ term ∈ row.a, term.1 ≠ column)
    (absentB : ∀ term ∈ row.b, term.1 ≠ column)
    (absentC : ∀ term ∈ row.c, term.1 ≠ column) :
    Algebraic.Holds (overrideAt base column value) row ↔
      Algebraic.Holds base row := by
  unfold Algebraic.Holds
  rw [linearEval_overrideAt base column value row.a absentA,
    linearEval_overrideAt base column value row.b absentB,
    linearEval_overrideAt base column value row.c absentC]

/-- Executable per-row guard: the row does not read the column. -/
def rowAvoidsColumn (column : Nat) (row : Numeric.Row) : Bool :=
  row.a.all (fun term => decide (term.1 ≠ column)) &&
    row.b.all (fun term => decide (term.1 ≠ column)) &&
      row.c.all (fun term => decide (term.1 ≠ column))

theorem holds_overrideAt_of_avoids (base : Nat → Field) (column : Nat)
    (value : Field) (row : Numeric.Row)
    (avoids : rowAvoidsColumn column row = true)
    (holds : Algebraic.Holds base row) :
    Algebraic.Holds (overrideAt base column value) row := by
  unfold rowAvoidsColumn at avoids
  simp only [Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq] at avoids
  exact (holds_overrideAt base column value row
    avoids.1.1 avoids.1.2 avoids.2).mpr holds

/-- Executable per-chunk guard used by generated leaf certificates: every
row outside `family` both holds on the background and avoids the column. -/
def chunkSupportsOverride (background : Nat → Field) (column : Nat)
    (family : String) (rows : List IndexedRow) : Bool :=
  rows.all fun row =>
    decide (row.family = family) ||
      (rowAvoidsColumn column row.row &&
        decide (Algebraic.Holds background row.row))

/-- Transport from chunk guards to the acceptance premise of a removal
counterexample: on the override assignment, every family other than the
removed one still holds on every one of its rows. -/
theorem familyHolds_overrideAt_of_chunks (artifact : Artifact)
    (background : Nat → Field) (column : Nat) (value : Field)
    (removed : String)
    (chunks : List (List IndexedRow))
    (assembled : artifact.rows = chunks.flatMap id)
    (supported : ∀ chunk ∈ chunks,
      chunkSupportsOverride background column removed chunk = true) :
    ∀ family, family ≠ removed →
      FamilyHolds artifact family (overrideAt background column value) := by
  intro family different row rowMember rowFamily
  have rowInChunks : row ∈ chunks.flatMap id := by
    rw [← assembled]; exact rowMember
  rcases List.mem_flatMap.mp rowInChunks with ⟨chunk, chunkMember, rowInChunk⟩
  have guard := supported chunk chunkMember
  unfold chunkSupportsOverride at guard
  rw [List.all_eq_true] at guard
  have rowGuard := guard row rowInChunk
  simp only [Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq] at rowGuard
  cases rowGuard with
  | inl sameFamily =>
      exact absurd (rowFamily ▸ sameFamily) different
  | inr avoidsAndHolds =>
      exact holds_overrideAt_of_avoids background column value row.row
        avoidsAndHolds.1 avoidsAndHolds.2


-- ── Scalar-geometry leaf predicates ─────────────────────────────────────

/-- `Artifact.rowWellFormed` with the geometry passed as scalars, so leaf
certificates never mention (and never force) the assembled artifact. -/
def rowWellFormedAt (totalRows columnCount : Nat) (row : IndexedRow) : Bool :=
  decide (row.sourceIndex < totalRows ∧ row.family ≠ "") &&
    Artifact.termsWellFormed columnCount row.row.a &&
      Artifact.termsWellFormed columnCount row.row.b &&
        Artifact.termsWellFormed columnCount row.row.c

theorem rowWellFormedAt_eq (artifact : Artifact) (row : IndexedRow) :
    rowWellFormedAt artifact.totalRows artifact.columnCount row =
      Artifact.rowWellFormed artifact row := rfl

-- ── Assembly: chunk facts to full coverage and well-formedness ───────────

/-- The artifact rows are exactly the concatenation of the chunk lists. -/
theorem artifactRows_eq_flatMap_id (wire : Wire) :
    artifactRows wire =
      ((List.range wire.chunkCount).map (rowsChunk wire)).flatMap id := by
  unfold artifactRows
  rw [List.flatMap_def]
  simp [List.flatMap_def]

/-- Full source-index coverage from per-chunk index censuses plus the chunk
arithmetic of the wire. -/
theorem covers_indexes_of_chunks (wire : Wire)
    (censuses : ∀ k, k < wire.chunkCount →
      (rowsChunk wire k).map (fun row => row.sourceIndex) =
        List.range' (wire.chunkStart k) (wire.chunkLength k))
    (fullChunks : ∀ k, k + 1 < wire.chunkCount →
      wire.chunkLength k = wire.chunkRows)
    (lastChunk : wire.chunkCount ≠ 0 →
      (wire.chunkCount - 1) * wire.chunkRows +
        wire.chunkLength (wire.chunkCount - 1) = wire.totalRows)
    (leadChunks : wire.chunkCount ≠ 0 →
      (wire.chunkCount - 1) * wire.chunkRows ≤ wire.totalRows)
    (noChunks : wire.chunkCount = 0 → wire.totalRows = 0) :
    (artifactRows wire).map (fun row => row.sourceIndex) =
      List.range wire.totalRows := by
  unfold artifactRows
  rw [flatMap_map]
  have pointwise :
      ((List.range wire.chunkCount).flatMap fun k =>
          (rowsChunk wire k).map fun row => row.sourceIndex) =
        (List.range wire.chunkCount).flatMap fun k =>
          List.range' (k * wire.chunkRows) (wire.chunkLength k) := by
    apply List.flatMap_congr
    intro k membership
    exact censuses k (List.mem_range.mp membership)
  rw [pointwise]
  exact range'_flatMap_with_tail wire.chunkCount wire.chunkRows wire.totalRows
    wire.chunkLength fullChunks lastChunk leadChunks noChunks

/-- Full coverage from index censuses plus per-chunk family membership. -/
theorem coversFullRelation_of_chunks (wire : Wire)
    (censuses : ∀ k, k < wire.chunkCount →
      (rowsChunk wire k).map (fun row => row.sourceIndex) =
        List.range' (wire.chunkStart k) (wire.chunkLength k))
    (fullChunks : ∀ k, k + 1 < wire.chunkCount →
      wire.chunkLength k = wire.chunkRows)
    (lastChunk : wire.chunkCount ≠ 0 →
      (wire.chunkCount - 1) * wire.chunkRows +
        wire.chunkLength (wire.chunkCount - 1) = wire.totalRows)
    (leadChunks : wire.chunkCount ≠ 0 →
      (wire.chunkCount - 1) * wire.chunkRows ≤ wire.totalRows)
    (noChunks : wire.chunkCount = 0 → wire.totalRows = 0)
    (membership : ∀ k, k < wire.chunkCount →
      (rowsChunk wire k).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) :
    (sourceArtifactOf wire).CoversFullRelation := by
  constructor
  · exact covers_indexes_of_chunks wire censuses fullChunks lastChunk
      leadChunks noChunks
  · intro row rowMember
    have inChunks : row ∈ (List.range wire.chunkCount).flatMap (rowsChunk wire) :=
      rowMember
    rcases List.mem_flatMap.mp inChunks with ⟨k, kMember, rowInChunk⟩
    have chunkAll := membership k (List.mem_range.mp kMember)
    rw [List.all_eq_true] at chunkAll
    simpa using chunkAll row rowInChunk

/-- Complete well-formedness from bounded chunk facts. The scalar conjuncts
arrive as one small decidable pack; the row facts arrive per chunk against
the scalar geometry; monotonicity is inherited from the index census. -/
theorem wellFormed_of_chunks (wire : Wire)
    (scalars : (sourceArtifactOf wire).schema = Artifact.supportedSchema ∧
      (sourceArtifactOf wire).profile ≠ "" ∧
      (sourceArtifactOf wire).scope ∈ Artifact.scopes ∧
      (sourceArtifactOf wire).diagnosticDigest ≠ "" ∧
      (sourceArtifactOf wire).fieldModulus = Artifact.goldilocksModulusDecimal ∧
      0 < (sourceArtifactOf wire).totalRows ∧
      0 < (sourceArtifactOf wire).columnCount ∧
      0 < (sourceArtifactOf wire).publicInputCount ∧
      (sourceArtifactOf wire).publicInputCount ≤ (sourceArtifactOf wire).columnCount ∧
      (sourceArtifactOf wire).constantOneColumn < (sourceArtifactOf wire).publicInputCount ∧
      (sourceArtifactOf wire).completeFamilies.Nodup ∧
      (sourceArtifactOf wire).completeFamilies.all
        (fun family => decide (family ≠ "")) = true)
    (indexCensus : (artifactRows wire).map (fun row => row.sourceIndex) =
      List.range wire.totalRows)
    (rowFacts : ∀ k, k < wire.chunkCount →
      (rowsChunk wire k).all
        (rowWellFormedAt wire.totalRows wire.columnCount) = true)
    (presence : (sourceArtifactOf wire).completeFamilies.all
      (fun family =>
        (sourceArtifactOf wire).rows.any
          (fun row => decide (row.family = family))) = true) :
    (sourceArtifactOf wire).WellFormed := by
  obtain ⟨schema, profile, scope, digest, modulus, totalPos, columnPos,
    publicPos, publicLe, constantLt, nodup, nonempty⟩ := scalars
  refine ⟨schema, profile, scope, digest, modulus, totalPos, columnPos,
    publicPos, publicLe, constantLt, ?_, nodup, nonempty, presence, ?_, ?_⟩
  · -- rows ≠ []
    intro empty
    have censusEmpty := indexCensus
    rw [show (sourceArtifactOf wire).rows = artifactRows wire from rfl] at empty
    rw [empty] at censusEmpty
    have rangeEmpty : List.range wire.totalRows = [] := censusEmpty.symm
    have zero : wire.totalRows = 0 := by
      simpa [List.range_eq_nil] using rangeEmpty
    have pos : 0 < wire.totalRows := totalPos
    omega
  · -- strictly increasing source rows
    have census' : (artifactRows wire).map (fun row => row.sourceIndex) =
        List.range' 0 wire.totalRows := by
      rw [indexCensus, List.range_eq_range']
    exact strictlyIncreasing_of_index_census (artifactRows wire) 0
      wire.totalRows census'
  · -- every row is well formed
    rw [show (sourceArtifactOf wire).rows = artifactRows wire from rfl,
      List.all_eq_true]
    intro row rowMember
    rcases List.mem_flatMap.mp rowMember with ⟨k, kMember, rowInChunk⟩
    have chunkAll := rowFacts k (List.mem_range.mp kMember)
    rw [List.all_eq_true] at chunkAll
    have fact := chunkAll row rowInChunk
    rw [← rowWellFormedAt_eq (sourceArtifactOf wire) row]
    exact fact


-- ── Counterexample assembly from shared leaves ──────────────────────────

/-- Combined per-chunk guard for every override of a classification batch:
each row either belongs to the pair's family or avoids the pair's column.
One leaf per chunk serves all families. -/
def chunkGuardsOverrides (pairs : List (Nat × String))
    (rows : List IndexedRow) : Bool :=
  rows.all fun row => pairs.all fun pair =>
    decide (row.family = pair.2) || rowAvoidsColumn pair.1 row.row

theorem chunkSupportsOverride_of_guards
    (background : Nat → Field) (pairs : List (Nat × String))
    (column : Nat) (family : String) (rows : List IndexedRow)
    (guards : chunkGuardsOverrides pairs rows = true)
    (holds : rows.all
      (fun row => decide (Algebraic.Holds background row.row)) = true)
    (member : (column, family) ∈ pairs) :
    chunkSupportsOverride background column family rows = true := by
  unfold chunkGuardsOverrides at guards
  unfold chunkSupportsOverride
  rw [List.all_eq_true] at guards holds ⊢
  intro row rowMember
  have rowGuards := guards row rowMember
  rw [List.all_eq_true] at rowGuards
  have pairGuard := rowGuards (column, family) member
  have rowHolds := holds row rowMember
  simp only [Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq] at pairGuard ⊢
  cases pairGuard with
  | inl same => exact Or.inl same
  | inr avoids =>
      exact Or.inr ⟨avoids, by simpa using rowHolds⟩

/-- Background assignment as a function over canonical residues. -/
def backgroundFn (values : Array Nat) : Nat → Field :=
  fun index => ((values.getD index 0 : Nat) : Field)

/-- The counterexample values: the background with one column overridden. -/
def overriddenValues (values : Array Nat) (column value : Nat) : List Field :=
  (List.range values.size).map fun index =>
    if index = column then (value : Field) else backgroundFn values index

def mkCounterexample (values : Array Nat) (column value : Nat)
    (removed : String) : RemovalCounterexample :=
  { removedFamily := removed
    values := overriddenValues values column value }

theorem overriddenValues_length (values : Array Nat) (column value : Nat) :
    (overriddenValues values column value).length = values.size := by
  simp [overriddenValues]

theorem overriddenValues_getD (values : Array Nat) (column value : Nat)
    (index : Nat) (inRange : column < values.size) :
    (overriddenValues values column value).getD index 0 =
      overrideAt (backgroundFn values) column (value : Field) index := by
  unfold overriddenValues overrideAt
  by_cases bounded : index < values.size
  · rw [List.getD_eq_getElem?_getD]
    rw [List.getElem?_map]
    simp [bounded]
  · rw [List.getD_eq_getElem?_getD]
    rw [List.getElem?_map]
    have outOfRange : values.size ≤ index := Nat.le_of_not_lt bounded
    have notColumn : index ≠ column := by omega
    simp [bounded, notColumn, backgroundFn, Array.getD_eq_getD_getElem?,
      Array.getElem?_eq_none outOfRange]

theorem mkCounterexample_assignment (values : Array Nat)
    (column value : Nat) (removed : String)
    (inRange : column < values.size) :
    (mkCounterexample values column value removed).assignment =
      overrideAt (backgroundFn values) column (value : Field) := by
  funext index
  unfold mkCounterexample RemovalCounterexample.assignment
  exact overriddenValues_getD values column value index inRange

/-- Complete validity of a generated removal counterexample from bounded
leaves: the shared background-holds and override-guard chunk facts, one
membership leaf and one violation leaf for the removed family, and small
scalar facts. -/
theorem mkCounterexample_valid (wire : Wire) (values : Array Nat)
    (pairs : List (Nat × String)) (column value : Nat) (removed : String)
    (plan : List String)
    (planFamilies : ∀ family ∈ plan,
      family ∈ (sourceArtifactOf wire).completeFamilies)
    (sizeEq : values.size = wire.columnCount)
    (inRange : column < values.size)
    (constantOne :
      overrideAt (backgroundFn values) column (value : Field)
        wire.constantOneColumn = 1)
    (pairMember : (column, removed) ∈ pairs)
    (guards : ∀ k, k < wire.chunkCount →
      chunkGuardsOverrides pairs (rowsChunk wire k) = true)
    (holds : ∀ k, k < wire.chunkCount →
      (rowsChunk wire k).all
        (fun row => decide (Algebraic.Holds (backgroundFn values) row.row)) =
          true)
    (violated : IndexedRow) (violatedChunk : Nat)
    (violatedInChunk : violatedChunk < wire.chunkCount ∧
      violated ∈ rowsChunk wire violatedChunk)
    (violation : ¬ Algebraic.Holds
      (overrideAt (backgroundFn values) column (value : Field)) violated.row) :
    (mkCounterexample values column value removed).Valid
      (sourceArtifactOf wire) plan := by
  have assignmentEq := mkCounterexample_assignment values column value removed inRange
  refine ⟨planFamilies, ?_, ?_, ?_, ?_⟩
  · -- length
    show (overriddenValues values column value).length =
      (sourceArtifactOf wire).columnCount
    rw [overriddenValues_length]
    exact sizeEq
  · -- constant one
    rw [assignmentEq]
    exact constantOne
  · -- acceptance of every retained family
    rw [assignmentEq]
    intro family familyMember
    have familyFacts :=
      Nightstream.SuperNeo.CheckPlan.mem_without_iff.mp familyMember
    refine familyHolds_overrideAt_of_chunks (sourceArtifactOf wire)
      (backgroundFn values) column (value : Field) removed
      ((List.range wire.chunkCount).map (rowsChunk wire))
      (artifactRows_eq_flatMap_id wire) ?_ family familyFacts.2
    intro chunk chunkMember
    rcases List.mem_map.mp chunkMember with ⟨k, kMember, rfl⟩
    have kBound := List.mem_range.mp kMember
    exact chunkSupportsOverride_of_guards (backgroundFn values) pairs column
      removed (rowsChunk wire k) (guards k kBound) (holds k kBound) pairMember
  · -- the target fails
    rw [assignmentEq]
    intro target
    apply violation
    apply target.2 violated
    show violated ∈ artifactRows wire
    unfold artifactRows
    exact List.mem_flatMap.mpr
      ⟨violatedChunk, List.mem_range.mpr violatedInChunk.1, violatedInChunk.2⟩

-- ── Shared background assignments ────────────────────────────────────────

/-- Decode one string-encoded assignment of canonical `u64` residues. -/
def decodeAssignment (payload : String) : Option (Array Nat) :=
  readU64s (decodeBase64 payload)

end Nightstream.Assurance.CompactSourceArtifact
