import Nightstream.Assurance.ConstraintMinimization
import Nightstream.Implementation.R1CS.Core.SeededPhi81

/-!
String-payload decoding for complete source artifacts.

Assurance tier: executable expansion. A multi-million-row source relation
cannot ride Lean list literals, so the generated artifact carries base64
string payloads (row term counts, term columns, value-table indices, a
value table) plus compact seeded Phi81 blocks and a family range table.
`expand` decodes them natively into the exact `Artifact` value consumed by
the existing minimization theorems; nothing downstream changes.

Owns: base64 and little-endian payload decoding; efficient seeded-block row
expansion (one Phi81 rotation step per message row, extensionally pinned to
`SeededPhi81.Block.rows` by the conformance tests); row assembly in strict
column order; family assignment from the reviewed range table; fail-closed
`none` on any malformation.

Does not own: the minimization theorems (`ConstraintMinimization`), the
sampler semantics (`SeededPhi81`), Rust emitter conformance (drift gates),
or any removal authority. The expanded value must still pass `WellFormed`,
`CoversFullRelation`, and the exact validation of the consuming theorems.

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
and newlines are tolerated; the byte length checks downstream stay the
authority on payload shape. -/
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

/-- One matrix payload: per-row term counts, absolute term columns in strict
row-major column order, and per-term indices into the shared value table.
Seeded Phi81 blocks stay compact; the payload may also carry ordinary terms
on seeded rows, merged in strict column order. -/
structure MatrixWire where
  rowCounts : String
  columns : String
  valueIndexes : String
  seededBlocks : List SeededPhi81.Block

/-- One family and the half-open source-row ranges it owns. -/
structure FamilyRanges where
  name : String
  ranges : List (Nat × Nat)
deriving DecidableEq, Repr

/-- Complete string-payload source artifact. -/
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
  a : MatrixWire
  b : MatrixWire
  c : MatrixWire

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

/-- Expand one matrix payload into per-row strictly column-sorted term lists.
Fails closed on any shape mismatch, out-of-range column or value index,
unsorted payload, invalid seeded block, or duplicate column. -/
def MatrixWire.expandRows (wire : MatrixWire) (totalRows columnCount : Nat)
    (valueTable : Array Nat) : Option (Array (List (Nat × Nat))) := Id.run do
  let some rowCounts := readU16s (decodeBase64 wire.rowCounts) | return none
  let some columns := readU32s (decodeBase64 wire.columns) | return none
  let some valueIndexes := readU16s (decodeBase64 wire.valueIndexes) | return none
  if rowCounts.size != totalRows then
    return none
  if columns.size != valueIndexes.size then
    return none
  let mut seededTerms : Array (Option (List (Nat × Nat))) :=
    Array.replicate totalRows none
  for block in wire.seededBlocks do
    let some blockRows := expandSeededBlock block | return none
    let mut local_ := 0
    for terms in blockRows do
      let row := block.rowStart + local_
      if row >= totalRows then
        return none
      if (seededTerms[row]!).isSome then
        return none
      let sorted := terms.qsort (fun left right => left.1 < right.1)
      seededTerms := seededTerms.set! row (some sorted.toList)
      local_ := local_ + 1
  let mut rows : Array (List (Nat × Nat)) := Array.emptyWithCapacity totalRows
  let mut cursor := 0
  for row in [0:totalRows] do
    let count := rowCounts[row]!
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
    match seededTerms[row]! with
    | none => rows := rows.push terms.toList
    | some seeded =>
        let some merged := mergeSortedTerms terms.toList seeded | return none
        rows := rows.push merged
  if cursor != columns.size then
    return none
  return some rows

/-- Family name for every source row, from the reviewed range table. Fails
closed unless the ranges exactly tile `[0, totalRows)` without overlap. -/
def familyNames (families : List FamilyRanges) (totalRows : Nat) :
    Option (Array String) := Id.run do
  let mut names : Array String := Array.replicate totalRows ""
  let mut covered := 0
  for family in families do
    if family.name = "" then
      return none
    for range in family.ranges do
      if range.1 >= range.2 || range.2 > totalRows then
        return none
      for row in [range.1:range.2] do
        if names[row]! != "" then
          return none
        names := names.set! row family.name
        covered := covered + 1
  if covered != totalRows then
    return none
  return some names

/-- Decode and expand the complete artifact. `none` is the fail-closed
outcome; the expanded value still faces `WellFormed` and coverage checks in
the consuming theorems. -/
def expand (wire : Wire) : Option Artifact := Id.run do
  let some valueTable := readU64s (decodeBase64 wire.valueTable) | return none
  if valueTable.any (fun value => value == 0 || value >= Numeric.modulus) then
    return none
  let some aRows := wire.a.expandRows wire.totalRows wire.columnCount valueTable | return none
  let some bRows := wire.b.expandRows wire.totalRows wire.columnCount valueTable | return none
  let some cRows := wire.c.expandRows wire.totalRows wire.columnCount valueTable | return none
  let some names := familyNames wire.families wire.totalRows | return none
  let mut rows : Array IndexedRow := Array.emptyWithCapacity wire.totalRows
  for row in [0:wire.totalRows] do
    rows := rows.push
      { sourceIndex := row
        family := names[row]!
        row := ⟨aRows[row]!, bRows[row]!, cRows[row]!⟩ }
  return some
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
      rows := rows.toList }

/-- Override a background assignment at named columns. Generated removal
counterexamples share one string-encoded background assignment and carry
only their mutated columns. -/
def applyOverrides (base : Array Nat) (overrides : List (Nat × Nat)) :
    Option (Array Nat) := Id.run do
  let mut out := base
  for override in overrides do
    if override.1 >= out.size then
      return none
    out := out.set! override.1 override.2
  return some out

/-- Decode one string-encoded assignment of canonical `u64` residues. -/
def decodeAssignment (payload : String) : Option (Array Nat) :=
  readU64s (decodeBase64 payload)

end Nightstream.Assurance.CompactSourceArtifact
