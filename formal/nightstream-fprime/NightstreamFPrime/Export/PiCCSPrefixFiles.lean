import NightstreamFPrime.Export.Stage1.PiCCSPublicReplay
import NightstreamFPrime.Export.Stage1.PiCCSFreshPrefix

/-!
Read saved fresh and carried prefixes, and retain their next ordered fold.
Legacy JSONL framing and binary range coverage are checked here. The caller
owns challenge derivation and source provenance. Binary readers check the
first row; consumers must decode every remaining byte with decodeFields.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.PiCCSPrefixFiles

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Layout
open NightstreamFPrime.Export.Stage1
open NightstreamFPrime.Export.Codec

private def checked {Alpha : Type} (value : Except String Alpha) : IO Alpha :=
  match value with
  | .ok value => pure value
  | .error error => throw (IO.userError error)

private def extensionValue (value : K) : Value :=
  .array [.atom value.c0.val, .atom value.c1.val]

private def decodeExtension (value : Lean.Json) : Except String K := do
  let words ← PiCCSInputCheck.decodeVector 2 PiCCSInputCheck.decodeField value
  return ⟨words.get 0, words.get 1⟩

/-- Two canonical little-endian 64-bit words per extension-field value. -/
def fieldBytes (values : Array K) : ByteArray := Id.run do
  let mut result := ByteArray.empty
  for value in values do
    for word in [value.c0.val.toUInt64, value.c1.val.toUInt64] do
      for byte in [:8] do
        result := result.push ((word >>> (8 * byte).toUInt64).toUInt8)
  return result

private def decodeWord (bytes : ByteArray) (offset : Nat) : Except String F := do
  let mut word : UInt64 := 0
  for byte in [:8] do
    word := word ||| ((bytes.get! (offset + byte)).toUInt64 <<< (8 * byte).toUInt64)
  if canonical : word.toNat < goldilocksModulus then
    return ⟨word.toNat, canonical⟩
  else throw "noncanonical Goldilocks word in prefix"

/-- Decode complete field values without reducing external words modulo p. -/
def decodeFields (bytes : ByteArray) : Except String (Array K) := do
  unless bytes.size % 16 == 0 do throw "partial extension-field value in prefix"
  let mut values := #[]
  for index in [:bytes.size / 16] do
    let low ← decodeWord bytes (16 * index)
    let high ← decodeWord bytes (16 * index + 8)
    values := values.push ⟨low, high⟩
  return values

/-- A contiguous range of rows. Each firstValues array has the declared width. -/
structure Chunk where
  path : System.FilePath
  first : Nat
  finish : Nat
  firstValues : Array K

private def binaryPath (directory : System.FilePath) (first finish : Nat) : System.FilePath :=
  directory / s!"{first}-{finish}.bin"

private def manifest (kind width count : Nat) (challenges : List K)
    (ranges : Array (Nat × Nat)) : Value :=
  .array [.atom 1, .atom challenges.length, .atom kind, .atom width, .atom count,
    .array (challenges.map extensionValue),
    .array (ranges.toList.map fun (first, finish) => .array [.atom first, .atom finish])]

/-- Record completed file ranges with the shared manifest encoding.
The caller owns file completion, geometry and challenge derivation. -/
def writeManifest (directory : System.FilePath) (kind width count : Nat)
    (challenges : List K) (ranges : Array (Nat × Nat)) : IO Unit :=
  IO.FS.writeFile (directory / "manifest.json")
    ((manifest kind width count challenges ranges).render ++ "\n")

/-- Check exact file identities, ranges, lengths and first rows. The consumer
must decode the rest of each file to check all field encodings. -/
def read (directory : System.FilePath) (expectedKind expectedWidth expectedCount : Nat)
    (challenges : List K) : IO (Array Chunk) := do
  unless expectedWidth > 0 do throw (IO.userError "zero prefix row width")
  let encoded ← checked (Lean.Json.parse (← IO.FS.readFile (directory / "manifest.json")))
  let ranges ← checked (do
    match (← encoded.getArr?).toList with
    | [schema, consumed, kind, width, count, coins, ranges] =>
        unless (← schema.getNat?) == 1 && (← consumed.getNat?) == challenges.length &&
            (← kind.getNat?) == expectedKind && (← width.getNat?) == expectedWidth &&
            (← count.getNat?) == expectedCount do
          throw "binary prefix profile differs from the selected input"
        let coins ← PiCCSInputCheck.decodeVector challenges.length decodeExtension coins
        unless decide (coins.toList = challenges) do
          throw "binary prefix challenges differ from the Lean transcript"
        (← ranges.getArr?).mapM fun range => do
          match (← range.getArr?).toList with
          | [first, finish] => return (← first.getNat?, ← finish.getNat?)
          | _ => throw "expected two binary prefix range fields"
    | _ => throw "expected seven binary prefix manifest fields")
  let mut next := 0
  let mut chunks := #[]
  for (first, finish) in ranges do
    unless first == next && first < finish && finish ≤ expectedCount do
      throw (IO.userError "binary prefix ranges have a gap, overlap or invalid extent")
    let path := binaryPath directory first finish
    let info ← path.metadata
    unless info.type == .file && info.byteSize.toNat == (finish - first) * expectedWidth * 16 do
      throw (IO.userError "binary prefix file has a wrong type or byte count")
    let input ← IO.FS.Handle.mk path .read
    let bytes ← input.read (expectedWidth * 16).toUSize
    unless bytes.size == expectedWidth * 16 do throw (IO.userError "truncated binary prefix first row")
    let firstValues ← checked (decodeFields bytes)
    chunks := chunks.push ⟨path, first, finish, firstValues⟩
    next := finish
  unless next == expectedCount do throw (IO.userError "binary prefix coverage is incomplete")
  unless (← directory.readDir).size == ranges.size + 1 do
    throw (IO.userError "binary prefix directory has missing or extra files")
  return chunks

private structure LegacyProfile where
  kind : Nat
  originalRows : Nat
  records : Nat
  width : Nat

private def legacyProfile (kind : Nat) : IO LegacyProfile := do
  unless kind ≤ 2 do throw (IO.userError "expected Pad, matrix or fresh prefix kind")
  let rows := (PerApplicationMatrixProgram.matrixProgram
    Poseidon2HashChainV1Package.application).rowCount
  return {
    kind := kind
    originalRows := rows
    records := if kind == 0 then PiCCSSourceImages.blockCount else (rows + 1) / 2
    width := if kind == 0 then ringDegree / 2
      else if kind == 1 then 1 else Spec.ProductionRelation.matrixCount }

private def itemsPerRecord (profile : LegacyProfile) : Nat :=
  if profile.kind == 2 then 1 else profile.width

private def outputWidth (profile : LegacyProfile) : Nat :=
  if profile.kind == 2 then profile.width else 1

private def legacyHeader (profile : LegacyProfile) (challenge : K) (value : Lean.Json) :
    Except String (Nat × Nat) := do
  let fields ← value.getArr?
  let (first, finish, coin) ← if profile.kind == 2 then
      match fields.toList with
      | [schema, consumed, ports, rows, first, finish, coin] => do
          unless (← schema.getNat?) == 1 && (← consumed.getNat?) == 1 &&
              (← ports.getNat?) == profile.width && (← rows.getNat?) == profile.originalRows do
            throw "legacy fresh prefix profile differs"
          pure (← first.getNat?, ← finish.getNat?, coin)
      | _ => throw "expected seven legacy fresh prefix header fields"
    else
      match fields.toList with
      | [schema, consumed, kind, width, bound, first, finish, coin] => do
          unless (← schema.getNat?) == 1 && (← consumed.getNat?) == 1 &&
              (← kind.getNat?) == profile.kind && (← width.getNat?) == profile.width &&
              (← bound.getNat?) == profile.records do
            throw "legacy carried prefix profile differs"
          pure (← first.getNat?, ← finish.getNat?, coin)
      | _ => throw "expected eight legacy carried prefix header fields"
  unless decide ((← decodeExtension coin) = challenge) do
    throw "legacy prefix challenge differs from the Lean transcript"
  unless first < finish && finish ≤ profile.records do throw "invalid legacy prefix extent"
  return (first, finish)

private def legacyRow (width index : Nat) (value : Lean.Json) : Except String (Array K) := do
  match (← value.getArr?).toList with
  | [position, values] =>
      unless (← position.getNat?) == index do throw "legacy prefix record index is not consecutive"
      return (← PiCCSInputCheck.decodeVector width decodeExtension values).toArray
  | _ => throw "expected indexed legacy prefix record"

private def legacyChunks (profile : LegacyProfile) (directories : List System.FilePath)
    (challenge : K) : IO (Array Chunk) := do
  let mut chunks := #[]
  for directory in directories do
    for entry in ← directory.readDir do
      if profile.kind != 2 && entry.fileName == "moments.json" then continue
      unless entry.path.extension == some "jsonl" do throw (IO.userError "unexpected legacy prefix file")
      let input ← IO.FS.Handle.mk entry.path .read
      let (first, finish) ← checked (legacyHeader profile challenge
        (← checked (Lean.Json.parse (← input.getLine))))
      let firstValues ← checked (legacyRow profile.width first
        (← checked (Lean.Json.parse (← input.getLine))))
      chunks := chunks.push ⟨entry.path, first, finish, firstValues⟩
  chunks := chunks.qsort (fun left right => decide (left.first < right.first))
  let mut next := 0
  for chunk in chunks do
    unless chunk.first == next do throw (IO.userError "legacy prefix ranges have a gap or overlap")
    next := chunk.finish
  unless next == profile.records do throw (IO.userError "legacy prefix coverage is incomplete")
  return chunks

private def pairBytes (width : Nat) (challenge : K) (low high : Array K) :
    IO ByteArray := do
  if lowSize : low.size = width then
    if highSize : high.size = width then
      return fieldBytes (PiCCSFreshPrefix.pairRow ⟨low, lowSize⟩ ⟨high, highSize⟩ challenge).toArray
    else throw (IO.userError "wrong high prefix width")
  else throw (IO.userError "wrong low prefix width")

private def foldChunk (profile : LegacyProfile) (chunk : Chunk)
    (nextValues : Option (Array K)) (firstChallenge secondChallenge : K) : IO ByteArray := do
  let input ← IO.FS.Handle.mk chunk.path .read
  let extent ← checked (legacyHeader profile firstChallenge
    (← checked (Lean.Json.parse (← input.getLine))))
  unless extent == (chunk.first, chunk.finish) do throw (IO.userError "legacy prefix header changed")
  let mut pending : Option (Array K) := none
  let mut result := ByteArray.empty
  for index in [chunk.first:chunk.finish] do
    let values ← checked (legacyRow profile.width index
      (← checked (Lean.Json.parse (← input.getLine))))
    if index == chunk.first then
      unless decide (values = chunk.firstValues) do throw (IO.userError "legacy prefix first record changed")
    if profile.kind == 2 then
      if index % 2 == 0 then pending := some values
      else if let some low := pending then
        result := result ++ (← pairBytes (outputWidth profile) secondChallenge low values)
        pending := none
    else
      for lane in [:values.size] do
        let position := profile.width * index + lane
        let value := #[values.getD lane K.zero]
        if position % 2 == 0 then pending := some value
        else if let some low := pending then
          result := result ++ (← pairBytes (outputWidth profile) secondChallenge low value)
          pending := none
  unless (← input.getLine).trimAscii.toString == "[]" && (← input.getLine).isEmpty do
    throw (IO.userError "legacy prefix has missing terminator or extra data")
  if let some low := pending then
    let high ← match nextValues with
      | some values => pure (if profile.kind == 2 then values else #[values.getD 0 K.zero])
      | none => do
          unless chunk.finish == profile.records do throw (IO.userError "missing boundary prefix value")
          pure (Array.replicate (outputWidth profile) K.zero)
    result := result ++ (← pairBytes (outputWidth profile) secondChallenge low high)
  let first := (itemsPerRecord profile * chunk.first + 1) / 2
  let finish := (itemsPerRecord profile * chunk.finish + 1) / 2
  unless result.size == (finish - first) * outputWidth profile * 16 do
    throw (IO.userError "folded prefix byte count differs from its range")
  return result

/-- Fold the complete legacy family through the second challenge. Input files
define work chunks; output files and the completion manifest use numeric order. -/
def foldLegacy (kind : Nat) (directories : List System.FilePath)
    (outputDirectory : System.FilePath) (firstChallenge secondChallenge : K) : IO Unit := do
  unless !(← outputDirectory.pathExists) do throw (IO.userError "output directory already exists")
  let profile ← legacyProfile kind
  let chunks ← legacyChunks profile directories firstChallenge
  IO.FS.createDirAll outputDirectory
  let workers := max 1 (((← IO.getEnv "LEAN_NUM_THREADS").bind String.toNat?).getD 1)
  let mut ranges : Array (Nat × Nat) := #[]
  for batch in [:(chunks.size + workers - 1) / workers] do
    let mut tasks : Array (Chunk × Task (Except IO.Error ByteArray)) := #[]
    for index in [batch * workers:min chunks.size ((batch + 1) * workers)] do
      if bound : index < chunks.size then
        let chunk := chunks[index]'bound
        let nextValues := (chunks[index + 1]?).map Chunk.firstValues
        let task ← IO.asTask (prio := Task.Priority.dedicated) do
          foldChunk profile chunk nextValues firstChallenge secondChallenge
        tasks := tasks.push (chunk, task)
    for (chunk, task) in tasks do
      let bytes ← match ← IO.wait task with
        | .ok bytes => pure bytes
        | .error error => throw error
      let first := (itemsPerRecord profile * chunk.first + 1) / 2
      let finish := (itemsPerRecord profile * chunk.finish + 1) / 2
      if first < finish then
        IO.FS.writeBinFile (binaryPath outputDirectory first finish) bytes
        ranges := ranges.push (first, finish)
  let count := (itemsPerRecord profile * profile.records + 1) / 2
  writeManifest outputDirectory kind (outputWidth profile) count [firstChallenge, secondChallenge] ranges

private def foldBinaryChunk (width count : Nat) (chunk : Chunk)
    (nextValues : Option (Array K)) (challenge : K) : IO ByteArray := do
  let input ← IO.FS.Handle.mk chunk.path .read
  let mut pending : Option (Array K) := none
  let mut result := ByteArray.empty
  for index in [chunk.first:chunk.finish] do
    let bytes ← input.read (width * 16).toUSize
    unless bytes.size == width * 16 do throw (IO.userError "truncated binary prefix row")
    let values ← checked (decodeFields bytes)
    if index == chunk.first then
      unless decide (values = chunk.firstValues) do throw (IO.userError "binary prefix first row changed")
    if index % 2 == 0 then pending := some values
    else if let some low := pending then
      result := result ++ (← pairBytes width challenge low values)
      pending := none
  unless (← input.read 1).isEmpty do throw (IO.userError "extra binary prefix bytes")
  if let some low := pending then
    let high ← match nextValues with
      | some values => pure values
      | none => do
          unless chunk.finish == count do throw (IO.userError "missing boundary binary prefix row")
          pure (Array.replicate width K.zero)
    result := result ++ (← pairBytes width challenge low high)
  let first := (chunk.first + 1) / 2
  let finish := (chunk.finish + 1) / 2
  unless result.size == (finish - first) * width * 16 do
    throw (IO.userError "folded binary prefix byte count differs from its range")
  return result

/-- Consume one challenge from a complete binary prefix. Every row is decoded,
including a leading odd row whose pair belongs to the preceding file.
Only a missing endpoint beyond count is zero; singleton prefixes still fold. -/
def foldBinary (kind width count : Nat) (inputDirectory outputDirectory : System.FilePath)
    (challenges : List K) (nextChallenge : K) : IO Unit := do
  unless !(← outputDirectory.pathExists) do throw (IO.userError "output directory already exists")
  let chunks ← read inputDirectory kind width count challenges
  IO.FS.createDirAll outputDirectory
  let workers := max 1 (((← IO.getEnv "LEAN_NUM_THREADS").bind String.toNat?).getD 1)
  let mut ranges : Array (Nat × Nat) := #[]
  for batch in [:(chunks.size + workers - 1) / workers] do
    let mut tasks : Array (Chunk × Task (Except IO.Error ByteArray)) := #[]
    for index in [batch * workers:min chunks.size ((batch + 1) * workers)] do
      if bound : index < chunks.size then
        let chunk := chunks[index]'bound
        let nextValues := (chunks[index + 1]?).map Chunk.firstValues
        let task ← IO.asTask (prio := Task.Priority.dedicated) do
          foldBinaryChunk width count chunk nextValues nextChallenge
        tasks := tasks.push (chunk, task)
    for (chunk, task) in tasks do
      let bytes ← match ← IO.wait task with
        | .ok bytes => pure bytes
        | .error error => throw error
      let first := (chunk.first + 1) / 2
      let finish := (chunk.finish + 1) / 2
      if first < finish then
        IO.FS.writeBinFile (binaryPath outputDirectory first finish) bytes
        ranges := ranges.push (first, finish)
  writeManifest outputDirectory kind width ((count + 1) / 2)
    (challenges ++ [nextChallenge]) ranges

end NightstreamFPrime.Export.PiCCSPrefixFiles
