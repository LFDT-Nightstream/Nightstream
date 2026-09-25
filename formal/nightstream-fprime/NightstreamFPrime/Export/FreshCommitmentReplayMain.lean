import NightstreamFPrime.Export.Stage1.FreshCommitmentBlock
import NightstreamFPrime.Export.Stage1.PiDECCommitmentFold
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Setup
import NightstreamFPrime.Export.Codec

open NightstreamFPrime.Spec
open NightstreamFPrime.Export
open NightstreamFPrime.Export.Stage1
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingArithmetic (StoredRing)

private abbrev NativeRows := Vector PiDECNativeProduct.Accumulator
  Poseidon2HashChainV1Setup.verifierRows
private abbrev Rows := Vector StoredRing Poseidon2HashChainV1Setup.verifierRows

private def zero : NativeRows := Vector.replicate _ PiDECNativeProduct.Accumulator.zero
private def add (left right : NativeRows) : NativeRows :=
  left.zipWith PiDECNativeProduct.Accumulator.add right

private def checked {Alpha : Type} (value : Except String Alpha) : IO Alpha :=
  match value with
  | .ok result => pure result
  | .error error => throw (IO.userError error)

private def report (fields : List (String × Lean.Json)) : IO Unit := do
  IO.println (Lean.Json.mkObj fields).compress
  (← IO.getStdout).flush

private def field (code : UInt8) : F :=
  if code == 1 then 1 else if code == 255 then -1 else 0

private def compute (bytes : ByteArray) (first finish : Nat) :
    Except String (NativeRows × Nat) := do
  let mut accumulated := zero
  let mut populated := 0
  for block in [first:finish] do
    if bound : block < Poseidon2HashChainV1Setup.messageColumns then
      let digit : StoredRing := Vector.ofFn fun lane =>
        field (bytes.get! (block * ringDegree + lane.val))
      unless digit.toArray.all (fun value => value == 0) do
        let prepared := PiDECNativeProduct.prepareDigit digit
        accumulated := Vector.ofFn fun row =>
          FreshCommitmentBlock.accumulatePrepared
            Poseidon2HashChainV1Setup.productionSetup row ⟨block, bound⟩ prepared
            (accumulated.get row)
        populated := populated + 1
    else throw "fresh commitment block exceeds the selected key"
  return (accumulated, populated)

private def writeResult (path : System.FilePath) (first finish : Nat) (rows : Rows) : IO Unit := do
  let value := Codec.Value.array [
    .atom 1, .atom Poseidon2HashChainV1Setup.messageColumns, .atom first, .atom finish,
    .array (List.ofFn fun row : Fin Poseidon2HashChainV1Setup.verifierRows =>
      .array (List.ofFn fun lane : Fin ringDegree => .atom ((rows.get row).get lane).val))]
  IO.FS.writeFile path (value.render ++ "\n")

private def replay (inputPath outputPath : System.FilePath) (first finish : Nat) : IO UInt32 := do
  if ← outputPath.pathExists then throw (IO.userError "fresh commitment output already exists")
  unless first < finish && finish ≤ Poseidon2HashChainV1Setup.messageColumns do
    throw (IO.userError "fresh commitment range is outside the selected key")
  let started ← IO.monoNanosNow
  let bytes ← IO.FS.readBinFile inputPath
  unless bytes.size == Poseidon2HashChainV1Setup.messageColumns * ringDegree do
    throw (IO.userError "fresh carrier has the wrong complete byte count")
  let logicalWidth := PerApplicationFixedPoint.logicalWidth Poseidon2HashChainV1Package.application
  for index in [:bytes.size] do
    let code := bytes.get! index
    unless code == 0 || code == 1 || code == 255 do
      throw (IO.userError s!"non-unit fresh coefficient at {index}")
    unless index < logicalWidth || code == 0 do
      throw (IO.userError "fresh carrier has nonzero tail padding")
  let workers := max 1 (((← IO.getEnv "LEAN_NUM_THREADS").bind String.toNat?).getD 1)
  let width := (finish - first + workers - 1) / workers
  let ready ← IO.monoNanosNow
  let mut tasks := #[]
  for worker in [:workers] do
    let start := first + worker * width
    let stop := min finish (start + width)
    if start < stop then tasks := tasks.push (Task.spawn fun _ => compute bytes start stop)
  let mut accumulated := zero
  let mut populated := 0
  for task in tasks do
    let (part, count) ← checked (← IO.wait task)
    accumulated := add accumulated part
    populated := populated + count
  let computed ← IO.monoNanosNow
  writeResult outputPath first finish (accumulated.map PiDECNativeProduct.Accumulator.finish)
  report [("event", .str "fresh_commitment_range_saved"), ("first", Lean.toJson first),
    ("finish", Lean.toJson finish), ("populated", Lean.toJson populated),
    ("workers", Lean.toJson workers), ("read_validate_ns", Lean.toJson (ready - started)),
    ("compute_ns", Lean.toJson (computed - ready)),
    ("write_ns", Lean.toJson ((← IO.monoNanosNow) - computed))]
  return 0

private def decodeRow (value : Lean.Json) : Except String StoredRing := do
  let entries ← value.getArr?
  let values ← entries.mapM fun entry => do
    let word ← entry.getNat?
    if bound : word < goldilocksModulus then pure (⟨word, bound⟩ : F)
    else throw "noncanonical fresh commitment field"
  if size : values.size = ringDegree then return ⟨values, size⟩
  else throw "fresh commitment row has the wrong size"

private def merge (outputPath : System.FilePath) (paths : List String) : IO UInt32 := do
  if ← outputPath.pathExists then throw (IO.userError "fresh commitment output already exists")
  let mut next := 0
  let mut parts : Array Rows := #[]
  for path in paths do
    let value ← checked (Lean.Json.parse (← IO.FS.readFile path))
    let fields ← checked value.getArr?
    unless fields.size == 5 && (← checked fields[0]!.getNat?) == 1 &&
        (← checked fields[1]!.getNat?) == Poseidon2HashChainV1Setup.messageColumns do
      throw (IO.userError "invalid fresh commitment range header")
    let first ← checked fields[2]!.getNat?
    let finish ← checked fields[3]!.getNat?
    unless first == next && first < finish && finish ≤ Poseidon2HashChainV1Setup.messageColumns do
      throw (IO.userError "fresh commitment ranges have a gap, overlap or invalid endpoint")
    let rows ← checked do (← fields[4]!.getArr?).mapM decodeRow
    if size : rows.size = Poseidon2HashChainV1Setup.verifierRows then
      parts := parts.push ⟨rows, size⟩
    else throw (IO.userError "fresh commitment has the wrong key-row count")
    next := finish
  unless next == Poseidon2HashChainV1Setup.messageColumns do
    throw (IO.userError "fresh commitment ranges do not cover the complete carrier")
  let rows : Rows := Vector.ofFn fun row =>
    PiDECCommitmentFold.sum fun part : Fin parts.size => (parts[part]).get row
  writeResult outputPath 0 next rows
  report [("event", .str "fresh_commitment_complete"), ("ranges", Lean.toJson parts.size),
    ("blocks", Lean.toJson next), ("coefficients", Lean.toJson (rows.size * ringDegree))]
  return 0

def main (arguments : List String) : IO UInt32 :=
  match arguments with
  | "merge" :: output :: ranges => merge output ranges
  | [input, output, first, finish] =>
      match first.toNat?, finish.toNat? with
      | some first, some finish => replay input output first finish
      | _, _ => throw (IO.userError "fresh commitment bounds must be natural numbers")
  | _ => do
      IO.eprintln "usage: replayFreshCommitment <complete-carrier> <new-output> <first-block> <finish-block>"
      IO.eprintln "   or: replayFreshCommitment merge <new-output> <ordered-range>..."
      return 2
