import NightstreamFPrime.Export.Stage1.CachedAssignmentProducts
import NightstreamFPrime.Export.Stage1.CachedAssignmentPlan
import NightstreamFPrime.Export.Stage1.PerApplicationAssignmentTransportExecution
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Package

open NightstreamFPrime.Spec
open NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Export.Stage1
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

private def report (fields : List (String × Lean.Json)) : IO Unit := do
  IO.println (Lean.Json.mkObj fields).compress
  (← IO.getStdout).flush

private def readPhysical (path : System.FilePath) (count : Nat) : IO (Array F) := do
  let bytes ← IO.FS.readBinFile path
  unless bytes.size == count * 8 do
    throw (IO.userError "physical file size differs from the selected layout")
  let mut values := Array.mkEmpty count
  for index in [:count] do
    let mut word : UInt64 := 0
    for byte in [:8] do
      word := word ||| ((bytes.get! (index * 8 + byte)).toUInt64 <<< (8 * byte).toUInt64)
    if bound : word.toNat < goldilocksModulus then
      values := values.push ⟨word.toNat, bound⟩
    else throw (IO.userError s!"noncanonical physical word at {index}")
  return values

private def signedBytes (values : List F) : Except String ByteArray := do
  let mut bytes := ByteArray.empty
  for value in values do
    if value == 0 then bytes := bytes.push 0
    else if value == 1 then bytes := bytes.push 1
    else if value == -1 then bytes := bytes.push 255
    else throw "canonical assignment contains a non-unit coordinate"
  return bytes

private def checked {Alpha : Type} (value : Except String Alpha) : IO Alpha :=
  match value with
  | .ok result => pure result
  | .error error => throw (IO.userError error)

private def writeBlock (path : System.FilePath)
    (entry : CanonicalBlockAssignment.BlockValue) : IO Unit := do
  let output ← IO.FS.Handle.mk path .write
  for index in [:entry.block.slotCount] do
    if bound : index < entry.block.slotCount then
      let value := entry.source (entry.block.source ⟨index, bound⟩)
      output.write (← checked (signedBytes (LowNormSlot.encode entry.block.kind value)))
    else throw (IO.userError "assignment slot exceeds its canonical block")
  output.flush

private def run (physicalPath directory : System.FilePath) (first finish : Nat) : IO UInt32 := do
  if ← directory.pathExists then throw (IO.userError "assignment output directory already exists")
  unless first < finish && finish ≤ PerApplicationAssignmentPlan.canonicalKinds.length do
    throw (IO.userError "assignment block range is outside the canonical schedule")
  let started ← IO.monoNanosNow
  let program := Poseidon2HashChainV1Package.application
  let values ← readPhysical physicalPath (PerApplicationPackage.directFinalLayout program).totalColumnCount
  let base : PerApplicationAssignmentTransportExecution.BaseValues program :=
    fun index => values[index.val]?.getD 0
  let products := CachedAssignmentProducts.prepare program
  let raw := CachedAssignmentProducts.rawValues products base
  let widths := CachedAssignmentPlan.prepareWidths program
  let schedule := CachedAssignmentPlan.expand widths raw
  let logicalWidth := PerApplicationFixedPoint.logicalWidth program
  unless ProductionAssignment.publicWidth + CanonicalBlockAssignment.coordinateCount schedule ==
      logicalWidth do
    throw (IO.userError "canonical assignment schedule has a wrong total width")
  IO.FS.createDir directory
  let publicValues := (List.ofFn (encodedHashCells raw.outputDigest))
  let publicBytes ← checked (signedBytes publicValues)
  IO.FS.writeBinFile (directory / "public.bin") publicBytes
  let prepared ← IO.monoNanosNow
  report [("event", .str "assignment_ready"), ("physical_fields", Lean.toJson values.size),
    ("logical_width", Lean.toJson logicalWidth), ("blocks", Lean.toJson schedule.length),
    ("prepare_ns", Lean.toJson (prepared - started))]
  let mut ordinal := 0
  let mut offset := ProductionAssignment.publicWidth
  let mut records : Array Lean.Json := #[]
  for entry in schedule do
    if first ≤ ordinal && ordinal < finish then
      let blockStarted ← IO.monoNanosNow
      let name := s!"block-{ordinal}.bin"
      writeBlock (directory / name) entry
      let record := Lean.Json.mkObj [
        ("ordinal", Lean.toJson ordinal), ("first", Lean.toJson offset),
        ("finish", Lean.toJson (offset + entry.coordinateCount)),
        ("slots", Lean.toJson entry.block.slotCount), ("file", .str name),
        ("compute_write_ns", Lean.toJson ((← IO.monoNanosNow) - blockStarted))]
      records := records.push record
      IO.FS.writeFile (directory / "manifest.json") ((Lean.Json.mkObj [
        ("schema", Lean.toJson (1 : Nat)), ("logical_width", Lean.toJson logicalWidth),
        ("physical_fields", Lean.toJson values.size),
        ("public_width", Lean.toJson publicBytes.size),
        ("first_block", Lean.toJson first), ("finish_block", Lean.toJson finish),
        ("blocks", .arr records)]).compress ++ "\n")
      report [("event", .str "assignment_block_saved"), ("block", record)]
    ordinal := ordinal + 1
    offset := offset + entry.coordinateCount
  return 0

private def profileSlot (physicalPath : System.FilePath) (block slot : Nat) : IO UInt32 := do
  let started ← IO.monoNanosNow
  let program := Poseidon2HashChainV1Package.application
  let values ← readPhysical physicalPath (PerApplicationPackage.directFinalLayout program).totalColumnCount
  let loaded ← IO.monoNanosNow
  let base : PerApplicationAssignmentTransportExecution.BaseValues program :=
    fun index => values[index.val]?.getD 0
  let products := CachedAssignmentProducts.prepare program
  let raw := CachedAssignmentProducts.rawValues products base
  let widths := CachedAssignmentPlan.prepareWidths program
  let schedule := (CachedAssignmentPlan.expand widths raw).toArray
  let some entry := schedule[block]? | throw (IO.userError "profile block out of range")
  let prepared ← IO.monoNanosNow
  if bound : slot < entry.block.slotCount then
    let source := entry.block.source ⟨slot, bound⟩
    let indexed ← IO.monoNanosNow
    let value := entry.source source
    let read ← IO.monoNanosNow
    let bytes ← checked (signedBytes (LowNormSlot.encode entry.block.kind value))
    let encoded ← IO.monoNanosNow
    report [("event", .str "assignment_slot_profile"), ("block", Lean.toJson block),
      ("slot", Lean.toJson slot), ("source", Lean.toJson source.val),
      ("value", Lean.toJson value.val), ("bytes", Lean.toJson (bytes.data.map UInt8.toNat)),
      ("load_ns", Lean.toJson (loaded - started)), ("prepare_ns", Lean.toJson (prepared - loaded)),
      ("source_index_ns", Lean.toJson (indexed - prepared)),
      ("source_value_ns", Lean.toJson (read - indexed)), ("encode_ns", Lean.toJson (encoded - read))]
    return 0
  else throw (IO.userError "profile slot out of range")

def main (arguments : List String) : IO UInt32 :=
  match arguments with
  | ["profile", physical, block, slot] =>
      match block.toNat?, slot.toNat? with
      | some block, some slot => profileSlot physical block slot
      | _, _ => throw (IO.userError "profile indices must be natural numbers")
  | [physical, directory, first, finish] =>
      match first.toNat?, finish.toNat? with
      | some first, some finish => run physical directory first finish
      | _, _ => do
          IO.eprintln "assignment block indices must be natural numbers"
          return 2
  | _ => do
      IO.eprintln "usage: replayFreshAssignment <physical-values> <new-directory> <first-block> <finish-block>"
      return 2
