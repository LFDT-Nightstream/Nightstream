import NightstreamFPrime.Export.Stage1.FreshRowsCheck

open NightstreamFPrime.Spec
open NightstreamFPrime.Layout
open NightstreamFPrime.Export.Stage1

private def checked {Alpha : Type} (value : Except String Alpha) : IO Alpha :=
  match value with
  | .ok result => pure result
  | .error error => throw (IO.userError error)

private def report (fields : List (String × Lean.Json)) : IO Unit := do
  IO.println (Lean.Json.mkObj fields).compress
  (← IO.getStdout).flush

private def canonicalField (value : Lean.Json) : Except String F := do
  let word ← value.getNat?
  if bound : word < goldilocksModulus then return ⟨word, bound⟩
  else throw "noncanonical caller fresh public field"

/-- The source is the generated caller's next fresh public input. The prior
physical public input at caller[3] has a different role and a different width. -/
private def nextPublic (callerPath : System.FilePath) : IO (Array F) := do
  let caller ← checked (Lean.Json.parse (← IO.FS.readFile callerPath))
  let fields ← checked caller.getArr?
  unless fields.size == 5 && (← checked fields[0]!.getNat?) == 1 do
    throw (IO.userError "expected a complete schema-one recursive caller")
  let derived ← checked fields[4]!.getArr?
  unless derived.size == 7 do
    throw (IO.userError "recursive caller derived fields have the wrong size")
  let publicValues ← checked do (← derived[2]!.getArr?).mapM canonicalField
  unless publicValues.size == ProductionAssignment.publicWidth do
    throw (IO.userError "next fresh public input has the wrong width")
  return publicValues

private def run (carrierPath callerPath outputPath : System.FilePath) : IO UInt32 := do
  if ← outputPath.pathExists then throw (IO.userError "fresh row-check output already exists")
  let started ← IO.monoNanosNow
  let bytes ← IO.FS.readBinFile carrierPath
  unless bytes.size == PiCCSSourceImages.shape.carrierWidth do
    throw (IO.userError "fresh carrier has the wrong complete byte count")
  let publicValues ← nextPublic callerPath
  let logicalWidth := PiCCSSourceImages.logicalWidth
  for index in [:bytes.size] do
    let code := bytes.get! index
    unless code == 0 || code == 1 || code == 255 do
      throw (IO.userError s!"noncanonical signed-unit byte at {index}")
    unless index < logicalWidth || code == 0 do
      throw (IO.userError s!"fresh carrier has nonzero tail at {index}")
  for index in [:publicValues.size] do
    unless FreshRowsCheck.field (bytes.get! index) == publicValues[index]! do
      throw (IO.userError s!"fresh public input differs from caller at {index}")
  let application := Poseidon2HashChainV1Package.application
  let program := PerApplicationMatrixProgram.matrixProgram application
  let cache := PiDECCanonicalSourceCache.stored application
  let sourceRow := fun source => cache[source]?
  let read := FreshRowsCheck.logicalRead bytes
  let ready ← IO.monoNanosNow
  report [("event", .str "fresh_rows_ready"),
    ("carrier_coefficients", Lean.toJson bytes.size),
    ("logical_width", Lean.toJson logicalWidth),
    ("active_rows", Lean.toJson program.rowCount),
    ("canonical_blocks", Lean.toJson program.blocks.length),
    ("read_validate_prepare_ns", Lean.toJson (ready - started))]
  let workers := max 1 (((← IO.getEnv "LEAN_NUM_THREADS").bind String.toNat?).getD 1)
  -- Split each block over the hardware-derived worker count. A unit is one
  -- complete Poseidon invocation or one row in the other canonical families.
  for (block, index) in program.blocks.zipIdx do
    let units := FreshRowsCheck.checkBlockUnits block
    let chunk := (units + workers - 1) / workers
    let parts := workers
    let blockStarted ← IO.monoNanosNow
    report [("event", .str "fresh_rows_block_started"),
      ("block", Lean.toJson index), ("rows", Lean.toJson block.rowCount),
      ("units", Lean.toJson units), ("tasks", Lean.toJson parts)]
    let tasks := (Array.range parts).map fun part =>
      let first := part * chunk
      Task.spawn (prio := Task.Priority.dedicated) fun _ =>
        FreshRowsCheck.checkBlockRange block sourceRow read first (min chunk (units - first))
    for part in [:tasks.size] do
      unless ← IO.wait tasks[part]! do
        throw (IO.userError s!"canonical scalar production row check failed in block {index}, part {part}")
    report [("event", .str "fresh_rows_block_checked"),
      ("block", Lean.toJson index), ("rows", Lean.toJson block.rowCount),
      ("elapsed_ns", Lean.toJson ((← IO.monoNanosNow) - blockStarted))]
  let computed ← IO.monoNanosNow
  let result := Lean.Json.mkObj [
    ("event", .str "fresh_canonical_rows_checked"),
    ("status", .str "passed"),
    ("active_rows", Lean.toJson program.rowCount),
    ("padding_rows", Lean.toJson (2 ^ NightstreamFPrime.Lifecycle.cubeVariables - program.rowCount)),
    ("matrix_count", Lean.toJson NightstreamFPrime.Spec.ProductionRelation.matrixCount),
    ("carrier_coefficients", Lean.toJson bytes.size),
    ("public_coefficients", Lean.toJson publicValues.size),
    ("tail_coefficients", Lean.toJson (bytes.size - logicalWidth)),
    ("canonical_blocks", Lean.toJson program.blocks.length),
    ("read_validate_prepare_ns", Lean.toJson (ready - started)),
    ("check_ns", Lean.toJson (computed - ready))]
  IO.FS.writeFile outputPath (result.compress ++ "\n")
  IO.println result.compress
  return 0

def main (arguments : List String) : IO UInt32 :=
  match arguments with
  | [carrier, caller, output] => run carrier caller output
  | _ => do
      IO.eprintln "usage: checkFreshRows <fresh-carrier.bin> <generated-caller.json> <new-result.json>"
      return 2
