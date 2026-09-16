import NightstreamFPrime.Export.Stage1.StridedArrayAll
import NightstreamFPrime.Export.Stage1.StoredCompactRowExecution
import NightstreamFPrime.Export.Stage1.StoredInstructionExecution
import NightstreamFPrime.Export.Stage1.StoredPhysicalPlan
import NightstreamFPrime.Export.Stage1.StoredPermutationExecution

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Export
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Export.Stage1
open StoredWitnessExecution (asEnv)

private def checked {Alpha : Type} (value : Except String Alpha) : IO Alpha :=
  match value with
  | .ok result => pure result
  | .error error => throw (IO.userError error)

private def report (fields : List (String × Lean.Json)) : IO Unit := do
  IO.println (Lean.Json.mkObj fields).compress
  (← IO.getStdout).flush

private def field (value : Lean.Json) : Except String F := do
  let word ← value.getNat?
  if bound : word < goldilocksModulus then pure ⟨word, bound⟩
  else throw "noncanonical caller field"

private def witnessRole (role : Nat) : Bool :=
  role == Data.Role.witness || role == Data.Role.piDecWitness ||
    role == Data.Role.runningTransitionWitness || role == PerApplicationPackage.Role.applicationLocal

private def seed (layout : PhysicalLayout) (caller : Lean.Json) : IO (Array F) := do
  let fields ← checked caller.getArr?
  unless fields.size == 5 && (← checked fields[0]!.getNat?) == 1 do
    throw (IO.userError "expected a complete schema-one caller")
  let privateWords ← checked do (← fields[2]!.getArr?).mapM field
  let publicWords ← checked do (← fields[3]!.getArr?).mapM field
  let inputCount := (layout.privateSegments.filter fun segment => !witnessRole segment.role
    ).foldl (fun total segment => total + segment.length) 0
  unless privateWords.size == inputCount && publicWords.size == layout.publicColumnCount &&
      layout.constantColumn == layout.privateColumnCount &&
      layout.totalColumnCount == layout.constantColumn + 1 + publicWords.size do
    throw (IO.userError "caller width differs from the selected physical layout")
  let mut values := Array.replicate layout.totalColumnCount (0 : F)
  let mut cursor := 0
  for segment in layout.privateSegments do
    unless segment.start + segment.length ≤ layout.privateColumnCount do
      throw (IO.userError "private segment exceeds its selected bound")
    unless witnessRole segment.role do
      for index in [:segment.length] do
        values := values.set! (segment.start + index) privateWords[cursor + index]!
      cursor := cursor + segment.length
  values := values.set! layout.constantColumn 1
  for index in [:publicWords.size] do
    values := values.set! (layout.constantColumn + 1 + index) publicWords[index]!
  return values

private def requireWrite (values : Array F) (start count : Nat) : Except String Unit :=
  if start + count ≤ values.size then .ok () else .error "physical write exceeds its selected bound"

private def compact (templates : Array CompactRowTemplate) (target : Nat)
    (invocation : CompactRowInvocation) (values : Array F) : Except String (Array F) := do
  let some template := templates[invocation.templateIndex]?
    | throw "missing canonical compact template"
  unless compactInputColumn invocation.inputRanges template.outputInput == target do
    throw "compact output differs from its scheduled target"
  requireWrite values target 1
  requireWrite values invocation.localStart template.localColumnCount
  match StoredCompactRowExecution.execute
      (compactInputColumn invocation.inputRanges) invocation.localStart template values with
  | some result => return result
  | none => throw s!"compact row failed at {target}"

private def executeEvent (pilot : CircuitPackage) (templates : Array CompactRowTemplate)
    (event : StoredPhysicalPlan.Event) (values : Array F) : Except String (Array F) := do
  match event with
  | .hash chain ordinal =>
      let invocation : PermutationInvocation := {
        phase := chain.phase
        rowStart := chain.rowStart + ordinal * pilot.poseidon.recipesPerPermutation
        witnessStart := invocationLocalStart pilot chain ordinal
        inputs := List.ofFn fun lane : Fin 8 =>
          Rows.sparseCombination (invocationInput pilot chain ordinal lane.val) }
      requireWrite values invocation.witnessStart 592
      return StoredPermutationExecution.execute invocation values
  | .permutation invocation =>
      requireWrite values invocation.witnessStart 592
      return StoredPermutationExecution.execute invocation values
  | .compact target invocation => compact templates target invocation values
  | .batch batch =>
      requireWrite values batch.start (batch.recipes.length + batch.hints.length)
      let result := StoredWitnessExecution.executeRecipes values batch.start batch.recipes
      return StoredWitnessExecution.executeHints result
        (batch.start + batch.recipes.length) batch.hints
  | .instruction instruction =>
      requireWrite values instruction.target 1
      return StoredInstructionExecution.execute instruction values

private def run (callerPath outputPath : System.FilePath) : IO UInt32 := do
  if ← outputPath.pathExists then throw (IO.userError "physical output already exists")
  let started ← IO.monoNanosNow
  let caller ← checked (Lean.Json.parse (← IO.FS.readFile callerPath))
  let plan ← StoredPhysicalPlan.prepare
  let pilot := plan.pilot.val
  let mut values ← seed plan.layout caller
  let prepared ← IO.monoNanosNow
  report [("event", .str "physical_plan_ready"),
    ("fields", Lean.toJson values.size), ("events", Lean.toJson plan.events.size),
    ("assertions", Lean.toJson plan.assertions.size), ("prepare_ns", Lean.toJson (prepared - started))]
  let mut previous : Option Nat := none
  for event in plan.events do
    if let some target := previous then
      unless target < event.target do
        throw (IO.userError "physical event targets are not strictly increasing")
    values ← checked (executeEvent pilot plan.templates event values)
    previous := some event.target
  let computed ← IO.monoNanosNow
  report [("event", .str "physical_values_ready"), ("fields", Lean.toJson values.size),
    ("compute_ns", Lean.toJson (computed - prepared))]
  let checkPackage := { pilot with compactRowTemplates := plan.templates.toList }
  let workers := max 1 (((← IO.getEnv "LEAN_NUM_THREADS").bind String.toNat?).getD 1)
  let tasks := (Array.range workers).map fun worker =>
    Task.spawn (fun _ => StridedArrayAll.worker plan.rowEvents
      (fun event => event.check checkPackage (asEnv values)) workers worker)
      (prio := Task.Priority.dedicated)
  let mut eventRowsHold := true
  for task in tasks do
    let holds ← IO.wait task
    eventRowsHold := eventRowsHold && holds
  unless eventRowsHold do
    throw (IO.userError "physical event row failed in final array")
  let rowsChecked ← IO.monoNanosNow
  report [("event", .str "physical_event_rows_passed"),
    ("events", Lean.toJson plan.events.size), ("workers", Lean.toJson workers),
    ("row_check_ns", Lean.toJson (rowsChecked - computed))]
  for row in plan.assertions do
    unless StoredPhysicalRowCheck.sparseRow row (asEnv values) do
      throw (IO.userError s!"physical assertion failed at row {row.rowIndex}")
  report [("event", .str "physical_assertions_passed"),
    ("assertions", Lean.toJson plan.assertions.size),
    ("assert_ns", Lean.toJson ((← IO.monoNanosNow) - rowsChecked))]
  let checkedAt ← IO.monoNanosNow
  let output ← IO.FS.Handle.mk outputPath .write
  for value in values do
    let word := value.val.toUInt64
    let mut bytes := ByteArray.empty
    for byte in [:8] do
      bytes := bytes.push ((word >>> (8 * byte).toUInt64).toUInt8)
    output.write bytes
  output.flush
  report [("event", .str "physical_values_saved"), ("fields", Lean.toJson values.size),
    ("bytes", Lean.toJson (values.size * 8)),
    ("write_ns", Lean.toJson ((← IO.monoNanosNow) - checkedAt))]
  return 0

def main (arguments : List String) : IO UInt32 :=
  match arguments with
  | [caller, output] => run caller output
  | _ => do
      IO.eprintln "usage: replayPhysicalWitness <independent-caller> <physical-values-output>"
      return 2
