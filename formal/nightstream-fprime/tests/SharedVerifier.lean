import tests.AxiomAudit
import tests.PerApplicationEmitterFixture
import NightstreamFPrime.Export.SharedVerifier

/-! Checks the export against the existing hash-chain and identity programs.
No application definition, circuit, layout, or proof is added here. -/

namespace NightstreamFPrime.Tests.SharedVerifier

open NightstreamFPrime.Export
open NightstreamFPrime.Export.Stage1
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.MatrixProgram

private def ordinaryReference : NightstreamFPrime.Lifecycle.Stage1.Application.Program :=
  { Poseidon2HashChainV1Package.application with compactHashChain := none }

private def require (condition : Bool) (message : String) : Except String Unit :=
  if condition then .ok () else .error message

private def natural (value : Lean.Json) (field : String) : Except String Nat := do
  (← value.getObjVal? field).getNat?

private def array (value : Lean.Json) (field : String) : Except String (List Lean.Json) := do
  pure (← (← value.getObjVal? field).getArr?).toList

private def dimension (parameters : List Nat) (value : Lean.Json) : Except String Nat := do
  let coefficients ← (← value.getArr?).toList.mapM Lean.Json.getNat?
  match parameters, coefficients with
  | [w, l, r], [constant, cw, cl, cr] => pure (constant + cw*w + cl*l + cr*r)
  | _, _ => throw "malformed manifest dimension"

private def fieldDimension (parameters : List Nat) (value : Lean.Json) (field : String) :
    Except String Nat := do
  dimension parameters (← value.getObjVal? field)

private def updateValue : Codec.Value → List Nat → Nat → Except String Codec.Value
  | _, [], value => pure (.atom value)
  | .atom _, _ :: _, _ => throw "relocation enters an atom"
  | .array values, index :: rest, value => do
      let some child := values[index]? | throw "relocation index is out of range"
      pure (.array (values.set index (← updateValue child rest value)))

private partial def equalValue : Codec.Value → Codec.Value → Bool
  | .atom left, .atom right => left == right
  | .array left, .array right =>
      left.length == right.length && (left.zip right).all (fun pair => equalValue pair.1 pair.2)
  | _, _ => false

private def relocationsFor (parameters : List Nat) (manifest : Lean.Json) (field : String)
    (blockStart blockCount : Nat) (source : Codec.Value) : Except String Codec.Value := do
  let mut result := source
  for relocation in ← array manifest field do
    let path ← (← array relocation "path").mapM Lean.Json.getNat?
    match path with
    | [] => throw "empty matrix relocation path"
    | block :: rest =>
      if blockStart ≤ block && block < blockStart + blockCount then
        let value ← fieldDimension parameters relocation "value"
        result ← updateValue result ((block - blockStart) :: rest) value
  pure result

private partial def codecValue (value : Lean.Json) : Except String Codec.Value := do
  match value with
  | .arr values => pure (.array (← values.toList.mapM codecValue))
  | _ => pure (.atom (← value.getNat?))

private def sourceRuns (parameters : List Nat) (values : List Lean.Json) :
    Except String (List AffineRuns.Run) := do
  let mut runs := []
  for value in values do
    let first ← fieldDimension parameters value "first"
    let count ← fieldDimension parameters value "count"
    let step ← natural value "step"
    if count != 0 then runs := ⟨first, step, count⟩ :: runs
  pure runs.reverse

/-- Compare whole affine index streams without expanding their slot values. -/
private partial def equalRuns : List AffineRuns.Run → List AffineRuns.Run → Bool
  | [], [] => true
  | [], _ | _, [] => false
  | left :: ls, right :: rs =>
    if left.count = 0 then equalRuns ls (right :: rs)
    else if right.count = 0 then equalRuns (left :: ls) rs
    else
      let count := min left.count right.count
      if left.first != right.first || (count > 1 && left.step != right.step) then false
      else
        let ls := if count = left.count then ls else
          { left with first := left.first + count * left.step, count := left.count - count } :: ls
        let rs := if count = right.count then rs else
          { right with first := right.first + count * right.step, count := right.count - count } :: rs
        equalRuns ls rs

private def programValue (compiled : PiRlcWideSampler.RangePlan.Compiled)
    (application : NightstreamFPrime.Lifecycle.Stage1.Application.Program) : Except String Program :=
  Wide.PhysicalMatrixSource.program application compiled

private def applicationValue (application : NightstreamFPrime.Lifecycle.Stage1.Application.Program) :
    Except String Program :=
  Wide.PhysicalMatrixSource.relocateProgram
    (PerApplicationPackage.directAddedPrivateColumnCount application)
    (Wide.ReusedMatrixPrograms.applicationProgram application)

private def replaceApplication (source replacement : Codec.Value) (start count : Nat) : Except String Codec.Value := do
  let .array blocks := source | throw "matrix program is not an array"
  let .array application := replacement | throw "application program is not an array"
  require (start + count ≤ blocks.length) "application replacement exceeds program"
  return .array (blocks.take start ++ application ++ blocks.drop (start + count))

private def valueAt : Codec.Value → List Nat → Except String Nat
  | .atom value, [] => pure value
  | .array values, index :: rest => do
      let some value := values[index]? | throw "matrix relocation is out of range"
      valueAt value rest
  | _, _ => throw "matrix relocation does not select an atom"

private def applicationChild (manifest : Lean.Json) : Except String Lean.Json := do
  let children ← array manifest "children"
  let some child := children.find? (fun child => child.getObjValAs? Bool "replaceable" == .ok true)
    | throw "manifest omits the application child"
  return child

private def checkApplication (manifest : Lean.Json) (reference : Program)
    (application : NightstreamFPrime.Lifecycle.Stage1.Application.Program)
    (expected : Program) : Except String Unit := do
  let parameters := [application.witnessWordCount,
    ApplicationRetainedBlocks.localCount application,
    PerApplicationPackage.directApplicationRowCount application]
  let prefixRows ← Wide.PhysicalRelabel.row Data.physicalLayout.rowCount
  let physicalPlan := Wide.ApplicationPackage.plan application prefixRows
  let privateCount := application.witnessWordCount + physicalPlan.privateCount
  let applicationProgram ← applicationValue application
  let geometry ← manifest.getObjVal? "geometry"
  for (name, value) in [
      ("source_rows", prefixRows + physicalPlan.rowCount + 5),
      ("source_private", Layout.Stage1.Wide.SourceOrder.privateColumns + privateCount),
      ("source_constant", Layout.Stage1.Wide.SourceOrder.constantColumn + privateCount),
      ("source_total", Layout.Stage1.Wide.SourceOrder.totalColumns + privateCount),
      ("logical_rows", expected.rowCount),
      ("logical_width", Wide.RetainedLayout.logicalWidth application)] do
    require ((← fieldDimension parameters geometry name) == value)
      s!"manifest dimension differs: {name}"
  let children ← array manifest "children"
  require (children.length == PerApplicationProductionPlan.canonicalKinds.length)
    "manifest omits a required child"
  let mut blockStart := 0
  let mut rowStart := 0
  for (kind, child) in PerApplicationProductionPlan.canonicalKinds.zip children do
    let count ← natural child "block_count"
    require ((← natural child "block_start") == blockStart)
      "matrix children do not cover their ordered blocks"
    let rows := ((expected.blocks.drop blockStart).take count).foldl (fun total block => total + block.rowCount) 0
    require ((← fieldDimension parameters child "row_start") == rowStart &&
      (← fieldDimension parameters child "row_count") == rows)
      "matrix child row ownership differs"
    if kind == .application then
      require (count == applicationProgram.blocks.length)
        "application connector block count differs"
    blockStart := blockStart + count
    rowStart := rowStart + rows
  require (blockStart == expected.blocks.length && rowStart == expected.rowCount)
    "matrix children omit program rows"
  let child ← applicationChild manifest
  let start ← natural child "block_start"
  let count ← natural child "block_count"
  let shared ← relocationsFor parameters manifest "matrix_relocations" 0 reference.blocks.length
    (Program.format.encode reference)
  let template ← codecValue (← manifest.getObjVal? "application_matrix_template")
  let connector ← relocationsFor parameters manifest "application_matrix_relocations" 0 count template
  require (equalValue connector (Program.format.encode applicationProgram))
    "application matrix connector differs"
  let actual ← replaceApplication shared connector start count
  require (equalValue actual (Program.format.encode expected))
    "complete wide matrix relocation differs"
  let transport ← Wide.AssignmentTransport.plan application
    (Layout.Stage1.Wide.SourceOrder.totalColumns + privateCount)
  let blocks ← array manifest "assignment_blocks"
  require (blocks.length == transport.blocks.length) "manifest omits a retained assignment block"
  for (expected, metadata) in transport.blocks.zip blocks do
    require ((← fieldDimension parameters metadata "slot_count") == expected.count)
      "manifest retained slot count differs"
    let actual ← sourceRuns parameters (← array metadata "source_runs")
    require (equalRuns actual expected.sources)
      s!"assignment relocation differs at block {← natural metadata "opcode"}"
  for (field, expected) in [("phi81_value_sources", transport.valueSources),
      ("phi81_challenge_sources", transport.challengeSources)] do
    let actual ← sourceRuns parameters (← array manifest field)
    require (equalRuns actual expected) s!"wide source relocation differs: {field}"

private def checkSelected (compiled : PiRlcWideSampler.RangePlan.Compiled)
    (manifest : Lean.Json) (ordinary : Program) : Except String Unit := do
  let selected ← manifest.getObjVal? "selected_reference"
  let application := Poseidon2HashChainV1Package.application
  let program ← programValue compiled application
  require ((← natural selected "logical_rows") == program.rowCount &&
    (← natural selected "logical_width") == Wide.RetainedLayout.logicalWidth application)
    "selected reference geometry differs"
  let child ← applicationChild manifest
  let start ← natural child "block_start"
  let applicationProgram ← applicationValue application
  let selectedApplication ← codecValue (← selected.getObjVal? "application_matrix")
  require (equalValue selectedApplication (Program.format.encode applicationProgram))
    "selected reference application matrix differs"
  require (equalValue selectedApplication (Program.format.encode
    ⟨(program.blocks.drop start).take applicationProgram.blocks.length⟩))
    "selected application child is misplaced"
  let localIndex := Wide.AssignmentTransport.commonKinds.idxOf .applicationLocal
  require ((← natural selected "application_local_index") == localIndex)
    "selected application local index differs"
  let localBlock ← Wide.AssignmentTransport.commonBlock application .applicationLocal
  require (equalValue (← codecValue (← selected.getObjVal? "application_local")) localBlock.encode)
    "selected reference assignment differs"
  let template ← codecValue (← manifest.getObjVal? "application_matrix_template")
  let replaced ← replaceApplication (Program.format.encode program) template start applicationProgram.blocks.length
  let mut actual := replaced
  let mut paths : List (List Nat) := []
  for relocation in ← array selected "matrix_relocations" do
    let path ← (← array relocation "path").mapM Lean.Json.getNat?
    require (!path.isEmpty && !paths.contains path) "duplicate or empty selected relocation"
    let before ← natural relocation "selected"
    let after ← natural relocation "ordinary"
    require ((← valueAt actual path) == before) "selected field differs before conversion"
    actual ← updateValue actual path after
    paths := path :: paths
  require (equalValue actual (Program.format.encode ordinary))
    "selected-to-ordinary conversion differs from complete wide program"
  let mut restored := actual
  for relocation in ← array selected "matrix_relocations" do
    let path ← (← array relocation "path").mapM Lean.Json.getNat?
    let before ← natural relocation "selected"
    let after ← natural relocation "ordinary"
    require ((← valueAt restored path) == after) "ordinary field differs before restoration"
    restored ← updateValue restored path before
  require (equalValue restored replaced) "selected conversion changed undeclared fields"

private def checked {α : Type} (phase : String) (action : Unit → Except String α) : IO α := do
  IO.println s!"wide shared verifier: {phase}"
  (← IO.getStdout).flush
  match action () with
  | .error error => throw (IO.userError error)
  | .ok value => return value

def check : IO Unit := do
  let compiled ← checked "range compiler" fun _ =>
    match PiRlcWideSampler.RangePlan.compile? with
    | some compiled => .ok compiled
    | none => .error "wide range compilation failed"
  let manifest ← Export.SharedVerifier.prepare
  let ordinary ← checked "ordinary reference program" fun _ => programValue compiled ordinaryReference
  checked "selected-to-ordinary matrix comparison" fun _ => checkSelected compiled manifest ordinary
  checked "ordinary reference dimensions and runs" fun _ =>
    checkApplication manifest ordinary ordinaryReference ordinary
  let fixture := PerApplicationEmitterFixture.program ()
  let fixtureProgram ← checked "identity application program" fun _ => programValue compiled fixture
  checked "identity application dimensions and runs" fun _ =>
    checkApplication manifest ordinary fixture fixtureProgram
  IO.println "wide shared verifier manifest checks passed"

#audit_axioms NightstreamFPrime.Export.Stage1.Wide.PhysicalMatrixSource.schedule_correct
#audit_axioms NightstreamFPrime.Export.Stage1.Wide.MatrixProgram.fixedPoint_exact
#audit_axioms NightstreamFPrime.Export.Stage1.Wide.MatrixProjection.column_eq
#audit_axioms NightstreamFPrime.Export.Stage1.Wide.FixedPoint.plan_fixedPoint
#audit_axioms NightstreamFPrime.Export.Stage1.Wide.AssignmentTransport.commonBlock_source

end NightstreamFPrime.Tests.SharedVerifier
