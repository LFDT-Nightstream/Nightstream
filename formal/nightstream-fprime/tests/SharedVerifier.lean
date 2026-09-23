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

private def checkApplication (manifest : Lean.Json)
    (application : NightstreamFPrime.Lifecycle.Stage1.Application.Program) : Except String Unit := do
  let parameters := [application.witnessWordCount,
    ApplicationRetainedBlocks.localCount application,
    PerApplicationPackage.directApplicationRowCount application]
  let geometry ← manifest.getObjVal? "geometry"
  let sourceLayout := PerApplicationPackage.directFinalLayout application
  for (name, expected) in [
      ("source_rows", sourceLayout.rowCount),
      ("source_private", sourceLayout.privateColumnCount),
      ("source_constant", sourceLayout.constantColumn),
      ("source_total", sourceLayout.totalColumnCount),
      ("logical_rows", PerApplicationCanonicalPackage.directStructuralRowCount application),
      ("logical_width", PerApplicationCanonicalPackage.directLogicalWidth application)] do
    require ((← fieldDimension parameters geometry name) == expected)
      s!"manifest dimension differs: {name}"
  let children ← array manifest "children"
  require (children.length == PerApplicationProductionPlan.canonicalKinds.length)
    "manifest omits a required child"
  let mut expectedBlockStart := 0
  let mut expectedRowStart := 0
  for (kind, metadata) in PerApplicationProductionPlan.canonicalKinds.zip children do
    let reference := PerApplicationMatrixProgram.blockProgram
      ordinaryReference kind
    let expected := PerApplicationMatrixProgram.blockProgram application kind
    let blockStart ← natural metadata "block_start"
    let blockCount ← natural metadata "block_count"
    require (blockStart == expectedBlockStart && blockCount == reference.blocks.length)
      "manifest matrix child range differs"
    require ((← fieldDimension parameters metadata "row_start") == expectedRowStart &&
        (← fieldDimension parameters metadata "row_count") == expected.rowCount)
      "manifest matrix row owner differs"
    let source := Program.format.encode reference
    if kind != .application then
      let actual ← relocationsFor parameters manifest "matrix_relocations" blockStart blockCount source
      require (equalValue actual (Program.format.encode expected))
        s!"shared matrix relocation differs at child {← natural metadata "opcode"}"
    else
      let template ← codecValue (← manifest.getObjVal? "application_matrix_template")
      let actual ← relocationsFor parameters manifest "application_matrix_relocations" 0 blockCount template
      require (equalValue actual (Program.format.encode expected))
        "application matrix connector differs"
    expectedBlockStart := expectedBlockStart + blockCount
    expectedRowStart := expectedRowStart + expected.rowCount
  let blocks ← array manifest "assignment_blocks"
  require (blocks.length == PerApplicationAssignmentPlan.canonicalKinds.length)
    "manifest omits a retained assignment block"
  for (kind, metadata) in PerApplicationAssignmentPlan.canonicalKinds.zip blocks do
    let expected := PerApplicationAssignmentBlocks.BlockPlan.ofKind application kind
    require ((← fieldDimension parameters metadata "slot_count") == expected.slotCount)
      "manifest retained slot count differs"
    let actual ← sourceRuns parameters (← array metadata "source_runs")
    require (equalRuns actual expected.sourceRuns)
      s!"assignment relocation differs at block {← natural metadata "opcode"}"
  let actual ← sourceRuns parameters (← array manifest "phi81_value_sources")
  require (equalRuns actual (PerApplicationAssignmentTransport.phi81ValueSources application))
    "Phi81 native-value source relocation differs"

def check : IO Unit := do
  let result : Except String Unit := do
    let manifest ← Export.SharedVerifier.value ()
    let selected ← manifest.getObjVal? "selected_reference"
    let application := Poseidon2HashChainV1Package.application
    require ((← natural selected "logical_rows") ==
      PerApplicationCanonicalPackage.directStructuralRowCount application)
      "selected reference row count differs"
    require ((← natural selected "logical_width") ==
      PerApplicationCanonicalPackage.directLogicalWidth application)
      "selected reference width differs"
    require (equalValue (← codecValue (← selected.getObjVal? "application_matrix"))
      (Program.format.encode (PerApplicationMatrixProgram.applicationProgram application)))
      "selected reference matrix differs"
    require (equalValue (← codecValue (← selected.getObjVal? "application_local"))
      (PerApplicationAssignmentBlocks.BlockPlan.format.encode
        (PerApplicationAssignmentBlocks.BlockPlan.ofKind application .applicationLocal)))
      "selected reference assignment differs"
    checkApplication manifest ordinaryReference
    checkApplication manifest (PerApplicationEmitterFixture.program ())
  match result with
  | .error error => throw (IO.userError error)
  | .ok () => IO.println "shared verifier manifest checks passed"

#audit_axioms NightstreamFPrime.Export.Stage1.PerApplicationPackage.shiftSparseRow_holds
#audit_axioms NightstreamFPrime.Export.Stage1.PerApplicationPackage.directFinalLayout_eq_finalLayout
#audit_axioms NightstreamFPrime.Export.Stage1.PerApplicationSourceProjection.base_column
#audit_axioms NightstreamFPrime.Export.Stage1.PerApplicationSourceProjection.pilot_column
#audit_axioms NightstreamFPrime.Export.Stage1.PerApplicationMatrixProgram.matrixProgram_blocks
#audit_axioms NightstreamFPrime.Export.Stage1.PerApplicationCanonicalPackage.matrixProgram_exact
#audit_axioms NightstreamFPrime.Export.Stage1.ApplicationRetainedGeometry.completeLogicalWidth_eq_applicationCounts
#audit_axioms NightstreamFPrime.Export.Stage1.ApplicationRetainedGeometry.carrierWidth_le_twoPow28_iff
#audit_axioms NightstreamFPrime.Export.Stage1.PerApplicationFixedPoint.plan_fixedPoint
#audit_axioms NightstreamFPrime.Export.Stage1.PerApplicationAssignmentBlocks.sourceRuns_expand

end NightstreamFPrime.Tests.SharedVerifier
