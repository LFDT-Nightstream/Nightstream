import NightstreamFPrime.Export.Stage1.Wide.PhysicalPackage
import NightstreamFPrime.Export.Stage1.PerApplicationPackage

/-! Extend the wide physical prefix with the same application and next-state
binding. Only source positions change; the application circuit is unchanged. -/

namespace NightstreamFPrime.Export.Stage1.Wide.ApplicationPackage

open NightstreamFPrime.Export.Package NightstreamFPrime.Layout
open Layout.Stage1.Wide

abbrev Program := Lifecycle.Stage1.Application.Program

def columns (program : Program) : Stage1.ApplicationPackage.Columns program.witnessWordCount where
  input := Layout.Stage1.ApplicationInputs.inputColumn
  output := Layout.Stage1.ApplicationInputs.outputColumn
  witness := fun index => SourceOrder.privateColumns + index.val

def plan (program : Program) (rowStart : Nat) : Stage1.ApplicationPackage.Plan :=
  Stage1.ApplicationPackage.ofProgram program (columns program)
    (SourceOrder.privateColumns + program.witnessWordCount) rowStart

def insertApplication (count : Nat) : PhysicalRelabel.Map where
  column := fun column => .ok (if column < SourceOrder.privateColumns then column else column + count)
  row := .ok

def ofBase (program : Program) (base : CircuitPackage) :
    Except String (CircuitPackage × Stage1.ApplicationPackage.Plan) := do
  let application := plan program base.layout.rowCount
  let count := program.witnessWordCount + application.privateCount
  let insert := insertApplication count
  let nextStart := base.layout.rowCount + application.rowCount
  let nextRows ← (NextPreimagePackage.assertionRows (PerApplicationPackage.nextPreimageRowStart program)).mapM
    PhysicalRelabel.prefixMap.assertion
  let layout : PhysicalLayout := {
    base.layout with
    rowCount := nextStart + 5
    privateColumnCount := base.layout.privateColumnCount + count
    constantColumn := base.layout.constantColumn + count
    totalColumnCount := base.layout.totalColumnCount + count
    privateSegments := base.layout.privateSegments ++
      [⟨PerApplicationPackage.Role.applicationWitness, SourceOrder.privateColumns, program.witnessWordCount⟩,
       ⟨PerApplicationPackage.Role.applicationLocal, application.privateStart, application.privateCount⟩]
    publicSegments := ← base.layout.publicSegments.mapM insert.segment }
  let result : CircuitPackage := { base with
    schemaVersion := 8
    layout := layout
    relation := productionCcsRelation layout.rowCount layout.totalColumnCount Lifecycle.cubeVariables
    hashChains := ← base.hashChains.mapM insert.chain
    permutationInvocations := ← base.permutationInvocations.mapM insert.permutation
    compactRowInvocations := ← base.compactRowInvocations.mapM insert.compact
    witnessBatches := (← base.witnessBatches.mapM insert.batch) ++ application.witnessBatches
    witnessInstructions := (← base.witnessInstructions.mapM insert.instruction) ++ application.witnessInstructions
    assertionRows := (← base.assertionRows.mapM insert.assertion) ++ application.assertionRows ++ nextRows }
  return (TerminalPackage.install result, application)

def package (program : Program) : Except String (CircuitPackage × Stage1.ApplicationPackage.Plan) := do
  ofBase program (← PhysicalPackage.circuitPackage ())

end NightstreamFPrime.Export.Stage1.Wide.ApplicationPackage
