import NightstreamFPrime.Export.PilotData
import NightstreamFPrime.Export.Stage1.ApplicationPackage
import NightstreamFPrime.Export.TypedWriter

/-! Typed streaming writers shared by the canonical package emitter. -/

namespace NightstreamFPrime.Export.Main

open NightstreamFPrime.Layout
open NightstreamFPrime.Circuit
open NightstreamFPrime.Export
open NightstreamFPrime.Export.Codec

def progress (message : String) : IO Unit := do
  IO.println message
  IO.getStdout >>= IO.FS.Stream.flush

def writeValue (handle : IO.FS.Handle) (value : Value) : IO Unit := do
  let _ ← value.writeCanonical handle
  pure ()

def writeList {α : Type} (handle : IO.FS.Handle) (format : Format α)
    (values : List α) : IO Unit :=
  writeListCanonical handle format values

def comma (handle : IO.FS.Handle) : IO Unit :=
  writeByte handle 44

def writePermutationTemplate (handle : IO.FS.Handle) : IO Unit := do
  let template := PilotData.permutationTemplate ()
  writeByte handle 91
  writeValue handle (.atom template.inputCount)
  comma handle
  writeValue handle (.atom template.localColumnCount)
  comma handle
  writeValue handle (.atom template.outputLocalStart)
  comma handle
  writeListWith handle (TypedWriter.writeTemplateRow handle) template.rows
  writeByte handle 93

def writeCompactRowTemplate (handle : IO.FS.Handle)
    (template : Package.CompactRowTemplate) : IO Unit := do
  writeByte handle 91
  writeValue handle (.atom template.inputCount)
  comma handle
  writeValue handle (.atom template.localColumnCount)
  comma handle
  writeValue handle (.atom template.outputInput)
  comma handle
  TypedWriter.writeExpr handle template.outputRecipe
  comma handle
  writeListWith handle (TypedWriter.writeCompactTemplateRow handle)
    template.rows
  writeByte handle 93

/-- Stream the exact `ApplicationPackage.Plan.format` field order without
constructing duplicate codec trees for its application-sized row lists. -/
def writeApplicationPackagePlan (handle : IO.FS.Handle)
    (plan : Stage1.ApplicationPackage.Plan) : IO Unit := do
  writeByte handle 91
  writeValue handle (.atom plan.schemaVersion)
  comma handle
  writeValue handle (.atom plan.witnessWordCount)
  comma handle
  writeList handle nat plan.inputColumns
  comma handle
  writeList handle nat plan.witnessColumns
  comma handle
  writeList handle nat plan.outputColumns
  comma handle
  writeValue handle (.atom plan.privateStart)
  comma handle
  writeValue handle (.atom plan.privateCount)
  comma handle
  writeValue handle (.atom plan.rowStart)
  comma handle
  writeValue handle (.atom plan.rowCount)
  comma handle
  writeList handle Package.HashChain.format plan.hashChains
  comma handle
  writeListWith handle (TypedWriter.writePermutationInvocation handle)
    plan.permutationInvocations
  comma handle
  writeListWith handle (writeCompactRowTemplate handle)
    plan.compactRowTemplates
  comma handle
  writeListWith handle (TypedWriter.writeCompactRowInvocation handle)
    plan.compactRowInvocations
  comma handle
  writeListWith handle (TypedWriter.writeWitnessBatch handle)
    plan.witnessBatches
  comma handle
  writeListWith handle (TypedWriter.writeWitnessInstruction handle)
    plan.witnessInstructions
  comma handle
  writeListWith handle (TypedWriter.writeSparseRow handle) plan.assertionRows
  writeByte handle 93

end NightstreamFPrime.Export.Main
