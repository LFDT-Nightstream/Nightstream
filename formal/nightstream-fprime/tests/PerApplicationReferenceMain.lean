import NightstreamFPrime.Export.Main
import tests.PerApplicationEmitterFixture

/-!
Test-only full-package byte oracle. It uses the canonical generic codecs for
each item and bounded canonical generators. It does not call `TypedWriter` or
the production per-application package writer.
-/

namespace NightstreamFPrime.Tests.PerApplicationReferenceMain

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export
open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Export.Stage1

abbrev Program := NightstreamFPrime.Lifecycle.Stage1.Application.Program

def writeEncoded {α : Type} (handle : IO.FS.Handle) (format : Format α)
    (value : α) : IO Unit := do
  let _ ← (format.encode value).writeCanonical handle
  pure ()

def writeAtom (handle : IO.FS.Handle) (value : Nat) : IO Unit := do
  let _ ← (Value.atom value).writeCanonical handle
  pure ()

def comma (handle : IO.FS.Handle) : IO Unit :=
  writeByte handle 44

def writeItems {α : Type} (handle : IO.FS.Handle) (format : Format α)
    (first : Bool) (values : List α) : IO Bool :=
  writeArrayItemsWith handle (writeEncoded handle format) first values

def writeShiftedPermutation (program : Program) (handle : IO.FS.Handle)
    (invocation : Package.PermutationInvocation) : IO Unit :=
  writeEncoded handle Package.PermutationInvocation.format
    (PerApplicationPackage.shiftPermutationInvocation program invocation)

partial def writePermutationBlocks (program : Program)
    (handle : IO.FS.Handle) (first : Bool) :
    List PermutationPlan.Block → IO Bool
  | [] => pure first
  | block :: rest => do
      let first ← writeArrayItemsWith handle
        (writeShiftedPermutation program handle) first block.expand
      writePermutationBlocks program handle first rest

def writePermutationInvocations (program : Program) (handle : IO.FS.Handle)
    (blocks : List PermutationPlan.Block) : IO Unit := do
  writeByte handle 91
  let _first ← writePermutationBlocks program handle true blocks
  writeByte handle 93

def writeShiftedCompactInvocation (program : Program)
    (handle : IO.FS.Handle)
    (invocation : Package.CompactRowInvocation) : IO Unit :=
  writeEncoded handle Package.CompactRowInvocation.format
    (PerApplicationPackage.shiftCompactRowInvocation program invocation)

partial def writeCompactBlocks (program : Program) (handle : IO.FS.Handle)
    (first : Bool) : List PackagePlan.CompactInvocationBlock → IO Bool
  | [] => pure first
  | block :: rest => do
      let first ← writeArrayItemsWith handle
        (writeShiftedCompactInvocation program handle) first block.expand
      writeCompactBlocks program handle first rest

def writeCompactInvocations (program : Program)
    (handle : IO.FS.Handle) : IO Unit := do
  writeByte handle 91
  let _first ← writeCompactBlocks program handle true
    PackagePlan.canonicalCompactBlocks
  writeByte handle 93

def orderedWitnessTasks (tasks : Main.PreparedWitnessTasks) :
    List Main.PreparedWitnessTask :=
  [tasks.initialClaim, tasks.sumcheck, tasks.evalK, tasks.evalA, tasks.ccs,
    tasks.norm, tasks.finalIdentity]

partial def writePreparedGroups {α : Type} (handle : IO.FS.Handle)
    (writeItem : α → IO Unit)
    (select : Main.PreparedWitnessGroup → List α) (first : Bool) :
    List Main.PreparedWitnessTask → IO Bool
  | [] => pure first
  | task :: rest => do
      let prepared ← Main.preparedWitnessGroup task
      let first ← writeArrayItemsWith handle writeItem first (select prepared)
      writePreparedGroups handle writeItem select first rest

def writeShiftedBatch (program : Program) (handle : IO.FS.Handle)
    (batch : WitnessBatch) : IO Unit :=
  writeEncoded handle Package.WitnessBatch.format
    (PerApplicationPackage.shiftBatch program batch)

partial def writeWitnessBlocks (program : Program) (handle : IO.FS.Handle)
    (first : Bool) : List WitnessPlan.Block → IO Bool
  | [] => pure first
  | block :: rest => do
      let first ← writeArrayItemsWith handle
        (writeShiftedBatch program handle) first block.expand
      writeWitnessBlocks program handle first rest

def writeWitnessBatches (program : Program) (handle : IO.FS.Handle)
    (tasks : Main.PreparedWitnessTasks)
    (application : ApplicationPackage.Plan) : IO Unit := do
  writeByte handle 91
  let first ← writeArrayItemsWith handle (writeShiftedBatch program handle)
    true (Data.liftPilotBatches (PilotData.priorWordBatches ()))
  let first ← writePreparedGroups handle (writeShiftedBatch program handle)
    Main.PreparedWitnessGroup.batches first (orderedWitnessTasks tasks)
  let first ← writeWitnessBlocks program handle first
    (WitnessPlan.canonicalBlocks Data.logicalWidth Data.publicFits)
  let _first ← writeItems handle Package.WitnessBatch.format first
    application.witnessBatches
  writeByte handle 93

def writeShiftedInstruction (program : Program) (handle : IO.FS.Handle)
    (instruction : Package.WitnessInstruction) : IO Unit :=
  writeEncoded handle Package.WitnessInstruction.format
    (PerApplicationPackage.shiftWitnessInstruction program instruction)

def writeShiftedRow (program : Program) (handle : IO.FS.Handle)
    (row : Package.SparseRow) : IO Unit :=
  writeEncoded handle Package.SparseRow.format
    (PerApplicationPackage.shiftSparseRow program row)

partial def writePreparedRows {α : Type} (handle : IO.FS.Handle)
    (writeItem : α → IO Unit) (select : Main.PreparedRowBlock → List α)
    (first : Bool) : List Main.PreparedRowTask → IO Bool
  | [] => pure first
  | task :: rest => do
      let prepared ← Main.preparedRowBlock task
      let first ← writeArrayItemsWith handle writeItem first (select prepared)
      writePreparedRows handle writeItem select first rest

partial def writePreparedRowBlocks {α : Type} (handle : IO.FS.Handle)
    (writeItem : α → IO Unit) (select : Main.PreparedRowBlock → List α)
    (first : Bool) : List Main.PreparedRowBlock → IO Bool
  | [] => pure first
  | prepared :: rest => do
      let first ← writeArrayItemsWith handle writeItem first (select prepared)
      writePreparedRowBlocks handle writeItem select first rest

partial def writePreparedSources {α : Type} (handle : IO.FS.Handle)
    (writeItem : α → IO Unit) (select : Main.PreparedRowBlock → List α)
    (first : Bool) : List Main.PreparedRowSourceTask → IO Bool
  | [] => pure first
  | task :: rest => do
      let prepared ← Main.preparedRowSource task
      let first ← writePreparedRowBlocks handle writeItem select first prepared
      writePreparedSources handle writeItem select first rest

def writeWitnessInstructions (program : Program) (handle : IO.FS.Handle)
    (witnessTasks : Main.PreparedWitnessTasks)
    (rowTasks : Main.PreparedRowTasks)
    (application : ApplicationPackage.Plan) : IO Unit := do
  let writer := writeShiftedInstruction program handle
  writeByte handle 91
  let first ← writeArrayItemsWith handle writer true
    (Data.liftPilotInstructions (PilotData.witnessInstructions ()))
  let first ← writePreparedRows handle writer
    Main.PreparedRowBlock.witnessInstructions first [rowTasks.statementBinding]
  let first ← writePreparedGroups handle writer
    Main.PreparedWitnessGroup.witnessInstructions first
    (orderedWitnessTasks witnessTasks)
  let first ← writePreparedSources handle writer
    Main.PreparedRowBlock.witnessInstructions first rowTasks.piRlcSources
  let first ← writePreparedRows handle writer
    Main.PreparedRowBlock.witnessInstructions first
    [rowTasks.piDec, rowTasks.runningTransition]
  let _first ← writeItems handle Package.WitnessInstruction.format first
    application.witnessInstructions
  writeByte handle 93

def writeAssertionRows (program : Program) (handle : IO.FS.Handle)
    (witnessTasks : Main.PreparedWitnessTasks)
    (rowTasks : Main.PreparedRowTasks)
    (application : ApplicationPackage.Plan) : IO Unit := do
  let writer := writeShiftedRow program handle
  writeByte handle 91
  let first ← writeArrayItemsWith handle writer true
    (Data.liftPilotRows (PilotData.assertionRows ()))
  let first ← writePreparedRows handle writer
    Main.PreparedRowBlock.assertionRows first [rowTasks.statementBinding]
  let first ← writePreparedGroups handle writer
    Main.PreparedWitnessGroup.assertionRows first
    (orderedWitnessTasks witnessTasks)
  let first ← writePreparedSources handle writer
    Main.PreparedRowBlock.assertionRows first rowTasks.piRlcSources
  let first ← writePreparedRows handle writer
    Main.PreparedRowBlock.assertionRows first
    [rowTasks.piDec, rowTasks.runningTransition]
  let first ← writeItems handle Package.SparseRow.format first
    application.assertionRows
  let _first ← writeItems handle Package.SparseRow.format first
    (NextPreimagePackage.assertionRows
      (PerApplicationPackage.nextPreimageRowStart program))
  writeByte handle 93

def writeInnerPackage (program : Program) (handle : IO.FS.Handle)
    (permutationBlocks : Main.PreparedPermutationBlocks)
    (witnessTasks : Main.PreparedWitnessTasks)
    (rowTasks : Main.PreparedRowTasks)
    (application : ApplicationPackage.Plan) : IO Unit := do
  writeByte handle 91
  writeAtom handle 8
  comma handle
  writeEncoded handle Package.Profile.format PilotData.profile
  comma handle
  writeEncoded handle Package.PoseidonSchedule.format PilotData.poseidonSchedule
  comma handle
  writeEncoded handle Package.PhysicalLayout.format
    (PerApplicationPackage.directFinalLayout program)
  comma handle
  writeEncoded handle Package.CcsRelation.format
    (PerApplicationCanonicalPackage.directRecursiveRelation program)
  comma handle
  writeEncoded handle Package.PermutationTemplate.format
    (PilotData.permutationTemplate ())
  comma handle
  writeEncoded handle (list Package.HashChain.format)
    ([Data.priorChain, Data.outputChain].map
      (PerApplicationPackage.shiftHashChain program))
  comma handle
  writePermutationInvocations program handle permutationBlocks.blocks
  comma handle
  writeByte handle 91
  let _first ← writeItems handle Package.CompactRowTemplate.format true
    (Data.compactRowTemplates ())
  writeByte handle 93
  comma handle
  writeCompactInvocations program handle
  comma handle
  writeWitnessBatches program handle witnessTasks application
  comma handle
  writeWitnessInstructions program handle witnessTasks rowTasks application
  comma handle
  writeAssertionRows program handle witnessTasks rowTasks application
  comma handle
  writeEncoded handle (option Package.TerminalLayout.format)
    (some (PerApplicationCanonicalPackage.directTerminalLayout program))
  writeByte handle 93

def writeSealedPackage (program : Program)
    (_fits : PerApplicationFixedPoint.FitsTwoPow28 program)
    (handle : IO.FS.Handle) : IO Unit := do
  let permutationTask ← IO.asTask Main.preparePermutationBlocks
  let witnessTasks ← Main.prepareWitnessGroups
  let rowTasks ← Main.prepareRowBlocks
  let application := PerApplicationPackage.directApplicationPlan program
  let permutationBlocks ← Main.preparedPermutationBlocks permutationTask
  writeByte handle 91
  writeAtom handle PerApplicationCanonicalPackage.sealedPackageSchema
  comma handle
  writeInnerPackage program handle permutationBlocks witnessTasks rowTasks
    application
  comma handle
  writeEncoded handle MatrixProgram.Program.format
    (PerApplicationMatrixProgram.matrixProgram program)
  comma handle
  writeEncoded handle ApplicationPackage.Plan.format application
  comma handle
  writeEncoded handle MatrixProgram.IndexRange.format
    (PerApplicationCanonicalPackage.nextPreimageRange program)
  comma handle
  writeAtom handle PerApplicationCanonicalPackage.logicalPublicInputCount
  writeByte handle 93

def emit (path : System.FilePath) : IO Unit := do
  if let some parent := path.parent then
    IO.FS.createDirAll parent
  let handle ← IO.FS.Handle.mk path .write
  writeSealedPackage (PerApplicationEmitterFixture.program ())
    (PerApplicationEmitterFixture.fits ()) handle
  handle.write (ByteArray.empty.push 10)
  handle.flush
  IO.println s!"emitted_per_application_reference={path}"

def run (arguments : List String) : IO UInt32 := do
  match arguments with
  | [path] | ["--", path] =>
      let start ← IO.monoMsNow
      emit ⟨path⟩
      let finish ← IO.monoMsNow
      IO.println s!"per_application_reference_ms={finish - start}"
      pure 0
  | _ =>
      IO.eprintln "usage: emitPerApplicationReferenceFixture <path>"
      pure 2

end NightstreamFPrime.Tests.PerApplicationReferenceMain

def main (arguments : List String) : IO UInt32 :=
  NightstreamFPrime.Tests.PerApplicationReferenceMain.run arguments
