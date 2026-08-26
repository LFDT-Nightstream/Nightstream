import NightstreamFPrime.Export.Stage1.Data
import NightstreamFPrime.Export.Stage1.PackagePlan

/-! Executable entry point for the canonical Stage 1 circuit-package emitter. -/

namespace NightstreamFPrime.Export.Main

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
  writeList handle Package.TemplateRow.format template.rows
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
  writeValue handle (Package.exprFormat.encode template.outputRecipe)
  comma handle
  writeList handle Package.CompactTemplateRow.format template.rows
  writeByte handle 93

def writeWitnessBatch (handle : IO.FS.Handle)
    (batch : NightstreamFPrime.Circuit.WitnessBatch) : IO Unit := do
  writeByte handle 91
  writeValue handle (.atom batch.start)
  comma handle
  writeList handle Package.exprFormat batch.recipes
  comma handle
  writeList handle Package.hintFormat batch.hints
  writeByte handle 93

def writeWitnessBatchItems (handle : IO.FS.Handle) (first : Bool)
    (batches : List NightstreamFPrime.Circuit.WitnessBatch) : IO Bool :=
  writeArrayItemsWith handle (writeWitnessBatch handle) first batches

partial def writePiRlcLaneBatchItems (handle : IO.FS.Handle)
    (first : Bool) (source round lane : Nat) : IO Bool := do
  if laneBound : lane < 4 then
    let first ← writeWitnessBatchItems handle first
      (Stage1.WitnessProgram.piRlcDigestLaneBatches Stage1.Data.logicalWidth
        Stage1.Data.publicFits source round ⟨lane, laneBound⟩)
    writePiRlcLaneBatchItems handle first source round (lane + 1)
  else
    pure first

partial def writePiRlcRoundBatchItems (handle : IO.FS.Handle)
    (first : Bool) (source round : Nat) : IO Bool := do
  if round < 8 then
    let first ← writePiRlcLaneBatchItems handle first source round 0
    writePiRlcRoundBatchItems handle first source (round + 1)
  else
    pure first

partial def writePiRlcSourceBatchItems (handle : IO.FS.Handle)
    (first : Bool) (source : Nat) : IO Bool := do
  if source < 17 then
    progress s!"emitter_stage=witness_sampler_source_{source}"
    let first ← writePiRlcRoundBatchItems handle first source 0
    writePiRlcSourceBatchItems handle first (source + 1)
  else
    pure first

/-- Stream `WitnessProgram.batches` in its exact concatenation and indexed
sampler order without constructing the complete list first. -/
def writeWitnessBatches (handle : IO.FS.Handle) : IO Unit := do
  writeByte handle 91
  progress "emitter_stage=witness_initial_claim"
  let first ← writeWitnessBatchItems handle true
    (Stage1.WitnessProgram.initialClaimBatches Stage1.Data.logicalWidth
      Stage1.Data.publicFits)
  progress "emitter_stage=witness_sumcheck"
  let first ← writeWitnessBatchItems handle first
    (Stage1.WitnessProgram.sumcheckBatches Stage1.Data.logicalWidth
      Stage1.Data.publicFits)
  progress "emitter_stage=witness_eval_k"
  let first ← writeWitnessBatchItems handle first
    (Stage1.WitnessProgram.evalKBatches Stage1.Data.logicalWidth
      Stage1.Data.publicFits)
  progress "emitter_stage=witness_eval_a"
  let first ← writeWitnessBatchItems handle first
    (Stage1.WitnessProgram.evalABatches Stage1.Data.logicalWidth
      Stage1.Data.publicFits)
  progress "emitter_stage=witness_ccs"
  let first ← writeWitnessBatchItems handle first
    (Stage1.WitnessProgram.ccsBatches Stage1.Data.logicalWidth
      Stage1.Data.publicFits)
  progress "emitter_stage=witness_norm"
  let first ← writeWitnessBatchItems handle first
    (Stage1.WitnessProgram.normBatches Stage1.Data.logicalWidth
      Stage1.Data.publicFits)
  progress "emitter_stage=witness_final_identity"
  let first ← writeWitnessBatchItems handle first
    (Stage1.WitnessProgram.finalIdentityBatches Stage1.Data.logicalWidth
      Stage1.Data.publicFits)
  let _first ← writePiRlcSourceBatchItems handle first 0
  writeByte handle 93

def writePermutationActionShape (handle : IO.FS.Handle)
    (shape : Stage1.PermutationPlan.ActionShape) : IO Unit := do
  match shape with
  | .absorb input =>
      writeByte handle 91
      writeValue handle (.atom 0)
      comma handle
      writeList handle Package.exprFormat input
      writeByte handle 93
  | .squeezeK =>
      writeByte handle 91
      writeValue handle (.atom 1)
      writeByte handle 93

def writePermutationActionBlock (handle : IO.FS.Handle)
    (block : Stage1.PermutationPlan.ActionBlock) : IO Unit := do
  writeByte handle 91
  writeValue handle (.atom block.phase)
  comma handle
  writeValue handle (.atom block.rowStart)
  comma handle
  writeValue handle (.atom block.witnessStart)
  comma handle
  writeList handle Package.exprFormat block.initialState
  comma handle
  writeListWith handle (writePermutationActionShape handle)
    block.actionShapes
  writeByte handle 93

def writeDirectPermutationBlock (handle : IO.FS.Handle)
    (block : Stage1.PermutationPlan.DirectBlock) : IO Unit := do
  writeByte handle 91
  writeValue handle (.atom block.phase)
  comma handle
  writeValue handle (.atom block.rowStart)
  comma handle
  writeValue handle (.atom block.witnessStart)
  comma handle
  writeList handle Package.exprFormat block.state
  writeByte handle 93

def writePermutationBlock (handle : IO.FS.Handle)
    (block : Stage1.PermutationPlan.Block) : IO Unit := do
  writeByte handle 91
  match block with
  | .actions actionBlock =>
      writeValue handle (.atom 0)
      comma handle
      writePermutationActionBlock handle actionBlock
  | .direct directBlock =>
      writeValue handle (.atom 1)
      comma handle
      writeDirectPermutationBlock handle directBlock
  writeByte handle 93

/-- Stream the schema-7 package with empty permutation- and compact-invocation
fields. This is exactly the static package in `PackagePlan.canonical ()`, but
it never builds one artifact-wide codec value. -/
def writeStaticPackage (handle : IO.FS.Handle) : IO Unit := do
  writeByte handle 91
  writeValue handle (.atom 7)
  comma handle
  writeValue handle (Package.Profile.format.encode PilotData.profile)
  comma handle
  writeValue handle
    (Package.PoseidonSchedule.format.encode PilotData.poseidonSchedule)
  comma handle
  writeValue handle
    (Package.PhysicalLayout.format.encode Stage1.Data.physicalLayout)
  comma handle
  writeValue handle (Package.CcsRelation.format.encode
    (Package.productionCcsRelation Stage1.Data.physicalLayout.rowCount
      Stage1.Data.physicalLayout.totalColumnCount
        NightstreamFPrime.Lifecycle.cubeVariables))
  comma handle
  progress "emitter_stage=permutation_template"
  writePermutationTemplate handle
  comma handle
  writeList handle Package.HashChain.format
    [Stage1.Data.priorChain, Stage1.Data.outputChain]
  comma handle
  writeList handle Package.PermutationInvocation.format []
  comma handle
  progress "emitter_stage=compact_templates"
  writeListWith handle (writeCompactRowTemplate handle)
    (Stage1.Data.compactRowTemplates ())
  comma handle
  writeList handle Package.CompactRowInvocation.format []
  comma handle
  progress "emitter_stage=witness_batches"
  writeWitnessBatches handle
  comma handle
  progress "emitter_stage=ordinary_rows"
  let components := Stage1.PackagePlan.staticComponents ()
  progress "emitter_stage=witness_instructions"
  writeList handle Package.WitnessInstruction.format
    components.witnessInstructions
  comma handle
  progress "emitter_stage=assertion_rows"
  writeList handle Package.SparseRow.format components.assertionRows
  comma handle
  writeValue handle
    ((option Package.TerminalLayout.format).encode none)
  writeByte handle 93

/-- Stream `PackagePlan.format.encode (PackagePlan.canonical ())` in canonical
field order. -/
def writeCanonicalPlan (handle : IO.FS.Handle) : IO Unit := do
  writeByte handle 91
  writeValue handle (.atom 8)
  comma handle
  writeStaticPackage handle
  comma handle
  progress "emitter_stage=permutation_blocks"
  writeListWith handle (writePermutationBlock handle)
    (Stage1.PermutationPlan.canonicalBlocks ())
  comma handle
  progress "emitter_stage=compact_blocks"
  writeList handle Stage1.PackagePlan.CompactInvocationBlock.format
    Stage1.PackagePlan.canonicalCompactBlocks
  writeByte handle 93

def emit (path : System.FilePath) : IO Unit := do
  progress "emitter_stage=stream"
  if let some parent := path.parent then
    IO.FS.createDirAll parent
  let handle ← IO.FS.Handle.mk path .write
  writeCanonicalPlan handle
  handle.write (ByteArray.empty.push 10)
  handle.flush
  IO.println s!"emitted={path}"

def run (arguments : List String) : IO UInt32 := do
  match arguments with
  | [path] =>
      emit ⟨path⟩
      pure 0
  | ["--", path] =>
      emit ⟨path⟩
      pure 0
  | _ =>
      IO.eprintln "usage: lake exe emit -- <output-path>"
      pure 2

end NightstreamFPrime.Export.Main

def main (arguments : List String) : IO UInt32 :=
  NightstreamFPrime.Export.Main.run arguments
