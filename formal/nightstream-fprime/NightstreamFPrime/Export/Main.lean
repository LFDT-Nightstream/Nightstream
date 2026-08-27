import NightstreamFPrime.Export.Stage1.Data
import NightstreamFPrime.Export.Stage1.OrdinaryRowPlan
import NightstreamFPrime.Export.Stage1.PackagePlan
import NightstreamFPrime.Export.Stage1.PiCCSPackets

/-! Executable entry point for the canonical Stage 1 circuit-package emitter. -/

namespace NightstreamFPrime.Export.Main

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

def exprNodeCount : Expr → Nat
  | .var _ => 1
  | .const _ => 1
  | .add left right => 1 + exprNodeCount left + exprNodeCount right
  | .mul left right => 1 + exprNodeCount left + exprNodeCount right

def hintNodeCount : Hint → Nat
  | .bit source _ => 1 + exprNodeCount source
  | .inverseOrZero source => 1 + exprNodeCount source
  | .quotientFive source => 1 + exprNodeCount source
  | .remainderFive source => 1 + exprNodeCount source

def witnessBatchNodeCount (batch : WitnessBatch) : Nat :=
  (batch.recipes.map exprNodeCount).sum +
    (batch.hints.map hintNodeCount).sum

structure PreparedWitnessGroup where
  batches : List WitnessBatch
  witnessInstructions : List Package.WitnessInstruction
  assertionRows : List Package.SparseRow
  batchCount : Nat
  nodeCount : Nat

abbrev PreparedWitnessTask := Task (Except IO.Error PreparedWitnessGroup)

def prepareWitnessGroup (build : Unit → Stage1.PiCCSPackets.Packet) :
    IO PreparedWitnessGroup := do
  let packet := build ()
  let classified := Stage1.Rows.classifyRowsTR packet.rows
  pure {
    batches := packet.batches
    witnessInstructions := classified.1
    assertionRows := classified.2
    batchCount := packet.batches.length
    nodeCount := (packet.batches.map witnessBatchNodeCount).sum }

def preparedWitnessGroup (task : PreparedWitnessTask) :
    IO PreparedWitnessGroup :=
  match task.get with
  | .ok group => pure group
  | .error error => throw error

def writePreparedWitnessGroup (handle : IO.FS.Handle) (label : String)
    (first : Bool) (task : PreparedWitnessTask) : IO Bool := do
  let prepared ← preparedWitnessGroup task
  progress s!"emitter_stage=witness_{label}_batches_{prepared.batchCount}_nodes_{prepared.nodeCount}"
  writeWitnessBatchItems handle first prepared.batches

structure PreparedWitnessTasks where
  initialClaim : PreparedWitnessTask
  sumcheck : PreparedWitnessTask
  evalK : PreparedWitnessTask
  evalA : PreparedWitnessTask
  ccs : PreparedWitnessTask
  norm : PreparedWitnessTask
  finalIdentity : PreparedWitnessTask

def prepareWitnessGroups : IO PreparedWitnessTasks := do
  let initialClaim ← IO.asTask (prepareWitnessGroup fun _ =>
    Stage1.PiCCSPackets.initialClaim Stage1.Data.logicalWidth
      Stage1.Data.publicFits)
  let sumcheck ← IO.asTask (prepareWitnessGroup fun _ =>
    Stage1.PiCCSPackets.sumcheck Stage1.Data.logicalWidth
      Stage1.Data.publicFits)
  let evalK ← IO.asTask (prepareWitnessGroup fun _ =>
    Stage1.PiCCSPackets.evalK Stage1.Data.logicalWidth
      Stage1.Data.publicFits)
  let evalA ← IO.asTask (prepareWitnessGroup fun _ =>
    Stage1.PiCCSPackets.evalA Stage1.Data.logicalWidth
      Stage1.Data.publicFits)
  let ccs ← IO.asTask (prepareWitnessGroup fun _ =>
    Stage1.PiCCSPackets.ccs Stage1.Data.logicalWidth
      Stage1.Data.publicFits)
  let norm ← IO.asTask (prepareWitnessGroup fun _ =>
    Stage1.PiCCSPackets.norm Stage1.Data.logicalWidth
      Stage1.Data.publicFits)
  let finalIdentity ← IO.asTask (prepareWitnessGroup fun _ =>
    Stage1.PiCCSPackets.finalIdentity Stage1.Data.logicalWidth
      Stage1.Data.publicFits)
  pure { initialClaim, sumcheck, evalK, evalA, ccs, norm, finalIdentity }

def writePreparedWitnessGroups (handle : IO.FS.Handle)
    (tasks : PreparedWitnessTasks) (first : Bool) : IO Bool := do
  let first ← writePreparedWitnessGroup handle "initial_claim" first
    tasks.initialClaim
  let first ← writePreparedWitnessGroup handle "sumcheck" first tasks.sumcheck
  let first ← writePreparedWitnessGroup handle "eval_k" first tasks.evalK
  let first ← writePreparedWitnessGroup handle "eval_a" first tasks.evalA
  let first ← writePreparedWitnessGroup handle "ccs" first tasks.ccs
  let first ← writePreparedWitnessGroup handle "norm" first tasks.norm
  writePreparedWitnessGroup handle "final_identity" first
    tasks.finalIdentity

/-- Stream the PiCCS prefix of `WitnessProgram.batches` in exact order. The
PiRLC sampler suffix is represented by the outer witness-plan field. -/
def writeWitnessBatches (handle : IO.FS.Handle)
    (tasks : PreparedWitnessTasks) : IO Unit := do
  writeByte handle 91
  let _first ← writePreparedWitnessGroups handle tasks true
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

def permutationActionShapeNodeCount :
    Stage1.PermutationPlan.ActionShape → Nat
  | .absorb input => (input.map exprNodeCount).sum
  | .squeezeK => 0

def permutationBlockNodeCount : Stage1.PermutationPlan.Block → Nat
  | .actions block =>
      (block.initialState.map exprNodeCount).sum +
        (block.actionShapes.map permutationActionShapeNodeCount).sum
  | .direct block => (block.state.map exprNodeCount).sum

structure PreparedPermutationBlocks where
  blocks : List Stage1.PermutationPlan.Block
  blockCount : Nat
  nodeCount : Nat

abbrev PreparedPermutationTask :=
  Task (Except IO.Error PreparedPermutationBlocks)

def preparePermutationBlocks : IO PreparedPermutationBlocks := do
  let blocks := Stage1.PermutationPlan.canonicalBlocks ()
  pure {
    blocks := blocks
    blockCount := blocks.length
    nodeCount := (blocks.map permutationBlockNodeCount).sum }

def preparedPermutationBlocks (task : PreparedPermutationTask) :
    IO PreparedPermutationBlocks :=
  match task.get with
  | .ok blocks => pure blocks
  | .error error => throw error

partial def writeExpandedPermutationBlockItems (handle : IO.FS.Handle)
    (first : Bool) : List Stage1.PermutationPlan.Block → IO Bool
  | [] => pure first
  | block :: rest => do
      let first ← writeArrayItemsWith handle
        (fun invocation => writeValue handle
          (Package.PermutationInvocation.format.encode invocation))
        first block.expand
      writeExpandedPermutationBlockItems handle first rest

def writeExpandedPermutationInvocations (handle : IO.FS.Handle)
    (blocks : List Stage1.PermutationPlan.Block) : IO Unit := do
  writeByte handle 91
  let _first ← writeExpandedPermutationBlockItems handle true blocks
  writeByte handle 93

partial def writeExpandedCompactBlockItems (handle : IO.FS.Handle)
    (first : Bool) : List Stage1.PackagePlan.CompactInvocationBlock → IO Bool
  | [] => pure first
  | block :: rest => do
      let first ← writeArrayItemsWith handle
        (fun invocation => writeValue handle
          (Package.CompactRowInvocation.format.encode invocation))
        first block.expand
      writeExpandedCompactBlockItems handle first rest

def writeExpandedCompactInvocations (handle : IO.FS.Handle) : IO Unit := do
  writeByte handle 91
  let _first ← writeExpandedCompactBlockItems handle true
    Stage1.PackagePlan.canonicalCompactBlocks
  writeByte handle 93

partial def writeExpandedWitnessBlockItems (handle : IO.FS.Handle)
    (first : Bool) : List Stage1.WitnessPlan.Block → IO Bool
  | [] => pure first
  | block :: rest => do
      let first ← writeWitnessBatchItems handle first block.expand
      writeExpandedWitnessBlockItems handle first rest

def writeExpandedWitnessBatches (handle : IO.FS.Handle)
    (tasks : PreparedWitnessTasks) : IO Unit := do
  writeByte handle 91
  let first ← writePreparedWitnessGroups handle tasks true
  let _first ← writeExpandedWitnessBlockItems handle first
    (Stage1.WitnessPlan.canonicalBlocks
      Stage1.Data.logicalWidth Stage1.Data.publicFits)
  writeByte handle 93

structure PreparedRowBlock where
  witnessInstructions : List Package.WitnessInstruction
  assertionRows : List Package.SparseRow

abbrev PreparedRowTask := Task (Except IO.Error PreparedRowBlock)

structure PreparedRowTasks where
  statementBinding : PreparedRowTask
  piRlc : List PreparedRowTask
  piDec : PreparedRowTask

def prepareRowBlock
    (block : Stage1.OrdinaryRowPlan.Block) : IO PreparedRowBlock := do
  let classified := Stage1.Rows.classifyRowsTR
    (block.rows Stage1.Data.logicalWidth Stage1.Data.publicFits)
  pure {
    witnessInstructions := classified.1
    assertionRows := classified.2 }

def prepareRowBlocks : IO PreparedRowTasks := do
  let statementBinding ← IO.asTask
    (prepareRowBlock .statementBinding)
  let piRlc ← (Stage1.OrdinaryRowPlan.piRlcBlocks ()).mapM fun block =>
    IO.asTask (prepareRowBlock block)
  let piDec ← IO.asTask
    (prepareRowBlock (Stage1.OrdinaryRowPlan.piDecBlock ()))
  pure { statementBinding, piRlc, piDec }

def preparedRowBlock (task : PreparedRowTask) : IO PreparedRowBlock :=
  match task.get with
  | .ok block => pure block
  | .error error => throw error

partial def writePreparedWitnessItems (handle : IO.FS.Handle)
    (first : Bool) : List PreparedRowTask → IO Bool
  | [] => pure first
  | task :: rest => do
      let prepared ← preparedRowBlock task
      let first ← writeArrayItemsWith handle
        (fun instruction =>
          writeValue handle (Package.WitnessInstruction.format.encode
            instruction))
        first prepared.witnessInstructions
      writePreparedWitnessItems handle first rest

def writePreparedPacketWitnessItems (handle : IO.FS.Handle)
    (first : Bool) (task : PreparedWitnessTask) : IO Bool := do
  let prepared ← preparedWitnessGroup task
  writeArrayItemsWith handle
    (fun instruction =>
      writeValue handle (Package.WitnessInstruction.format.encode
        instruction))
    first prepared.witnessInstructions

def writePreparedPiCCSWitnessItems (handle : IO.FS.Handle)
    (first : Bool) (tasks : PreparedWitnessTasks) : IO Bool := do
  let first ← writePreparedPacketWitnessItems handle first tasks.initialClaim
  let first ← writePreparedPacketWitnessItems handle first tasks.sumcheck
  let first ← writePreparedPacketWitnessItems handle first tasks.evalK
  let first ← writePreparedPacketWitnessItems handle first tasks.evalA
  let first ← writePreparedPacketWitnessItems handle first tasks.ccs
  let first ← writePreparedPacketWitnessItems handle first tasks.norm
  writePreparedPacketWitnessItems handle first tasks.finalIdentity

def writePreparedWitnessInstructions (handle : IO.FS.Handle)
    (witnessTasks : PreparedWitnessTasks)
    (rowTasks : PreparedRowTasks) : IO Unit := do
  writeByte handle 91
  let first ← writePreparedWitnessItems handle true
    [rowTasks.statementBinding]
  let first ← writePreparedPiCCSWitnessItems handle first witnessTasks
  let first ← writePreparedWitnessItems handle first rowTasks.piRlc
  let _first ← writePreparedWitnessItems handle first [rowTasks.piDec]
  writeByte handle 93

partial def writePreparedAssertionItems (handle : IO.FS.Handle)
    (first : Bool) : List PreparedRowTask → IO Bool
  | [] => pure first
  | task :: rest => do
      let prepared ← preparedRowBlock task
      let first ← writeArrayItemsWith handle
        (fun row => writeValue handle (Package.SparseRow.format.encode row))
        first prepared.assertionRows
      writePreparedAssertionItems handle first rest

def writePreparedPacketAssertionItems (handle : IO.FS.Handle)
    (first : Bool) (task : PreparedWitnessTask) : IO Bool := do
  let prepared ← preparedWitnessGroup task
  writeArrayItemsWith handle
    (fun row => writeValue handle (Package.SparseRow.format.encode row))
    first prepared.assertionRows

def writePreparedPiCCSAssertionItems (handle : IO.FS.Handle)
    (first : Bool) (tasks : PreparedWitnessTasks) : IO Bool := do
  let first ← writePreparedPacketAssertionItems handle first tasks.initialClaim
  let first ← writePreparedPacketAssertionItems handle first tasks.sumcheck
  let first ← writePreparedPacketAssertionItems handle first tasks.evalK
  let first ← writePreparedPacketAssertionItems handle first tasks.evalA
  let first ← writePreparedPacketAssertionItems handle first tasks.ccs
  let first ← writePreparedPacketAssertionItems handle first tasks.norm
  writePreparedPacketAssertionItems handle first tasks.finalIdentity

def writePreparedAssertionRows (handle : IO.FS.Handle)
    (witnessTasks : PreparedWitnessTasks)
    (rowTasks : PreparedRowTasks) : IO Unit := do
  writeByte handle 91
  let first ← writeArrayItemsWith handle
    (fun row => writeValue handle (Package.SparseRow.format.encode row))
    true (Stage1.Data.liftPilotRows (PilotData.assertionRows ()))
  let first ← writePreparedAssertionItems handle first
    [rowTasks.statementBinding]
  let first ← writePreparedPiCCSAssertionItems handle first witnessTasks
  let first ← writePreparedAssertionItems handle first rowTasks.piRlc
  let _first ← writePreparedAssertionItems handle first [rowTasks.piDec]
  writeByte handle 93

def writePackagePrefix (handle : IO.FS.Handle) : IO Unit := do
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

def writePackageTail (handle : IO.FS.Handle)
    (witnessTasks : PreparedWitnessTasks)
    (rowTasks : PreparedRowTasks) : IO Unit := do
  comma handle
  progress "emitter_stage=ordinary_row_tasks"
  progress "emitter_stage=witness_instructions"
  writePreparedWitnessInstructions handle witnessTasks rowTasks
  comma handle
  progress "emitter_stage=assertion_rows"
  writePreparedAssertionRows handle witnessTasks rowTasks
  comma handle
  writeValue handle
    ((option Package.TerminalLayout.format).encode none)
  writeByte handle 93

/-! Stream the schema-7 static payload of the schema-8 plan. -/
def writeStaticPackage (handle : IO.FS.Handle)
    (witnessTasks : PreparedWitnessTasks)
    (rowTasks : PreparedRowTasks) : IO Unit := do
  writePackagePrefix handle
  writeList handle Package.PermutationInvocation.format []
  comma handle
  progress "emitter_stage=compact_templates"
  writeListWith handle (writeCompactRowTemplate handle)
    (Stage1.Data.compactRowTemplates ())
  comma handle
  writeList handle Package.CompactRowInvocation.format []
  comma handle
  progress "emitter_stage=witness_batches"
  writeWitnessBatches handle witnessTasks
  writePackageTail handle witnessTasks rowTasks

/-! Stream the exact semantic package selected by `PackagePlan.canonical_expand`.
This reference is for independent conformance checks, not production loading. -/
def writeExpandedPackage (handle : IO.FS.Handle) : IO Unit := do
  progress "emitter_stage=expanded_parallel_preparation"
  let permutationTask ← IO.asTask preparePermutationBlocks
  let witnessTasks ← prepareWitnessGroups
  let rowTasks ← prepareRowBlocks
  writePackagePrefix handle
  let permutationBlocks ← preparedPermutationBlocks permutationTask
  progress s!"emitter_stage=expanded_permutation_invocations_{permutationBlocks.blockCount}_nodes_{permutationBlocks.nodeCount}"
  writeExpandedPermutationInvocations handle permutationBlocks.blocks
  comma handle
  progress "emitter_stage=compact_templates"
  writeListWith handle (writeCompactRowTemplate handle)
    (Stage1.Data.compactRowTemplates ())
  comma handle
  progress "emitter_stage=expanded_compact_invocations"
  writeExpandedCompactInvocations handle
  comma handle
  progress "emitter_stage=expanded_witness_batches"
  writeExpandedWitnessBatches handle witnessTasks
  writePackageTail handle witnessTasks rowTasks

/-- Stream `PackagePlan.format.encode (PackagePlan.canonical ())` in canonical
field order. -/
def writeCanonicalPlan (handle : IO.FS.Handle) : IO Unit := do
  progress "emitter_stage=parallel_preparation"
  let permutationTask ← IO.asTask preparePermutationBlocks
  let witnessTasks ← prepareWitnessGroups
  let rowTasks ← prepareRowBlocks
  writeByte handle 91
  writeValue handle (.atom 8)
  comma handle
  writeStaticPackage handle witnessTasks rowTasks
  comma handle
  let permutationBlocks ← preparedPermutationBlocks permutationTask
  progress s!"emitter_stage=permutation_blocks_{permutationBlocks.blockCount}_nodes_{permutationBlocks.nodeCount}"
  writeListWith handle (writePermutationBlock handle)
    permutationBlocks.blocks
  comma handle
  progress "emitter_stage=compact_blocks"
  writeList handle Stage1.PackagePlan.CompactInvocationBlock.format
    Stage1.PackagePlan.canonicalCompactBlocks
  comma handle
  progress "emitter_stage=witness_blocks"
  writeList handle Stage1.WitnessPlan.Block.format
    (Stage1.WitnessPlan.canonicalBlocks
      Stage1.Data.logicalWidth Stage1.Data.publicFits)
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

def emitExpanded (path : System.FilePath) : IO Unit := do
  progress "emitter_stage=expanded_stream"
  if let some parent := path.parent then
    IO.FS.createDirAll parent
  let handle ← IO.FS.Handle.mk path .write
  writeExpandedPackage handle
  handle.write (ByteArray.empty.push 10)
  handle.flush
  IO.println s!"emitted_expanded={path}"

def run (arguments : List String) : IO UInt32 := do
  match arguments with
  | [path] =>
      emit ⟨path⟩
      pure 0
  | ["--expanded", path] =>
      emitExpanded ⟨path⟩
      pure 0
  | ["--", path] =>
      emit ⟨path⟩
      pure 0
  | ["--", "--expanded", path] =>
      emitExpanded ⟨path⟩
      pure 0
  | _ =>
      IO.eprintln "usage: lake exe emit -- [--expanded] <output-path>"
      pure 2

end NightstreamFPrime.Export.Main

def main (arguments : List String) : IO UInt32 :=
  NightstreamFPrime.Export.Main.run arguments
