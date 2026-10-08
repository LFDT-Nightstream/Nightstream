import NightstreamFPrime.Export.Stage1.Data
import NightstreamFPrime.Export.Stage1.OrdinaryRowPlan
import NightstreamFPrime.Export.Stage1.PackagePlan
import NightstreamFPrime.Export.Stage1.PerApplicationCanonicalPackage
import NightstreamFPrime.Export.Stage1.PerApplicationAssignmentTransport
import NightstreamFPrime.Export.Stage1.PerApplicationCachedBlocks
import NightstreamFPrime.Export.Stage1.PerApplicationCachedShift
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Package
import NightstreamFPrime.Export.Stage1.NebulaMemoryV1Package
import NightstreamFPrime.Export.Stage1.PiCCSPackets
import NightstreamFPrime.Export.TypedWriter
import NightstreamFPrime.Export.MatrixProgram.Program

/-! Executable entry point for the canonical Stage 1 circuit-package emitter. -/

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

def writeWitnessBatch (handle : IO.FS.Handle)
    (batch : NightstreamFPrime.Circuit.WitnessBatch) : IO Unit :=
  TypedWriter.writeWitnessBatch handle batch

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

structure PreparedRowBlock where
  witnessInstructions : List Package.WitnessInstruction
  assertionRows : List Package.SparseRow

abbrev PreparedRowTask := Task (Except IO.Error PreparedRowBlock)
abbrev PreparedRowSourceTask :=
  Task (Except IO.Error (List PreparedRowBlock))

structure PreparedRowTasks where
  statementBinding : PreparedRowTask
  piRlcSources : List PreparedRowSourceTask
  piDec : PreparedRowTask
  runningTransition : PreparedRowTask

def prepareRowBlock
    (block : Stage1.OrdinaryRowPlan.Block) : IO PreparedRowBlock := do
  let classified := Stage1.Rows.classifyRowsTR
    (block.rows Stage1.Data.logicalWidth Stage1.Data.publicFits)
  pure {
    witnessInstructions := classified.1
    assertionRows := classified.2 }

def prepareRowBlockDeferred
    (build : Unit → Stage1.OrdinaryRowPlan.Block) : IO PreparedRowBlock := do
  prepareRowBlock (build ())

def preparePiRlcSource (source : Nat) : IO (List PreparedRowBlock) :=
  (Stage1.OrdinaryRowPlan.piRlcSourceBlocks source).mapM prepareRowBlock

def prepareRowBlocks : IO PreparedRowTasks := do
  let statementBinding ← IO.asTask
    (prepareRowBlock .statementBinding)
  let piRlcSources ←
      (List.range Stage1.PiRLCSamplerInvocations.sourceCount).mapM
        fun source =>
          IO.asTask (preparePiRlcSource source)
            (prio := Task.Priority.dedicated)
  let piDec ← IO.asTask
    (prepareRowBlockDeferred Stage1.OrdinaryRowPlan.piDecBlock)
  let runningTransition ← IO.asTask
    (prepareRowBlockDeferred
      Stage1.OrdinaryRowPlan.runningTransitionBlock)
  pure { statementBinding, piRlcSources, piDec, runningTransition }

def preparedRowBlock (task : PreparedRowTask) : IO PreparedRowBlock :=
  match task.get with
  | .ok block => pure block
  | .error error => throw error

def preparedRowSource (task : PreparedRowSourceTask) :
    IO (List PreparedRowBlock) :=
  match task.get with
  | .ok blocks => pure blocks
  | .error error => throw error

/-! ## Verifier-selected per-application sealed package -/

def writeShiftedPermutationInvocation
    (shift : Stage1.PerApplicationCachedShift.Context)
    (handle : IO.FS.Handle) (invocation : Package.PermutationInvocation) :
    IO Unit :=
  TypedWriter.writePermutationInvocation handle
    (Stage1.PerApplicationCachedShift.shiftPermutationInvocation shift invocation)

partial def writeShiftedPermutationBlockItems
    (shift : Stage1.PerApplicationCachedShift.Context)
    (handle : IO.FS.Handle) (first : Bool) :
    List Stage1.PermutationPlan.Block → IO Bool
  | [] => pure first
  | block :: rest => do
      let first ← writeArrayItemsWith handle
        (writeShiftedPermutationInvocation shift handle) first block.expand
      writeShiftedPermutationBlockItems shift handle first rest

def writeShiftedPermutationInvocations
    (shift : Stage1.PerApplicationCachedShift.Context)
    (handle : IO.FS.Handle) (blocks : List Stage1.PermutationPlan.Block) :
    IO Unit := do
  writeByte handle 91
  let _first ← writeShiftedPermutationBlockItems shift handle true blocks
  writeByte handle 93

def writeShiftedCompactInvocation
    (shift : Stage1.PerApplicationCachedShift.Context)
    (handle : IO.FS.Handle) (invocation : Package.CompactRowInvocation) :
    IO Unit :=
  TypedWriter.writeCompactRowInvocation handle
    (Stage1.PerApplicationCachedShift.shiftCompactRowInvocation shift invocation)

partial def writeShiftedCompactBlockItems
    (shift : Stage1.PerApplicationCachedShift.Context)
    (handle : IO.FS.Handle) (first : Bool) :
    List Stage1.PackagePlan.CompactInvocationBlock → IO Bool
  | [] => pure first
  | block :: rest => do
      let first ← writeArrayItemsWith handle
        (writeShiftedCompactInvocation shift handle) first block.expand
      writeShiftedCompactBlockItems shift handle first rest

def writeShiftedCompactInvocations
    (shift : Stage1.PerApplicationCachedShift.Context)
    (handle : IO.FS.Handle) : IO Unit := do
  writeByte handle 91
  let _first ← writeShiftedCompactBlockItems shift handle true
    Stage1.PackagePlan.canonicalCompactBlocks
  writeByte handle 93

def writeShiftedWitnessBatchItems
    (shift : Stage1.PerApplicationCachedShift.Context)
    (handle : IO.FS.Handle) (first : Bool) (batches : List WitnessBatch) :
    IO Bool :=
  writeArrayItemsWith handle (fun batch =>
    TypedWriter.writeWitnessBatch handle
      (Stage1.PerApplicationCachedShift.shiftBatch shift batch)) first batches

def writeShiftedPreparedWitnessGroup
    (shift : Stage1.PerApplicationCachedShift.Context)
    (handle : IO.FS.Handle) (first : Bool) (task : PreparedWitnessTask) :
    IO Bool := do
  let prepared ← preparedWitnessGroup task
  writeShiftedWitnessBatchItems shift handle first prepared.batches

def writeShiftedPreparedWitnessGroups
    (shift : Stage1.PerApplicationCachedShift.Context)
    (handle : IO.FS.Handle) (first : Bool) (tasks : PreparedWitnessTasks) :
    IO Bool := do
  let first ← writeShiftedPreparedWitnessGroup shift handle first
    tasks.initialClaim
  let first ← writeShiftedPreparedWitnessGroup shift handle first
    tasks.sumcheck
  let first ← writeShiftedPreparedWitnessGroup shift handle first tasks.evalK
  let first ← writeShiftedPreparedWitnessGroup shift handle first tasks.evalA
  let first ← writeShiftedPreparedWitnessGroup shift handle first tasks.ccs
  let first ← writeShiftedPreparedWitnessGroup shift handle first tasks.norm
  writeShiftedPreparedWitnessGroup shift handle first tasks.finalIdentity

partial def writeShiftedWitnessBlockItems
    (shift : Stage1.PerApplicationCachedShift.Context)
    (handle : IO.FS.Handle) (first : Bool) :
    List Stage1.WitnessPlan.Block → IO Bool
  | [] => pure first
  | block :: rest => do
      let first ← writeShiftedWitnessBatchItems shift handle first block.expand
      writeShiftedWitnessBlockItems shift handle first rest

def writePerApplicationWitnessBatches
    (shift : Stage1.PerApplicationCachedShift.Context)
    (handle : IO.FS.Handle) (tasks : PreparedWitnessTasks)
    (application : Stage1.ApplicationPackage.Plan) : IO Unit := do
  writeByte handle 91
  let first ← writeShiftedWitnessBatchItems shift handle true
    (Stage1.Data.liftPilotBatches (PilotData.priorWordBatches ()))
  let first ← writeShiftedPreparedWitnessGroups shift handle first tasks
  let first ← writeShiftedWitnessBlockItems shift handle first
    (Stage1.WitnessPlan.canonicalBlocks
      Stage1.Data.logicalWidth Stage1.Data.publicFits)
  let _first ← writeArrayItemsWith handle
    (TypedWriter.writeWitnessBatch handle) first application.witnessBatches
  writeByte handle 93

def writeShiftedWitnessInstruction
    (shift : Stage1.PerApplicationCachedShift.Context)
    (handle : IO.FS.Handle) (instruction : Package.WitnessInstruction) :
    IO Unit :=
  TypedWriter.writeWitnessInstruction handle
    (Stage1.PerApplicationCachedShift.shiftWitnessInstruction shift instruction)

partial def writeShiftedPreparedWitnessItems
    (shift : Stage1.PerApplicationCachedShift.Context)
    (handle : IO.FS.Handle) (first : Bool) : List PreparedRowTask → IO Bool
  | [] => pure first
  | task :: rest => do
      let prepared ← preparedRowBlock task
      let first ← writeArrayItemsWith handle
        (writeShiftedWitnessInstruction shift handle) first
        prepared.witnessInstructions
      writeShiftedPreparedWitnessItems shift handle first rest

partial def writeShiftedPreparedWitnessBlockItems
    (shift : Stage1.PerApplicationCachedShift.Context)
    (handle : IO.FS.Handle) (first : Bool) :
    List PreparedRowBlock → IO Bool
  | [] => pure first
  | prepared :: rest => do
      let first ← writeArrayItemsWith handle
        (writeShiftedWitnessInstruction shift handle) first
        prepared.witnessInstructions
      writeShiftedPreparedWitnessBlockItems shift handle first rest

partial def writeShiftedPreparedWitnessSourceItems
    (shift : Stage1.PerApplicationCachedShift.Context)
    (handle : IO.FS.Handle) (first : Bool) :
    List PreparedRowSourceTask → IO Bool
  | [] => pure first
  | task :: rest => do
      let prepared ← preparedRowSource task
      let first ← writeShiftedPreparedWitnessBlockItems shift handle first
        prepared
      writeShiftedPreparedWitnessSourceItems shift handle first rest

def writeShiftedPacketWitnessItems
    (shift : Stage1.PerApplicationCachedShift.Context)
    (handle : IO.FS.Handle) (first : Bool) (task : PreparedWitnessTask) :
    IO Bool := do
  let prepared ← preparedWitnessGroup task
  writeArrayItemsWith handle (writeShiftedWitnessInstruction shift handle)
    first prepared.witnessInstructions

def writeShiftedPiCCSWitnessItems
    (shift : Stage1.PerApplicationCachedShift.Context)
    (handle : IO.FS.Handle) (first : Bool) (tasks : PreparedWitnessTasks) :
    IO Bool := do
  let first ← writeShiftedPacketWitnessItems shift handle first
    tasks.initialClaim
  let first ← writeShiftedPacketWitnessItems shift handle first tasks.sumcheck
  let first ← writeShiftedPacketWitnessItems shift handle first tasks.evalK
  let first ← writeShiftedPacketWitnessItems shift handle first tasks.evalA
  let first ← writeShiftedPacketWitnessItems shift handle first tasks.ccs
  let first ← writeShiftedPacketWitnessItems shift handle first tasks.norm
  writeShiftedPacketWitnessItems shift handle first tasks.finalIdentity

def writePerApplicationWitnessInstructions
    (shift : Stage1.PerApplicationCachedShift.Context)
    (handle : IO.FS.Handle) (witnessTasks : PreparedWitnessTasks)
    (rowTasks : PreparedRowTasks)
    (application : Stage1.ApplicationPackage.Plan) : IO Unit := do
  writeByte handle 91
  let first ← writeArrayItemsWith handle
    (writeShiftedWitnessInstruction shift handle) true
    (Stage1.Data.liftPilotInstructions (PilotData.witnessInstructions ()))
  let first ← writeShiftedPreparedWitnessItems shift handle first
    [rowTasks.statementBinding]
  let first ← writeShiftedPiCCSWitnessItems shift handle first witnessTasks
  let first ← writeShiftedPreparedWitnessSourceItems shift handle first
    rowTasks.piRlcSources
  let first ← writeShiftedPreparedWitnessItems shift handle first
    [rowTasks.piDec, rowTasks.runningTransition]
  let _first ← writeArrayItemsWith handle
    (TypedWriter.writeWitnessInstruction handle) first
    application.witnessInstructions
  writeByte handle 93

def writeShiftedSparseRow
    (shift : Stage1.PerApplicationCachedShift.Context)
    (handle : IO.FS.Handle) (row : Package.SparseRow) : IO Unit :=
  TypedWriter.writeSparseRow handle
    (Stage1.PerApplicationCachedShift.shiftSparseRow shift row)

partial def writeShiftedPreparedAssertionItems
    (shift : Stage1.PerApplicationCachedShift.Context)
    (handle : IO.FS.Handle) (first : Bool) : List PreparedRowTask → IO Bool
  | [] => pure first
  | task :: rest => do
      let prepared ← preparedRowBlock task
      let first ← writeArrayItemsWith handle
        (writeShiftedSparseRow shift handle) first prepared.assertionRows
      writeShiftedPreparedAssertionItems shift handle first rest

partial def writeShiftedPreparedAssertionBlockItems
    (shift : Stage1.PerApplicationCachedShift.Context)
    (handle : IO.FS.Handle) (first : Bool) :
    List PreparedRowBlock → IO Bool
  | [] => pure first
  | prepared :: rest => do
      let first ← writeArrayItemsWith handle
        (writeShiftedSparseRow shift handle) first prepared.assertionRows
      writeShiftedPreparedAssertionBlockItems shift handle first rest

partial def writeShiftedPreparedAssertionSourceItems
    (shift : Stage1.PerApplicationCachedShift.Context)
    (handle : IO.FS.Handle) (first : Bool) :
    List PreparedRowSourceTask → IO Bool
  | [] => pure first
  | task :: rest => do
      let prepared ← preparedRowSource task
      let first ← writeShiftedPreparedAssertionBlockItems shift handle first
        prepared
      writeShiftedPreparedAssertionSourceItems shift handle first rest

def writeShiftedPacketAssertionItems
    (shift : Stage1.PerApplicationCachedShift.Context)
    (handle : IO.FS.Handle) (first : Bool) (task : PreparedWitnessTask) :
    IO Bool := do
  let prepared ← preparedWitnessGroup task
  writeArrayItemsWith handle (writeShiftedSparseRow shift handle) first
    prepared.assertionRows

def writeShiftedPiCCSAssertionItems
    (shift : Stage1.PerApplicationCachedShift.Context)
    (handle : IO.FS.Handle) (first : Bool) (tasks : PreparedWitnessTasks) :
    IO Bool := do
  let first ← writeShiftedPacketAssertionItems shift handle first
    tasks.initialClaim
  let first ← writeShiftedPacketAssertionItems shift handle first tasks.sumcheck
  let first ← writeShiftedPacketAssertionItems shift handle first tasks.evalK
  let first ← writeShiftedPacketAssertionItems shift handle first tasks.evalA
  let first ← writeShiftedPacketAssertionItems shift handle first tasks.ccs
  let first ← writeShiftedPacketAssertionItems shift handle first tasks.norm
  writeShiftedPacketAssertionItems shift handle first tasks.finalIdentity

def writePerApplicationAssertionRows
    (shift : Stage1.PerApplicationCachedShift.Context)
    (handle : IO.FS.Handle) (witnessTasks : PreparedWitnessTasks)
    (rowTasks : PreparedRowTasks)
    (application : Stage1.ApplicationPackage.Plan) : IO Unit := do
  writeByte handle 91
  let first ← writeArrayItemsWith handle (writeShiftedSparseRow shift handle)
    true (Stage1.Data.liftPilotRows (PilotData.assertionRows ()))
  let first ← writeShiftedPreparedAssertionItems shift handle first
    [rowTasks.statementBinding]
  let first ← writeShiftedPiCCSAssertionItems shift handle first witnessTasks
  let first ← writeShiftedPreparedAssertionSourceItems shift handle first
    rowTasks.piRlcSources
  let first ← writeShiftedPreparedAssertionItems shift handle first
    [rowTasks.piDec, rowTasks.runningTransition]
  let first ← writeArrayItemsWith handle (TypedWriter.writeSparseRow handle)
    first application.assertionRows
  let _first ← writeArrayItemsWith handle (TypedWriter.writeSparseRow handle)
    first (Stage1.NextPreimagePackage.assertionRows
      (Stage1.PerApplicationPackage.nextPreimageRowStart shift.program))
  writeByte handle 93

def writePerApplicationInnerPackage
    (shift : Stage1.PerApplicationCachedShift.Context)
    (handle : IO.FS.Handle) (permutationBlocks : PreparedPermutationBlocks)
    (witnessTasks : PreparedWitnessTasks) (rowTasks : PreparedRowTasks)
    (application : Stage1.ApplicationPackage.Plan) : IO Unit := do
  let relation :=
    Stage1.PerApplicationCanonicalPackage.directRecursiveRelation shift.program
  writeByte handle 91
  writeValue handle (.atom 8)
  comma handle
  writeValue handle (Package.Profile.format.encode PilotData.profile)
  comma handle
  writeValue handle
    (Package.PoseidonSchedule.format.encode PilotData.poseidonSchedule)
  comma handle
  writeValue handle (Package.PhysicalLayout.format.encode
    (Stage1.PerApplicationPackage.directFinalLayout shift.program))
  comma handle
  writeValue handle (Package.CcsRelation.format.encode relation)
  comma handle
  writePermutationTemplate handle
  comma handle
  writeList handle Package.HashChain.format
    ([Stage1.Data.priorChain, Stage1.Data.outputChain].map
      (Stage1.PerApplicationCachedShift.shiftHashChain shift))
  comma handle
  writeShiftedPermutationInvocations shift handle permutationBlocks.blocks
  comma handle
  writeListWith handle (writeCompactRowTemplate handle)
    (Stage1.Data.compactRowTemplates ())
  comma handle
  writeShiftedCompactInvocations shift handle
  comma handle
  writePerApplicationWitnessBatches shift handle witnessTasks application
  comma handle
  writePerApplicationWitnessInstructions shift handle witnessTasks rowTasks
    application
  comma handle
  writePerApplicationAssertionRows shift handle witnessTasks rowTasks
    application
  comma handle
  writeValue handle ((option Package.TerminalLayout.format).encode
    (some (Stage1.PerApplicationCanonicalPackage.directTerminalLayout
      shift.program)))
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

/-- Stream the assignment-transport codec fields. Field order and child codecs are exactly those
of `PerApplicationAssignmentTransport.Plan.format`. -/
def writePerApplicationAssignmentTransport
    (program : Lifecycle.Stage1.Application.Program)
    (handle : IO.FS.Handle) : IO Unit := do
  writeByte handle 91
  writeValue handle
    (.atom Stage1.PerApplicationAssignmentTransport.schema)
  comma handle
  writeValue handle
    (Stage1.PerApplicationAssignmentBlocks.format.encode
      (Stage1.PerApplicationAssignmentBlocks.canonical program))
  comma handle
  writeValue handle
    (Stage1.PerApplicationAssignmentTransport.Phi81QuotientRecipe.format.encode
      (Stage1.PerApplicationAssignmentTransport.phi81QuotientRecipe program))
  comma handle
  writeValue handle
    (Stage1.PerApplicationAssignmentPlan.BlockKind.format.encode
      .pilotOutputDigest)
  comma handle
  writeListWith handle (TypedWriter.writeExpr handle)
    (Stage1.PerApplicationAssignmentTransport.outputDigestExpressions program)
  writeByte handle 93

/-- Stream the exact `sealedPackageValue` field order without constructing its
artifact-sized codec tree. The selected application remains a Lean value; no
runtime field selects its rows or layout. -/
def writePerApplicationSealedPackage
    (program : Lifecycle.Stage1.Application.Program)
    (_fits : Stage1.PerApplicationFixedPoint.FitsTwoPow28 program)
    (handle : IO.FS.Handle) : IO Unit := do
  progress "emitter_stage=per_application_parallel_preparation"
  let permutationTask ← IO.asTask preparePermutationBlocks
  let witnessTasks ← prepareWitnessGroups
  let rowTasks ← prepareRowBlocks
  let shift := Stage1.PerApplicationCachedShift.Context.ofProgram program
  let application := Stage1.PerApplicationPackage.directApplicationPlan program
  let permutationBlocks ← preparedPermutationBlocks permutationTask
  writeByte handle 91
  writeValue handle
    (.atom Stage1.PerApplicationCanonicalPackage.sealedPackageSchema)
  comma handle
  writePerApplicationInnerPackage shift handle permutationBlocks
    witnessTasks rowTasks application
  comma handle
  writeValue handle (MatrixProgram.Program.format.encode
    (Stage1.PerApplicationMatrixProgram.matrixProgram program))
  comma handle
  writeApplicationPackagePlan handle application
  comma handle
  writePerApplicationAssignmentTransport program handle
  comma handle
  writeValue handle (MatrixProgram.IndexRange.format.encode
    (Stage1.PerApplicationCanonicalPackage.nextPreimageRange program))
  comma handle
  writeValue handle
    (.atom Stage1.PerApplicationCanonicalPackage.logicalPublicInputCount)
  writeByte handle 93

def emitPerApplication
    (program : Lifecycle.Stage1.Application.Program)
    (fits : Stage1.PerApplicationFixedPoint.FitsTwoPow28 program)
    (path : System.FilePath) : IO Unit := do
  progress "emitter_stage=per_application_stream"
  if let some parent := path.parent then
    IO.FS.createDirAll parent
  let handle ← IO.FS.Handle.mk path .write
  writePerApplicationSealedPackage program fits handle
  handle.write (ByteArray.empty.push 10)
  handle.flush
  IO.println s!"emitted_per_application={path}"

/-- Emit the final physical package without its sealed metadata envelope.
This is the Lean-owned row-expansion reference for exact Rust A/B/C
comparison. -/
def emitPerApplicationExpanded
    (program : Lifecycle.Stage1.Application.Program)
    (_fits : Stage1.PerApplicationFixedPoint.FitsTwoPow28 program)
    (path : System.FilePath) : IO Unit := do
  progress "emitter_stage=per_application_expanded_stream"
  if let some parent := path.parent then
    IO.FS.createDirAll parent
  let permutationTask ← IO.asTask preparePermutationBlocks
  let witnessTasks ← prepareWitnessGroups
  let rowTasks ← prepareRowBlocks
  let shift := Stage1.PerApplicationCachedShift.Context.ofProgram program
  let application := Stage1.PerApplicationPackage.directApplicationPlan program
  let permutationBlocks ← preparedPermutationBlocks permutationTask
  let handle ← IO.FS.Handle.mk path .write
  writePerApplicationInnerPackage shift handle permutationBlocks
    witnessTasks rowTasks application
  handle.write (ByteArray.empty.push 10)
  handle.flush
  IO.println s!"emitted_per_application_expanded={path}"

/-- Emit the selected package with its exact matrix and assignment plans. -/
def emit (path : System.FilePath) : IO Unit :=
  emitPerApplication Stage1.Poseidon2HashChainV1Package.application
    Stage1.Poseidon2HashChainV1Package.fits path

/-- Emit the same physical rows for independent matrix conformance checks. -/
def emitExpanded (path : System.FilePath) : IO Unit :=
  emitPerApplicationExpanded Stage1.Poseidon2HashChainV1Package.application
    Stage1.Poseidon2HashChainV1Package.fits path

def run (arguments : List String) : IO UInt32 := do
  match arguments with
  | [path] | ["--", path] =>
      emit ⟨path⟩
      pure 0
  | ["--expanded", path] | ["--", "--expanded", path] =>
      emitExpanded ⟨path⟩
      pure 0
  | ["--application", "nebula-memory-v1", path]
  | ["--", "--application", "nebula-memory-v1", path] =>
      emitPerApplication Stage1.NebulaMemoryV1Package.application
        Stage1.NebulaMemoryV1Package.fits ⟨path⟩
      pure 0
  | _ =>
      IO.eprintln "usage: lake exe emit -- [--expanded | --application nebula-memory-v1] <output-path>"
      pure 2

end NightstreamFPrime.Export.Main
