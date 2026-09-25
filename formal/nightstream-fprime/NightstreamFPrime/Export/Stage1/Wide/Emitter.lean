import NightstreamFPrime.Export.Stage1.Wide.ApplicationPackage
import NightstreamFPrime.Export.Stage1.Wide.PhysicalMatrixSource
import NightstreamFPrime.Export.Stage1.Wide.AssignmentTransport
import NightstreamFPrime.Export.Stage1.Wide.AuthorityStream
import NightstreamFPrime.Export.Stage1.Wide.SetupBinding
import NightstreamFPrime.Export.Stage1.Wide.BaseStepFixture
import NightstreamFPrime.Export.ParityEmitter
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Package
import NightstreamFPrime.Export.Main

/-! Stream the wide physical package through the existing typed writers.
The ordinary source rows are shared with witness generation and the matrix
consumer. This entry point does not change the production identity pins. -/

namespace NightstreamFPrime.Export.Stage1.Wide.Emitter

open NightstreamFPrime.Export.Codec NightstreamFPrime.Export.Package
open NightstreamFPrime.Export.Main

def prepareCommon : IO CircuitPackage := do
  let groups ← prepareWitnessGroups
  let statement ← IO.asTask (prepareRowBlock .statementBinding)
  let piDec ← IO.asTask (prepareRowBlockDeferred OrdinaryRowPlan.piDecBlock)
  let running ← IO.asTask (prepareRowBlockDeferred OrdinaryRowPlan.runningTransitionBlock)
  let permutations ← IO.asTask (pure ((PermutationPlan.piCcsBlocks ()).flatMap PermutationPlan.Block.expand))
  let base := Data.circuitPackageOf [] [] [] []
  progress "wide_prepared=pilot"
  let mut batches := base.witnessBatches
  let mut instructions := base.witnessInstructions
  let mut assertions := base.assertionRows
  let bound ← preparedRowBlock statement
  instructions := instructions ++ bound.witnessInstructions
  assertions := assertions ++ bound.assertionRows
  for (label, task) in [("initial", groups.initialClaim), ("sumcheck", groups.sumcheck),
      ("eval_k", groups.evalK), ("eval_a", groups.evalA), ("ccs", groups.ccs),
      ("norm", groups.norm), ("final", groups.finalIdentity)] do
    let packet ← preparedWitnessGroup task
    batches := batches ++ packet.batches
    instructions := instructions ++ packet.witnessInstructions
    assertions := assertions ++ packet.assertionRows
    progress s!"wide_prepared={label}"
  for (label, task) in [("pi_dec", piDec), ("running", running)] do
    let packet ← preparedRowBlock task
    instructions := instructions ++ packet.witnessInstructions
    assertions := assertions ++ packet.assertionRows
    progress s!"wide_prepared={label}"
  let permutationInvocations ← match permutations.get with
    | .ok value => pure value
    | .error error => throw error
  progress s!"wide_prepared=permutations count={permutationInvocations.length}"
  let piDecBatches := WitnessProgram.piDecBatches Data.logicalWidth Data.publicFits
  progress s!"wide_prepared=pi_dec_hints count={piDecBatches.length}"
  return { base with
    permutationInvocations := permutationInvocations
    compactRowTemplates := PiRLCCombinationTemplates.templates
    compactRowInvocations := PiRLCCombinationInvocations.invocations
    witnessBatches := batches ++ piDecBatches ++
      WitnessProgram.directRunningTransitionBatches Data.logicalWidth Data.publicFits
    witnessInstructions := instructions
    assertionRows := assertions }

def writeCircuit (handle : IO.FS.Handle) (value : CircuitPackage) : IO Unit := do
  writeByte handle 91
  writeValue handle (.atom value.schemaVersion)
  comma handle
  writeValue handle (Profile.format.encode value.profile)
  comma handle
  writeValue handle (PoseidonSchedule.format.encode value.poseidon)
  comma handle
  writeValue handle (PhysicalLayout.format.encode value.layout)
  comma handle
  writeValue handle (CcsRelation.format.encode value.relation)
  comma handle
  writePermutationTemplate handle
  comma handle
  writeList handle HashChain.format value.hashChains
  comma handle
  writeListWith handle (TypedWriter.writePermutationInvocation handle) value.permutationInvocations
  comma handle
  writeListWith handle (writeCompactRowTemplate handle) value.compactRowTemplates
  comma handle
  writeListWith handle (TypedWriter.writeCompactRowInvocation handle) value.compactRowInvocations
  comma handle
  writeListWith handle (TypedWriter.writeWitnessBatch handle) value.witnessBatches
  comma handle
  writeListWith handle (TypedWriter.writeWitnessInstruction handle) value.witnessInstructions
  comma handle
  writeListWith handle (TypedWriter.writeSparseRow handle) value.assertionRows
  comma handle
  writeValue handle ((option TerminalLayout.format).encode value.terminal)
  writeByte handle 93

def run (arguments : List String) : IO UInt32 := do
  let (path, context) ← match arguments with
    | [path] => pure (path, none)
    | [path, w0, w1, w2, w3] =>
      match ParityEmitter.parseVerifierKey w0 w1 w2 w3 with
      | .ok value => pure (path, some value)
      | .error message => throw (IO.userError message)
    | _ => throw (IO.userError "expected output path and optional four fixture-context words")
  let start ← IO.monoMsNow
  progress "wide_stage=common"
  let common ← prepareCommon
  progress s!"wide_stage=relocate common_instructions={common.witnessInstructions.length}"
  let base ← match PhysicalPackage.ofCommon common with
    | .ok value => pure value
    | .error message => throw (IO.userError message)
  progress s!"wide_stage=application base_rows={base.layout.rowCount}"
  let (package, application) ← match ApplicationPackage.ofBase Poseidon2HashChainV1Package.application base with
    | .ok value => pure value
    | .error message => throw (IO.userError message)
  let some compiled := Layout.PiRlcWideSampler.RangePlan.compile?
    | throw (IO.userError "wide range compiler rejected the sampler")
  let matrix ← match PhysicalMatrixSource.program Poseidon2HashChainV1Package.application compiled with
    | .ok value => pure value
    | .error message => throw (IO.userError message)
  let width := RetainedLayout.logicalWidth Poseidon2HashChainV1Package.application
  let transport ← match AssignmentTransport.plan Poseidon2HashChainV1Package.application
      package.layout.totalColumnCount with
    | .ok value => pure value
    | .error message => throw (IO.userError message)
  unless transport.coordinateCount = width do
    throw (IO.userError s!"transport width {transport.coordinateCount} differs from matrix width {width}")
  let parts := AuthorityStream.ofChildren package matrix application transport
  let package := parts.package
  progress s!"wide_physical_rows={package.layout.rowCount} columns={package.layout.totalColumnCount}"
  let handle ← IO.FS.Handle.mk ⟨path⟩ .write
  writeCircuit handle package
  writeByte handle 10
  handle.flush
  let matrixHandle ← IO.FS.Handle.mk ⟨path ++ ".matrix.json"⟩ .write
  writeValue matrixHandle (Layout.MatrixProgram.Program.format.encode parts.matrix)
  writeByte matrixHandle 10
  matrixHandle.flush
  let sealedHandle ← IO.FS.Handle.mk ⟨path ++ ".sealed.json"⟩ .write
  writeByte sealedHandle 91
  writeValue sealedHandle (.atom PerApplicationCanonicalPackage.sealedPackageSchema)
  comma sealedHandle
  writeCircuit sealedHandle package
  comma sealedHandle
  writeValue sealedHandle (Layout.MatrixProgram.Program.format.encode parts.matrix)
  comma sealedHandle
  writeApplicationPackagePlan sealedHandle parts.application
  comma sealedHandle
  writeValue sealedHandle parts.transport.encode
  comma sealedHandle
  writeValue sealedHandle (Layout.MatrixProgram.IndexRange.format.encode
    (AuthorityStream.nextPreimageRange parts))
  comma sealedHandle
  writeValue sealedHandle (.atom PerApplicationCanonicalPackage.logicalPublicInputCount)
  writeByte sealedHandle 93
  writeByte sealedHandle 10
  sealedHandle.flush
  progress "wide_stage=binding"
  let binding := SetupBinding.bindingValues parts
  ParityEmitter.emit "wide_binding_fixture" (SetupBinding.bindingFixtureValue binding)
    ⟨path ++ ".binding.json"⟩
  let fixtureContext := context.getD (SetupBinding.contextDigest binding.2.context)
  ParityEmitter.emit "wide_base_fixture" (← BaseStepFixture.valueIO fixtureContext) ⟨path ++ ".base.json"⟩
  progress s!"wide_logical_rows={matrix.rowCount} logical_coordinates={width}"
  let stop ← IO.monoMsNow
  progress s!"wide_physical_package={path} elapsed_ms={stop - start}"
  return 0

end NightstreamFPrime.Export.Stage1.Wide.Emitter

def main (arguments : List String) : IO UInt32 :=
  NightstreamFPrime.Export.Stage1.Wide.Emitter.run arguments
