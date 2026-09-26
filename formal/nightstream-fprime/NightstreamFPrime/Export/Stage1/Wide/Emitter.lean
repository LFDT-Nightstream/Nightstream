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
The written parts are the result of `AuthorityStream.prepare`, so the emitted
archive, matrix program and transport are the ones that `PackageAuthority`
and `Wide.PackageCompleteness` describe. -/

namespace NightstreamFPrime.Export.Stage1.Wide.Emitter

open NightstreamFPrime.Export.Codec NightstreamFPrime.Export.Package
open NightstreamFPrime.Export.Main

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

/-- Emit exactly the parts built by the proved pure constructor. -/
def prepare : IO AuthorityStream.Parts := do
  progress "wide_stage=prepare"
  let some compiled := Layout.PiRlcWideSampler.RangePlan.compile?
    | throw (IO.userError "wide range compiler rejected the sampler")
  match AuthorityStream.prepare compiled with
  | .ok parts => pure parts
  | .error message => throw (IO.userError message)

def writeSealed (handle : IO.FS.Handle) (parts : AuthorityStream.Parts) : IO Unit := do
  writeByte handle 91
  writeValue handle (.atom PerApplicationCanonicalPackage.sealedPackageSchema)
  comma handle
  writeCircuit handle parts.package
  comma handle
  writeValue handle (Layout.MatrixProgram.Program.format.encode parts.matrix)
  comma handle
  writeApplicationPackagePlan handle parts.application
  comma handle
  writeValue handle parts.transport.encode
  comma handle
  writeValue handle (Layout.MatrixProgram.IndexRange.format.encode
    (AuthorityStream.nextPreimageRange parts))
  comma handle
  writeValue handle (.atom PerApplicationCanonicalPackage.logicalPublicInputCount)
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
  let parts ← prepare
  let matrix := parts.matrix
  let width := RetainedLayout.logicalWidth Poseidon2HashChainV1Package.application
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
  writeSealed sealedHandle parts
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

/-- The selected command emits the same typed children as the proved builder. -/
def emitSelected (path : System.FilePath) (expanded : Bool) : IO Unit := do
  if let some parent := path.parent then IO.FS.createDirAll parent
  let parts ← prepare
  let handle ← IO.FS.Handle.mk path .write
  if expanded then writeCircuit handle parts.package else writeSealed handle parts
  writeByte handle 10
  handle.flush
  progress s!"emitted_selected_wide={path}"

def bindingFixtureIO : IO Codec.Value := do
  return SetupBinding.bindingFixture (← prepare)

end NightstreamFPrime.Export.Stage1.Wide.Emitter
