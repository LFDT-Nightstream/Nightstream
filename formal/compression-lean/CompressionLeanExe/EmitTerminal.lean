import CompressionLean.Terminal.Outputs

/-!
Owns the terminal artifact writer. It streams the canonical encoding of
`envelope`: the package field by field, then the output map, so a reader
decodes the package with the F′ package decoder.
-/

namespace CompressionLeanExe.EmitTerminal

open CompressionLean CompressionLean.Terminal
open NightstreamFPrime.Export
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Export.Codec

def progress (message : String) : IO Unit := do
  IO.println message
  (← IO.getStdout).flush

def field (handle : IO.FS.Handle) (value : Value) : IO Unit := do
  let _ ← value.writeCanonical handle
  writeByte handle 44

/-- The field order of `CircuitPackage.format`. -/
def writePackage (handle : IO.FS.Handle) (package : CircuitPackage) : IO Unit := do
  writeByte handle 91
  field handle (.atom package.schemaVersion)
  field handle (Profile.format.encode package.profile)
  field handle (PoseidonSchedule.format.encode package.poseidon)
  field handle (PhysicalLayout.format.encode package.layout)
  field handle (CcsRelation.format.encode package.relation)
  field handle (PermutationTemplate.format.encode package.permutation)
  progress "emitter_stage=hash_chains"
  writeListCanonical handle HashChain.format package.hashChains
  writeByte handle 44
  progress "emitter_stage=permutation_invocations"
  writeListCanonical handle PermutationInvocation.format
    package.permutationInvocations
  writeByte handle 44
  progress "emitter_stage=compact_templates"
  writeListCanonical handle CompactRowTemplate.format package.compactRowTemplates
  writeByte handle 44
  progress "emitter_stage=compact_invocations"
  writeListCanonical handle CompactRowInvocation.format
    package.compactRowInvocations
  writeByte handle 44
  progress "emitter_stage=witness_batches"
  writeListCanonical handle WitnessBatch.format package.witnessBatches
  writeByte handle 44
  progress "emitter_stage=witness_instructions"
  writeListCanonical handle WitnessInstruction.format package.witnessInstructions
  writeByte handle 44
  progress "emitter_stage=assertion_rows"
  writeListCanonical handle SparseRow.format package.assertionRows
  writeByte handle 44
  let _ ← ((option TerminalLayout.format).encode package.terminal).writeCanonical
    handle
  writeByte handle 93

/-- The field order of `envelope`. -/
def write (handle : IO.FS.Handle) : IO Unit := do
  writeByte handle 91
  field handle (.atom 1)
  writePackage handle (circuitPackage ())
  writeByte handle 44
  let _ ← outputs.value.writeCanonical handle
  writeByte handle 93
  writeByte handle 10

def main (arguments : List String) : IO UInt32 := do
  match arguments with
  | [path] | ["--", path] =>
      let handle ← IO.FS.Handle.mk path .write
      write handle
      handle.flush
      progress s!"emitted_terminal={path}"
      pure 0
  | _ =>
      IO.eprintln "usage: lake exe emitTerminal -- <output-path>"
      pure 2

end CompressionLeanExe.EmitTerminal

def main (arguments : List String) : IO UInt32 :=
  CompressionLeanExe.EmitTerminal.main arguments
