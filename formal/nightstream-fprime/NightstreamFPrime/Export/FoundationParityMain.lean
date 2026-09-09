import NightstreamFPrime.Export.Codec
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CoefficientKernel
import NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.Radix

/-! Executable foundation vectors from the active semantics. Each JSON line
contains one matrix row or one checked scalar split; no complete output is
retained in memory. These are test inputs, not circuit-package authority. -/

open NightstreamFPrime.Spec
open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CoefficientKernel
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.Radix

private def emitBar (path : System.FilePath) : IO Unit := do
  let handle ← IO.FS.Handle.mk path .write
  handle.putStrLn (Value.array [.atom goldilocksModulus, .atom ringDegree]).render
  for output in List.finRange ringDegree do
    let row := (List.finRange ringDegree).map fun input =>
      Value.atom (nativeBarEntry output input).val
    handle.putStrLn (Value.array row).render
  handle.flush

private def writeSplit (handle : IO.FS.Handle) (value : F) : IO Unit := do
  let digits := match splitScalarChecked value with
    | none => []
    | some split => (List.finRange productionGlobalParams.k).map fun index =>
        Value.atom (split index).val
  handle.putStrLn (Value.array [.atom value.val, .array digits]).render

private def emitSplit (path : System.FilePath) : IO Unit := do
  let handle ← IO.FS.Handle.mk path .write
  handle.putStrLn (Value.array [.atom goldilocksModulus,
    .atom productionGlobalParams.b, .atom productionGlobalParams.k,
    .atom combinedBound]).render
  -- The complete strict-B domain, with zero once and both signs thereafter.
  for magnitude in [0:combinedBound] do
    let value := fieldOfNat magnitude
    writeSplit handle value
    if magnitude != 0 then writeSplit handle (-value)
  -- First rejected magnitude, its neighbour, and both centered field extremes.
  for magnitude in [combinedBound, combinedBound + 1, goldilocksModulus / 2] do
    let value := fieldOfNat magnitude
    writeSplit handle value
    writeSplit handle (-value)
  handle.flush

def main (arguments : List String) : IO UInt32 := do
  let some path := (match arguments with
    | [path] | ["--", path] => some (System.FilePath.mk path)
    | _ => none)
    | IO.eprintln "expected one output directory"
      return 2
  IO.FS.createDirAll path
  emitBar (path / "bar.jsonl")
  emitSplit (path / "split.jsonl")
  IO.println s!"emitted_foundation_parity={path}"
  return 0
