import NightstreamFPrime.Export.Codec
import NightstreamFPrime.Circuit.Basic
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CoefficientKernel
import NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.Radix

/-! Executable foundation vectors from the active semantics. Each JSON line
contains one arithmetic case, matrix row or checked scalar split; no complete output is
retained in memory. These are test inputs, not circuit-package authority. -/

open NightstreamFPrime.Spec
open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CoefficientKernel
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.Radix

private def extensionValue (value : K) : Value :=
  .array [.atom value.c0.val, .atom value.c1.val]

private def emitArithmetic (directory : System.FilePath) : IO Unit := do
  -- Zero/one, the extension nonresidue, the production B boundary, 32-bit
  -- carries, centered signs, modulus reduction and the largest u64 input.
  let inputs := [0, 1, 2, 7, combinedBound - 1, combinedBound, combinedBound + 1,
    2 ^ 32 - 1, 2 ^ 32, 2 ^ 32 + 1, goldilocksModulus / 2,
    goldilocksModulus - 2, goldilocksModulus - 1, goldilocksModulus,
    goldilocksModulus + 1, 2 ^ 64 - 1]
  let base ← IO.FS.Handle.mk (directory / "field.jsonl") .write
  base.putStrLn (Value.array [.atom goldilocksModulus, .atom inputs.length]).render
  for left in inputs do
    let a := fieldOfNat left
    let inverse := NightstreamFPrime.Circuit.Hint.inverse a
    for right in inputs do
      let b := fieldOfNat right
      base.putStrLn (Value.array ([left, right, a.val, b.val,
        (a + b).val, (a - b).val, (a * b).val, (-a).val, inverse.val].map Value.atom)).render
  base.flush
  -- Both basis directions and both signs exercise coefficient order and
  -- the X² = 7 term. Test all pairs of these boundary extension values.
  let values : List K := ((inputs.map fieldOfNat).flatMap fun value =>
    [⟨value, 0⟩, ⟨0, value⟩, ⟨value, value⟩, ⟨value, -value⟩]).eraseDups
  let extension ← IO.FS.Handle.mk (directory / "extension.jsonl") .write
  extension.putStrLn (Value.array [.atom goldilocksModulus, .atom 7, .atom values.length]).render
  for a in values do
    let conjugate : K := ⟨a.c0, -a.c1⟩
    let normInverse := NightstreamFPrime.Circuit.Hint.inverse (a.c0 * a.c0 - 7 * a.c1 * a.c1)
    let inverse : K := ⟨conjugate.c0 * normInverse, conjugate.c1 * normInverse⟩
    if a ≠ K.zero ∧ K.mul a inverse ≠ K.one then
      throw (IO.userError "extension inverse does not satisfy the active multiplication relation")
    for b in values do
      extension.putStrLn (Value.array ([a, b, K.add a b, K.sub a b, K.mul a b,
        K.sub K.zero a, conjugate, inverse].map extensionValue)).render
  extension.flush

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
  emitArithmetic path
  emitBar (path / "bar.jsonl")
  emitSplit (path / "split.jsonl")
  IO.println s!"emitted_foundation_parity={path}"
  return 0
