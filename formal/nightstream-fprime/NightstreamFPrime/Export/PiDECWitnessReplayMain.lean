import NightstreamFPrime.Export.Codec
import NightstreamFPrime.Export.Stage1.PiRLCParity
import NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.StoredSplit

/-!
PiDEC private witness replay from a complete Lean parent range. The stored
split computes all sixteen digits before lossless signed-mask encoding.
Rust children are comparison targets and are not inputs to this executable.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.PiDECWitnessReplay

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra
open NightstreamFPrime.Spec.Folding.Nifs.StoredAssignmentArithmetic (StoredAssignment)
open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Export.Stage1

private def checked {Alpha : Type} (value : Except String Alpha) : IO Alpha :=
  match value with
  | .ok result => pure result
  | .error error => throw (IO.userError error)

private def naturals (line : String) : Except String (List Nat) := do
  let fields ← (← Lean.Json.parse line).getArr?
  fields.toList.mapM Lean.Json.getNat?

private def decodeBlock (line : String) :
    Except String (Nat × StoredAssignment ringDegree) := do
  let fields ← (← Lean.Json.parse line).getArr?
  match fields.toList with
  | [block, values] =>
      let block ← block.getNat?
      let words ← values.getArr?
      let mut values : Array F := #[]
      for word in words do
        let value ← word.getNat?
        unless value < goldilocksModulus do throw "noncanonical parent coefficient"
        values := values.push (Radix.fieldOfNat value)
      if size : values.size = ringDegree then
        return (block, ⟨values, size⟩)
      else throw "expected 54 parent coefficients"
  | _ => throw "expected parent block and coefficient array"

private def childMasks (values : StoredAssignment ringDegree) : Except String (Nat × Nat) := do
  let mut positive := 0
  let mut negative := 0
  for lane in [:ringDegree] do
    let value := values.toArray[lane]!
    if value = 1 then positive := positive ||| (2 ^ lane)
    else if value = -1 then negative := negative ||| (2 ^ lane)
    else unless value = 0 do throw "computed digit is outside the signed-unit set"
  return (positive, negative)

private def computeBlock (block : Nat) (parent : StoredAssignment ringDegree) :
    IO (Option Value) := do
  let some children := StoredSplit.splitChecked parent
    | throw (IO.userError s!"parent norm exceeds the strict B bound at block {block}")
  let mut entries := []
  for child in [:productionGlobalParams.k] do
    let values := children.toArray[child]!
    let (positive, negative) ← checked (childMasks values)
    if positive ||| negative != 0 then
      entries := entries ++ [.array [.atom child, .atom positive, .atom negative]]
  if entries.isEmpty then return none
  else return some (.array [.atom block, .array entries])

private def writeValue (output : IO.FS.Handle) (value : Value) : IO Unit :=
  output.putStr (value.render ++ "\n")

private def writeCompleted (output : IO.FS.Handle)
    (tasks : Array (PiRLCParity.PreparedTask (Option Value))) : IO Unit := do
  for task in tasks do
    let result ← PiRLCParity.prepared task
    if let some value := result then writeValue output value

private def replay (parentPath outputPath : System.FilePath) (start finish : Nat) : IO UInt32 := do
  unless !(← outputPath.pathExists) do throw (IO.userError "output already exists")
  let parent ← IO.FS.Handle.mk parentPath .read
  let header ← checked (naturals (← parent.getLine))
  let (blocks, parentStart, parentEnd) ← match header with
    | [1, blocks, first, last] => pure (blocks, first, last)
    | _ => throw (IO.userError "expected a Lean PiRLC range header")
  unless parentStart ≤ start && start < finish && finish ≤ parentEnd && parentEnd ≤ blocks do
    throw (IO.userError "replay range is outside the complete parent range")
  let workers := max 1 (((← IO.getEnv "LEAN_NUM_THREADS").bind String.toNat?).getD 1)
  let output ← IO.FS.Handle.mk outputPath .write
  writeValue output (.array [.atom 1, .atom ringDegree, .atom productionGlobalParams.k,
    .atom blocks, .atom start, .atom finish])
  let mut next := parentStart
  let mut complete := false
  let mut computed := 0
  let mut pending := #[]
  while !complete do
    let line ← parent.getLine
    if line.isEmpty then throw (IO.userError "missing parent terminator")
    if line.trimAscii.toString == "[]" then complete := true
    else
      let (block, values) ← checked (decodeBlock line)
      unless next ≤ block && block < parentEnd do
        throw (IO.userError "duplicate or out-of-range parent block")
      next := block + 1
      if start ≤ block && block < finish then
        pending := pending.push (← IO.asTask (computeBlock block values))
        if pending.size ≥ workers then
          writeCompleted output pending
          pending := #[]
        computed := computed + 1
  unless (← parent.getLine).isEmpty do throw (IO.userError "extra data after parent terminator")
  writeCompleted output pending
  writeValue output (.array [])
  output.flush
  IO.println s!"pidec_Lean_witness_range=passed start={start} end={finish} computed_blocks={computed} child_coefficients={(finish - start) * ringDegree * productionGlobalParams.k} workers={workers}"
  return 0

end NightstreamFPrime.Export.PiDECWitnessReplay

def main (arguments : List String) : IO UInt32 := do
  let arguments := if arguments.head? == some "--" then arguments.tail else arguments
  match arguments with
  | [parentPath, outputPath, start, finish] =>
      match start.toNat?, finish.toNat? with
      | some start, some finish =>
          NightstreamFPrime.Export.PiDECWitnessReplay.replay parentPath outputPath start finish
      | _, _ => throw (IO.userError "range endpoints must be natural numbers")
  | _ =>
      IO.eprintln "usage: replayPiDECWitness <Lean-parent-range> <new-output> <start-block> <end-block>"
      return 2
