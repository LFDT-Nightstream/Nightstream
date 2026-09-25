import NightstreamFPrime.Export.Stage1.Wide.PiRLCInputCheck
import NightstreamFPrime.Export.SignedUnitSourceInput

/-!
Replay PiRLC from original signed-unit source blocks and an accepted C input.
The expected Rust parent is not an argument. A source file stores nonzero
blocks in increasing order; omitted blocks contain zero in every source.
Output ranges retain their complete extent, including zero blocks and tails.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.PiRLCWitnessReplay

open NightstreamFPrime.Spec
open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Export.Stage1
open PiRLCNonzero (SourceCount)
open PiRLCPartialTrace

private def checked {Alpha : Type} (value : Except String Alpha) : IO Alpha :=
  match value with
  | .ok value => pure value
  | .error error => throw (IO.userError error)

private def naturals (line : String) : Except String (List Nat) := do
  let values ← (← Lean.Json.parse line).getArr?
  values.toList.mapM Lean.Json.getNat?

private def writeValue (output : IO.FS.Handle) (value : Value) : IO Unit :=
  output.putStr (value.render ++ "\n")

private def computeBlock (challenges : Fin SourceCount → RingF)
    (tables : FixedArray (FixedArray MaterializedRingF ringDegree) SourceCount)
    (block : Nat) (values : Fin SourceCount → MaterializedRingF) :
    IO (Nat × MaterializedRingF × Nat) := do
  let before ← IO.monoNanosNow
  let some result := (PiRLCWitnessBlock.preparedWitnessBlockPartials
      challenges tables values).getLast?
    | throw (IO.userError "missing PiRLC block result")
  let after ← IO.monoNanosNow
  return (block, result, after - before)

private def writeCompleted (output : IO.FS.Handle)
    (tasks : Array (PiRLCParity.PreparedTask (Nat × MaterializedRingF × Nat))) : IO Nat := do
  let mut nanos := 0
  for task in tasks do
    let (block, result, elapsed) ← PiRLCParity.prepared task
    writeValue output (.array [.atom block,
      .array (result.toList.map fun value => .atom value.val)])
    nanos := nanos + elapsed
  return nanos

private def replay (ccsPath sourcePath outputPath : System.FilePath)
    (start finish : Nat) : IO UInt32 := do
  unless !(← outputPath.pathExists) do throw (IO.userError "output already exists")
  let ccs ← checked (PiCCSInputCheck.parse (← IO.FS.readFile ccsPath))
  let some batch := Wide.PiRLCInputCheck.sampled ccs
    | throw (IO.userError "C rejected before PiRLC replay")
  let tables := PiRLCWitnessBlock.prepareWitnessActions batch.challenges
  let workers := max 1 (((← IO.getEnv "LEAN_NUM_THREADS").bind String.toNat?).getD 1)
  let source ← IO.FS.Handle.mk sourcePath .read
  let header ← checked (naturals (← source.getLine))
  let blocks ← match header with
    | [1, 54, 17, blocks] => pure blocks
    | _ => throw (IO.userError "expected the selected source-capture header")
  unless start < finish && finish ≤ blocks do
    throw (IO.userError "invalid consecutive replay range")
  let output ← IO.FS.Handle.mk outputPath .write
  writeValue output (.array [.atom 1, .atom blocks, .atom start, .atom finish])
  let mut next := 0
  let mut complete := false
  let mut computed := 0
  let mut kernelNanos := 0
  let mut pending := #[]
  while !complete do
    let line ← source.getLine
    if line.isEmpty then throw (IO.userError "missing source terminator")
    if line.trimAscii.toString == "[]" then
      complete := true
    else
      let (block, masks) ← checked (SignedUnitSourceInput.decodeBlock line)
      let values := SignedUnitSourceInput.sourceBlock masks
      unless next ≤ block && block < blocks do
        throw (IO.userError "duplicate or out-of-range source block")
      next := block + 1
      if start ≤ block && block < finish then
        pending := pending.push (← IO.asTask (computeBlock batch.challenges tables block values))
        if pending.size ≥ workers then
          kernelNanos := kernelNanos + (← writeCompleted output pending)
          pending := #[]
        computed := computed + 1
  unless (← source.getLine).isEmpty do
    throw (IO.userError "extra data after source terminator")
  kernelNanos := kernelNanos + (← writeCompleted output pending)
  writeValue output (.array [])
  output.flush
  IO.println s!"pirlc_Lean_witness_range=passed start={start} end={finish} computed_blocks={computed} coefficients={(finish - start) * ringDegree} workers={workers} kernel_nanos={kernelNanos}"
  return 0

end NightstreamFPrime.Export.PiRLCWitnessReplay

def main (arguments : List String) : IO UInt32 := do
  let arguments := if arguments.head? == some "--" then arguments.tail else arguments
  match arguments with
  | [ccsPath, sourcePath, outputPath, start, finish] =>
      match start.toNat?, finish.toNat? with
      | some start, some finish =>
          NightstreamFPrime.Export.PiRLCWitnessReplay.replay
            ccsPath sourcePath outputPath start finish
      | _, _ => throw (IO.userError "range endpoints must be natural numbers")
  | _ =>
      IO.eprintln "usage: replayPiRLCWitness <C-input> <source-capture> <new-output> <start-block> <end-block>"
      return 2
