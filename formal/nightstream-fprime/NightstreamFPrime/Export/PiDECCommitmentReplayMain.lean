import NightstreamFPrime.Export.Codec
import NightstreamFPrime.Export.Stage1.PiDECCommitmentFold
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Setup
import NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.StoredSplit

/-!
Accumulate a selected range of PiDEC commitment contributions from a Lean
parent file. Keys and signed digits are computed here. Expected commitments
are not inputs. Missing parent blocks denote zero; all sixteen children and
twenty-two key rows are included in the returned partial sum.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.PiDECCommitmentReplay

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra
open NightstreamFPrime.Spec.Folding.Nifs.StoredAssignmentArithmetic (StoredAssignment)
open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Export.Stage1

private abbrev Products := Vector
  (Vector (StoredAssignment ringDegree) productionGlobalParams.k)
  Poseidon2HashChainV1Setup.verifierRows

private def zero : Products := Vector.replicate _
  (Vector.replicate _ PiDECCommitmentFold.zero)

private def add (left right : Products) : Products :=
  left.zipWith (fun a b => a.zipWith PiDECCommitmentFold.add b) right

private def checked {Alpha : Type} (value : Except String Alpha) : IO Alpha :=
  match value with
  | .ok result => pure result
  | .error error => throw (IO.userError error)

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
      if size : values.size = ringDegree then return (block, ⟨values, size⟩)
      else throw "expected 54 parent coefficients"
  | _ => throw "expected parent block and coefficient array"

private def computeBlock (block : Nat) (parent : StoredAssignment ringDegree) :
    IO Products := do
  let some children := StoredSplit.splitChecked parent
    | throw (IO.userError s!"parent exceeds the strict B bound at block {block}")
  if live : block < Poseidon2HashChainV1Setup.messageColumns then
    return Vector.ofFn fun row => PiDECCommitmentBlock.contributions
      Poseidon2HashChainV1Setup.productionSetup row ⟨block, live⟩ children
  else throw (IO.userError "block is outside the selected fixed key")

private def collect (initial : Products)
    (tasks : Array (Task (Except IO.Error Products))) : IO Products := do
  let mut result := initial
  for task in tasks do
    match ← IO.wait task with
    | .ok value => result := add result value
    | .error error => throw error
  return result

private def writeResult (outputPath : System.FilePath) (blocks start finish : Nat)
    (accumulated : Products) : IO Unit := do
  let value := Value.array [.atom 1, .atom blocks, .atom start, .atom finish,
    .array (List.ofFn fun row : Fin Poseidon2HashChainV1Setup.verifierRows =>
      .array (List.ofFn fun child : Fin productionGlobalParams.k =>
        .array (List.ofFn fun lane : Fin ringDegree =>
          .atom (((accumulated.get row).get child).get lane).val)))]
  IO.FS.writeFile outputPath (value.render ++ "\n")

private def decodeVector {Alpha : Type} (count : Nat)
    (decode : Lean.Json → Except String Alpha) (value : Lean.Json) :
    Except String (Vector Alpha count) := do
  let entries ← (← value.getArr?).mapM decode
  if size : entries.size = count then return ⟨entries, size⟩
  else throw s!"expected {count} entries"

private def decodeCoefficient (value : Lean.Json) : Except String F := do
  let word ← value.getNat?
  unless word < goldilocksModulus do throw "noncanonical commitment coefficient"
  return Radix.fieldOfNat word

private def decodeRange (text : String) : Except String (Nat × Nat × Products) := do
  let fields ← (← Lean.Json.parse text).getArr?
  match fields.toList with
  | [schema, blocks, start, finish, values] =>
      unless (← schema.getNat?) = 1 &&
          (← blocks.getNat?) = Poseidon2HashChainV1Setup.messageColumns do
        throw "expected a selected Lean commitment range"
      let products ← decodeVector Poseidon2HashChainV1Setup.verifierRows
        (decodeVector productionGlobalParams.k (decodeVector ringDegree decodeCoefficient)) values
      return (← start.getNat?, ← finish.getNat?, products)
  | _ => throw "expected a commitment range header and values"

/-- Combine only complete, contiguous Lean ranges. The final coefficient sums
use the same audited `PiDECCommitmentFold.sum` as the block replay. -/
private def merge (outputPath : System.FilePath) (paths : List String) : IO UInt32 := do
  unless !(← outputPath.pathExists) do throw (IO.userError "output already exists")
  let started ← IO.monoMsNow
  let blocks := Poseidon2HashChainV1Setup.messageColumns
  let mut cursor := 0
  let mut parts : Array Products := #[]
  for path in paths do
    let (start, finish, products) ← checked (decodeRange (← IO.FS.readFile path))
    unless start = cursor && start < finish && finish ≤ blocks do
      throw (IO.userError "commitment ranges have a gap, overlap or invalid endpoint")
    parts := parts.push products
    cursor := finish
  unless cursor = blocks do throw (IO.userError "commitment ranges do not cover the complete carrier")
  let accumulated : Products := Vector.ofFn fun row => Vector.ofFn fun child =>
    PiDECCommitmentFold.sum fun index : Fin parts.size =>
      ((parts[index]).get row).get child
  writeResult outputPath blocks 0 blocks accumulated
  let finished ← IO.monoMsNow
  IO.println s!"pidec_Lean_commitments=passed ranges={parts.size} blocks={blocks} rows={Poseidon2HashChainV1Setup.verifierRows} children={productionGlobalParams.k} coefficients={Poseidon2HashChainV1Setup.verifierRows * productionGlobalParams.k * ringDegree} read_sum_write_ms={finished - started}"
  return 0

private def replay (parentPath outputPath : System.FilePath) (start finish : Nat) :
    IO UInt32 := do
  unless !(← outputPath.pathExists) do throw (IO.userError "output already exists")
  let started ← IO.monoMsNow
  let input ← IO.FS.Handle.mk parentPath .read
  let headerLine ← input.getLine
  let header ← checked do
    (← (← Lean.Json.parse headerLine).getArr?).toList.mapM Lean.Json.getNat?
  let (blocks, parentStart, parentEnd) ← match header with
    | [1, blocks, first, last] => pure (blocks, first, last)
    | _ => throw (IO.userError "expected a Lean PiRLC range header")
  unless blocks = Poseidon2HashChainV1Setup.messageColumns &&
      parentStart ≤ start && start < finish && finish ≤ parentEnd && parentEnd ≤ blocks do
    throw (IO.userError "replay range is outside the selected parent range")
  let workers := max 1 (((← IO.getEnv "LEAN_NUM_THREADS").bind String.toNat?).getD 1)
  let mut next := parentStart
  let mut complete := false
  let mut computed := 0
  let mut pending := #[]
  let mut accumulated := zero
  while !complete do
    let line ← input.getLine
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
          accumulated ← collect accumulated pending
          pending := #[]
        computed := computed + 1
  unless (← input.getLine).isEmpty do throw (IO.userError "extra data after parent terminator")
  accumulated ← collect accumulated pending
  writeResult outputPath blocks start finish accumulated
  let finished ← IO.monoMsNow
  IO.println s!"pidec_Lean_commitment_range=passed start={start} end={finish} computed_blocks={computed} rows={Poseidon2HashChainV1Setup.verifierRows} children={productionGlobalParams.k} workers={workers} compute_read_write_ms={finished - started}"
  return 0

end NightstreamFPrime.Export.PiDECCommitmentReplay

def main (arguments : List String) : IO UInt32 := do
  let arguments := if arguments.head? == some "--" then arguments.tail else arguments
  match arguments with
  | "merge" :: outputPath :: paths =>
      NightstreamFPrime.Export.PiDECCommitmentReplay.merge outputPath paths
  | [parentPath, outputPath, start, finish] =>
      match start.toNat?, finish.toNat? with
      | some start, some finish =>
          NightstreamFPrime.Export.PiDECCommitmentReplay.replay parentPath outputPath start finish
      | _, _ => throw (IO.userError "range endpoints must be natural numbers")
  | _ =>
      IO.eprintln "usage: replayPiDECCommitment <Lean-parent-range> <new-output> <start-block> <end-block>"
      IO.eprintln "   or: replayPiDECCommitment merge <new-output> <Lean-range>..."
      return 2
