import NightstreamFPrime.Export.Codec
import NightstreamFPrime.Export.PiDECParentRange
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

private abbrev NativeProducts := Vector
  (Vector PiDECNativeProduct.Accumulator productionGlobalParams.k)
  Poseidon2HashChainV1Setup.verifierRows

private def zero : NativeProducts := Vector.replicate _
  (Vector.replicate _ PiDECNativeProduct.Accumulator.zero)

private def add (left right : NativeProducts) : NativeProducts :=
  left.zipWith (fun a b => a.zipWith PiDECNativeProduct.Accumulator.add b) right

private def finishProducts (values : NativeProducts) : Products :=
  values.map (fun row => row.map PiDECNativeProduct.Accumulator.finish)

private def checked {Alpha : Type} (value : Except String Alpha) : IO Alpha :=
  match value with
  | .ok result => pure result
  | .error error => throw (IO.userError error)

private def computeBlock (initial : NativeProducts) (block : Nat)
    (parent : StoredAssignment ringDegree) : IO NativeProducts := do
  let some children := StoredSplit.splitChecked parent
    | throw (IO.userError s!"parent exceeds the strict B bound at block {block}")
  if live : block < Poseidon2HashChainV1Setup.messageColumns then
    let preparedChildren := children.map PiDECNativeProduct.prepareDigit
    return Vector.ofFn fun row => PiDECCommitmentBlock.accumulatePreparedContributions
      Poseidon2HashChainV1Setup.productionSetup row ⟨block, live⟩ preparedChildren (initial.get row)
  else throw (IO.userError "block is outside the selected fixed key")

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

/-- Parent blocks per work item: large enough to amortize scheduling, and small
enough that faster cores take more items. -/
private def chunkBlocks : Nat := 128

/-- Every chunk sums its contiguous blocks; the chunk sums are added in order.
All sums are canonical field additions, so the grouping does not change them. -/
private def replay (parentPath outputPath : System.FilePath) (start finish : Nat) :
    IO UInt32 := do
  unless !(← outputPath.pathExists) do throw (IO.userError "output already exists")
  let started ← IO.monoMsNow
  let (blocks, records) ← PiDECParentRange.select parentPath start finish
  let workers ← ParallelChunks.workers
  let tasks ← ParallelChunks.start workers ((records.size + chunkBlocks - 1) / chunkBlocks)
    fun chunk => do
      let mut sum := zero
      for (block, values) in records[chunk * chunkBlocks:(chunk + 1) * chunkBlocks] do
        sum ← computeBlock sum block values
      return sum
  let mut accumulated := zero
  for task in tasks do
    match ← IO.wait task with
    | .ok (sum, _, _) => accumulated := add accumulated sum
    | .error error => throw error
  writeResult outputPath blocks start finish (finishProducts accumulated)
  let finished ← IO.monoMsNow
  IO.println s!"pidec_Lean_commitment_range=passed start={start} end={finish} computed_blocks={records.size} rows={Poseidon2HashChainV1Setup.verifierRows} children={productionGlobalParams.k} workers={workers} compute_read_write_ms={finished - started}"
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
