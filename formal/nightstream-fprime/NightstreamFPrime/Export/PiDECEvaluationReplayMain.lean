import NightstreamFPrime.Export.Codec
import NightstreamFPrime.Export.PiDECParentRange
import NightstreamFPrime.Export.Stage1.PiCCSInputCheck
import NightstreamFPrime.Export.Stage1.PiDECEvaluationBatch
import NightstreamFPrime.Export.Stage1.PiDECPadWeightedProduct
import NightstreamFPrime.Export.Stage1.PiCCSTensorWeights
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Setup
import NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.StoredSplit

/-!
Compute a complete Pad range from the original Lean R parent stream. The
accepted C execution supplies the common point. Lean splits each present
parent block, computes both weighted products and sums every child value.
Omitted blocks denote zero. Native or expected evaluations are not inputs.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.PiDECEvaluationReplay

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra
open NightstreamFPrime.Spec.Folding.Nifs.StoredAssignmentArithmetic (StoredAssignment)
open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Export.Stage1
open NightstreamFPrime.Export.Stage1.PiRLCPartialTrace (MaterializedRingK)

private abbrev Products := Vector MaterializedRingK productionGlobalParams.k

private def checked {Alpha : Type} (value : Except String Alpha) : IO Alpha :=
  match value with
  | .ok result => pure result
  | .error error => throw (IO.userError error)

private def computeBlock (point : CubePoint K Lifecycle.cubeVariables)
    (tables : Array K × Array K)
    (block : Nat) (parent : StoredAssignment ringDegree) : IO Products := do
  let some children := StoredSplit.splitChecked parent
    | throw (IO.userError s!"parent exceeds the strict B bound at block {block}")
  let weights := Vector.ofFn fun lane : Fin ringDegree =>
    PiCCSTensorWeights.lookup extensionOps point.coordinates tables
      (block * ringDegree + lane.val)
  return PiDECPadWeightedProduct.products weights children

private def writeResult (outputPath : System.FilePath) (blocks start finish : Nat)
    (point : CubePoint K Lifecycle.cubeVariables) (accumulated : Products) : IO Unit := do
  let encodeK := fun value : K => Value.array [.atom value.c0.val, .atom value.c1.val]
  let value := Value.array [.atom 1, .atom blocks, .atom start, .atom finish,
    .array (point.coordinates.map encodeK),
    .array (List.ofFn fun child : Fin productionGlobalParams.k =>
      .array (List.ofFn fun lane : Fin ringDegree =>
        encodeK ((accumulated.get child).toRing lane)))]
  IO.FS.writeFile outputPath (value.render ++ "\n")

private def decodeVector {Alpha : Type} (count : Nat)
    (decode : Lean.Json → Except String Alpha) (value : Lean.Json) :
    Except String (Vector Alpha count) := do
  let entries ← (← value.getArr?).mapM decode
  if size : entries.size = count then return ⟨entries, size⟩
  else throw s!"expected {count} entries"

private def decodeField (value : Lean.Json) : Except String F := do
  let word ← value.getNat?
  unless word < goldilocksModulus do throw "noncanonical Pad field coefficient"
  return Radix.fieldOfNat word

private def decodeK (value : Lean.Json) : Except String K := do
  let pair ← decodeVector 2 decodeField value
  return ⟨pair.get ⟨0, by decide⟩, pair.get ⟨1, by decide⟩⟩

private def decodeRing (value : Lean.Json) : Except String MaterializedRingK := do
  let values ← decodeVector ringDegree decodeK value
  return PiRLCPartialTrace.FixedArray.ofFn values.get

private def decodeRange (text : String) :
    Except String (Nat × Nat × CubePoint K Lifecycle.cubeVariables × Products) := do
  let fields ← (← Lean.Json.parse text).getArr?
  match fields.toList with
  | [schema, blocks, start, finish, coordinates, values] =>
      unless (← schema.getNat?) = 1 &&
          (← blocks.getNat?) = Poseidon2HashChainV1Setup.messageColumns do
        throw "expected a selected Lean Pad range"
      let point ← decodeVector Lifecycle.cubeVariables decodeK coordinates
      let products ← decodeVector productionGlobalParams.k decodeRing values
      return (← start.getNat?, ← finish.getNat?,
        { coordinates := point.toList, dimension := by simp }, products)
  | _ => throw "expected a Pad range header, point and values"

/-- Add only complete contiguous ranges at the same independently derived
point. The final sum is computed by the audited Lean batch accumulator. -/
private def mergePad (outputPath : System.FilePath) (paths : List String) : IO UInt32 := do
  unless !(← outputPath.pathExists) do throw (IO.userError "output already exists")
  let started ← IO.monoMsNow
  let blocks := Poseidon2HashChainV1Setup.messageColumns
  let mut cursor := 0
  let mut commonPoint : Option (CubePoint K Lifecycle.cubeVariables) := none
  let mut parts : Array Products := #[]
  for path in paths do
    let (start, finish, point, products) ← checked (decodeRange (← IO.FS.readFile path))
    unless start = cursor && start < finish && finish ≤ blocks do
      throw (IO.userError "Pad ranges have a gap, overlap or invalid endpoint")
    match commonPoint with
    | none => commonPoint := some point
    | some expected =>
        unless point.coordinates = expected.coordinates do
          throw (IO.userError "Pad ranges have different points")
    parts := parts.push products
    cursor := finish
  unless cursor = blocks do throw (IO.userError "Pad ranges do not cover the complete carrier")
  let some point := commonPoint | throw (IO.userError "missing Pad ranges")
  let accumulated := PiDECEvaluationBatch.sum parts.size fun index =>
    if live : index < parts.size then parts[index]
    else PiDECEvaluationBatch.zero productionGlobalParams.k
  writeResult outputPath blocks 0 blocks point accumulated
  IO.println s!"pidec_Lean_pad_complete=computed ranges={parts.size} blocks={blocks} children={productionGlobalParams.k} field_words={productionGlobalParams.k * ringDegree * 2} read_sum_write_ms={(← IO.monoMsNow) - started}"
  return 0

/-- Parent blocks per work item: large enough to amortize scheduling, and small
enough that faster cores take more items. -/
private def chunkBlocks : Nat := 128

/-- Validate the complete source stream while computing only the requested
range. An empty sparse range still emits its complete extent and zero sum.
Every chunk sums its contiguous blocks and the chunk sums are added in order;
field addition does not depend on the grouping. All partial addition and field
arithmetic remain in Lean. -/
private def pad (ccsPath parentPath outputPath : System.FilePath)
    (start finish : Nat) : IO UInt32 := do
  unless !(← outputPath.pathExists) do throw (IO.userError "output already exists")
  let started ← IO.monoNanosNow
  let ccs ← checked (PiCCSInputCheck.parse (← IO.FS.readFile ccsPath))
  let phase ← IO.wait (Task.spawn fun _ => PiCCSInputCheck.execute ccs)
  unless phase.accepted do throw (IO.userError "C input rejected")
  let tables := PiCCSTensorWeights.prepare extensionOps phase.point.coordinates
  let pointReady ← IO.monoNanosNow
  let (blocks, records) ← PiDECParentRange.select parentPath start finish
  let workers ← ParallelChunks.workers
  let tasks ← ParallelChunks.start workers ((records.size + chunkBlocks - 1) / chunkBlocks)
    fun chunk => do
      let mut sum := PiDECEvaluationBatch.zero productionGlobalParams.k
      for (block, values) in records[chunk * chunkBlocks:(chunk + 1) * chunkBlocks] do
        sum := PiDECEvaluationBatch.add sum (← computeBlock phase.point tables block values)
      return sum
  let mut accumulated := PiDECEvaluationBatch.zero productionGlobalParams.k
  for task in tasks do
    match ← IO.wait task with
    | .ok (sum, _, _) => accumulated := PiDECEvaluationBatch.add accumulated sum
    | .error error => throw error
  let computedAt ← IO.monoNanosNow
  writeResult outputPath blocks start finish phase.point accumulated
  let finished ← IO.monoNanosNow
  IO.println s!"pidec_Lean_pad_range=computed accepted_ccs=true blocks={blocks} start={start} end={finish} computed_blocks={records.size} children={productionGlobalParams.k} extension_values={productionGlobalParams.k * ringDegree} field_words={productionGlobalParams.k * ringDegree * 2} workers={workers} ccs_nanos={pointReady - started} read_compute_add_nanos={computedAt - pointReady} encode_write_nanos={finished - computedAt} total_ms={(finished - started) / 1000000}"
  return 0

end NightstreamFPrime.Export.PiDECEvaluationReplay

def main (arguments : List String) : IO UInt32 := do
  let arguments := if arguments.head? == some "--" then arguments.tail else arguments
  match arguments with
  | "merge-pad" :: outputPath :: paths =>
      NightstreamFPrime.Export.PiDECEvaluationReplay.mergePad outputPath paths
  | ["pad", ccsPath, parentPath, outputPath, start, finish] =>
      match start.toNat?, finish.toNat? with
      | some start, some finish =>
          NightstreamFPrime.Export.PiDECEvaluationReplay.pad
            ccsPath parentPath outputPath start finish
      | _, _ => throw (IO.userError "range endpoints must be natural numbers")
  | _ =>
      IO.eprintln "usage: replayPiDECEvaluation pad <C-input> <Lean-parent-range> <new-output> <start-block> <end-block>"
      return 2
