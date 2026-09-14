import NightstreamFPrime.Export.Codec
import NightstreamFPrime.Export.Stage1.PiCCSInputCheck
import NightstreamFPrime.Export.Stage1.PiDECEvaluationBatch
import NightstreamFPrime.Export.Stage1.PiDECPadWeightedProduct
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

private def decodeBlock (line : String) :
    Except String (Nat × StoredAssignment ringDegree) := do
  let fields ← (← Lean.Json.parse line).getArr?
  match fields.toList with
  | [block, coefficients] =>
      let block ← block.getNat?
      let words ← coefficients.getArr?
      let mut values : Array F := #[]
      for word in words do
        let value ← word.getNat?
        unless value < goldilocksModulus do throw "noncanonical parent coefficient"
        values := values.push (Radix.fieldOfNat value)
      if size : values.size = ringDegree then return (block, ⟨values, size⟩)
      else throw "expected 54 parent coefficients"
  | _ => throw "expected parent block and coefficient array"

private def computeBlock (point : CubePoint K Lifecycle.cubeVariables)
    (block : Nat) (parent : StoredAssignment ringDegree) : IO Products := do
  let some children := StoredSplit.splitChecked parent
    | throw (IO.userError s!"parent exceeds the strict B bound at block {block}")
  let weights := Vector.ofFn fun lane : Fin ringDegree =>
    PiDECEvaluationWeights.weight point (block * ringDegree + lane.val)
  return PiDECPadWeightedProduct.products weights children

private def collect (initial : Products)
    (tasks : Array (Task (Except IO.Error Products))) : IO Products := do
  let mut result := initial
  for task in tasks do
    match ← IO.wait task with
    | .ok value => result := PiDECEvaluationBatch.add result value
    | .error error => throw error
  return result

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

/-- Validate the complete source stream while computing only the requested
range. An empty sparse range still emits its complete extent and zero sum.
All partial addition and field arithmetic remain in Lean. -/
private def pad (ccsPath parentPath outputPath : System.FilePath)
    (start finish : Nat) : IO UInt32 := do
  unless !(← outputPath.pathExists) do throw (IO.userError "output already exists")
  let started ← IO.monoNanosNow
  let ccs ← checked (PiCCSInputCheck.parse (← IO.FS.readFile ccsPath))
  let phase ← IO.wait (Task.spawn fun _ => PiCCSInputCheck.execute ccs)
  unless phase.accepted do throw (IO.userError "C input rejected")
  let pointReady ← IO.monoNanosNow
  let input ← IO.FS.Handle.mk parentPath .read
  let headerLine ← input.getLine
  let header ← checked do
    (← (← Lean.Json.parse headerLine).getArr?).toList.mapM Lean.Json.getNat?
  let (blocks, parentStart, parentEnd) ← match header with
    | [1, blocks, first, last] => pure (blocks, first, last)
    | _ => throw (IO.userError "expected a Lean PiRLC range header")
  unless blocks = Poseidon2HashChainV1Setup.messageColumns &&
      parentStart ≤ start && start < finish && finish ≤ parentEnd && parentEnd ≤ blocks do
    throw (IO.userError "Pad range is outside the selected parent range")
  let workers := max 1 (((← IO.getEnv "LEAN_NUM_THREADS").bind String.toNat?).getD 1)
  let mut next := parentStart
  let mut complete := false
  let mut computed := 0
  let mut pending := #[]
  let mut accumulated := PiDECEvaluationBatch.zero productionGlobalParams.k
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
        pending := pending.push (← IO.asTask (computeBlock phase.point block values))
        if pending.size ≥ workers then
          accumulated ← collect accumulated pending
          pending := #[]
        computed := computed + 1
  unless (← input.getLine).isEmpty do throw (IO.userError "extra data after parent terminator")
  accumulated ← collect accumulated pending
  let computedAt ← IO.monoNanosNow
  writeResult outputPath blocks start finish phase.point accumulated
  let finished ← IO.monoNanosNow
  IO.println s!"pidec_Lean_pad_range=computed accepted_ccs=true blocks={blocks} start={start} end={finish} computed_blocks={computed} children={productionGlobalParams.k} extension_values={productionGlobalParams.k * ringDegree} field_words={productionGlobalParams.k * ringDegree * 2} workers={workers} ccs_nanos={pointReady - started} read_compute_add_nanos={computedAt - pointReady} encode_write_nanos={finished - computedAt} total_ms={(finished - started) / 1000000}"
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
