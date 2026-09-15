import NightstreamFPrime.Export.Codec
import NightstreamFPrime.Export.Stage1.PiCCSInputCheck
import NightstreamFPrime.Export.Stage1.PiDECMatrixRangeSum
import NightstreamFPrime.Export.Stage1.PerApplicationCanonicalPackage
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Package
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Setup

/-!
Merge selected matrix ranges and the complete Pad result at the point derived
by accepted C execution. Validate canonical fields, dimensions, exact counts
and contiguous coverage before writing the complete evaluation family.
All addition runs in Lean.
The range producers retain ownership of their computed partial values.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.PiDECMatrixMerge

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.ProductionRelation
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Export.Stage1
open NightstreamFPrime.Export.Stage1.PiRLCPartialTrace (MaterializedRingK)
open NightstreamFPrime.Export.Stage1.PiDECMatrixRangeSum (Values)

private def checked {Alpha : Type} (value : Except String Alpha) : IO Alpha :=
  match value with
  | .ok result => pure result
  | .error error => throw (IO.userError error)

private def decodeVector {Alpha : Type} (count : Nat)
    (decode : Lean.Json → Except String Alpha) (value : Lean.Json) :
    Except String (Vector Alpha count) := do
  let entries ← value.getArr?
  if size : entries.size = count then (Vector.mk entries size).mapM decode
  else throw s!"expected {count} entries, got {entries.size}"

private def decodeField (value : Lean.Json) : Except String F := do
  let word ← value.getNat?
  if canonical : word < goldilocksModulus then return ⟨word, canonical⟩
  else throw "noncanonical matrix field coefficient"

private def decodeK (value : Lean.Json) : Except String K := do
  let pair ← decodeVector 2 decodeField value
  return ⟨pair.get ⟨0, by decide⟩, pair.get ⟨1, by decide⟩⟩

private def decodeRing (value : Lean.Json) : Except String MaterializedRingK := do
  let values ← decodeVector ringDegree decodeK value
  return PiRLCPartialTrace.FixedArray.ofFn values.get

private def decodeRange (rowCount : Nat) (text : String) :
    Except String (Nat × Nat × CubePoint K Lifecycle.cubeVariables × Values) := do
  let fields ← (← Lean.Json.parse text).getArr?
  match fields.toList with
  | [schema, rows, first, finish, coordinates, values] =>
      unless (← schema.getNat?) = 1 && (← rows.getNat?) = rowCount do
        throw "expected a selected Lean matrix range"
      let point ← decodeVector Lifecycle.cubeVariables decodeK coordinates
      let values ← decodeVector productionGlobalParams.k
        (decodeVector matrixCount decodeRing) values
      return (← first.getNat?, ← finish.getNat?,
        { coordinates := point.toList, dimension := by simp }, values)
  | _ => throw "expected a matrix range header, point and child values"

private def decodePad (expectedPoint : CubePoint K Lifecycle.cubeVariables)
    (text : String) : Except String (Vector MaterializedRingK productionGlobalParams.k) := do
  let fields ← (← Lean.Json.parse text).getArr?
  match fields.toList with
  | [schema, blocks, first, finish, coordinates, values] =>
      let expectedBlocks := Poseidon2HashChainV1Setup.messageColumns
      unless (← schema.getNat?) = 1 && (← blocks.getNat?) = expectedBlocks &&
          (← first.getNat?) = 0 && (← finish.getNat?) = expectedBlocks do
        throw "expected a complete selected Lean Pad result"
      let point ← decodeVector Lifecycle.cubeVariables decodeK coordinates
      unless point.toList = expectedPoint.coordinates do
        throw "Pad point differs from accepted C execution"
      decodeVector productionGlobalParams.k decodeRing values
  | _ => throw "expected a complete Pad header, point and child values"

private def writeResult (outputPath : System.FilePath)
    (point : CubePoint K Lifecycle.cubeVariables)
    (pad : Vector MaterializedRingK productionGlobalParams.k) (values : Values) : IO Unit := do
  let encodeK := fun value : K => Value.array [.atom value.c0.val, .atom value.c1.val]
  let result := Value.array [.atom 1, .atom Poseidon2HashChainV1Setup.messageColumns,
    .array (point.coordinates.map encodeK),
    .array (List.ofFn fun child : Fin productionGlobalParams.k =>
      .array (List.ofFn fun output : Fin ringDegree =>
        encodeK ((pad.get child).toRing output))),
    .array (List.ofFn fun child : Fin productionGlobalParams.k =>
      .array (List.ofFn fun port : Fin matrixCount =>
        .array (List.ofFn fun output : Fin ringDegree =>
          encodeK (((values.get child).get port).toRing output))))]
  IO.FS.writeFile outputPath (result.render ++ "\n")

/-- The C execution selects the point before any partial is accepted.
Partial ranges must form one positive, ordered partition of the complete
selected row domain. Only the current partial and accumulated sum are retained. -/
def merge (ccsPath padPath outputPath : System.FilePath)
    (partPaths : List String) : IO UInt32 := do
  unless !(← outputPath.pathExists) do throw (IO.userError "output already exists")
  let started ← IO.monoNanosNow
  let ccs ← checked (PiCCSInputCheck.parse (← IO.FS.readFile ccsPath))
  let phase ← IO.wait (Task.spawn fun _ => PiCCSInputCheck.execute ccs)
  unless phase.accepted do throw (IO.userError "C input rejected")
  let pad ← checked (decodePad phase.point (← IO.FS.readFile padPath))
  -- directStructuralRowCount_eq identifies this executable count with the selected plan.
  let rowCount := PerApplicationCanonicalPackage.directStructuralRowCount
    Poseidon2HashChainV1Package.application
  let mut cursor := 0
  let mut partCount := 0
  let mut accumulated := PiDECMatrixRangeSum.zero
  for path in partPaths do
    let (first, finish, point, values) ← checked
      (decodeRange rowCount (← IO.FS.readFile path))
    unless first = cursor && first < finish && finish ≤ rowCount do
      throw (IO.userError s!"matrix ranges have a gap, overlap or invalid endpoint: {path}")
    unless point.coordinates = phase.point.coordinates do
      throw (IO.userError s!"matrix range point differs from accepted C execution: {path}")
    accumulated := PiDECMatrixRangeSum.add accumulated values
    cursor := finish
    partCount := partCount + 1
  unless cursor = rowCount do
    throw (IO.userError "matrix ranges do not cover the complete selected row domain")
  writeResult outputPath phase.point pad accumulated
  IO.println (Lean.Json.mkObj [
    ("event", .str "matrix_merge_complete"), ("accepted_ccs", .bool true),
    ("ranges", Lean.toJson partCount), ("rows", Lean.toJson rowCount),
    ("children", Lean.toJson productionGlobalParams.k),
    ("matrices", Lean.toJson matrixCount),
    ("evaluation_words", Lean.toJson (productionGlobalParams.k * (1 + matrixCount) * ringDegree * 2)),
    ("point_words", Lean.toJson (Lifecycle.cubeVariables * 2)),
    ("total_ns", Lean.toJson ((← IO.monoNanosNow) - started))]).compress
  return 0

end NightstreamFPrime.Export.PiDECMatrixMerge

def main (arguments : List String) : IO UInt32 := do
  let arguments := if arguments.head? == some "--" then arguments.tail else arguments
  match arguments with
  | ccsPath :: padPath :: outputPath :: partPaths =>
      NightstreamFPrime.Export.PiDECMatrixMerge.merge ccsPath padPath outputPath partPaths
  | _ => throw (IO.userError "usage: mergePiDECMatrix <C-input> <complete-Pad> <new-output> <matrix-parts...>")
