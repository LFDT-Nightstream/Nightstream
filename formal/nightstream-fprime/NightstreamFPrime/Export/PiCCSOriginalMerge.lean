import NightstreamFPrime.Export.Stage1.PiCCSOriginalMatrixBatch
import NightstreamFPrime.Export.Stage1.PiCCSOriginalReads
import NightstreamFPrime.Export.Stage1.PiCCSInputCheck
import NightstreamFPrime.Export.Codec

/-!
Merge retained original-source evaluations in their complete canonical ranges.
The caller derives the point from the original public input and all Lean rounds.
Files are checked for shape, canonical fields, point, and complete coverage;
their calculation provenance belongs to the source-bound replay evidence.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.PiCCSOriginalMerge

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.ProductionRelation
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Layout
open NightstreamFPrime.Export.Stage1
open NightstreamFPrime.Export.Stage1.PiRLCPartialTrace (MaterializedRingK)
open NightstreamFPrime.Export.Codec

private def checked {Alpha : Type} (value : Except String Alpha) : IO Alpha :=
  match value with
  | .ok result => pure result
  | .error error => throw (IO.userError error)

private def decodeK (value : Lean.Json) : Except String K := do
  let words ← PiCCSInputCheck.decodeVector 2 PiCCSInputCheck.decodeField value
  return ⟨words.get 0, words.get 1⟩

private def decodeRing (value : Lean.Json) : Except String MaterializedRingK := do
  let words ← PiCCSInputCheck.decodeVector ringDegree decodeK value
  return PiRLCPartialTrace.FixedArray.ofFn words.get

private def decodeMatrices (value : Lean.Json) : Except String PiCCSOriginalMatrixBatch.Batch := do
  let sources ← PiCCSInputCheck.decodeVector productionShape.sourceCount
    (PiCCSInputCheck.decodeVector matrixCount decodeRing) value
  return Vector.ofFn fun code =>
    let pair : Fin productionShape.sourceCount × Fin matrixCount := Fin.decodeProd code
    (sources.get pair.1).get pair.2

private def collect {width : Nat} (domain : Nat) (point : PaperAlgebra.Point)
    (decodeValues : Lean.Json → Except String (Vector MaterializedRingK width))
    (paths : List String) : IO (Vector MaterializedRingK width) := do
  let mut cursor := 0
  let mut total := PiDECEvaluationBatch.zero width
  for path in paths do
    let text ← IO.FS.readFile path
    let fields ← checked do
      (← Lean.Json.parse text).getArr?
    unless fields.size == 6 do throw (IO.userError "expected six original evaluation range fields")
    unless (← checked (fields[0]!.getNat?)) == 1 &&
        (← checked (fields[1]!.getNat?)) == domain do
      throw (IO.userError "wrong original evaluation range schema or domain")
    let first ← checked (fields[2]!.getNat?)
    let finish ← checked (fields[3]!.getNat?)
    unless first == cursor && first < finish && finish ≤ domain do
      throw (IO.userError "original evaluation ranges have a gap, overlap or invalid endpoint")
    let savedPoint ← checked (PiCCSInputCheck.decodeVector cubeVariables decodeK fields[4]!)
    unless savedPoint.toList == point.coordinates do
      throw (IO.userError "range point differs from the complete Lean transcript")
    let values ← checked (decodeValues fields[5]!)
    total := PiDECEvaluationBatch.add total values
    cursor := finish
  unless cursor == domain do throw (IO.userError "original evaluation ranges are incomplete")
  return total

def merge (point : PaperAlgebra.Point) (output : System.FilePath)
    (padPaths matrixPaths : List String) : IO UInt32 := do
  unless !(← output.pathExists) do throw (IO.userError "output already exists")
  let started ← IO.monoNanosNow
  let program ← IO.wait (Task.spawn fun _ =>
    PerApplicationMatrixProgram.matrixProgram Poseidon2HashChainV1Package.application)
  let pad ← collect PiCCSSourceImages.blockCount point
    (PiCCSInputCheck.decodeVector productionShape.sourceCount decodeRing) padPaths
  let matrices ← collect program.rowCount point decodeMatrices matrixPaths
  let encodeK := fun value : K => Value.array [.atom value.c0.val, .atom value.c1.val]
  let encodeRing := fun ring : MaterializedRingK => Value.array
    (List.ofFn fun lane : Fin ringDegree => encodeK (ring.toRing lane))
  let value := Value.array [.atom 1, .array (point.coordinates.map encodeK),
    .array (List.ofFn fun source : Fin productionShape.sourceCount => encodeRing (pad.get source)),
    .array (List.ofFn fun source : Fin productionShape.sourceCount =>
      .array (List.ofFn fun port : Fin matrixCount =>
        encodeRing (matrices.get (Fin.encodeProd (source, port)))))]
  IO.FS.writeFile output (value.render ++ "\n")
  IO.println (Lean.Json.mkObj [("event", .str "original_evaluations_merged"),
    ("sources", Lean.toJson productionShape.sourceCount),
    ("matrices_per_source", Lean.toJson matrixCount),
    ("field_words", Lean.toJson (productionShape.sourceCount * (matrixCount + 1) * ringDegree * 2)),
    ("elapsed_ns", Lean.toJson ((← IO.monoNanosNow) - started))]).compress
  return 0


/-- Retain a complete Pad sum for its independent family comparison. -/
def mergePad (point : PaperAlgebra.Point) (output : System.FilePath)
    (paths : List String) : IO UInt32 := do
  unless !(← output.pathExists) do throw (IO.userError "output already exists")
  let pad ← collect PiCCSSourceImages.blockCount point
    (PiCCSInputCheck.decodeVector productionShape.sourceCount decodeRing) paths
  let encodeK := fun value : K => Value.array [.atom value.c0.val, .atom value.c1.val]
  let encodeRing := fun ring : MaterializedRingK => Value.array
    (List.ofFn fun lane : Fin ringDegree => encodeK (ring.toRing lane))
  let value := Value.array [.atom 1, .atom PiCCSSourceImages.blockCount, .atom 0,
    .atom PiCCSSourceImages.blockCount, .array (point.coordinates.map encodeK),
    .array (List.ofFn fun source : Fin productionShape.sourceCount => encodeRing (pad.get source))]
  IO.FS.writeFile output (value.render ++ "\n")
  IO.println "original_complete_pad_merged"
  return 0

end NightstreamFPrime.Export.PiCCSOriginalMerge
