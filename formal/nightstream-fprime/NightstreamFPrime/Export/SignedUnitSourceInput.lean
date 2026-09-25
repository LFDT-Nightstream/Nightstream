import NightstreamFPrime.Export.Stage1.PiRLCWitnessBlock

/-!
Owns the existing original-witness capture format shared by PiRLC and PiCCS
replay. Records contain signed-unit masks, not prover outputs. Omitted source
entries and blocks are zero. The loader checks ordered complete file framing.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.SignedUnitSourceInput

open NightstreamFPrime.Spec
open NightstreamFPrime.Export.Stage1
open PiRLCNonzero (SourceCount)
open PiRLCPartialTrace

/-- Read one original signed-unit coefficient from its two masks. -/
def scalar (masks : Array (Nat × Nat)) (source : Fin SourceCount)
    (lane : Fin ringDegree) : F :=
  let pair := masks[source.val]?.getD (0, 0)
  if pair.1.testBit lane.val then 1
  else if pair.2.testBit lane.val then -1 else 0

/-- The original PiRLC source block, with no mixed-witness input. -/
def sourceBlock (masks : Array (Nat × Nat)) (source : Fin SourceCount) :
    MaterializedRingF :=
  FixedArray.ofFn (scalar masks source)

/-- Decode the existing ordered nonzero-source record and reject invalid masks. -/
def decodeBlock (line : String) :
    Except String (Nat × Array (Nat × Nat)) := do
  let values ← (← Lean.Json.parse line).getArr?
  match values.toList with
  | [index, entries] =>
      let block ← index.getNat?
      let entries ← entries.getArr?
      let mut values : Array (Nat × Nat) := Array.replicate SourceCount (0, 0)
      let mut next := 0
      for entry in entries do
        let fields ← entry.getArr?
        let fields ← fields.toList.mapM Lean.Json.getNat?
        match fields with
        | [source, positive, negative] =>
            unless next ≤ source && source < SourceCount do
              throw "source indices must be unique and increasing"
            unless positive < 2 ^ ringDegree && negative < 2 ^ ringDegree &&
                (positive &&& negative) == 0 && (positive ||| negative) != 0 do
              throw "invalid signed-unit source masks"
            values := values.set! source (positive, negative)
            next := source + 1
        | _ => throw "expected source index and two masks"
      unless !entries.isEmpty do throw "zero source blocks must be omitted"
      return (block, values)
  | _ => throw "expected a block index and source entries"

private def checked {Alpha : Type} (value : Except String Alpha) : IO Alpha :=
  match value with
  | .ok value => pure value
  | .error error => throw (IO.userError error)

/-- Load the same source capture for random matrix-column reads. The caller
supplies the selected carrier block count. Every input record is validated,
including records outside the first arithmetic range. -/
def read (path : System.FilePath) (blocks : Nat) :
    IO (Array (Array (Nat × Nat)) × Nat) := do
  let input ← IO.FS.Handle.mk path .read
  let header ← checked (Lean.Json.parse (← input.getLine))
  let header ← checked header.getArr?
  let header ← checked (header.toList.mapM Lean.Json.getNat?)
  unless header == [1, ringDegree, SourceCount, blocks] do
    throw (IO.userError "source header does not match the selected carrier")
  let mut result := Array.replicate blocks #[]
  let mut next := 0
  let mut records := 0
  let mut complete := false
  while !complete do
    let line ← input.getLine
    if line.isEmpty then throw (IO.userError "missing source terminator")
    if line.trimAscii.toString == "[]" then
      complete := true
    else
      let (block, masks) ← checked (decodeBlock line)
      unless next ≤ block && block < blocks do
        throw (IO.userError "duplicate or out-of-range source block")
      result := result.set! block masks
      next := block + 1
      records := records + 1
  unless (← input.getLine).isEmpty do
    throw (IO.userError "extra data after source terminator")
  return (result, records)

end NightstreamFPrime.Export.SignedUnitSourceInput
