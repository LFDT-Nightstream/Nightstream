import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Setup
import NightstreamFPrime.Export.Stage1.PiDECParentIntRead
import NightstreamFPrime.Export.ParallelLines

/-!
Read the independently replayed PiRLC parent for indexed PiDEC evaluation.
The input format is the existing sparse parent range stream. Range coverage,
canonical field words, coefficient bounds and record order are checked before
the centered integer parent is returned. Missing records denote the existing
zero block. The canonical field input format is unchanged.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.PiDECParentInput

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra
open NightstreamFPrime.Export.Stage1

abbrev ParentBlocks := Vector (Vector Int ringDegree)
  Poseidon2HashChainV1Setup.messageColumns

private def checked {Alpha : Type} (value : Except String Alpha) : IO Alpha :=
  match value with
  | .ok result => pure result
  | .error error => throw (IO.userError error)

/-- Use the same canonical field decoder as the Pad range runner, and check
the strict parent bound before any child digit is requested. -/
private def decodeBlock (line : String) :
    Except String (Nat × Vector Int ringDegree) := do
  let fields ← (← Lean.Json.parse line).getArr?
  match fields.toList with
  | [block, coefficients] =>
      let block ← block.getNat?
      let words ← coefficients.getArr?
      let mut values : Array Int := #[]
      for word in words do
        let value ← word.getNat?
        unless value < goldilocksModulus do throw "noncanonical parent coefficient"
        let coefficient := Radix.fieldOfNat value
        unless centeredMagnitude coefficient < Radix.combinedBound do
          throw "parent exceeds the strict B bound"
        values := values.push (ZMod.valMinAbs (n := goldilocksModulus) coefficient)
      if size : values.size = ringDegree then return (block, ⟨values, size⟩)
      else throw "expected 54 parent coefficients"
  | _ => throw "expected parent block and coefficient array"

/-- One decoded line of a parent range file. -/
private inductive ParentLine where
  | header (line : String)
  | terminator
  | block (record : Except String (Nat × Vector Int ringDegree))

instance : Inhabited ParentLine := ⟨.terminator⟩

/-- Decode one range file on the configured worker threads, then check its
header, the increasing record order, the terminator and the end of the file. -/
private def readRange (path : System.FilePath) :
    IO (Nat × Nat × Array (Vector Int ringDegree) × Nat) := do
  let lines : Array (Nat × ParentLine) ←
    ParallelLines.decode path (← ParallelChunks.workers) fun offset line =>
      if offset == 0 then ParentLine.header line
      else if line.trimAscii.toString == "[]" then ParentLine.terminator
      else ParentLine.block (decodeBlock line)
  let some (_, ParentLine.header headerLine) := lines[0]?
    | throw (IO.userError "missing Lean PiRLC range header")
  let header ← checked do
    (← (← Lean.Json.parse headerLine).getArr?).toList.mapM Lean.Json.getNat?
  let (blocks, first, last) ← match header with
    | [1, blocks, first, last] => pure (blocks, first, last)
    | _ => throw (IO.userError "expected a Lean PiRLC range header")
  unless blocks = Poseidon2HashChainV1Setup.messageColumns &&
      first < last && last ≤ blocks do
    throw (IO.userError "invalid selected parent range")
  let zero : Vector Int ringDegree := Vector.replicate ringDegree 0
  let mut values := Array.replicate (last - first) zero
  let mut next := first
  let mut records := 0
  let mut complete := false
  for (_, line) in lines[1:] do
    if complete then throw (IO.userError "extra data after parent terminator")
    match line with
    | ParentLine.terminator => complete := true
    | ParentLine.block record =>
        let (block, coefficients) ← checked record
        unless next ≤ block && block < last do
          throw (IO.userError "duplicate or out-of-range parent block")
        values := values.set! (block - first) coefficients
        next := block + 1
        records := records + 1
    | ParentLine.header _ => throw (IO.userError "unexpected parent range header")
  unless complete do throw (IO.userError "missing parent terminator")
  return (first, last, values, records)

/-- Read the ordered range files one at a time; each file is decoded in parallel. -/
def read (paths : List String) : IO (ParentBlocks × Nat) := do
  let mut complete : Array (Vector Int ringDegree) := #[]
  let mut cursor := 0
  let mut records := 0
  for path in paths do
    let (first, last, values, count) ← readRange path
    unless first = cursor && values.size = last - first do
      throw (IO.userError "parent ranges have a gap, overlap or wrong length")
    complete := complete ++ values
    cursor := last
    records := records + count
  unless cursor = Poseidon2HashChainV1Setup.messageColumns do
    throw (IO.userError "parent ranges do not cover the complete carrier")
  if size : complete.size = Poseidon2HashChainV1Setup.messageColumns then
    return (⟨complete, size⟩, records)
  else throw (IO.userError "wrong complete parent size")

end NightstreamFPrime.Export.PiDECParentInput
