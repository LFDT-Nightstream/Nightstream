import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Setup
import NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.StoredSplit

/-!
Read the independently replayed PiRLC parent for indexed PiDEC evaluation.
The input format is the existing sparse parent range stream. Range coverage,
canonical field words, coefficient bounds and record order are checked before
the stored parent is returned. Missing records denote the existing zero block.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.PiDECParentInput

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra
open NightstreamFPrime.Spec.Folding.Nifs.StoredAssignmentArithmetic (StoredAssignment)
open NightstreamFPrime.Export.Stage1

abbrev ParentBlocks := Vector (StoredAssignment ringDegree)
  Poseidon2HashChainV1Setup.messageColumns

private def checked {Alpha : Type} (value : Except String Alpha) : IO Alpha :=
  match value with
  | .ok result => pure result
  | .error error => throw (IO.userError error)

/-- Use the same canonical field decoder as the Pad range runner, and check
the strict parent bound before any child digit is requested. -/
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
        let coefficient := Radix.fieldOfNat value
        unless centeredMagnitude coefficient < Radix.combinedBound do
          throw "parent exceeds the strict B bound"
        values := values.push coefficient
      if size : values.size = ringDegree then return (block, ⟨values, size⟩)
      else throw "expected 54 parent coefficients"
  | _ => throw "expected parent block and coefficient array"

private def readRange (path : System.FilePath) :
    IO (Nat × Nat × Array (StoredAssignment ringDegree) × Nat) := do
  let input ← IO.FS.Handle.mk path .read
  let headerLine ← input.getLine
  let header ← checked do
    (← (← Lean.Json.parse headerLine).getArr?).toList.mapM Lean.Json.getNat?
  let (blocks, first, last) ← match header with
    | [1, blocks, first, last] => pure (blocks, first, last)
    | _ => throw (IO.userError "expected a Lean PiRLC range header")
  unless blocks = Poseidon2HashChainV1Setup.messageColumns &&
      first < last && last ≤ blocks do
    throw (IO.userError "invalid selected parent range")
  let zero : StoredAssignment ringDegree := Vector.replicate ringDegree 0
  let mut values := Array.replicate (last - first) zero
  let mut next := first
  let mut records := 0
  let mut complete := false
  while !complete do
    let line ← input.getLine
    if line.isEmpty then throw (IO.userError "missing parent terminator")
    if line.trimAscii.toString == "[]" then complete := true
    else
      let (block, coefficients) ← checked (decodeBlock line)
      unless next ≤ block && block < last do
        throw (IO.userError "duplicate or out-of-range parent block")
      values := values.set! (block - first) coefficients
      next := block + 1
      records := records + 1
  unless (← input.getLine).isEmpty do
    throw (IO.userError "extra data after parent terminator")
  return (first, last, values, records)

/-- Retain one parent, with complete contiguous coverage. Parallel range
decoding uses the existing Lean runtime worker count; concatenation retains
the supplied canonical order. No Rust witness or evaluation is read here. -/
def read (paths : List String) : IO (ParentBlocks × Nat) := do
  let workers := max 1 (((← IO.getEnv "LEAN_NUM_THREADS").bind String.toNat?).getD 1)
  let mut pending : Array (Task (Except IO.Error
    (Nat × Nat × Array (StoredAssignment ringDegree) × Nat))) := #[]
  let mut complete : Array (StoredAssignment ringDegree) := #[]
  let mut cursor := 0
  let mut records := 0
  let mut remaining := paths
  while !remaining.isEmpty do
    let batch := remaining.take workers
    remaining := remaining.drop workers
    for path in batch do
      pending := pending.push (← IO.asTask (readRange path))
    for task in pending do
      let result ← IO.wait task
      let (first, last, values, count) ← match result with
        | .ok value => pure value
        | .error error => throw error
      unless first = cursor && values.size = last - first do
        throw (IO.userError "parent ranges have a gap, overlap or wrong length")
      complete := complete ++ values
      cursor := last
      records := records + count
    pending := #[]
  unless cursor = Poseidon2HashChainV1Setup.messageColumns do
    throw (IO.userError "parent ranges do not cover the complete carrier")
  if size : complete.size = Poseidon2HashChainV1Setup.messageColumns then
    return (⟨complete, size⟩, records)
  else throw (IO.userError "wrong complete parent size")

end NightstreamFPrime.Export.PiDECParentInput
