import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Setup
import NightstreamFPrime.Export.ParallelLines
import NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.StoredSplit

/-!
Read the records of one sparse PiRLC parent range file that a range command
computes. The lines are decoded on the configured worker threads; the header,
the selected range, canonical field words, increasing record order, the
terminator and the end of the file are then checked in file order over the
complete stream. Omitted blocks denote zero.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.PiDECParentRange

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra
open NightstreamFPrime.Spec.Folding.Nifs.StoredAssignmentArithmetic (StoredAssignment)
open NightstreamFPrime.Export.Stage1

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

/-- One decoded line of a parent range file. -/
private inductive ParentLine where
  | header (line : String)
  | terminator
  | block (record : Except String (Nat × StoredAssignment ringDegree))

instance : Inhabited ParentLine := ⟨.terminator⟩

/-- Check the complete parent range file and return the carrier width and the
records with `start ≤ block < finish`, in block order. -/
def select (path : System.FilePath) (start finish : Nat) :
    IO (Nat × Array (Nat × StoredAssignment ringDegree)) := do
  let lines : Array (Nat × ParentLine) ←
    ParallelLines.decode path (← ParallelChunks.workers) fun offset line =>
      if offset == 0 then ParentLine.header line
      else if line.trimAscii.toString == "[]" then ParentLine.terminator
      else ParentLine.block (decodeBlock line)
  let some (_, ParentLine.header headerLine) := lines[0]?
    | throw (IO.userError "missing Lean PiRLC range header")
  let header ← checked do
    (← (← Lean.Json.parse headerLine).getArr?).toList.mapM Lean.Json.getNat?
  let (blocks, parentStart, parentEnd) ← match header with
    | [1, blocks, first, last] => pure (blocks, first, last)
    | _ => throw (IO.userError "expected a Lean PiRLC range header")
  unless blocks = Poseidon2HashChainV1Setup.messageColumns &&
      parentStart ≤ start && start < finish && finish ≤ parentEnd && parentEnd ≤ blocks do
    throw (IO.userError "selected range is outside the parent range")
  let mut selected := #[]
  let mut next := parentStart
  let mut complete := false
  for (_, line) in lines[1:] do
    if complete then throw (IO.userError "extra data after parent terminator")
    match line with
    | ParentLine.terminator => complete := true
    | ParentLine.block record =>
        let (block, values) ← checked record
        unless next ≤ block && block < parentEnd do
          throw (IO.userError "duplicate or out-of-range parent block")
        next := block + 1
        if start ≤ block && block < finish then selected := selected.push (block, values)
    | ParentLine.header _ => throw (IO.userError "unexpected parent range header")
  unless complete do throw (IO.userError "missing parent terminator")
  return (blocks, selected)

end NightstreamFPrime.Export.PiDECParentRange
