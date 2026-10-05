import NightstreamFPrime.Export.PiDECParentRange

/-! Regression checks for the parent range reader of the PiDEC commitment and
Pad range commands: it returns exactly the selected records of a valid file and
rejects broken framing, order, coefficients and ranges anywhere in the file. -/

namespace NightstreamFPrime.Tests.PiDECParentRange

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra
open NightstreamFPrime.Export
open NightstreamFPrime.Export.Stage1

private def check (ok : Bool) (message : String) : IO Unit :=
  unless ok do throw (IO.userError message)

private def header (first last : Nat) : String :=
  s!"[1,{Poseidon2HashChainV1Setup.messageColumns},{first},{last}]\n"

/-- One record line whose coefficients are `seed`, `seed + 1`, and so on. -/
private def record (block seed : Nat) (count : Nat := ringDegree) : String :=
  s!"[{block},{(List.range count).map (· + seed)}]\n"

private def valid : IO Unit := IO.FS.withTempDir fun directory => do
  let path := directory / "valid.jsonl"
  IO.FS.writeFile path
    (header 10 20 ++ record 11 1 ++ record 12 2 ++ record 15 3 ++ record 19 4 ++ "[]\n")
  let (blocks, records) ← PiDECParentRange.select path 12 19
  check (blocks == Poseidon2HashChainV1Setup.messageColumns) "wrong carrier width"
  check (records.map (·.1) == #[12, 15]) "wrong selected blocks"
  check (records.all fun (block, values) =>
      values.toArray == (Array.range ringDegree).map
        fun lane => Radix.fieldOfNat (lane + if block == 12 then 2 else 3))
    "wrong selected coefficients"
  let (_, records) ← PiDECParentRange.select path 16 19
  check records.isEmpty "an empty selection returned records"

private def broken : IO Unit := IO.FS.withTempDir fun directory => do
  let cases := [
    ("missing terminator", header 10 20 ++ record 11 1),
    ("data after terminator", header 10 20 ++ "[]\n" ++ record 11 1),
    ("empty line after terminator", header 10 20 ++ "[]\n\n"),
    ("decreasing blocks", header 10 20 ++ record 12 1 ++ record 11 1 ++ "[]\n"),
    ("duplicate block", header 10 20 ++ record 12 1 ++ record 12 1 ++ "[]\n"),
    ("block before the parent range", header 10 20 ++ record 9 1 ++ "[]\n"),
    ("block after the parent range", header 10 20 ++ record 20 1 ++ "[]\n"),
    ("unselected bad block", header 10 20 ++ record 13 1 ++ record 18 1 53 ++ "[]\n"),
    ("noncanonical coefficient", header 10 20 ++ record 13 (goldilocksModulus - 1) ++ "[]\n"),
    ("wrong schema", "[2,1605616,10,20]\n[]\n"),
    ("wrong carrier width", "[1,54,10,20]\n[]\n"),
    ("selection after the parent range", header 10 13 ++ "[]\n"),
    ("empty file", "")]
  for (name, text) in cases do
    let path := directory / "broken.jsonl"
    IO.FS.writeFile path text
    match ← (PiDECParentRange.select path 12 14).toBaseIO with
    | .error _ => pure ()
    | .ok _ => throw (IO.userError s!"parent range file with {name} was accepted")
  let path := directory / "empty-selection.jsonl"
  IO.FS.writeFile path (header 10 20 ++ "[]\n")
  match ← (PiDECParentRange.select path 14 14).toBaseIO with
  | .error _ => pure ()
  | .ok _ => throw (IO.userError "an empty selected range was accepted")

#eval valid
#eval broken

end NightstreamFPrime.Tests.PiDECParentRange
