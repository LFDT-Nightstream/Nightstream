import NightstreamFPrime.Export.ParityEmitter
import NightstreamFPrime.Export.Stage1.PiDECInputCheck

/-! Check supplied PiDEC mutations after one actual accepted C/R execution.
The canonical decoder owns encoding failures; the paper verifier and exact
output comparison own public failures. No synthetic parent is substituted. -/

namespace NightstreamFPrime.Tests.PiDECActualMutations

open NightstreamFPrime.Export.Stage1

private def checked {Alpha : Type} (value : Except String Alpha) : IO Alpha :=
  match value with
  | .ok result => pure result
  | .error error => throw (IO.userError error)

private def run (p0 p1 p2 p3 inputPath messagesPath badInputPath mutationsDir : String) :
    IO UInt32 := do
  let identity ← checked (Export.ParityEmitter.parseVerifierKey p0 p1 p2 p3)
  let input ← checked (PiCCSInputCheck.parse (← IO.FS.readFile inputPath))
  let messages ← checked (PiCCSInputCheck.parseRunning (← IO.FS.readFile messagesPath))
  let execution ← PiRLCInputCheck.checkIO input identity
  let some parent := execution.parent
    | throw (IO.userError "honest C/R execution rejected")
  if !PiDECInputCheck.accepted parent messages then
    throw (IO.userError "honest PiDEC input rejected")
  if PiDECInputCheck.accepted (PiDECInputCheck.unbounded parent) messages then
    throw (IO.userError "PiDEC accepted the public bound B")
  let entries := (← (System.FilePath.mk mutationsDir).readDir).toList.mergeSort
    (fun left right => left.fileName ≤ right.fileName)
  let mut publicCount := 0
  let mut encodingCount := 0
  for entry in entries do
    let value := PiCCSInputCheck.parseRunning (← IO.FS.readFile entry.path)
    if entry.fileName.startsWith "encoding_" then
      match value with
      | .ok _ => throw (IO.userError s!"decoder accepted {entry.fileName}")
      | .error error =>
          encodingCount := encodingCount + 1
          IO.println s!"lean_pi_dec_mutation={entry.fileName} rejected_by=decoder reason={error}"
    else if entry.fileName.startsWith "public_" then
      let changed ← checked value
      if PiDECInputCheck.accepted parent changed then
        throw (IO.userError s!"PiDEC accepted {entry.fileName}")
      publicCount := publicCount + 1
      IO.println s!"lean_pi_dec_mutation={entry.fileName} rejected_by=public_check"
    else
      throw (IO.userError s!"mutation has no expected owner: {entry.fileName}")
  if publicCount = 0 || encodingCount = 0 then
    throw (IO.userError "public and encoding mutations are both required")
  let badInput ← checked (PiCCSInputCheck.parse (← IO.FS.readFile badInputPath))
  let rejectedPrefix ← PiRLCInputCheck.checkIO badInput identity
  if rejectedPrefix.parent.isSome then
    throw (IO.userError "rejected C input supplied a D parent")
  IO.println s!"lean_pi_dec_mutations=passed public={publicCount} encoding={encodingCount} unbounded=1 rejected_C_stops_D=1"
  pure 0

def main (arguments : List String) : IO UInt32 :=
  match arguments with
  | [p0, p1, p2, p3, inputPath, messagesPath, badInputPath, mutationsDir] =>
      run p0 p1 p2 p3 inputPath messagesPath badInputPath mutationsDir
  | _ => do
      IO.eprintln "usage: PiDECActualMutations <package[4]> <PiCCS-input> <children> <bad-PiCCS-input> <mutations-dir>"
      pure 2

end NightstreamFPrime.Tests.PiDECActualMutations

def main := NightstreamFPrime.Tests.PiDECActualMutations.main
