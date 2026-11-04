import NightstreamFPrime.Export.ParallelChunks

set_option autoImplicit false

namespace NightstreamFPrime.Export.ParallelLines

private def partBytes : Nat := 4 * 2 ^ 20

private partial def nextLineStart (bytes : ByteArray) (position : Nat) : Nat :=
  if position < bytes.size then
    if bytes.get! position == 10 then position + 1 else nextLineStart bytes (position + 1)
  else bytes.size

private partial def lineEnd (bytes : ByteArray) (position : Nat) : Nat :=
  if position < bytes.size then
    if bytes.get! position == 10 then position else lineEnd bytes (position + 1)
  else bytes.size

private partial def decodePart {Alpha : Type} (bytes : ByteArray)
    (decode : Nat → String → Alpha) (hi start : Nat) (results : Array (Nat × Alpha)) :
    Except String (Array (Nat × Alpha)) :=
  if start < hi && start < bytes.size then
    let finish := lineEnd bytes start
    match String.fromUTF8? (bytes.extract start finish) with
    | some line =>
        decodePart bytes decode hi (finish + 1) (results.push (start, decode start line))
    | none => .error s!"invalid UTF-8 in the line at byte {start}"
  else .ok results

def decodeBytes {Alpha : Type} (bytes : ByteArray) (size threads : Nat)
    (decode : Nat → String → Alpha) : IO (Array (Nat × Alpha)) := do
  let parts := max 1 ((bytes.size + size - 1) / max 1 size)
  let tasks ← ParallelChunks.start threads parts fun part => do
    let lo := bytes.size * part / parts
    let hi := bytes.size * (part + 1) / parts
    let start := if lo == 0 then 0 else nextLineStart bytes (lo - 1)
    match decodePart bytes decode hi start #[] with
    | .ok results => pure results
    | .error message => throw (IO.userError message)
  let mut results := #[]
  for task in tasks do
    match ← IO.wait task with
    | .ok (part, _, _) => results := results ++ part
    | .error error => throw error
  return results

def decode {Alpha : Type} (path : System.FilePath) (threads : Nat)
    (decode : Nat → String → Alpha) : IO (Array (Nat × Alpha)) := do
  try decodeBytes (← IO.FS.readBinFile path) partBytes threads decode
  catch error => throw (IO.userError s!"{path}: {error}")

end NightstreamFPrime.Export.ParallelLines
