import NightstreamFPrime.Export.ParallelChunks

/-!
Decode the lines of a text file on several threads. A line is the text between
line terminators, without its terminator; the text after the last terminator is
a line only when it is not empty, as with repeated `Handle.getLine`. Every line
must be valid UTF-8. Results keep file order, with each line's byte offset.
Callers keep their own framing, order and terminator checks.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.ParallelLines

/-- Bytes per work item: large enough that a part holds many lines, and small
enough that faster cores take more parts. -/
private def partBytes : Nat := 4 * 2 ^ 20

/-- The offset just after the first line terminator at or after `position`, or `size`. -/
private partial def nextLineStart (bytes : ByteArray) (position : Nat) : Nat :=
  if position < bytes.size then
    if bytes.get! position == 10 then position + 1 else nextLineStart bytes (position + 1)
  else bytes.size

/-- The offset of the first line terminator at or after `position`, or `size`. -/
private partial def lineEnd (bytes : ByteArray) (position : Nat) : Nat :=
  if position < bytes.size then
    if bytes.get! position == 10 then position else lineEnd bytes (position + 1)
  else bytes.size

/-- Decode the lines that start in `[lo, hi)`, in order. -/
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

/-- Decode each line of `bytes` with its byte offset, in parts of about `size`
bytes on `threads` threads. A part decodes the lines that start inside it. -/
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

/-- Read a file and decode each line with its byte offset, on `threads` threads. -/
def decode {Alpha : Type} (path : System.FilePath) (threads : Nat)
    (decode : Nat → String → Alpha) : IO (Array (Nat × Alpha)) := do
  try decodeBytes (← IO.FS.readBinFile path) partBytes threads decode
  catch error => throw (IO.userError s!"{path}: {error}")

end NightstreamFPrime.Export.ParallelLines
