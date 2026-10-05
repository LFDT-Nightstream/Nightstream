import NightstreamFPrime.Export.ParallelLines
import NightstreamFPrime.Export.SignedUnitSourceInput

/-! Regression checks for parallel line decoding: every part size gives the lines
and offsets of a sequential split, invalid UTF-8 is rejected, and the source
loader keeps its header, order, terminator and end-of-file checks. -/

namespace NightstreamFPrime.Tests.ParallelLines

open NightstreamFPrime.Export

private def check (ok : Bool) (message : String) : IO Unit :=
  unless ok do throw (IO.userError message)

/-- Lines without terminators and their byte offsets; no line after a final terminator. -/
private def sequentialLines (text : String) : Array (Nat × String) := Id.run do
  let pieces := text.splitOn "\n"
  let pieces := if pieces.getLast? == some "" then pieces.dropLast else pieces
  let mut offset := 0
  let mut lines := #[]
  for piece in pieces do
    lines := lines.push (offset, piece)
    offset := offset + piece.utf8ByteSize + 1
  return lines

private def splitting : IO Unit := do
  let texts := ["", "a", "a\n", "a\nbb\n", "\n\nx", "x\n\n", "αβ\nγ\n", "one\ntwo\nthree"]
  for text in texts do
    for size in [1:12] do
      for threads in [1, 3] do
        let lines ← ParallelLines.decodeBytes text.toUTF8 size threads fun _ line => line
        check (lines == sequentialLines text) s!"wrong lines for {repr text} in parts of {size}"
  let invalid := ByteArray.mk #[97, 10, 0xff, 10]
  match ← (ParallelLines.decodeBytes invalid 1 2 fun _ line => line).toBaseIO with
  | .error _ => pure ()
  | .ok _ => throw (IO.userError "invalid UTF-8 was accepted")

/-- The source loader accepts a valid file and rejects broken framing. -/
private def sourceFraming : IO Unit := IO.FS.withTempDir fun directory => do
  let header := "[1,54,17,3]\n"
  let valid := header ++ "[0,[[0,1,2]]]\n[2,[[3,4,0]]]\n[]\n"
  IO.FS.writeFile (directory / "valid.jsonl") valid
  let (masks, records) ← SignedUnitSourceInput.read (directory / "valid.jsonl") 3
  check (records == 2 && (masks[0]!)[0]! == (1, 2) && masks[1]! == #[] && (masks[2]!)[3]! == (4, 0))
    "valid source file decoded wrongly"
  let broken := [
    ("missing terminator", header ++ "[0,[[0,1,2]]]\n"),
    ("data after terminator", header ++ "[0,[[0,1,2]]]\n[]\n[1,[[0,1,2]]]\n"),
    ("empty line after terminator", header ++ "[]\n\n"),
    ("decreasing blocks", header ++ "[2,[[0,1,2]]]\n[1,[[0,1,2]]]\n[]\n"),
    ("overlapping masks", header ++ "[0,[[0,3,1]]]\n[]\n"),
    ("wrong header", "[1,54,17,4]\n[]\n")]
  for (name, text) in broken do
    let path := directory / "broken.jsonl"
    IO.FS.writeFile path text
    match ← (SignedUnitSourceInput.read path 3).toBaseIO with
    | .error _ => pure ()
    | .ok _ => throw (IO.userError s!"source file with {name} was accepted")

#eval splitting
#eval sourceFraming

end NightstreamFPrime.Tests.ParallelLines
