import CompressionLean.Ports.Circuit.Record
import CompressionLeanExe.Parse
import CompressionLeanExe.Digest
import CompressionLeanExe.Fixtures
import CompressionLeanExe.Key

namespace CompressionLeanExe.LowerCheck

open CompressionLean CompressionLean.Shrink CompressionLean.Circuit CompressionLeanExe.Parse NightstreamFPrime.Spec
open CompressionLeanExe.Digest CompressionLeanExe.Fixtures
open CompressionLean.Replay (Word)

def wire : Parser Wire := do
  match ← nat with
  | 0 => pure (.glue (← nat))
  | 1 => do
      let b ← nat
      pure (.block b (← nat))
  | 2 => pure (.pub (← nat))
  | tag => throw s!"unknown wire tag {tag}"

def traceSide : Parser (List (Wire × ℕ)) := do
  many (← nat) do
    let w ← wire
    pure (w, ← nat)

def check (mirror : Option (Array Instr)) : Parser String := do
  keyword "statement"
  let stmt ← many (← nat) glWord
  let instrs ← program
  let mirrored ← match mirror with
    | none => pure ""
    | some lean => do
        samePrograms lean instrs
        pure s!"; Lean's program is all {lean.size} of Rust's {instrs.size} instructions"
  let source := mirror.getD instrs
  let run := lower (F := Gl) source (stmt.map fun g => (g.toNat : Gl))
  keyword "cells"
  expect "glue cells" run.cells (← nat)
  keyword "blocks"
  expect "blocks" run.blocks (← nat)
  keyword "publics"
  let publics ← many (← nat) nat
  expect "statement words" (run.publics.map ZMod.val) publics
  keyword "rows"
  let count ← nat
  let rows := (run.calls.flatMap Call.computedRows).toArray
  expect "glue rows" rows.size count
  for h : r in [0:rows.size] do
    for s in List.finRange 7 do
      expect s!"row {r} side {s}" ((rows[r]'h.2.1).side s |>.map fun t => (t.1, t.2.val)) (← traceSide)
  pure s!"match: {instrs.size} instructions, {rows.size} rows, {run.cells} glue cells, {run.blocks} blocks, \
    {run.publics.length} statement words{mirrored}; digest {(runDigest run rows).toList.map Word.toNat}"

def arguments : List String → IO (Except String (String × Option (Array Instr) × Option Layer1Key))
  | [path] => pure (.ok (path, none, none))
  | [path, "shared"] => pure (.ok (path, some shared.program, none))
  | [path, "early"] => pure (.ok (path, some early.program, none))
  | [path, "layer1", input] => do
      pure <| (← readJson Layer1Key input).map fun key => (path, some (layer1 key).program, some key)
  | [path, "linear", input] => do
      pure <| (← readJson StageInput input).map fun input => (path, some (linear input).program, none)
  | [path, "honest", input] => do
      pure <| (← readJson StageInput input).map fun input => (path, some (honest input).program, none)
  | [path, "late", input] => do
      pure <| (← readJson StageInput input).map fun input => (path, some (late input).program, none)
  | [path, "lane_split"] => pure (.ok (path, some laneSplit.program, none))
  | [path, "openings", input] => do
      pure <| (← readJson OpeningsJson input).map fun input => (path, some (openings input).program, none)
  | _ => pure (.error "usage: lowerCheck <export.txt> [shared | early | layer1 <layer1.json> | \
      linear <linear.json> | honest <honest.json> | late <late.json> | lane_split | openings <openings.json>]")

end CompressionLeanExe.LowerCheck

open CompressionLeanExe.LowerCheck in
def main (args : List String) : IO UInt32 := do
  let (path, mirror, key) ← match ← arguments args with
    | .ok parsed => pure parsed
    | .error message => IO.eprintln message; return 2
  if let some key := key then
    let _ ← CompressionLeanExe.Key.printFacts key
    CompressionLeanExe.Key.printWidth key
  match ((check mirror).run 0).run (← IO.FS.readBinFile path) with
  | .ok (message, _) => IO.println message; return 0
  | .error message => IO.println s!"mismatch: {message}"; return 1
