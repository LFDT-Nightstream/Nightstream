import CompressionLean.Circuit.Record
import CompressionLeanExe.Parse
import CompressionLeanExe.Digest
import CompressionLeanExe.Fixtures
import CompressionLeanExe.Key

/-!
The `lowerCheck` executable. It reads one export of `neo-spartan`'s
`tests/internal/lowering.rs` (the statement, the IR, then Rust's trace),
lowers the IR in Goldilocks and compares the run with the trace: the glue
cells, the blocks, the statement words and every glue row, side by side and
term by term (`Call.computedRows`, which is `Call.rows`). It prints `match`,
the counts and the run digest (`runDigest`), or the first difference and
exits with 1.

With a program name it also writes that program with the Lean builder
(`Fixtures.lean`) and compares it with the IR, instruction by instruction
(`samePrograms`). It then lowers Lean's program, so the digest is that of
Lean's run:

- `shared`: the Lean side of `Shared`, every helper of `Algebra.lean`,
  `Hash.lean`, `Fold.lean` and `Sumcheck.lean`. The whole IR must match.
- `early`: the Lean side of `Early`, the helpers of `Gkr.lean` and
  `Early.lean` on a structure that the toy relation does not have. The whole
  IR must match.
- `layer1 <input.json>`: the Lean side of `Layer1::run` for the relation and
  the two `WhirKey`s in the file. The whole IR must match. It prints whether
  the key has `KeyFacts` (`keyFailure`) and whether it fits the Poseidon2
  hash chain (`poseidon2Fits`); the toy key has neither.
- `linear <input.json>`: the Lean side of `Linear`, the linear stage
  (`linearStage`) from the state after the early stage. The whole IR must
  match.
- `honest <input.json>`: the Lean side of `Honest`, the honest fold
  (`honestStage`) and the query points from the state before `μ`. The whole
  IR must match.
- `late <input.json>`: the Lean side of `Late`, `lateStage` from the toy
  proof's state after the setup queries, for the relation and duplex sizes
  in the file. The whole IR must match.
- `lane_split`: the Lean side of `LaneSplit`, `laneTables` on classes that
  the toy relation does not have. The whole IR must match.
- `openings <input.json>`: the Lean side of `Openings`, both openings
  (`openingsStage`) from the state after the query points. The whole IR must
  match.
-/

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

/-- Compare Lean's program `mirror` with the IR, then the lowered run with
the trace that follows the program. -/
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

/-- The export's path, Lean's program for it and the layer-1 key, from the arguments. -/
def arguments : List String → IO (Except String (String × Option (Array Instr) × Option Layer1Input))
  | [path] => pure (.ok (path, none, none))
  | [path, "shared"] => pure (.ok (path, some shared.program, none))
  | [path, "early"] => pure (.ok (path, some early.program, none))
  | [path, "layer1", input] => do
      pure <| (← readJson Layer1Input input).map fun key => (path, some (layer1 key).program, some key)
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
