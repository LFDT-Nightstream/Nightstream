import CompressionLean.Circuit.Record
import CompressionLean.Final.KeyFacts
import CompressionLeanExe.Parse
import CompressionLeanExe.Digest
import CompressionLeanExe.Fixtures

/-!
The `lowerCheck` executable. It reads one export of `neo-spartan`'s
`tests/internal/lowering.rs` (the statement, the IR, then Rust's trace),
lowers the IR in Goldilocks and compares the run with the trace: the glue
cells, the blocks, the statement words and every glue row, side by side and
term by term (`Call.computedRows`, which is `Call.rows`). It prints `match`,
the counts and the run digest (`runDigest`), or the first difference and
exits with 1.

With a program name it also writes that program with the Lean builder
(`Fixtures.lean`) and compares it with the IR, instruction by instruction, as
`Log` writes them (`line`). It then lowers Lean's program, so the digest is
that of Lean's run:

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

def handlesText {n : ℕ} (a : Fin n → Var) : String :=
  String.join ((List.finRange n).map fun k => s!" {a k}")

def termsText (terms : List (ℕ × Word)) : String :=
  s!" {terms.length}" ++ String.join (terms.map fun t => s!" {t.1} {t.2.toNat}")

/-- The line that `Log` writes for an instruction (`lowering.rs`). -/
def line : Instr → String
  | .const value => s!"c {value.toNat}"
  | .priv => "p"
  | .pub => "u"
  | .linear parts => s!"l{termsText parts}"
  | .mul a b => s!"m {a} {b}"
  | .extMul a b => s!"x{handlesText a}{handlesText b}"
  | .extInv a => s!"i{handlesText a}"
  | .assertZero a => s!"z {a}"
  | .bits a count => s!"b {a} {count}"
  | .hint count => s!"h {count}"
  | .permute state => s!"q{handlesText state}"
  | .r1cs rows columns =>
      s!"r {rows.length} {columns.length}" ++ String.join (columns.map (s!" {·}")) ++
        String.join (rows.map fun row => termsText row.a ++ termsText row.b ++ termsText row.c)

/-- Lean's program for an export, and whether it is the whole program or its
start. -/
structure Mirror where
  instrs : Array Instr
  whole : Bool

/-- Lean's instructions against Rust's, one by one. -/
def compareMirror (mirror : Mirror) (rust : Array Instr) : Except String String := do
  for (instr, k) in mirror.instrs.zipIdx do
    match rust[k]? with
    | none => throw s!"instruction {k}: Lean `{line instr}`, Rust's program has ended"
    | some r => unless line instr == line r do throw s!"instruction {k}: Lean `{line instr}`, Rust `{line r}`"
  if mirror.whole && mirror.instrs.size != rust.size then
    throw s!"Lean writes {mirror.instrs.size} instructions, Rust {rust.size}"
  pure s!"Lean's program is {if mirror.whole then "all" else "the first"} {mirror.instrs.size} of Rust's \
    {rust.size} instructions"

/-- Compare `mirror` with the IR, then the lowered run with the trace that
follows the program. -/
def check (mirror : Option Mirror) : Parser String := do
  keyword "statement"
  let stmt ← many (← nat) glWord
  let instrs ← program
  let mirrored ← match mirror.map (compareMirror · instrs) with
    | none => pure ""
    | some (.ok message) => pure s!"; {message}"
    | some (.error message) => throw message
  let source := ((mirror.filter (·.whole)).map (·.instrs)).getD instrs
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
def arguments : List String → IO (Except String (String × Option Mirror × Option Layer1Input))
  | [path] => pure (.ok (path, none, none))
  | [path, "shared"] => pure (.ok (path, some ⟨shared.program, true⟩, none))
  | [path, "early"] => pure (.ok (path, some ⟨early.program, true⟩, none))
  | [path, "layer1", input] => do
      let json ← IO.FS.readFile input
      pure <| (Lean.Json.parse json >>= Lean.fromJson? (α := Layer1Input)).map fun input =>
        (path, some ⟨(layer1 input).program, true⟩, some input)
  | [path, "linear", input] => do
      let json ← IO.FS.readFile input
      pure <| (Lean.Json.parse json >>= Lean.fromJson? (α := StageInput)).map fun input =>
        (path, some ⟨(linear input).program, true⟩, none)
  | [path, "honest", input] => do
      let json ← IO.FS.readFile input
      pure <| (Lean.Json.parse json >>= Lean.fromJson? (α := StageInput)).map fun input =>
        (path, some ⟨(honest input).program, true⟩, none)
  | [path, "late", input] => do
      let json ← IO.FS.readFile input
      pure <| (Lean.Json.parse json >>= Lean.fromJson? (α := StageInput)).map fun input =>
        (path, some ⟨(late input).program, true⟩, none)
  | [path, "lane_split"] => pure (.ok (path, some ⟨laneSplit.program, true⟩, none))
  | [path, "openings", input] => do
      let json ← IO.FS.readFile input
      pure <| (Lean.Json.parse json >>= Lean.fromJson? (α := OpeningsJson)).map fun input =>
        (path, some ⟨(openings input).program, true⟩, none)
  | _ => pure (.error "usage: lowerCheck <export.txt> [shared | early | layer1 <layer1.json> | \
      linear <linear.json> | honest <honest.json> | late <late.json> | lane_split | openings <openings.json>]")

end CompressionLeanExe.LowerCheck

open CompressionLeanExe.LowerCheck in
def main (args : List String) : IO UInt32 := do
  let (path, mirror, key) ← match ← arguments args with
    | .ok parsed => pure parsed
    | .error message => IO.eprintln message; return 2
  if let some key := key then
    IO.println (match CompressionLean.Final.keyFailure key with
      | some fact => s!"key fact fails: {fact}"
      | none => "key facts hold")
    IO.println (if CompressionLean.Final.poseidon2Fits key then "key width fits the Poseidon2 hash chain"
      else "key width does not fit the Poseidon2 hash chain")
  match ((check mirror).run 0).run (← IO.FS.readBinFile path) with
  | .ok (message, _) => IO.println message; return 0
  | .error message => IO.println s!"mismatch: {message}"; return 1
