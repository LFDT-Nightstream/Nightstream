import CompressionLean.Circuit.Record
import NightstreamFPrime.Spec.GoldilocksPrime

/-!
The `lowerCheck` executable. It reads one export of `neo-spartan`'s
`tests/internal/lowering.rs` (the statement, the IR, then Rust's trace),
lowers the IR in Goldilocks and compares the run with the trace: the glue
cells, the blocks, the statement words and every glue row, side by side and
term by term (`Call.computedRows`, which is `Call.rows`). It prints `match`
and the counts, or the first difference and exits with 1.
-/

namespace CompressionLean.LowerCheck

open CompressionLean.Shrink CompressionLean.Circuit NightstreamFPrime.Spec
open CompressionLean.Replay (Gl)

instance : Fact (Nat.Prime goldilocksModulus) := ⟨GoldilocksPrime.goldilocks_natPrime⟩

/-- Goldilocks: there the lowering's zero tests are Rust's. -/
abbrev G := ZMod goldilocksModulus

/-- Whitespace-separated tokens of the export. -/
abbrev Parser := StateT ℕ (ReaderT ByteArray (Except String))

def isSpace (b : UInt8) : Bool := b == 32 || b == 10 || b == 13 || b == 9

def skipSpaces (data : ByteArray) (pos : ℕ) : ℕ :=
  if h : pos < data.size then if isSpace data[pos] then skipSpaces data (pos + 1) else pos else pos
termination_by data.size - pos

def tokenEnd (data : ByteArray) (pos : ℕ) : ℕ :=
  if h : pos < data.size then if isSpace data[pos] then pos else tokenEnd data (pos + 1) else pos
termination_by data.size - pos

/-- The next token's bytes. -/
def token : Parser ByteArray := do
  let data ← readThe ByteArray
  let start := skipSpaces data (← get)
  let stop := tokenEnd data start
  set stop
  pure (data.extract start stop)

def text : Parser String := do
  pure (String.fromUTF8! (← token))

def nat : Parser ℕ := do
  let bytes ← token
  if bytes.size = 0 then throw "expected a number, found the end"
  bytes.foldlM (init := 0) fun value b =>
    if 48 ≤ b ∧ b ≤ 57 then pure (10 * value + (b - 48).toNat)
    else throw s!"expected a number, found {String.fromUTF8! bytes}"

def keyword (expected : String) : Parser Unit := do
  let got ← text
  if got ≠ expected then throw s!"expected {expected}, found {got}"

def glWord : Parser Gl := do
  let value ← nat
  if value < goldilocksModulus then pure (Gl.ofNat value)
  else throw s!"word {value} is not a canonical Goldilocks value"

def many {α : Type} (count : ℕ) (item : Parser α) : Parser (List α) :=
  (List.range count).mapM fun _ => item

def coordinates (n : ℕ) : Parser (Fin n → Var) := do
  let handles := (← many n nat).toArray
  pure fun k => handles.getD k 0

def term : Parser (ℕ × Gl) := do
  let index ← nat
  pure (index, ← glWord)

/-- One instruction after its opcode (`Log`, `lowering.rs`). -/
def instr : String → Parser Instr
  | "c" => do pure (.const (← glWord))
  | "p" => pure .priv
  | "u" => pure .pub
  | "l" => do pure (.linear (← many (← nat) term))
  | "m" => do
      let a ← nat
      pure (.mul a (← nat))
  | "x" => do
      let a ← coordinates 3
      pure (.extMul a (← coordinates 3))
  | "i" => do pure (.extInv (← coordinates 3))
  | "z" => do pure (.assertZero (← nat))
  | "b" => do
      let a ← nat
      pure (.bits a (← nat))
  | "h" => do pure (.hint (← nat))
  | "q" => do pure (.permute (← coordinates 16))
  | "r" => do
      let rows ← nat
      let columns ← many (← nat) nat
      let side := do many (← nat) term
      let rows ← many rows do
        let a ← side
        let b ← side
        pure (⟨a, b, ← side⟩ : R1csRow Gl)
      pure (.r1cs rows columns)
  | other => throw s!"unknown instruction {other}"

/-- The instructions up to `end`. -/
def program : Parser (Array Instr) := do
  let mut out := #[]
  for _ in [0:(← readThe ByteArray).size] do
    let op ← text
    if op = "end" then return out
    out := out.push (← instr op)
  throw "the program has no end"

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

def expect {α : Type} [BEq α] [Repr α] (what : String) (lean rust : α) : Parser Unit :=
  if lean == rust then pure () else throw s!"{what}: Lean {repr lean}, Rust {repr rust}"

/-- Compare the lowered run with the trace that follows the program. -/
def check : Parser String := do
  keyword "statement"
  let stmt ← many (← nat) glWord
  let instrs ← program
  let run := lower (F := G) instrs (stmt.map fun g => (g.toNat : G))
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
    {run.publics.length} statement words"

end CompressionLean.LowerCheck

open CompressionLean.LowerCheck in
def main (args : List String) : IO UInt32 := do
  let [path] := args | IO.eprintln "usage: lowerCheck <export.txt>"; return 2
  match (check.run 0).run (← IO.FS.readBinFile path) with
  | .ok (message, _) => IO.println message; return 0
  | .error message => IO.println s!"mismatch: {message}"; return 1
