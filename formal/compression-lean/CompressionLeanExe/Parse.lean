import CompressionLean.Ports.Circuit.Ir

/-!
Owns the format of the IR exports that `neo-spartan`'s
`tests/internal/lowering.rs` and `nightstream`'s
`tests/lifecycle_native/terminal_ir.rs` write: whitespace-separated tokens,
one instruction per opcode (`Log` in both files). It reads an export, writes
an instruction as `Log` does (`line`) and compares Lean's program with
Rust's (`samePrograms`). It also reads the JSON files of those tests
(`readJson`).
-/

namespace CompressionLeanExe.Parse

open CompressionLean CompressionLean.Shrink CompressionLean.Circuit
open CompressionLean.Replay (Word)
open NightstreamFPrime.Spec

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

def glWord : Parser Word := do
  let value ← nat
  if value < goldilocksModulus then pure (Word.ofNat value)
  else throw s!"word {value} is not a canonical Goldilocks value"

/-- `count` items, read in a loop: the terminal export has lists of millions. -/
def many {α : Type} (count : ℕ) (item : Parser α) : Parser (List α) := do
  let mut out := Array.mkEmpty count
  for _ in [0:count] do
    out := out.push (← item)
  return out.toList

def coordinates (n : ℕ) : Parser (Fin n → Var) := do
  let handles := (← many n nat).toArray
  pure fun k => handles.getD k 0

def term : Parser (ℕ × Word) := do
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
        pure (⟨a, b, ← side⟩ : R1csRow Word)
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

def expect {α : Type} [BEq α] [Repr α] (what : String) (lean rust : α) : Parser Unit :=
  if lean == rust then pure () else throw s!"{what}: Lean {repr lean}, Rust {repr rust}"

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

/-- Handles compare as their lists. -/
instance {n : ℕ} : BEq (Fin n → Var) := ⟨fun a b => List.ofFn a == List.ofFn b⟩

deriving instance BEq for R1csRow, Instr

/-- Lean's instructions against Rust's, one by one. -/
def samePrograms (lean rust : Array Instr) : Parser Unit := do
  for k in [0:max lean.size rust.size] do
    match lean[k]?, rust[k]? with
    | some l, some r => unless l == r do throw s!"instruction {k}: Lean `{line l}`, Rust `{line r}`"
    | _, _ => throw s!"Lean writes {lean.size} instructions, Rust {rust.size}"

/-- The JSON file at `path` as an `α`. -/
def readJson (α : Type) [Lean.FromJson α] (path : String) : IO (Except String α) := do
  pure (Lean.Json.parse (← IO.FS.readFile path) >>= Lean.fromJson?)

end CompressionLeanExe.Parse
