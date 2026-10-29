import CompressionLean.Circuit.Ir

namespace CompressionLean.Circuit.Parse

open CompressionLean.Shrink CompressionLean.Circuit
open CompressionLean.Replay (Gl)
open NightstreamFPrime.Spec

abbrev Parser := StateT ℕ (ReaderT ByteArray (Except String))

def isSpace (b : UInt8) : Bool := b == 32 || b == 10 || b == 13 || b == 9

def skipSpaces (data : ByteArray) (pos : ℕ) : ℕ :=
  if h : pos < data.size then if isSpace data[pos] then skipSpaces data (pos + 1) else pos else pos
termination_by data.size - pos

def tokenEnd (data : ByteArray) (pos : ℕ) : ℕ :=
  if h : pos < data.size then if isSpace data[pos] then pos else tokenEnd data (pos + 1) else pos
termination_by data.size - pos

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

def many {α : Type} (count : ℕ) (item : Parser α) : Parser (List α) := do
  let mut out := Array.mkEmpty count
  for _ in [0:count] do
    out := out.push (← item)
  return out.toList

def coordinates (n : ℕ) : Parser (Fin n → Var) := do
  let handles := (← many n nat).toArray
  pure fun k => handles.getD k 0

def term : Parser (ℕ × Gl) := do
  let index ← nat
  pure (index, ← glWord)

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

def program : Parser (Array Instr) := do
  let mut out := #[]
  for _ in [0:(← readThe ByteArray).size] do
    let op ← text
    if op = "end" then return out
    out := out.push (← instr op)
  throw "the program has no end"

def expect {α : Type} [BEq α] [Repr α] (what : String) (lean rust : α) : Parser Unit :=
  if lean == rust then pure () else throw s!"{what}: Lean {repr lean}, Rust {repr rust}"

end CompressionLean.Circuit.Parse
