import CompressionLean.Circuit.Record
import CompressionLean.Circuit.Verifier
import NightstreamFPrime.Spec.GoldilocksPrime

/-!
The `lowerCheck` executable. It reads one export of `neo-spartan`'s
`tests/internal/lowering.rs` (the statement, the IR, then Rust's trace),
lowers the IR in Goldilocks and compares the run with the trace: the glue
cells, the blocks, the statement words and every glue row, side by side and
term by term (`Call.computedRows`, which is `Call.rows`). It prints `match`
and the counts, or the first difference and exits with 1.

With a program name it also writes that program with the Lean builder and
compares it with the IR, instruction by instruction, as `Log` writes them
(`line`):

- `shared`: the Lean side of `Shared`, every helper of `Algebra.lean`,
  `Hash.lean`, `Fold.lean` and `Sumcheck.lean`. The whole IR must match.
- `layer1 <input.json>`: the Lean side of `Layer1::run` for the relation in
  the file. Its instructions must be the first ones of the IR: the verifier
  is ported up to `verifier.rs:151`.
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

/-! ## Lean's own programs -/

def handlesText {n : ℕ} (a : Fin n → Var) : String :=
  String.join ((List.finRange n).map fun k => s!" {a k}")

def termsText (terms : List (ℕ × Gl)) : String :=
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

/-- `Shared::run` (`lowering.rs`): every shared helper, in Rust's order. -/
def shared : Build Unit := do
  let words ← (Array.range 14).mapM fun _ => Build.priv
  let x ← privateExts 4
  let scaled ← Build.scale words[0]! 3
  let lifted ← Algebra.lift scaled
  let weighted ← Algebra.extScaleConstant lifted 9
  let dot ← Algebra.dot (x.extract 0 2) (x.extract 2 4)
  let eq ← Algebra.eqEval (x.extract 0 2) (x.extract 2 4)
  let table ← Algebra.eqTable (x.extract 0 2)
  let at_ ← Algebra.eqAt (x.extract 1 4) 5
  let value ← Algebra.interpolate table weighted
  let horner ← Algebra.horner x dot
  let k ← KxVar.fromK (scaled, words[1]!)
  let one ← KxVar.one
  let zero ← KxVar.zero
  let k ← k.mul ⟨eq, at_⟩
  let k ← k.add one
  let k ← k.scale horner
  let k ← k.oneMinus
  let k ← k.add zero
  let bits := Array.ofFn (← Build.bits words[2]! 5)
  let power ← Algebra.powerFromBits 7 bits
  let leaf ← hashLeaf words
  let short ← hashLeaf (words.extract 0 3)
  let node ← compress leaf short
  let root ← merkleRoot leaf (bits.extract 0 2) #[short, node]
  let rest : Transcript Unit := do
    Duplex.observeSlice words
    Duplex.observeExt k.re
    let first ← Duplex.sample
    let drawn ← Duplex.sampleExt
    let index ← Duplex.sampleBits 10
    let more ← (Array.range 3).mapM fun _ => Duplex.sampleExt
    Duplex.observe power
    let (point, last) ← sumcheckReplay #[#[x[0]!, x[1]!], #[x[2]!, x[3]!, value]] drawn
    let fold ← FoldTranscript.new
    let fold ← fold.absorb words
    let fold ← fold.absorbConstants #[1, 2, 3]
    let (digest, fold) ← fold.squeezeDigest
    let seed ← (FoldTranscript.fromState fun lane => words[lane.val % 14]!).handoff
    Build.assertExtEqual last point[1]!
    Build.assertExtEqual k.im more[2]!
    Build.assertEqual first index[3]!
    for lane in List.finRange 4 do
      Build.assertEqual (root lane) (seed lane)
      Build.assertEqual (digest lane) (fold.statePrefix lane)
  let _ ← rest.run (← Duplex.new)

/-- What `layer1` reads (`lowering.rs`, `layer1.json`): the replay's relation
and the query sets of the two openings. -/
structure Layer1Input where
  relation : Replay.Relation
  p1Rows : OpeningRows
  p0Rows : OpeningRows
deriving Lean.FromJson

/-- `statement_input` (`tests/internal/verifier.rs`): every statement word,
in `ClaimWords::words` order. -/
def statementInput (rel : Replay.Relation) : Build ClaimView := do
  let ring := (Array.range Replay.D).mapM fun _ => Build.pub
  let pair : Build KVar := do
    let re ← Build.pub
    pure (re, ← Build.pub)
  let commitment ← (Array.range rel.kappa).mapM fun _ => ring
  let publicInput ← (Array.range rel.publicBlocks).mapM fun _ => ring
  let point ← (Array.range rel.pointVariables).mapM fun _ => pair
  let evalK ← (Array.range Replay.D).mapM fun _ => pair
  let evalA ← (Array.range rel.matrices).mapM fun _ => (Array.range Replay.D).mapM fun _ => pair
  pure { commitment, publicInput, point, evalK, evalA }

/-- `Layer1::run` (`tests/internal/shrink.rs`): the seed and the statement as
statement words, the proof view, then the verifier. -/
def layer1 (input : Layer1Input) : Build Unit := do
  let seed ← (Array.range 4).mapM fun _ => Build.pub
  let statement ← statementInput input.relation
  let _ ← ProofView.read input.relation input.p1Rows input.p0Rows
  verify input.relation (fun lane => seed[lane]!) statement

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
    {run.publics.length} statement words{mirrored}"

/-- The export's path and Lean's program for it, from the arguments. -/
def arguments : List String → IO (Except String (String × Option Mirror))
  | [path] => pure (.ok (path, none))
  | [path, "shared"] => pure (.ok (path, some ⟨shared.program, true⟩))
  | [path, "layer1", input] => do
      let json ← IO.FS.readFile input
      pure <| (Lean.Json.parse json >>= Lean.fromJson? (α := Layer1Input)).map fun input =>
        (path, some ⟨(layer1 input).program, false⟩)
  | _ => pure (.error "usage: lowerCheck <export.txt> [shared | layer1 <layer1.json>]")

end CompressionLean.LowerCheck

open CompressionLean.LowerCheck in
def main (args : List String) : IO UInt32 := do
  let (path, mirror) ← match ← arguments args with
    | .ok parsed => pure parsed
    | .error message => IO.eprintln message; return 2
  match ((check mirror).run 0).run (← IO.FS.readBinFile path) with
  | .ok (message, _) => IO.println message; return 0
  | .error message => IO.println s!"mismatch: {message}"; return 1
