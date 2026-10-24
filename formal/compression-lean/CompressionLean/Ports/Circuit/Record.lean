import CompressionLean.Ports.Circuit.Ir
import CompressionLean.Fields

/-!
Owns the lowering of a program (`lower`): an exact port of the constraint
reading of `Backend` (`crates/neo-spartan/src/circuit/record.rs`) from the
instructions of `Ir.lean` to the shrink run of `Ports/Shrink/Layout.lean`.

- Forms (`record.rs::Form`) are `Σ coefficient·wire + constant`. `combine`
  merges the terms of weighted forms in first-wire order and drops zero
  coefficients (`record.rs::combine`). A form of at most four terms stays a
  form; a longer one gets a glue cell and one linear row
  (`record.rs::Recorder::form`).
- Constant folding: a constant side of a product stays linear
  (`record.rs::Recorder::mul`, `Recorder::ext_mul`), a constant extension value with an
  inverse is inverted (`record.rs::Recorder::ext_inverse`), and a zero constant needs no
  assertion row (`record.rs::Recorder::assert_zero`). A constant zero extension value and a
  nonzero constant assertion still get their rows, which cannot hold.
- Numbering: glue cells, blocks and statement words in creation order, rows
  in emission order (`Recorder`, `record.rs`).
- A permutation (`record.rs::Recorder::block`, `Recorder::permute`) is one block and 16 rows that
  tie the block's first 16 cells to the inputs. Its outputs are cells 166 to
  181 (`circuit/poseidon2.rs::OUTPUT`).

The lowering computes in a field `F`. `lowerCheck`
(`CompressionLeanExe/LowerCheck.lean`) runs it in Goldilocks, where its zero
tests are Rust's, against the trace of `neo-spartan`'s
`tests/internal/lowering.rs`. The proofs about it are in `Lowering/`.
-/

namespace CompressionLean.Circuit

open CompressionLean.Shrink
open CompressionLean.Replay (Word)

/-- The cells of a permutation block (`poseidon2::CELLS`,
`circuit/poseidon2.rs::CELLS`). -/
def blockCells : ℕ := 182

/-- Its first output cell (`poseidon2::OUTPUT`, `circuit/poseidon2.rs::OUTPUT`). -/
def outputCell : ℕ := 166

section Ring

variable {F : Type} [CommRing F]

/-- The coordinates of `a·b` in `F[x]/(x³ - x - 1)`. -/
def extProduct (a b : Fin 3 → F) : Fin 3 → F :=
  ![a 0 * b 0 + a 1 * b 2 + a 2 * b 1,
    a 0 * b 1 + a 1 * (b 0 + b 2) + a 2 * (b 1 + b 2),
    a 0 * b 2 + a 1 * b 1 + a 2 * (b 0 + b 2)]

end Ring

section Lowering

variable {F : Type} [Field F] [DecidableEq F]

/-- Add `t` to the first term on its wire, or append it
(`record.rs::combine`). -/
def addTerm (t : Wire × F) : List (Wire × F) → List (Wire × F)
  | [] => [t]
  | u :: rest => if u.1 = t.1 then (u.1, u.2 + t.2) :: rest else u :: addTerm t rest

/-- The terms of `Σ weight·form`, merged in first-wire order
(`record.rs::combine`). -/
def mergeTerms (parts : List (Form F × F)) : List (Wire × F) :=
  parts.foldl (fun terms p => p.1.terms.foldl (fun terms t => addTerm (t.1, t.2 * p.2) terms) terms) []

/-- `combine` (`record.rs::combine`): the merged terms without zero
coefficients, and the constant. -/
def combine (parts : List (Form F × F)) : List (Wire × F) × F :=
  ((mergeTerms parts).filter (·.2 ≠ 0), (parts.map fun p => p.1.constant * p.2).sum)

/-- The terms with negated coefficients. -/
def negate (terms : List (Wire × F)) : List (Wire × F) := terms.map fun t => (t.1, -t.2)

/-- The recorder's state (`Recorder`, `record.rs::Recorder`) and the forms of
the values. `publics` counts the statement words read; Rust's counter starts
at the first word's wire, 1. -/
structure State (F : Type) where
  values : Array (Form F) := #[]
  calls : Array (Call F) := #[]
  cells : ℕ := 0
  blocks : ℕ := 0
  publics : ℕ := 0

/-- The lowering monad. -/
abbrev Lower (F : Type) := StateM (State F)

/-- A new glue cell (`Recorder::cell`, `record.rs::Recorder::cell`). -/
def cell : Lower F (Form F) :=
  modifyGet fun s => (.wire (.glue s.cells), { s with cells := s.cells + 1 })

/-- `count` new glue cells; returns the first. -/
def cells (count : ℕ) : Lower F ℕ :=
  modifyGet fun s => (s.cells, { s with cells := s.cells + count })

/-- One glue row (`Recorder::row`, `record.rs::Recorder::row`). -/
def row (products : List (List (Wire × F) × List (Wire × F))) (linear : List (Wire × F))
    (three : products.length ≤ 3) : Lower F Unit :=
  modify fun s => { s with calls := s.calls.push (.row products linear three) }

/-- `Recorder::form` (`record.rs::Recorder::form`): at most four terms stay a form;
a longer one gets a cell and the row `cell - Σ terms - constant = 0`. -/
def form (terms : List (Wire × F)) (constant : F) : Lower F (Form F) :=
  if terms.length ≤ 4 then pure ⟨terms, constant⟩
  else do
    let c ← cell
    row [] (c.terms ++ negate terms ++ if constant = 0 then [] else [(.pub 0, -constant)]) (by simp)
    pure c

/-- `Recorder::linear` (`record.rs::Recorder::linear`). -/
def linear (parts : List (Form F × F)) : Lower F (Form F) :=
  let c := combine parts
  form c.1 c.2

/-- `mul` (`record.rs::Recorder::mul`). -/
def mul (a b : Form F) : Lower F (Form F) :=
  if a.isConstant then linear [(b, a.constant)]
  else if b.isConstant then linear [(a, b.constant)]
  else do
    let p ← cell
    row [(a.rowTerms, b.rowTerms)] p.terms (by simp)
    pure p

/-- `Σ forms` as row terms, the constant on the one column (`sum` in
`ext_rows`, `record.rs::Recorder::ext_rows`). -/
def sumTerms (forms : List (Form F)) : List (Wire × F) :=
  let c := combine (forms.map fun f => (f, 1))
  c.1 ++ if c.2 = 0 then [] else [(.pub 0, c.2)]

/-- The products of row `k` of `a·b` with `x³ = x + 1` (`ext_rows`,
`record.rs::Recorder::ext_rows`). -/
def extRow (a b : Fin 3 → Form F) : Fin 3 → List (List (Wire × F) × List (Wire × F)) :=
  let pair (i : Fin 3) (forms : List (Form F)) := ((a i).rowTerms, sumTerms forms)
  ![[pair 0 [b 0], pair 1 [b 2], pair 2 [b 1]],
    [pair 0 [b 1], pair 1 [b 0, b 2], pair 2 [b 1, b 2]],
    [pair 0 [b 2], pair 1 [b 1], pair 2 [b 0, b 2]]]

/-- The inverse in `F[x]/(x³ - x - 1)`: Cramer's rule on multiplication by
`a`, none when its determinant is zero. Over Goldilocks the determinant is
the norm of `a`, zero only at zero (`Lib.cubic_irreducible`), so this is
`try_inverse`. -/
def extInverse (a : Fin 3 → F) : Option (Fin 3 → F) :=
  let c0 := (a 0 + a 2) * (a 0 + a 2) - a 1 * (a 1 + a 2)
  let c1 := a 2 * (a 1 + a 2) - a 1 * (a 0 + a 2)
  let c2 := a 1 * a 1 - a 2 * (a 0 + a 2)
  let det := a 0 * c0 + a 2 * c1 + a 1 * c2
  if det = 0 then none else some ![c0 / det, c1 / det, c2 / det]

/-- Whether every coordinate is a constant. -/
def constants (a : Fin 3 → Form F) : Bool := (a 0).isConstant && (a 1).isConstant && (a 2).isConstant

/-- One product cell and its row (`record.rs::Recorder::ext_mul`). -/
def extCell (a b : Fin 3 → Form F) (k : Fin 3) : Lower F (Form F) := do
  let c ← cell
  row (extRow a b k) c.terms (by fin_cases k <;> simp [extRow])
  pure c

/-- `std::array::from_fn` over three coordinates: the calls in order. -/
def fromFn (m : Fin 3 → Lower F (Form F)) : Lower F (Fin 3 → Form F) := do
  let x ← m 0
  let y ← m 1
  let z ← m 2
  pure ![x, y, z]

/-- `ext_mul` (`record.rs::Recorder::ext_mul`): a constant side gives three linear
forms, by the columns of multiplication by it; otherwise three cells and
three rows. -/
def extMul (a b : Fin 3 → Form F) : Lower F (Fin 3 → Form F) :=
  if constants a || constants b then
    let c := if constants a then a else b
    let other := if constants a then b else a
    let column (j : Fin 3) := extProduct (fun i => (c i).constant) fun i => if i = j then 1 else 0
    fromFn fun k => linear [(other 0, column 0 k), (other 1, column 1 k), (other 2, column 2 k)]
  else fromFn (extCell a b)

/-- `ext_inverse` (`record.rs::Recorder::ext_inverse`): a constant with an inverse folds;
otherwise three cells and the rows `a·inverse = 1`. A constant zero falls
through, and its rows cannot hold. -/
def extInv (a : Fin 3 → Form F) : Lower F (Fin 3 → Form F) :=
  match if constants a then extInverse fun i => (a i).constant else none with
  | some v => pure fun i => .const (v i)
  | none => do
      let inverse ← fromFn fun _ => cell
      row (extRow a inverse 0) [(.pub 0, 1)] (by simp [extRow])
      row (extRow a inverse 1) [] (by simp [extRow])
      row (extRow a inverse 2) [] (by simp [extRow])
      pure inverse

/-- `assert_zero` (`record.rs::Recorder::assert_zero`). -/
def assertZero (a : Form F) : Lower F Unit :=
  if a.isConstant ∧ a.constant = 0 then pure () else row [] a.rowTerms (by simp)

/-- One boolean row per bit cell (`record.rs::Recorder::bits`). -/
def booleanRows : List ℕ → Lower F Unit
  | [] => pure ()
  | c :: rest => do
      row [([(.glue c, 1)], [(.glue c, 1)])] [(.glue c, 1)] (by simp)
      booleanRows rest

/-- `high` times each bit in turn (`record.rs::Recorder::bits`). -/
def chain (high : Form F) : List ℕ → Lower F (Form F)
  | [] => pure high
  | c :: rest => do chain (← mul high (.wire (.glue c))) rest

/-- `bits` (`record.rs::Recorder::bits`): `count` bit cells, a boolean row each, the
row `a = Σ 2^i·bit_i`, and for 64 bits the row by which the high 32 bits all
one force the low 32 to zero. The weights are Rust's up to 64 bits, the
widest split `record.rs` takes. -/
def bits (a : Form F) (count : ℕ) : Lower F (List (Form F)) := do
  let first ← cells count
  booleanRows (List.range' first count)
  row [] (a.rowTerms ++ (List.range count).map fun i => (.glue (first + i), -((2 : F) ^ i))) (by simp)
  if count = 64 then
    let high ← chain (.wire (.glue (first + 32))) (List.range' (first + 33) 31)
    row [(high.rowTerms, (List.range' first 32).map fun c => (.glue c, 1))] [] (by simp)
  pure ((List.range' first count).map fun c => .wire (.glue c))

/-- The rows that tie the first cells of block `b` to the inputs
(`record.rs::Recorder::block`). -/
def tieRows (b : ℕ) (inputs : Fin 16 → Form F) : List (Fin 16) → Lower F Unit
  | [] => pure ()
  | k :: rest => do
      row [] ((.block b k, 1) :: negate (inputs k).rowTerms) (by simp)
      tieRows b inputs rest

/-- A new block (`record.rs::Recorder::block`). -/
def block : Lower F ℕ :=
  modifyGet fun s => (s.blocks, { s with blocks := s.blocks + 1 })

/-- One permutation block (`Recorder::block`, `record.rs::Recorder::block`). -/
def permute (inputs : Fin 16 → Form F) : Lower F (Fin 16 → Form F) := do
  let b ← block
  tieRows b inputs (List.finRange 16)
  pure fun k => .wire (.block b (outputCell + k))

/-- The next statement word (`Recorder::public`, `record.rs::Recorder::public`): wire
`i + 1`, since wire `0` is the constant one. The lowering reads no
statement value; `lower` attaches them to the run. -/
def statementWord : Lower F (Form F) :=
  modifyGet fun s => (.wire (.pub (s.publics + 1)), { s with publics := s.publics + 1 })

/-- `r1cs` (`record.rs::Recorder::r1cs`): the system's rows over the column forms. -/
def r1cs (rows : List (R1csRow F)) (columns : List (Form F)) : Lower F Unit :=
  modify fun s => { s with calls := s.calls.push (.r1cs rows columns) }

/-- The form of value `v`; a handle past the run reads the zero form. -/
def readValue (v : Var) : Lower F (Form F) := do
  return (← get).values.getD v (.const 0)

/-- The forms of the values `vs`. The reads finish before the lowering
changes the state, so the value table is never copied. -/
def readValues (vs : List Var) : Lower F (List (Form F)) := do
  let values := (← get).values
  return vs.map fun v => values.getD v (.const 0)

/-- The forms of `n` values, as `readValues`. -/
def readCoordinates {n : ℕ} (a : Fin n → Var) : Lower F (Fin n → Form F) := do
  let values := (← get).values
  let forms := Array.ofFn fun k => values.getD (a k) (.const 0)
  return fun k => forms.getD k (.const 0)

/-- Append values. -/
def push (forms : List (Form F)) : Lower F Unit :=
  modify fun s => { s with values := s.values ++ forms.toArray }

/-- An R1CS row with its coefficients in `F`. -/
def R1csRow.inField (r : R1csRow Word) : R1csRow F :=
  ⟨r.a.map fun t => (t.1, Word.cast t.2), r.b.map fun t => (t.1, Word.cast t.2), r.c.map fun t => (t.1, Word.cast t.2)⟩

/-- One instruction (`Backend for Recorder`, `record.rs`). -/
def step : Instr → Lower F Unit
  | .const value => push [.const (Word.cast value)]
  | .priv => do push [← cell]
  | .pub => do push [← statementWord]
  | .linear parts => do
      let forms ← readValues (parts.map (·.1))
      push [← linear (forms.zip (parts.map fun p => Word.cast p.2))]
  | .mul a b => do push [← mul (← readValue a) (← readValue b)]
  | .extMul a b => do
      let p ← extMul (← readCoordinates a) (← readCoordinates b)
      push [p 0, p 1, p 2]
  | .extInv a => do
      let p ← extInv (← readCoordinates a)
      push [p 0, p 1, p 2]
  | .assertZero a => do assertZero (← readValue a)
  | .bits a count => do push (← bits (← readValue a) count)
  | .permute state => do push (List.ofFn (← permute (← readCoordinates state)))
  | .hint count => do
      let first ← cells count
      push ((List.range' first count).map fun c => .wire (.glue c))
  | .r1cs rows columns => do r1cs (rows.map R1csRow.inField) (← readValues columns)

/-- Lower the instructions from state `s` on. -/
def steps : List Instr → State F → State F
  | [], s => s
  | i :: rest, s => steps rest ((step i).run s).2

/-- **The lowering**: the run that `record.rs` records for program `L` on
the statement words `stmt`. The shape does not read `stmt`; word `i` of the
run is `stmt`'s word `i`, as the sink receives it (`Sink::public`, `record.rs::Sink::public`). -/
def lower (L : Array Instr) (stmt : List F) : Run F :=
  let s := steps L.toList {}
  ⟨s.calls.toList, s.cells, s.blocks, (List.range s.publics).map fun i => stmt.getD i 0⟩

end Lowering

end CompressionLean.Circuit
