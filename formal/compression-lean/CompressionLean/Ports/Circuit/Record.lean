import CompressionLean.Ports.Circuit.Ir
import CompressionLean.Fields

namespace CompressionLean.Circuit

open CompressionLean.Shrink
open CompressionLean.Replay (Word)

def blockCells : ℕ := 182

def outputCell : ℕ := 166

section Ring

variable {F : Type} [CommRing F]

def extProduct (a b : Fin 3 → F) : Fin 3 → F :=
  ![a 0 * b 0 + a 1 * b 2 + a 2 * b 1,
    a 0 * b 1 + a 1 * (b 0 + b 2) + a 2 * (b 1 + b 2),
    a 0 * b 2 + a 1 * b 1 + a 2 * (b 0 + b 2)]

end Ring

section Lowering

variable {F : Type} [Field F] [DecidableEq F]

def addTerm (t : Wire × F) : List (Wire × F) → List (Wire × F)
  | [] => [t]
  | u :: rest => if u.1 = t.1 then (u.1, u.2 + t.2) :: rest else u :: addTerm t rest

def mergeTerms (parts : List (Form F × F)) : List (Wire × F) :=
  parts.foldl (fun terms p => p.1.terms.foldl (fun terms t => addTerm (t.1, t.2 * p.2) terms) terms) []

def combine (parts : List (Form F × F)) : List (Wire × F) × F :=
  ((mergeTerms parts).filter (·.2 ≠ 0), (parts.map fun p => p.1.constant * p.2).sum)

def negate (terms : List (Wire × F)) : List (Wire × F) := terms.map fun t => (t.1, -t.2)

structure State (F : Type) where
  values : Array (Form F) := #[]
  calls : Array (Call F) := #[]
  cells : ℕ := 0
  blocks : ℕ := 0
  publics : ℕ := 0

abbrev Lower (F : Type) := StateM (State F)

def cell : Lower F (Form F) :=
  modifyGet fun s => (.wire (.glue s.cells), { s with cells := s.cells + 1 })

def cells (count : ℕ) : Lower F ℕ :=
  modifyGet fun s => (s.cells, { s with cells := s.cells + count })

def row (products : List (List (Wire × F) × List (Wire × F))) (linear : List (Wire × F))
    (three : products.length ≤ 3) : Lower F Unit :=
  modify fun s => { s with calls := s.calls.push (.row products linear three) }

def form (terms : List (Wire × F)) (constant : F) : Lower F (Form F) :=
  if terms.length ≤ 4 then pure ⟨terms, constant⟩
  else do
    let c ← cell
    row [] (c.terms ++ negate terms ++ if constant = 0 then [] else [(.pub 0, -constant)]) (by simp)
    pure c

def linear (parts : List (Form F × F)) : Lower F (Form F) :=
  let c := combine parts
  form c.1 c.2

def mul (a b : Form F) : Lower F (Form F) :=
  if a.isConstant then linear [(b, a.constant)]
  else if b.isConstant then linear [(a, b.constant)]
  else do
    let p ← cell
    row [(a.rowTerms, b.rowTerms)] p.terms (by simp)
    pure p

def sumTerms (forms : List (Form F)) : List (Wire × F) :=
  let c := combine (forms.map fun f => (f, 1))
  c.1 ++ if c.2 = 0 then [] else [(.pub 0, c.2)]

def extRow (a b : Fin 3 → Form F) : Fin 3 → List (List (Wire × F) × List (Wire × F)) :=
  let pair (i : Fin 3) (forms : List (Form F)) := ((a i).rowTerms, sumTerms forms)
  ![[pair 0 [b 0], pair 1 [b 2], pair 2 [b 1]],
    [pair 0 [b 1], pair 1 [b 0, b 2], pair 2 [b 1, b 2]],
    [pair 0 [b 2], pair 1 [b 1], pair 2 [b 0, b 2]]]

def extInverse (a : Fin 3 → F) : Option (Fin 3 → F) :=
  let c0 := (a 0 + a 2) * (a 0 + a 2) - a 1 * (a 1 + a 2)
  let c1 := a 2 * (a 1 + a 2) - a 1 * (a 0 + a 2)
  let c2 := a 1 * a 1 - a 2 * (a 0 + a 2)
  let det := a 0 * c0 + a 2 * c1 + a 1 * c2
  if det = 0 then none else some ![c0 / det, c1 / det, c2 / det]

def constants (a : Fin 3 → Form F) : Bool := (a 0).isConstant && (a 1).isConstant && (a 2).isConstant

def extCell (a b : Fin 3 → Form F) (k : Fin 3) : Lower F (Form F) := do
  let c ← cell
  row (extRow a b k) c.terms (by fin_cases k <;> simp [extRow])
  pure c

def fromFn (m : Fin 3 → Lower F (Form F)) : Lower F (Fin 3 → Form F) := do
  let x ← m 0
  let y ← m 1
  let z ← m 2
  pure ![x, y, z]

def extMul (a b : Fin 3 → Form F) : Lower F (Fin 3 → Form F) :=
  if constants a || constants b then
    let c := if constants a then a else b
    let other := if constants a then b else a
    let column (j : Fin 3) := extProduct (fun i => (c i).constant) fun i => if i = j then 1 else 0
    fromFn fun k => linear [(other 0, column 0 k), (other 1, column 1 k), (other 2, column 2 k)]
  else fromFn (extCell a b)

def extInv (a : Fin 3 → Form F) : Lower F (Fin 3 → Form F) :=
  match if constants a then extInverse fun i => (a i).constant else none with
  | some v => pure fun i => .const (v i)
  | none => do
      let inverse ← fromFn fun _ => cell
      row (extRow a inverse 0) [(.pub 0, 1)] (by simp [extRow])
      row (extRow a inverse 1) [] (by simp [extRow])
      row (extRow a inverse 2) [] (by simp [extRow])
      pure inverse

def assertZero (a : Form F) : Lower F Unit :=
  if a.isConstant ∧ a.constant = 0 then pure () else row [] a.rowTerms (by simp)

def booleanRows : List ℕ → Lower F Unit
  | [] => pure ()
  | c :: rest => do
      row [([(.glue c, 1)], [(.glue c, 1)])] [(.glue c, 1)] (by simp)
      booleanRows rest

def chain (high : Form F) : List ℕ → Lower F (Form F)
  | [] => pure high
  | c :: rest => do chain (← mul high (.wire (.glue c))) rest

def bits (a : Form F) (count : ℕ) : Lower F (List (Form F)) := do
  let first ← cells count
  booleanRows (List.range' first count)
  row [] (a.rowTerms ++ (List.range count).map fun i => (.glue (first + i), -((2 : F) ^ i))) (by simp)
  if count = 64 then
    let high ← chain (.wire (.glue (first + 32))) (List.range' (first + 33) 31)
    row [(high.rowTerms, (List.range' first 32).map fun c => (.glue c, 1))] [] (by simp)
  pure ((List.range' first count).map fun c => .wire (.glue c))

def tieRows (b : ℕ) (inputs : Fin 16 → Form F) : List (Fin 16) → Lower F Unit
  | [] => pure ()
  | k :: rest => do
      row [] ((.block b k, 1) :: negate (inputs k).rowTerms) (by simp)
      tieRows b inputs rest

def block : Lower F ℕ :=
  modifyGet fun s => (s.blocks, { s with blocks := s.blocks + 1 })

def permute (inputs : Fin 16 → Form F) : Lower F (Fin 16 → Form F) := do
  let b ← block
  tieRows b inputs (List.finRange 16)
  pure fun k => .wire (.block b (outputCell + k))

def statementWord : Lower F (Form F) :=
  modifyGet fun s => (.wire (.pub (s.publics + 1)), { s with publics := s.publics + 1 })

def r1cs (rows : List (R1csRow F)) (columns : List (Form F)) : Lower F Unit :=
  modify fun s => { s with calls := s.calls.push (.r1cs rows columns) }

def readValue (v : Var) : Lower F (Form F) := do
  return (← get).values.getD v (.const 0)

def readValues (vs : List Var) : Lower F (List (Form F)) := do
  let values := (← get).values
  return vs.map fun v => values.getD v (.const 0)

def readCoordinates {n : ℕ} (a : Fin n → Var) : Lower F (Fin n → Form F) := do
  let values := (← get).values
  let forms := Array.ofFn fun k => values.getD (a k) (.const 0)
  return fun k => forms.getD k (.const 0)

def push (forms : List (Form F)) : Lower F Unit :=
  modify fun s => { s with values := s.values ++ forms.toArray }

def R1csRow.inField (r : R1csRow Word) : R1csRow F :=
  ⟨r.a.map fun t => (t.1, Word.cast t.2), r.b.map fun t => (t.1, Word.cast t.2), r.c.map fun t => (t.1, Word.cast t.2)⟩

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

def steps : List Instr → State F → State F
  | [], s => s
  | i :: rest, s => steps rest ((step i).run s).2

def lower (L : Array Instr) (stmt : List F) : Run F :=
  let s := steps L.toList {}
  ⟨s.calls.toList, s.cells, s.blocks, (List.range s.publics).map fun i => stmt.getD i 0⟩

end Lowering

end CompressionLean.Circuit
