import CompressionLean.Circuit.Record

/-!
Owns the executor reading of a program (`Exec`): the values that its
instructions compute, and the checks that an accepting run passes.

- Value `v` is the `v`-th value of the run, as in `Ir.lean`. An instruction
  reads earlier values and appends its outputs.
- The tape holds the free values. Tape word `v` is the value of handle `v`,
  and the executor reads it only at the outputs of `priv`, `hint`, `bits` and
  `extInv`.
- The executor computes the other values: constants, the statement words in
  order, linear combinations, products, products in `F[x]/(x³ - x - 1)`, and
  the permutation `perm` of the 16 inputs of a block.
- The checks: every `assertZero` value is zero, every `bits` decomposition
  is a canonical binary decomposition (`Bits`), every `extInv` output is an
  inverse, and every row of every `r1cs` system holds.

The executor is arithmetic, as the recorder is: `select` is
`whenZero + bit·(whenOne - whenZero)` (`Ir.lean`).
-/

namespace CompressionLean.Circuit

open CompressionLean.Shrink
open CompressionLean.Replay (Word)

variable {F : Type} [CommRing F]

/-- `b` is a decomposition of `a` into `count` bits (`record.rs:409-437`): each
bit is 0 or 1, `a = Σ 2^i·b_i`, and 64 bits are canonical, so that when the
high 32 bits are all one, the low 32 bits are zero. -/
structure Bits (a : F) (count : ℕ) (b : ℕ → F) : Prop where
  boolean : ∀ i < count, b i = 0 ∨ b i = 1
  sum : a = ((List.range count).map fun i => 2 ^ i * b i).sum
  canonical : count = 64 → (∀ i, 32 ≤ i → i < 64 → b i = 1) → ∀ i < 32, b i = 0

/-- Row `r` of a fixed R1CS system holds on the column values `column`:
`(Σ a)·(Σ b) = Σ c` (`R1cs::sides`, `circuit/r1cs.rs:59-65`). -/
def R1csRow.Holds (column : ℕ → F) (r : R1csRow Word) : Prop :=
  let side (terms : List (ℕ × Word)) := (terms.map fun t => column t.1 * word t.2).sum
  side r.a * side r.b = side r.c

/-- The executor's state after a prefix of the program. -/
structure Exec.State (F : Type) where
  /-- The values so far; value `v` is handle `v`. -/
  values : List F
  /-- The statement words read. -/
  reads : ℕ
  /-- Every check so far holds. -/
  holds : Prop

namespace Exec

/-- No value, no statement word read, no check. -/
def start : State F := ⟨[], 0, True⟩

/-- Value `v`. A handle past the run reads zero, as the lowering reads the
zero form there. -/
def State.value (s : State F) (v : Var) : F := s.values.getD v 0

/-- Append `outputs` and add `check`. -/
def State.next (s : State F) (outputs : List F) (check : Prop) : State F :=
  ⟨s.values ++ outputs, s.reads, s.holds ∧ check⟩

variable (perm : List F → List F) (stmt : List F) (tape : ℕ → F)

/-- One instruction (`Backend for Recorder`, `record.rs:285-474`). Its first
output is handle `s.values.length`. -/
def step (s : State F) : Instr → State F
  | .const value => s.next [word value] True
  | .priv => s.next [tape s.values.length] True
  | .pub => ⟨s.values ++ [stmt.getD s.reads 0], s.reads + 1, s.holds⟩
  | .linear parts => s.next [(parts.map fun p => s.value p.1 * word p.2).sum] True
  | .mul a b => s.next [s.value a * s.value b] True
  | .extMul a b => s.next (List.ofFn (extProduct (fun i => s.value (a i)) fun i => s.value (b i))) True
  | .extInv a =>
      let inverse (k : Fin 3) := tape (s.values.length + k)
      s.next (List.ofFn inverse) (extProduct (fun i => s.value (a i)) inverse = ![1, 0, 0])
  | .assertZero a => s.next [] (s.value a = 0)
  | .bits a count =>
      let bit (i : ℕ) := tape (s.values.length + i)
      s.next ((List.range count).map bit) (Bits (s.value a) count bit)
  | .permute state =>
      let out := perm (List.ofFn fun k : Fin 16 => s.value (state k))
      s.next (List.ofFn fun k : Fin 16 => out.getD k 0) True
  | .hint count => s.next ((List.range count).map fun i => tape (s.values.length + i)) True
  | .r1cs rows columns => s.next [] (∀ r ∈ rows, R1csRow.Holds (fun j => (columns.map s.value).getD j 0) r)

/-- The state after the program `L`. -/
def run (L : Array Instr) : State F := L.toList.foldl (step perm stmt tape) start

end Exec

/-- **The executor accepts** program `L` on the statement words `stmt` and the
tape `tape`: every check of the run holds. Blocks compute `perm`. -/
def Exec (perm : List F → List F) (L : Array Instr) (stmt : List F) (tape : ℕ → F) : Prop :=
  (Exec.run perm stmt tape L).holds

end CompressionLean.Circuit
