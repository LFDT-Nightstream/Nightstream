import CompressionLean.Circuit.Record

namespace CompressionLean.Circuit

open CompressionLean.Shrink
open CompressionLean.Replay (Gl)

variable {F : Type} [CommRing F]

structure Bits (a : F) (count : ℕ) (b : ℕ → F) : Prop where
  boolean : ∀ i < count, b i = 0 ∨ b i = 1
  sum : a = ((List.range count).map fun i => 2 ^ i * b i).sum
  canonical : count = 64 → (∀ i, 32 ≤ i → i < 64 → b i = 1) → ∀ i < 32, b i = 0

def R1csRow.Holds (column : ℕ → F) (r : R1csRow Gl) : Prop :=
  let side (terms : List (ℕ × Gl)) := (terms.map fun t => column t.1 * word t.2).sum
  side r.a * side r.b = side r.c

structure Exec.State (F : Type) where
  values : List F
  reads : ℕ
  holds : Prop

namespace Exec

def start : State F := ⟨[], 0, True⟩

def State.value (s : State F) (v : Var) : F := s.values.getD v 0

def State.next (s : State F) (outputs : List F) (check : Prop) : State F :=
  ⟨s.values ++ outputs, s.reads, s.holds ∧ check⟩

variable (perm : List F → List F) (stmt : List F) (tape : ℕ → F)

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

def run (L : Array Instr) : State F := L.toList.foldl (step perm stmt tape) start

end Exec

def Exec (perm : List F → List F) (L : Array Instr) (stmt : List F) (tape : ℕ → F) : Prop :=
  (Exec.run perm stmt tape L).holds

end CompressionLean.Circuit
