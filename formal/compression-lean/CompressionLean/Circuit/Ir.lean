import CompressionLean.Shrink.Layout
import CompressionLean.Replay.Field

namespace CompressionLean.Circuit

open CompressionLean.Shrink
open CompressionLean.Replay (Gl)

abbrev Var := ℕ

inductive Instr where
  | const (value : Gl)
  | priv
  | pub
  | linear (parts : List (Var × Gl))
  | mul (a b : Var)
  | extMul (a b : Fin 3 → Var)
  | extInv (a : Fin 3 → Var)
  | assertZero (a : Var)
  | bits (a : Var) (count : ℕ)
  | permute (state : Fin 16 → Var)
  | hint (count : ℕ)
  | r1cs (rows : List (R1csRow Gl)) (columns : List Var)

def Instr.outputs : Instr → ℕ
  | .const _ | .priv | .pub | .linear _ | .mul _ _ => 1
  | .extMul _ _ | .extInv _ => 3
  | .assertZero _ | .r1cs _ _ => 0
  | .bits _ count | .hint count => count
  | .permute _ => 16

structure Builder where
  instrs : Array Instr := #[]
  values : ℕ := 0

abbrev Build := StateM Builder

namespace Build

def emit (i : Instr) : Build Var :=
  modifyGet fun p => (p.values, { instrs := p.instrs.push i, values := p.values + i.outputs })

def handles {n : ℕ} (first : Var) : Fin n → Var := fun k => first + k

def const (value : Gl) : Build Var := emit (.const value)
def priv : Build Var := emit .priv
def pub : Build Var := emit .pub
def linear (parts : List (Var × Gl)) : Build Var := emit (.linear parts)
def add (a b : Var) : Build Var := linear [(a, 1), (b, 1)]
def sub (a b : Var) : Build Var := linear [(a, 1), (b, -1)]
def scale (a : Var) (by_ : Gl) : Build Var := linear [(a, by_)]
def mul (a b : Var) : Build Var := emit (.mul a b)
def extMul (a b : Fin 3 → Var) : Build (Fin 3 → Var) := handles <$> emit (.extMul a b)
def extInv (a : Fin 3 → Var) : Build (Fin 3 → Var) := handles <$> emit (.extInv a)
def assertZero (a : Var) : Build Unit := do
  let _ ← emit (.assertZero a)
  pure ()
def bits (a : Var) (count : ℕ) : Build (Fin count → Var) := handles <$> emit (.bits a count)
def permute (state : Fin 16 → Var) : Build (Fin 16 → Var) := handles <$> emit (.permute state)
def hint (count : ℕ) : Build (Fin count → Var) := handles <$> emit (.hint count)
def r1cs (rows : List (R1csRow Gl)) (columns : List Var) : Build Unit := do
  let _ ← emit (.r1cs rows columns)
  pure ()

def extConstant (value : Fin 3 → Gl) : Build (Fin 3 → Var) := do
  let x ← const (value 0)
  let y ← const (value 1)
  let z ← const (value 2)
  pure ![x, y, z]

def coordinatewise (f : Var → Var → Build Var) (a b : Fin 3 → Var) : Build (Fin 3 → Var) := do
  let x ← f (a 0) (b 0)
  let y ← f (a 1) (b 1)
  let z ← f (a 2) (b 2)
  pure ![x, y, z]

def extAdd : (Fin 3 → Var) → (Fin 3 → Var) → Build (Fin 3 → Var) := coordinatewise add
def extSub : (Fin 3 → Var) → (Fin 3 → Var) → Build (Fin 3 → Var) := coordinatewise sub
def extScale (a : Fin 3 → Var) (by_ : Var) : Build (Fin 3 → Var) := coordinatewise mul a fun _ => by_

def select (bit whenZero whenOne : Var) : Build Var := do
  let difference ← sub whenOne whenZero
  let chosen ← mul bit difference
  add whenZero chosen

def assertEqual (a b : Var) : Build Unit := do assertZero (← sub a b)

def assertExtZero (a : Fin 3 → Var) : Build Unit := do
  assertZero (a 0)
  assertZero (a 1)
  assertZero (a 2)

def assertExtEqual (a b : Fin 3 → Var) : Build Unit := do assertExtZero (← extSub a b)

end Build

def Build.program {α : Type} (b : Build α) : Array Instr := (b.run {}).2.instrs

end CompressionLean.Circuit
