import CompressionLean.Ports.Shrink.Layout
import CompressionLean.Ports.Replay.Field

/-!
Owns the shrink program as data: a flat, branch-free list of instructions
over value handles (`Instr`), and the builder that emits it (`Build`).

- An instruction is one call of `Backend`
  (`crates/neo-spartan/src/circuit/mod.rs::Backend`) that `record.rs` lowers by
  itself. `add`, `sub` and `scale` are one `linear` instruction, as
  `Recorder::linear` reads them (`record.rs::Recorder::linear`; `add`, `sub`, `scale`). The other
  methods are combinations (`ext`, `coordinates`, `ext_constant`, `ext_add`, `ext_sub`, `ext_scale` and
  `select` of `record.rs::Recorder`; `mod.rs::Backend`), and the builder writes them so.
- Value `v` is the `v`-th value of the run: each instruction appends its
  `outputs` in order and reads only earlier values.
- The program never sees a value, so no instruction depends on one.
  Constants and coefficients are canonical Goldilocks words.

`select` is `whenZero + bit·(whenOne - whenZero)`, as `record.rs::Recorder::select`
computes it.
`Native::select` branches (`mod.rs::Native::select`); the two agree on a boolean
`bit`.
-/

namespace CompressionLean.Circuit

open CompressionLean.Shrink
open CompressionLean.Replay (Word)

/-- A value handle: value `v` is the `v`-th value of the run. -/
abbrev Var := ℕ

/-- One `Backend` call that `record.rs` lowers by itself. -/
inductive Instr where
  /-- `constant` (`record.rs::Recorder::constant`). -/
  | const (value : Word)
  /-- `private`: a proof word (`record.rs::Recorder::private`). -/
  | priv
  /-- `public`: the next statement word (`record.rs::Recorder::public`). -/
  | pub
  /-- `Σ weight·part`: `add`, `sub` and `scale` (`record.rs::Recorder::linear`). -/
  | linear (parts : List (Var × Word))
  /-- `mul` (`record.rs::Recorder::mul`). -/
  | mul (a b : Var)
  /-- `ext_mul` on three coordinates each (`record.rs::Recorder::ext_mul`). -/
  | extMul (a b : Fin 3 → Var)
  /-- `ext_inverse` (`record.rs::Recorder::ext_inverse`). -/
  | extInv (a : Fin 3 → Var)
  /-- `assert_zero` (`record.rs::Recorder::assert_zero`). -/
  | assertZero (a : Var)
  /-- `bits`: `count` little-endian bits (`record.rs::Recorder::bits`). -/
  | bits (a : Var) (count : ℕ)
  /-- `permute`: one Poseidon2 block (`record.rs::Recorder::permute`). -/
  | permute (state : Fin 16 → Var)
  /-- `hint`: `count` free proof words (`record.rs::Recorder::hint`). -/
  | hint (count : ℕ)
  /-- `r1cs`: a fixed system over value columns (`record.rs::Recorder::r1cs`). -/
  | r1cs (rows : List (R1csRow Word)) (columns : List Var)

/-- The values an instruction appends. -/
def Instr.outputs : Instr → ℕ
  | .const _ | .priv | .pub | .linear _ | .mul _ _ => 1
  | .extMul _ _ | .extInv _ => 3
  | .assertZero _ | .r1cs _ _ => 0
  | .bits _ count | .hint count => count
  | .permute _ => 16

/-- A program being written: its instructions and the number of values they
make. -/
structure Builder where
  instrs : Array Instr := #[]
  values : ℕ := 0

/-- Writes a program; each call appends instructions and returns the
handles of their outputs. -/
abbrev Build := StateM Builder

namespace Build

/-- Append `i`; its outputs are the handles from the returned one on. -/
def emit (i : Instr) : Build Var :=
  modifyGet fun p => (p.values, { instrs := p.instrs.push i, values := p.values + i.outputs })

/-- The handles `first, first + 1, …` of a run of outputs. -/
def handles {n : ℕ} (first : Var) : Fin n → Var := fun k => first + k

def const (value : Word) : Build Var := emit (.const value)
def priv : Build Var := emit .priv
def pub : Build Var := emit .pub
def linear (parts : List (Var × Word)) : Build Var := emit (.linear parts)
def add (a b : Var) : Build Var := linear [(a, 1), (b, 1)]
def sub (a b : Var) : Build Var := linear [(a, 1), (b, -1)]
def scale (a : Var) (by_ : Word) : Build Var := linear [(a, by_)]
def mul (a b : Var) : Build Var := emit (.mul a b)
def extMul (a b : Fin 3 → Var) : Build (Fin 3 → Var) := handles <$> emit (.extMul a b)
def extInv (a : Fin 3 → Var) : Build (Fin 3 → Var) := handles <$> emit (.extInv a)
def assertZero (a : Var) : Build Unit := do
  let _ ← emit (.assertZero a)
  pure ()
def bits (a : Var) (count : ℕ) : Build (Fin count → Var) := handles <$> emit (.bits a count)
def permute (state : Fin 16 → Var) : Build (Fin 16 → Var) := handles <$> emit (.permute state)
def hint (count : ℕ) : Build (Fin count → Var) := handles <$> emit (.hint count)
def r1cs (rows : List (R1csRow Word)) (columns : List Var) : Build Unit := do
  let _ ← emit (.r1cs rows columns)
  pure ()

/-- `ext_constant`: three constants (`record.rs::Recorder::ext_constant`). -/
def extConstant (value : Fin 3 → Word) : Build (Fin 3 → Var) := do
  let x ← const (value 0)
  let y ← const (value 1)
  let z ← const (value 2)
  pure ![x, y, z]

/-- Three coordinate calls in order (`record.rs::Recorder::ext_add`, `ext_sub`, `ext_scale`). -/
def coordinatewise (f : Var → Var → Build Var) (a b : Fin 3 → Var) : Build (Fin 3 → Var) := do
  let x ← f (a 0) (b 0)
  let y ← f (a 1) (b 1)
  let z ← f (a 2) (b 2)
  pure ![x, y, z]

def extAdd : (Fin 3 → Var) → (Fin 3 → Var) → Build (Fin 3 → Var) := coordinatewise add
def extSub : (Fin 3 → Var) → (Fin 3 → Var) → Build (Fin 3 → Var) := coordinatewise sub
def extScale (a : Fin 3 → Var) (by_ : Var) : Build (Fin 3 → Var) := coordinatewise mul a fun _ => by_

/-- `whenZero + bit·(whenOne - whenZero)` (`record.rs::Recorder::select`). -/
def select (bit whenZero whenOne : Var) : Build Var := do
  let difference ← sub whenOne whenZero
  let chosen ← mul bit difference
  add whenZero chosen

/-- The default assertions (`mod.rs::Backend`). -/
def assertEqual (a b : Var) : Build Unit := do assertZero (← sub a b)

def assertExtZero (a : Fin 3 → Var) : Build Unit := do
  assertZero (a 0)
  assertZero (a 1)
  assertZero (a 2)

def assertExtEqual (a b : Fin 3 → Var) : Build Unit := do assertExtZero (← extSub a b)

end Build

/-- The instructions that `b` writes. -/
def Build.program {α : Type} (b : Build α) : Array Instr := (b.run {}).2.instrs

end CompressionLean.Circuit
