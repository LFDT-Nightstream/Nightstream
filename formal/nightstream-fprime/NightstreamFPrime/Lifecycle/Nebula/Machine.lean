import NightstreamFPrime.Spec.Nebula.Reference

/-! Owns the semantics of the first Nebula memory application: a two-word
machine `(pc, acc)` whose program is in ROM. A step fetches the instruction
word `ROM[pc]` through port 0 and runs it; port 1 is its data port. An idle
step uses no port and keeps the state (spec §10). It does not own the circuit
or the memory rows.

Instruction word `v`: `op = v mod 4`, `arg = v / 4`.

| `op` | name  | effect                                   | port 1          |
|------|-------|------------------------------------------|-----------------|
| 0    | halt  | none                                     | inactive        |
| 1    | load  | `acc ← RAM[arg]`, `pc ← pc + 1`          | read `RAM[arg]` |
| 2    | store | `RAM[arg] ← acc`, `pc ← pc + 1`          | write `RAM[arg]`|
| 3    | loadi | `acc ← arg`, `pc ← pc + 1`               | inactive        |
-/

namespace NightstreamFPrime.Lifecycle.Nebula.Machine

open NightstreamFPrime.Spec.Nebula

/-- The machine state `(pc, acc)`. -/
structure State where
  pc : ℕ
  acc : ℕ
deriving DecidableEq

/-- The fetch access of a non-idle step: read `ROM[pc]`. -/
def fetch (s : State) (word : ℕ) : PortAccess := ⟨false, false, s.pc, word, word⟩

/-- The data access of instruction `word`, with `data` the value that the
port reads (for a store, the old value). -/
def dataPort (s : State) (word data : ℕ) : Option PortAccess :=
  match word % 4 with
  | 1 => some ⟨false, true, word / 4, data, data⟩
  | 2 => some ⟨true, true, word / 4, data, s.acc⟩
  | _ => none

/-- The state after instruction `word`, with `data` the value read by port 1. -/
def exec (s : State) (word data : ℕ) : State :=
  match word % 4 with
  | 0 => s
  | 1 => ⟨s.pc + 1, data⟩
  | 2 => ⟨s.pc + 1, s.acc⟩
  | _ => ⟨s.pc + 1, word / 4⟩

/-- One machine step over its two ports: idle, or one fetched instruction. -/
def Step (s : State) (ports : List (Option PortAccess)) (s' : State) : Prop :=
  (ports = [none, none] ∧ s' = s) ∨
    ∃ word data, ports = [some (fetch s word), dataPort s word data] ∧ s' = exec s word data

/-- The machine as a Nebula application of a plan with two ports. -/
def application (p : Plan) (twoPorts : p.bOps = 2) : Application p State where
  Step := Step
  idle := fun s => Or.inl ⟨by simp [idlePorts, twoPorts], rfl⟩

end NightstreamFPrime.Lifecycle.Nebula.Machine
