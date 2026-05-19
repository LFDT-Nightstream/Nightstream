-- Goldilocks field instance for zkLean.
--
-- This file owns the Lean-side concrete field used by the wasm zkVM. It does
-- not own the Rust runtime field (`neo_math::F = p3_goldilocks::Goldilocks`)
-- — that's a different type with the same algebraic structure. Coefficient
-- transmission from Rust to Lean happens through balanced-residue `Int`
-- conversion in the exporter, then `IntCast` lifts back into `Fq` here.
--
-- The `Nat.Prime` fact on `2^64 − 2^32 + 1` is taken as an axiom. Proving it
-- in Lean is mechanical (Pratt certificate: `p − 1 = 2^32 · 3 · 5 · 17 · 257
-- · 65537`, all Fermat primes) but not yet done. The zkLean examples follow
-- the same convention.

import Mathlib.Algebra.Field.ZMod
import zkLean.Semantics

namespace WasmCircuit

/-- Goldilocks prime: `2^64 − 2^32 + 1`. -/
def goldilocksPrime : Nat := 2 ^ 64 - 2 ^ 32 + 1

/-- Primality of the Goldilocks prime. Trust-debt axiom; closure plan is a
    Pratt certificate over `p − 1 = 2^32 · 3 · 5 · 17 · 257 · 65537`. -/
axiom goldilocks_prime : Nat.Prime goldilocksPrime

instance : Fact (Nat.Prime goldilocksPrime) := ⟨goldilocks_prime⟩

/-- The Lean-side Goldilocks field. Defined as `ZMod p` so we inherit mathlib's
    `Field`, `DecidableEq`, etc. -/
abbrev Fq : Type := ZMod goldilocksPrime

/-- `ZKField` instance for `Fq`. The parent classes (`Field`, `BEq`, `Inhabited`,
    `LawfulBEq`) are picked up via mathlib's existing instances on `ZMod p`;
    here we only supply the four zkLean-specific fields. -/
instance : ZKField Fq where
  hash x := Hashable.hash x.val
  field_to_bits {num_bits} (f : Fq) : Vector Fq num_bits :=
    Vector.ofFn (fun (i : Fin num_bits) =>
      if (f.val >>> i.val) &&& 1 = 1 then (1 : Fq) else (0 : Fq))
  field_to_nat f := f.val

end WasmCircuit
