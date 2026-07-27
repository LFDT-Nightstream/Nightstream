import Nightstream.Implementation.R1CS.Canonical.Poseidon2Eval

/-!
Contract: the reference Poseidon2 permutation on field values.

Owns: the `x⁷` S-box as an addition chain over values, the round-by-round
reference evaluator, and nothing else.

Does not own: any encoding, row, or column.  This module is deliberately free
of `Row`, `Layout` and `Satisfies` — it is the specification the encoding must
meet, so it must be readable without reference to how the encoding works.

## Provenance

Mirrors `absorb_words_then_permute_values` in
`crates/neo-fold-clean/src/engine/ccs_native/poseidon2.rs` statement for
statement.  `sbox7` uses the same addition chain the encoding does
(`1 → 2 → 4 → 6 → 7`) rather than `x^7` written directly, so that
`chain_implies_sbox7` is a substitution rather than an exponentiation argument.

Round constants are a parameter throughout, so every downstream theorem is
universally quantified over them.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.Poseidon2Reference

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Matrices
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Eval

/-- Field values, one per lane. -/
abbrev Values := Fin width → Nat

/-- `x⁷` by the addition chain `1 → 2 → 4 → 6 → 7`, in the same order and with
the same intermediate reductions the encoding emits.  This is production's
chain (`enforce_sbox_x7`). -/
def sbox7 (x : Nat) : Nat :=
  let square := x * x % goldilocksP
  let fourth := square * square % goldilocksP
  let sixth := square * fourth % goldilocksP
  x * sixth % goldilocksP

/-- One full round on values: add this round's constants, S-box every lane,
then apply the external layer. -/
def fullRoundValues (roundConstants : Fin width → Nat) (state : Values) : Values :=
  applyMatrixValues externalMatrix
    (fun lane => sbox7 ((roundConstants lane + state lane) % goldilocksP))

/-- One partial round on values: add this round's constant, S-box lane 0 only,
then apply the internal layer.  Lanes 1..7 pass through untouched. -/
def partialRoundValues (roundConstant : Nat) (state : Values) : Values :=
  applyMatrixValues internalMatrix
    (fun lane =>
      if lane.val = 0 then sbox7 ((roundConstant + state ⟨0, by decide⟩) % goldilocksP)
      else state lane)

/-! ## The reference permutation

Three phases, in the Rust order.  Each `Nat` argument counts rounds *already
taken*, so index `0` is the state entering that phase. -/

/-- After the pre-layer and `round` initial full rounds. -/
def refInitial (constants : Constants) (input : Values) : Nat → Values
  | 0 => applyMatrixValues externalMatrix input
  | round + 1 =>
      fullRoundValues (constants.initial round) (refInitial constants input round)

/-- After all initial full rounds and `round` partial rounds. -/
def refPartial (constants : Constants) (input : Values) : Nat → Values
  | 0 => refInitial constants input halfFullRounds
  | round + 1 =>
      partialRoundValues (constants.internal round)
        (refPartial constants input round)

/-- After all partial rounds and `round` terminal full rounds. -/
def refTerminal (constants : Constants) (input : Values) : Nat → Values
  | 0 => refPartial constants input partialRounds
  | round + 1 =>
      fullRoundValues (constants.terminal round)
        (refTerminal constants input round)

/-- The complete reference permutation. -/
def referencePermutation (constants : Constants) (input : Values) : Values :=
  refTerminal constants input halfFullRounds

end Nightstream.Implementation.R1CS.Canonical.Poseidon2Reference
