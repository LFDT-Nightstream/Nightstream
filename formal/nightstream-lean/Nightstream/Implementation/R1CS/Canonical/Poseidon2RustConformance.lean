import Nightstream.Implementation.R1CS.Artifacts.Poseidon2Goldilocks.RoundConstants
import Nightstream.Implementation.R1CS.Canonical.Poseidon2CanonicalConstants
import Nightstream.Implementation.R1CS.Canonical.Poseidon2Matrices

/-!
Contract: shipping Rust currently agrees with the protocol Lean defines.

Owns: the comparison between the generated Rust export and the Lean-owned
constant table and internal diagonal.

Does not own: anything the canonical encoding depends on. **Nothing in the
headline path imports this file**, and that is the point — it is a check on
Rust, not an input to Lean.

## Reading a failure here

If a theorem in this file stops holding, Rust has drifted from the protocol.
The repair is to change Rust, not to change
`Poseidon2CanonicalConstants`. Before the authority inversion the same drift
would have silently rewritten every canonical coefficient count instead of
failing anywhere.

That asymmetry is the entire content of the inversion. The values did not
change; which file gets to decide them did.

## Scope

`rust_matches_lean_*` compares the *selected table*. It says nothing about the
seed-to-constant generator, which stays a published TCB boundary
(`POSEIDON2-CONSTANT-DERIVATION`), and nothing about whether Rust's permutation
*uses* the table it exports — that is
`POSEIDON2-RUST-TARGET-BOUNDARY`, and it is not discharged here.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.Poseidon2RustConformance

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Matrices

/-! ## The round constants -/

theorem rust_matches_lean_initial :
    Artifacts.Poseidon2Goldilocks.RoundConstants.initial
      = Poseidon2CanonicalConstants.initial := by
  decide

theorem rust_matches_lean_internal :
    Artifacts.Poseidon2Goldilocks.RoundConstants.internal
      = Poseidon2CanonicalConstants.internal := by
  decide

theorem rust_matches_lean_terminal :
    Artifacts.Poseidon2Goldilocks.RoundConstants.terminal
      = Poseidon2CanonicalConstants.terminal := by
  decide

/-! ## The internal diagonal

`Poseidon2Matrices.internalDiag` was already Lean-owned — it is written as the
published design `[-2, 1, 2, 2⁻¹, 3, -2⁻¹, -3, -4]` and its two non-obvious
entries are proved to be the intended field elements. This checks that Rust's
exported diagonal is the same eight values. -/

theorem rust_matches_lean_diagonal :
    ∀ lane : Fin width,
      Artifacts.Poseidon2Goldilocks.RoundConstants.internalDiagonal.getD lane.val 0
        = internalDiag lane := by
  decide

/-- The comparison is against a table of the right length, so the diagonal check
above is not passing on `getD`'s default. -/
theorem rust_diagonal_length :
    Artifacts.Poseidon2Goldilocks.RoundConstants.internalDiagonal.length
      = width := by
  decide

end Nightstream.Implementation.R1CS.Canonical.Poseidon2RustConformance
