import Nightstream.Implementation.R1CS.Canonical.LinCombNormal
import Nightstream.Implementation.R1CS.Core.Projection.Polynomial

/-!
Contract: one Goldilocks-quadratic multiplication as emitted rows.

Owns: the four product rows of a `K` multiplication, their derived count, and
the proof that a satisfying assignment computes `K.mul` on the carried
coordinates.

Does not own: any projection identity, trace, or NIFS structure. Those are
built from this.

## Why this is the atom

A Lean-owned NIFS row program checks PiRLC projection identities, which are
polynomial evaluations over the quadratic extension `K`. Every such evaluation
decomposes into `K` multiplications and `K` additions. Additions are linear and
emit no row — they are terms in the carried combination, exactly as Poseidon2's
linear layers are. So the entire row cost of the projection check is a multiple
of this gadget, and pinning it fixes the multiplier before any identity is
written.

`K.mul ⟨l0, l1⟩ ⟨r0, r1⟩ = ⟨l0·r0 + 7·l1·r1, l0·r1 + l1·r0⟩` — the extension is
`X² = 7`. Schoolbook needs the four products `l0r0`, `l1r1`, `l0r1`, `l1r0`;
the two output coordinates are then linear in them and cost nothing.

Karatsuba would use three products at the cost of one extra addition chain.
That is a real choice with a real row saving, and it is deliberately **not**
taken here: the three-product form changes which intermediate values exist, so
it must be selected once, for the whole projection encoding, rather than per
gadget. Named `KMUL-PRODUCT-COUNT`.

## Scope

This is `K` arithmetic on *columns*, not on `Fin goldilocksP`. The assignment is
`Nat → Nat` as everywhere else in the canonical track, and coordinates are
canonical residues. `kMul_sound` relates the two by taking residues explicitly
rather than assuming the assignment lands in `F`.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KMul

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal

/-- The four auxiliary columns one `K` multiplication allocates. -/
structure Frame where
  lowLow : Nat
  highHigh : Nat
  lowHigh : Nat
  highLow : Nat

/-- A `K` value carried symbolically: one combination per coordinate. -/
structure Carried where
  low : LinComb
  high : LinComb

/-- One product row: `left · right = target`. -/
def productRow (left right : LinComb) (target : Nat) : Row where
  a := left
  b := right
  c := [(target, 1)]

/-- **The four rows.**  Every nonlinear step of a `K` multiplication, and
nothing else. -/
def rows (left right : Carried) (frame : Frame) : List Row :=
  [ productRow left.low right.low frame.lowLow,
    productRow left.high right.high frame.highHigh,
    productRow left.low right.high frame.lowHigh,
    productRow left.high right.low frame.highLow ]

/-- **The derived row count.**  Four, from the emitted list. -/
theorem rows_length (left right : Carried) (frame : Frame) :
    (rows left right frame).length = 4 := rfl

/-! ## The carried outputs

Linear in the four products, so they emit no row.  This is the same
never-materialize discipline the Poseidon2 encoding uses for its linear
layers. -/

/-- Low coordinate: `l0·r0 + 7·l1·r1`. -/
def outLow (frame : Frame) : LinComb :=
  [(frame.lowLow, 1), (frame.highHigh, 7)]

/-- High coordinate: `l0·r1 + l1·r0`. -/
def outHigh (frame : Frame) : LinComb :=
  [(frame.lowHigh, 1), (frame.highLow, 1)]

theorem out_emits_no_rows (frame : Frame) :
    (outLow frame).length + (outHigh frame).length = 4 := rfl

/-! ## Soundness

A satisfying assignment puts the schoolbook products on the four frame columns.
Stated on residues, so nothing is assumed about how the assignment was built. -/

theorem lcEval_singleton_col (z : Nat → Nat) (column : Nat) :
    lcEval z [(column, 1)] = z column % goldilocksP := by
  simp [lcEval]

/-- Multiplying by an already-reduced value is the same as reducing after.
The `X² = 7` coefficient forces this at the low output coordinate. -/
theorem mul_mod_right_reduce (a b : Nat) :
    a * (b % goldilocksP) % goldilocksP = a * b % goldilocksP := by
  rw [Nat.mul_comm a, Nat.mod_mul_mod, Nat.mul_comm]

/-- Value of a two-term combination, without unfolding `lcEval` elsewhere. -/
theorem lcEval_pair (z : Nat → Nat) (c1 k1 c2 k2 : Nat) :
    lcEval z [(c1, k1), (c2, k2)] = (k1 * z c1 + k2 * z c2) % goldilocksP := by
  simp [lcEval]

theorem product_of_row
    (z : Nat → Nat) (left right : LinComb) (target : Nat)
    (holds : RowHolds z (productRow left right target)) :
    lcEval z left * lcEval z right % goldilocksP = z target % goldilocksP := by
  rw [← lcEval_singleton_col z target]
  exact holds

/-- **Each frame column carries its product.** -/
theorem frame_products
    (z : Nat → Nat) (left right : Carried) (frame : Frame)
    (satisfied : Satisfies (rows left right frame) z) :
    lcEval z left.low * lcEval z right.low % goldilocksP
        = z frame.lowLow % goldilocksP
      ∧ lcEval z left.high * lcEval z right.high % goldilocksP
        = z frame.highHigh % goldilocksP
      ∧ lcEval z left.low * lcEval z right.high % goldilocksP
        = z frame.lowHigh % goldilocksP
      ∧ lcEval z left.high * lcEval z right.low % goldilocksP
        = z frame.highLow % goldilocksP := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;>
    exact product_of_row z _ _ _ (satisfied _ (by simp [rows]))

/-- **The low output is the extension's low coordinate.**  `l0·r0 + 7·l1·r1`,
with `X² = 7` supplying the seven. -/
theorem outLow_sound
    (z : Nat → Nat) (left right : Carried) (frame : Frame)
    (satisfied : Satisfies (rows left right frame) z) :
    lcEval z (outLow frame)
      = (lcEval z left.low * lcEval z right.low
          + 7 * (lcEval z left.high * lcEval z right.high)) % goldilocksP := by
  rcases frame_products z left right frame satisfied with ⟨ll, hh, _, _⟩
  rw [outLow, lcEval_pair, Nat.one_mul, Nat.add_mod, ← ll,
    ← mul_mod_right_reduce 7 (z frame.highHigh), ← hh,
    mul_mod_right_reduce, ← Nat.add_mod]

/-- **The high output is the extension's high coordinate.**  `l0·r1 + l1·r0`. -/
theorem outHigh_sound
    (z : Nat → Nat) (left right : Carried) (frame : Frame)
    (satisfied : Satisfies (rows left right frame) z) :
    lcEval z (outHigh frame)
      = (lcEval z left.low * lcEval z right.high
          + lcEval z left.high * lcEval z right.low) % goldilocksP := by
  rcases frame_products z left right frame satisfied with ⟨_, _, lh, hl⟩
  rw [outHigh, lcEval_pair, Nat.one_mul, Nat.one_mul, Nat.add_mod, ← lh, ← hl,
    ← Nat.add_mod]

end Nightstream.Implementation.R1CS.Canonical.KMul
