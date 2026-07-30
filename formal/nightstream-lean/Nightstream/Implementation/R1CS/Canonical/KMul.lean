import Nightstream.Implementation.R1CS.Canonical.LinCombNormal
import Nightstream.Implementation.R1CS.Core.Projection.Polynomial

/-!
Contract: one Goldilocks-quadratic multiplication as emitted rows.

Owns: the three Karatsuba product rows of a `K` multiplication, their derived
count, and the proof that a satisfying assignment computes `K.mul` on the
carried coordinates.

Does not own: any projection identity, trace, or NIFS structure. Those are
built from this.

## Why this is the atom

A Lean-owned NIFS row program checks PiRLC projection identities, which are
polynomial evaluations over the quadratic extension `K`. Every such evaluation
decomposes into `K` multiplications and `K` additions. Additions are linear and
emit no row — they are terms in the carried combination, exactly as Poseidon2's
linear layers are. So the entire row cost of the projection check is a multiple
of this gadget, and pinning it fixes the multiplier before any identity frame
layout depends on it.

## Why Karatsuba, decided here rather than later

`K.mul ⟨l0, l1⟩ ⟨r0, r1⟩ = ⟨l0·r0 + 7·l1·r1, l0·r1 + l1·r0⟩`, the extension
being `X² = 7`. Two expansions compute it:

| | rows | auxiliary columns | operand entries |
|---|---|---|---|
| schoolbook `l0r0, l1r1, l0r1, l1r0` | 4 | 4 | `4|cL| + 4|cR| + 4` |
| Karatsuba `l0r0, l1r1, (l0+l1)(r0+r1)` | **3** | **3** | `4|cL| + 4|cR| + 3` |

Karatsuba's third row takes *summed* operands, so it carries `2|cL|` and
`2|cR|` entries where a schoolbook row carries `|cL|` and `|cR|`. The entry
totals therefore come out one apart, not in schoolbook's favour, and `outHigh`
spends that one back on its extra term. Same coefficient count, one fewer row
and one fewer column: Karatsuba strictly dominates, so there is no tradeoff to
weigh and no need to consult the project's rows-first optimization order.

This is settled *now* because the three-product form changes which intermediate
values exist. Selecting it after identity frames are laid out would invalidate
them, exactly as re-choosing the Poseidon2 S-box chain would have.

`outHigh` recovers `l0r1 + l1r0` by subtraction, which in a `Nat` encoding is
the coefficient `goldilocksP - 1`. That stays linear, so it still emits no row.

## Scope

This is `K` arithmetic on *columns*, not on `Fin goldilocksP`. The assignment is
`Nat → Nat` as everywhere else in the canonical track. The soundness theorems
take residues explicitly rather than assuming the assignment lands in `F`.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KMul

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal

/-- The three auxiliary columns one `K` multiplication allocates. -/
structure Frame where
  lowLow : Nat
  highHigh : Nat
  cross : Nat

/-- A `K` value carried symbolically: one combination per coordinate. -/
structure Carried where
  low : LinComb
  high : LinComb

/-- The coordinate sum, as a combination.  Concatenation, so it emits no row. -/
def sumComb (value : Carried) : LinComb := value.low ++ value.high

/-- One product row: `left · right = target`. -/
def productRow (left right : LinComb) (target : Nat) : Row where
  a := left
  b := right
  c := [(target, 1)]

/-- **The three rows.**  Every nonlinear step of a `K` multiplication. -/
def rows (left right : Carried) (frame : Frame) : List Row :=
  [ productRow left.low right.low frame.lowLow,
    productRow left.high right.high frame.highHigh,
    productRow (sumComb left) (sumComb right) frame.cross ]

/-- **The derived row count.**  Three, from the emitted list. -/
theorem rows_length (left right : Carried) (frame : Frame) :
    (rows left right frame).length = 3 := rfl

/-! ## The carried outputs

Linear in the three products, so they emit no row. -/

/-- Low coordinate: `l0·r0 + 7·l1·r1`. -/
def outLow (frame : Frame) : LinComb :=
  [(frame.lowLow, 1), (frame.highHigh, 7)]

/-- High coordinate: `(l0+l1)(r0+r1) − l0·r0 − l1·r1`, with subtraction carried
as the coefficient `goldilocksP - 1`. -/
def outHigh (frame : Frame) : LinComb :=
  [(frame.cross, 1), (frame.lowLow, goldilocksP - 1),
    (frame.highHigh, goldilocksP - 1)]

/-! ## Modular plumbing -/

theorem lcEval_singleton_col (z : Nat → Nat) (column : Nat) :
    lcEval z [(column, 1)] = z column % goldilocksP := by
  simp [lcEval]

theorem mul_mod_right_reduce (a b : Nat) :
    a * (b % goldilocksP) % goldilocksP = a * b % goldilocksP := by
  rw [Nat.mul_comm a, Nat.mod_mul_mod, Nat.mul_comm]

/-- Value of a two-term combination. -/
theorem lcEval_pair (z : Nat → Nat) (c1 k1 c2 k2 : Nat) :
    lcEval z [(c1, k1), (c2, k2)] = (k1 * z c1 + k2 * z c2) % goldilocksP := by
  simp [lcEval]

/-- Value of a three-term combination. -/
theorem lcEval_triple (z : Nat → Nat) (c1 k1 c2 k2 c3 k3 : Nat) :
    lcEval z [(c1, k1), (c2, k2), (c3, k3)]
      = (k1 * z c1 + k2 * z c2 + k3 * z c3) % goldilocksP := by
  simp [lcEval]

/-- **The coordinate sum evaluates to the sum of the coordinates.**  This is
what makes Karatsuba's third row legitimate: its operands are concatenations,
and concatenation is addition on values. -/
theorem lcEval_sumComb (z : Nat → Nat) (value : Carried) :
    lcEval z (sumComb value)
      = (lcEval z value.low + lcEval z value.high) % goldilocksP := by
  rw [lcEval_eq_rawSum, sumComb, rawSum_append, lcEval_eq_rawSum,
    lcEval_eq_rawSum, Nat.add_mod]

/-- Scaling both sides of a congruence keeps it. -/
theorem scaled_congr (k a b : Nat)
    (agree : a % goldilocksP = b % goldilocksP) :
    k * a % goldilocksP = k * b % goldilocksP := by
  rw [← mul_mod_right_reduce k a, agree, mul_mod_right_reduce]

/-- Two-way congruence.  The `←` pattern is unambiguous here, which it is not
once three summands nest. -/
theorem add2_mod_congr (x X y Y : Nat)
    (hx : x % goldilocksP = X % goldilocksP)
    (hy : y % goldilocksP = Y % goldilocksP) :
    (x + y) % goldilocksP = (X + Y) % goldilocksP := by
  rw [Nat.add_mod, hx, hy, ← Nat.add_mod]

/-- Three-way congruence, so `outHigh` need not fight rewrite order. -/
theorem add3_mod_congr (x X y Y w W : Nat)
    (hx : x % goldilocksP = X % goldilocksP)
    (hy : y % goldilocksP = Y % goldilocksP)
    (hw : w % goldilocksP = W % goldilocksP) :
    (x + y + w) % goldilocksP = (X + Y + W) % goldilocksP :=
  add2_mod_congr _ _ _ _ (add2_mod_congr _ _ _ _ hx hy) hw

theorem product_of_row
    (z : Nat → Nat) (left right : LinComb) (target : Nat)
    (holds : RowHolds z (productRow left right target)) :
    lcEval z left * lcEval z right % goldilocksP = z target % goldilocksP := by
  rw [← lcEval_singleton_col z target]
  exact holds

/-! ## Soundness -/

/-- **Each frame column carries its Karatsuba product.** -/
theorem frame_products
    (z : Nat → Nat) (left right : Carried) (frame : Frame)
    (satisfied : Satisfies (rows left right frame) z) :
    lcEval z left.low * lcEval z right.low % goldilocksP
        = z frame.lowLow % goldilocksP
      ∧ lcEval z left.high * lcEval z right.high % goldilocksP
        = z frame.highHigh % goldilocksP
      ∧ lcEval z (sumComb left) * lcEval z (sumComb right) % goldilocksP
        = z frame.cross % goldilocksP := by
  refine ⟨?_, ?_, ?_⟩ <;>
    exact product_of_row z _ _ _ (satisfied _ (by simp [rows]))

/-- **The low output is the extension's low coordinate**, with `X² = 7`
supplying the seven.  Unchanged by the Karatsuba choice: `outLow` never used
the cross terms. -/
theorem outLow_sound
    (z : Nat → Nat) (left right : Carried) (frame : Frame)
    (satisfied : Satisfies (rows left right frame) z) :
    lcEval z (outLow frame)
      = (lcEval z left.low * lcEval z right.low
          + 7 * (lcEval z left.high * lcEval z right.high)) % goldilocksP := by
  rcases frame_products z left right frame satisfied with ⟨ll, hh, _⟩
  rw [outLow, lcEval_pair, Nat.one_mul, Nat.add_mod, ← ll,
    ← mul_mod_right_reduce 7 (z frame.highHigh), ← hh,
    mul_mod_right_reduce, ← Nat.add_mod]

/-- The Karatsuba identity as a `Nat` equation — no subtraction anywhere.  The
`goldilocksP - 1` coefficients turn into exact multiples of the prime, which the
final reduction discards. -/
theorem karatsuba_identity (a b c d : Nat) :
    (a + b) * (c + d) + (goldilocksP - 1) * (a * c)
        + (goldilocksP - 1) * (b * d)
      = a * d + b * c + goldilocksP * (a * c) + goldilocksP * (b * d) := by
  have expand : (a + b) * (c + d) = a * c + a * d + b * c + b * d := by
    rw [Nat.add_mul, Nat.mul_add, Nat.mul_add]; omega
  rw [expand]
  generalize a * c = ac
  generalize a * d = ad
  generalize b * c = bc
  generalize b * d = bd
  simp only [goldilocksP]
  omega

/-- **The high output is the extension's high coordinate.**  Recovered by
subtraction from the single cross product, which is the whole point of the
three-row form. -/
theorem outHigh_sound
    (z : Nat → Nat) (left right : Carried) (frame : Frame)
    (satisfied : Satisfies (rows left right frame) z) :
    lcEval z (outHigh frame)
      = (lcEval z left.low * lcEval z right.high
          + lcEval z left.high * lcEval z right.low) % goldilocksP := by
  rcases frame_products z left right frame satisfied with ⟨ll, hh, cross⟩
  rw [lcEval_sumComb, lcEval_sumComb, ← Nat.mul_mod] at cross
  rw [outHigh, lcEval_triple, Nat.one_mul]
  refine (add3_mod_congr _ _ _ _ _ _ cross.symm
    (scaled_congr _ _ _ ll.symm) (scaled_congr _ _ _ hh.symm)).trans ?_
  rw [karatsuba_identity]
  simp [Nat.add_mul_mod_self_left]

end Nightstream.Implementation.R1CS.Canonical.KMul
