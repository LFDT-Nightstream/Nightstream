import Nightstream.Implementation.R1CS.Canonical.KHorner
import Nightstream.Implementation.R1CS.Canonical.KMul
import Nightstream.Implementation.Lowering.Typed.Cost

/-!
Contract: the emitted row program for Π_DEC's inactive-`X` zero check.

Owns: the single row forcing a combination to zero, its derived count, its empty
column allocation, conservation, soundness, honest completeness, and cost.

## What Π_DEC asks for

`superneo_inactive_x_zero` (`paper/relations/mod.rs`) requires `X[r, c] = 0` for
every `c` in `[ceil(m_in / D), cols)`, with `D = 54`. Rust's own comment records
that "the circuit side enforces `X[r, c] == 0`", so this is a row obligation
rather than a decoder-side shape test.

`ceil(m_in / D)` selects *which* entries are checked; it does not change the
check. So this atom is parameterised by nothing at all — one combination in,
one row out — and the `m_in` arithmetic belongs to whatever enumerates the
inactive positions.

## One row, no columns

Forcing a value to zero needs no intermediate. The row is `value · 1 = 0`, with
the `1` read from the constant wire, so the receipt is one row and an **empty**
allocation.

## No field premise

Unlike the `b = 2` low-norm check, soundness here needs nothing about the field.
`lcEval` already returns a residue below the modulus, so `value · 1 ≡ 0` gives
`value = 0` on the nose — no zero-divisor reasoning, and in particular no
dependence on `EuclidPrime goldilocksP`.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KZeroCheck

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal

/-- **The emitted zero check.**  `value · 1 = 0`. -/
def zeroRows (value : LinComb) : List Row :=
  [⟨value, [(0, 1)], []⟩]

/-- **The derived row count**, from the emitted list. -/
theorem zeroRows_length (value : LinComb) : (zeroRows value).length = 1 := rfl

/-- The check allocates nothing. -/
def zeroColumns : List Nat := []

theorem zeroColumns_length : zeroColumns.length = 0 := rfl

theorem zeroColumns_nodup : zeroColumns.Nodup := List.nodup_nil

/-! ## Soundness -/

/-- **Satisfaction forces the combination to zero.**

No field premise: `lcEval` returns a residue below the modulus, so the row's
`value · 1 ≡ 0` gives equality on the nose. -/
theorem zeroRows_sound
    (z : Nat → Nat) (value : LinComb) (constantWire : z 0 = 1)
    (satisfied : Satisfies (zeroRows value) z) :
    lcEval z value = 0 := by
  have row : RowHolds z ⟨value, [(0, 1)], []⟩ :=
    satisfied _ (by simp [zeroRows])
  unfold RowHolds at row
  simp only at row
  have one : lcEval z [(0, 1)] = 1 := by
    simp only [lcEval, List.foldl, constantWire, Nat.zero_add, Nat.mul_one]
    exact Nat.mod_eq_of_lt (by decide)
  rw [one, Nat.mul_one, KHorner.lcEval_nil] at row
  have bound : lcEval z value < goldilocksP :=
    Nat.mod_lt _ (by decide)
  rw [Nat.mod_eq_of_lt bound] at row
  exact row

/-! ## Honest completeness

Nothing is allocated, so the honest assignment is the caller's own — there is no
witness to extend, which is the strongest form completeness can take. -/

/-- **A zero combination satisfies the row**, under the caller's own
assignment. -/
theorem zeroRows_honest
    (z : Nat → Nat) (value : LinComb) (constantWire : z 0 = 1)
    (isZero : lcEval z value = 0) :
    Satisfies (zeroRows value) z := by
  intro row member
  simp only [zeroRows, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl
  unfold RowHolds
  simp only
  have one : lcEval z [(0, 1)] = 1 := by
    simp only [lcEval, List.foldl, constantWire, Nat.zero_add, Nat.mul_one]
    exact Nat.mod_eq_of_lt (by decide)
  rw [one, Nat.mul_one, KHorner.lcEval_nil, isZero]
  exact Nat.zero_mod _

/-! ## Conservation

The row reads the checked combination and the constant wire, and writes nothing.
The constant-wire arm is real here — the row carries a literal `1` — which is
the opposite of the low-norm rows, whose operands carry no literal. -/

/-- **Every column is the checked combination's or the constant wire.** -/
theorem zeroRows_conservation
    (value : LinComb) (row : Row) (member : row ∈ zeroRows value)
    (column : Nat)
    (mentioned : Mentions row.a column ∨ Mentions row.b column
      ∨ Mentions row.c column) :
    Mentions value column ∨ column = 0 := by
  simp only [zeroRows, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl
  simp only at mentioned
  rcases mentioned with a | b | c
  · exact Or.inl a
  · refine Or.inr ?_
    simpa only [Mentions, List.map_cons, List.map_nil,
      List.mem_singleton] using b
  · simp only [Mentions, List.map_nil, List.not_mem_nil] at c

/-! ## Cost -/

/-- **The atom's cost.**  One row, nothing allocated. -/
def zeroCost : Lowering.Typed.Cost where
  recurringRows := 1
  committedColumns := 0
  publicColumns := 0
  auxiliaryColumns := 0

theorem zeroCost_rows (value : LinComb) :
    (zeroRows value).length = zeroCost.recurringRows :=
  zeroRows_length value

theorem zeroCost_columns :
    zeroColumns.length = zeroCost.auxiliaryColumns :=
  zeroColumns_length

/-! ## The `K`-valued lift

Pi_DEC's `validate_y_ring_padding_zero` requires every `y_ring` lane at index
`>= D = 54` to be zero in the extension. A `K` value is two coordinates, so
zeroing one costs **two** rows — the same "an equality in `K` is two physical
rows" fact `KEquality` records, arriving here for the padding check.

### Not the carrier-270 padding

This is the tail of a `y_ring` row beyond `D`. It is **not** the 13 padding
coordinates of the 270-wide public carrier, which are ring lanes 41-53 of the
fifth ring and are **not** inert. Two different things both called padding;
conflating them would import a live-coordinate warning into a check where the
tail really is zero. -/

/-- **The emitted `K`-valued zero check.**  One row per coordinate. -/
def carriedZeroRows (value : KMul.Carried) : List Row :=
  zeroRows value.low ++ zeroRows value.high

/-- **The derived row count.**  Two — a `K` zero is not one row. -/
theorem carriedZeroRows_length (value : KMul.Carried) :
    (carriedZeroRows value).length = 2 := rfl

/-- **Satisfaction forces both coordinates to zero.** -/
theorem carriedZeroRows_sound
    (z : Nat → Nat) (value : KMul.Carried) (constantWire : z 0 = 1)
    (satisfied : Satisfies (carriedZeroRows value) z) :
    KHorner.carriedValue z value = ⟨0, 0⟩ := by
  have lowSat : Satisfies (zeroRows value.low) z :=
    fun row member => satisfied row (List.mem_append_left _ member)
  have highSat : Satisfies (zeroRows value.high) z :=
    fun row member => satisfied row (List.mem_append_right _ member)
  unfold KHorner.carriedValue
  rw [zeroRows_sound z value.low constantWire lowSat,
    zeroRows_sound z value.high constantWire highSat]

/-- **A zero `K` value satisfies both rows**, under the caller's own
assignment. -/
theorem carriedZeroRows_honest
    (z : Nat → Nat) (value : KMul.Carried) (constantWire : z 0 = 1)
    (isZero : KHorner.carriedValue z value = ⟨0, 0⟩) :
    Satisfies (carriedZeroRows value) z := by
  have coords : lcEval z value.low = 0 ∧ lcEval z value.high = 0 := by
    unfold KHorner.carriedValue at isZero
    simp only [KHorner.Pair.mk.injEq] at isZero
    exact isZero
  intro row member
  rcases List.mem_append.1 member with inLow | inHigh
  · exact zeroRows_honest z value.low constantWire coords.1 row inLow
  · exact zeroRows_honest z value.high constantWire coords.2 row inHigh

/-- **The `K`-valued cost.**  Two rows, nothing allocated. -/
def carriedZeroCost : Lowering.Typed.Cost where
  recurringRows := 2
  committedColumns := 0
  publicColumns := 0
  auxiliaryColumns := 0

theorem carriedZeroCost_rows (value : KMul.Carried) :
    (carriedZeroRows value).length = carriedZeroCost.recurringRows :=
  carriedZeroRows_length value

/-! ## The check, over all padded lanes

`validate_y_ring_padding_zero` ranges over every lane past `D` in every
`y_ring` row. The check-level program is the per-lane atom concatenated, and its
cost is a **fold over per-lane receipts** rather than a formula — the number of
padded lanes is a property of the claim's shape, not a constant, so a closed
formula here would be a subtotal presented as a total. -/

/-- **The emitted padding check.** -/
def paddingRows (lanes : List KMul.Carried) : List Row :=
  lanes.flatMap carriedZeroRows

/-- **Every column is a coordinate's or the constant wire.**

The `K`-valued form of `zeroRows_conservation`: a `K` zero is two rows, so the
carried value contributes two combinations rather than one. -/
theorem carriedZeroRows_conservation
    (value : KMul.Carried) (row : Row) (member : row ∈ carriedZeroRows value)
    (column : Nat)
    (mentioned : Mentions row.a column ∨ Mentions row.b column
      ∨ Mentions row.c column) :
    Mentions value.low column ∨ Mentions value.high column ∨ column = 0 := by
  unfold carriedZeroRows at member
  rcases List.mem_append.1 member with inLow | inHigh
  · rcases zeroRows_conservation value.low row inLow column mentioned with
      inValue | wire
    · exact Or.inl inValue
    · exact Or.inr (Or.inr wire)
  · rcases zeroRows_conservation value.high row inHigh column mentioned with
      inValue | wire
    · exact Or.inr (Or.inl inValue)
    · exact Or.inr (Or.inr wire)

/-- **Every column of the padding check belongs to some padded lane**, or is the
constant wire. -/
theorem paddingRows_conservation
    (lanes : List KMul.Carried) (row : Row) (member : row ∈ paddingRows lanes)
    (column : Nat)
    (mentioned : Mentions row.a column ∨ Mentions row.b column
      ∨ Mentions row.c column) :
    ∃ lane ∈ lanes,
      Mentions lane.low column ∨ Mentions lane.high column ∨ column = 0 := by
  unfold paddingRows at member
  rcases List.mem_flatMap.1 member with ⟨lane, laneMember, rowMember⟩
  exact ⟨lane, laneMember,
    carriedZeroRows_conservation lane row rowMember column mentioned⟩

/-- **The derived row count, as a fold over per-lane receipts.** -/
theorem paddingRows_length (lanes : List KMul.Carried) :
    (paddingRows lanes).length = (lanes.map (fun _ => 2)).sum := by
  unfold paddingRows
  rw [List.length_flatMap]
  exact congrArg List.sum
    (List.map_congr_left (fun lane _ => carriedZeroRows_length lane))

/-- Two rows per padded lane, once the fold is evaluated. -/
theorem paddingRows_length_eq (lanes : List KMul.Carried) :
    (paddingRows lanes).length = 2 * lanes.length := by
  rw [paddingRows_length]
  induction lanes with
  | nil => rfl
  | cons lane rest inductionHypothesis =>
      simp only [List.map_cons, List.sum_cons, List.length_cons,
        inductionHypothesis]
      omega

/-- **Satisfaction forces every padded lane to zero.** -/
theorem paddingRows_sound
    (z : Nat → Nat) (lanes : List KMul.Carried) (constantWire : z 0 = 1)
    (satisfied : Satisfies (paddingRows lanes) z)
    (lane : KMul.Carried) (member : lane ∈ lanes) :
    KHorner.carriedValue z lane = ⟨0, 0⟩ :=
  carriedZeroRows_sound z lane constantWire
    (fun row rowMember =>
      satisfied row (List.mem_flatMap.2 ⟨lane, member, rowMember⟩))

/-- **All-zero lanes satisfy the check**, under the caller's own assignment. -/
theorem paddingRows_honest
    (z : Nat → Nat) (lanes : List KMul.Carried) (constantWire : z 0 = 1)
    (allZero : ∀ lane ∈ lanes, KHorner.carriedValue z lane = ⟨0, 0⟩) :
    Satisfies (paddingRows lanes) z := by
  intro row member
  rcases List.mem_flatMap.1 member with ⟨lane, laneMember, rowMember⟩
  exact carriedZeroRows_honest z lane constantWire (allZero lane laneMember)
    row rowMember

/-- **The check's cost**, folded over lanes.  Nothing is allocated at any
lane, so the auxiliary component stays zero however many lanes there are. -/
def paddingCost (lanes : List KMul.Carried) : Lowering.Typed.Cost where
  recurringRows := 2 * lanes.length
  committedColumns := 0
  publicColumns := 0
  auxiliaryColumns := 0

theorem paddingCost_rows (lanes : List KMul.Carried) :
    (paddingRows lanes).length = (paddingCost lanes).recurringRows :=
  paddingRows_length_eq lanes

end Nightstream.Implementation.R1CS.Canonical.KZeroCheck
