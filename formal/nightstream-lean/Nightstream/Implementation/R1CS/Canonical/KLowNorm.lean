import Nightstream.Implementation.R1CS.Canonical.KHorner
import Nightstream.Implementation.R1CS.Canonical.KMulHonest
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.NormRange
import Nightstream.Implementation.Lowering.Typed.Cost

/-!
Contract: the emitted row program for Π_DEC's `b = 2` low-norm check.

Owns: the two rows enforcing `x³ = x`, their derived count, their column
allocation, soundness in the row layer's `Nat` vocabulary, and honest
completeness.

Does **not** own, and does not prove: the bridge to
`PaperJoint.NormRange.cubicResidual`. The algebra on the far side of that bridge
is **already proved** — `cubicResidual_eq_zero_iff_strictNormTwo` classifies the
roots as exactly the strict centered window, conditional on
`BaseFieldNoZeroDivisors`. What is missing here is only the translation from
`lcEval`-over-`Nat` to `Concrete.F`, which is not written.

## What Π_DEC actually asks for

`validate_child_x_low_norm` (`paper/reductions/pi_dec.rs`) requires every active
packed entry of a child's `X` to satisfy `within_nc_bound(v, b)`, which is
`-(b-1) ≤ balanced(v) ≤ b-1`. The deployment fixes `b = 2` — the params module
is literally named `goldilocks_paper_b2` — so the window is `{-1, 0, 1}` and the
constraint is `x·(x-1)·(x+1) = 0`, equivalently `x³ = x`.

`b = 2` is a **protocol constant**, not application data, which is why this atom
can be built without selecting an application.

## Two rows, one column

The cube needs one intermediate. Nothing else is allocated: the second row
writes back to the operand itself, which is a shared read rather than an
allocation, so the receipt is two rows and one auxiliary column.

## The premise this will inherit

Soundness *here* is stated in `Nat` and needs no field premise: it says only
that satisfaction forces `x³ ≡ x`. Concluding `x ∈ {-1,0,1}` from that is
`NormRange`'s theorem and carries `BaseFieldNoZeroDivisors`, which
`baseFieldNoZeroDivisors_of_modulusEuclid` derives from `EuclidPrime
goldilocksP` — still a typed hypothesis on 297 occurrences tree-wide.
`ARITH-GOLDILOCKS-CERTIFICATE` supplies the kernel-checked Lucas arithmetic
toward discharging it; `ARITH-GOLDILOCKS-LUCAS-COST` prices what remains.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KLowNorm

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal

/-- **The emitted low-norm program.**  `x · x = s` then `s · x = x`. -/
def lowNormRows (value : LinComb) (squareColumn : Nat) : List Row :=
  [ ⟨value, value, [(squareColumn, 1)]⟩,
    ⟨[(squareColumn, 1)], value, value⟩ ]

/-- **The derived row count**, from the emitted list. -/
theorem lowNormRows_length (value : LinComb) (squareColumn : Nat) :
    (lowNormRows value squareColumn).length = 2 := rfl

/-- **The allocated column is one the rows actually use.**

The converse of conservation, and the direction nothing in this line had.
Conservation bounds the mentioned columns from above; without this, a recipe
could declare an allocation wider than the program it emits and
`Typed.Cost.auxiliaryColumns` would overcount with nothing objecting. -/
theorem lowNormRows_use_squareColumn (value : LinComb) (squareColumn : Nat) :
    ∃ row ∈ lowNormRows value squareColumn, Mentions row.c squareColumn :=
  ⟨⟨value, value, [(squareColumn, 1)]⟩, by simp [lowNormRows], by
    simp [Mentions]⟩

/-- The single column one check allocates. -/
def lowNormColumns (squareColumn : Nat) : List Nat := [squareColumn]

theorem lowNormColumns_length (squareColumn : Nat) :
    (lowNormColumns squareColumn).length = 1 := rfl

theorem lowNormColumns_nodup (squareColumn : Nat) :
    (lowNormColumns squareColumn).Nodup := by
  simp [lowNormColumns]

/-! ## Soundness

Stated in the row layer's own vocabulary: satisfaction forces the cube to equal
the value, modulo the prime. No field premise is needed for *this* step. -/

/-- **Satisfaction forces `x³ ≡ x`.** -/
theorem lowNormRows_sound
    (z : Nat → Nat) (value : LinComb) (squareColumn : Nat)
    (satisfied : Satisfies (lowNormRows value squareColumn) z) :
    lcEval z value * lcEval z value % goldilocksP * lcEval z value % goldilocksP
      = lcEval z value := by
  have squareRow : RowHolds z ⟨value, value, [(squareColumn, 1)]⟩ :=
    satisfied _ (by simp [lowNormRows])
  have cubeRow : RowHolds z ⟨[(squareColumn, 1)], value, value⟩ :=
    satisfied _ (by simp [lowNormRows])
  unfold RowHolds at squareRow cubeRow
  simp only at squareRow cubeRow
  rw [← squareRow] at cubeRow
  exact cubeRow

/-! ## Honest completeness

The witness writes the square to its column. Every value in the strict centered
window satisfies both rows, which is what makes the check complete rather than
merely sound. -/

/-- The honest witness: the square, on the allocated column. -/
def lowNormWitness (z : Nat → Nat) (value : LinComb) (squareColumn : Nat) :
    Nat → Nat :=
  fun column =>
    if column = squareColumn then
      lcEval z value * lcEval z value % goldilocksP
    else z column

theorem lowNormWitness_off_column
    (z : Nat → Nat) (value : LinComb) (squareColumn column : Nat)
    (distinct : column ≠ squareColumn) :
    lowNormWitness z value squareColumn column = z column := by
  unfold lowNormWitness
  rw [if_neg distinct]

/-- The witness leaves the checked combination alone, provided that
combination does not read the allocated column. -/
theorem lcEval_lowNormWitness
    (z : Nat → Nat) (value : LinComb) (squareColumn : Nat)
    (fresh : ¬ Mentions value squareColumn) :
    lcEval (lowNormWitness z value squareColumn) value = lcEval z value := by
  refine KMulHonest.lcEval_congr _ z value (fun column mentioned => ?_)
  exact lowNormWitness_off_column z value squareColumn column
    (fun equal => fresh (equal ▸ mentioned))

/-- **An in-window value satisfies both rows.**

The hypothesis is exactly the conclusion of `lowNormRows_sound`, so the check is
complete for precisely the values it accepts — no gap in either direction.

Freshness is a real hypothesis, not bookkeeping: if the checked combination read
its own square column, writing the square would change the value being squared
and neither row need hold. -/
theorem lowNormRows_honest
    (z : Nat → Nat) (value : LinComb) (squareColumn : Nat)
    (fresh : ¬ Mentions value squareColumn)
    (cube : lcEval z value * lcEval z value % goldilocksP * lcEval z value
      % goldilocksP = lcEval z value) :
    Satisfies (lowNormRows value squareColumn)
      (lowNormWitness z value squareColumn) := by
  have preserved := lcEval_lowNormWitness z value squareColumn fresh
  have squareValue :
      lcEval (lowNormWitness z value squareColumn) [(squareColumn, 1)]
        = lcEval z value * lcEval z value % goldilocksP := by
    rw [show lcEval (lowNormWitness z value squareColumn) [(squareColumn, 1)]
        = lowNormWitness z value squareColumn squareColumn % goldilocksP from by
      simp [lcEval]]
    unfold lowNormWitness
    rw [if_pos rfl, Nat.mod_mod]
  intro row member
  simp only [lowNormRows, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl
  · unfold RowHolds
    simp only
    rw [preserved, squareValue]
  · unfold RowHolds
    simp only
    rw [preserved, squareValue, cube]

/-! ## Toward the frozen root classification

`NormRange` already owns the algebra: `cubicResidual_eq_zero_iff_strictNormTwo`
says the roots of `(z+1)z(z-1)` are exactly the strict centered window. All that
separates `lowNormRows_sound` from it is a translation into `Concrete.F`.

That translation rests on one polynomial identity, and the identity is the part
worth isolating, because `Fin`'s `-1` is the complement `n - 1` and the naive
statement drowns in `Nat` subtraction.

Written with `n = m + 1` the identity is **subtraction-free on both sides**,
which is what makes it provable by distribution plus `omega` with the monomials
as atoms — no `ring` tactic exists here. -/

/-- **The cubic expansion, without any subtraction.**

Reading `m` as `n - 1`, this says `(x+1)·x·((n-1)+x) = (x²+x)·n + (x³ - x)`,
which is the step that turns `Fin`'s complement form of `-1` into a multiple of
the modulus plus `x³ - x`. -/
theorem cubic_expansion (x m : Nat) :
    (x + 1) * x * (m + x) + x = (x * x + x) * (m + 1) + x * x * x := by
  simp only [Nat.mul_add, Nat.one_mul, Nat.mul_one,
    Nat.mul_assoc, Nat.mul_comm, Nat.mul_left_comm]
  omega

/-- A residue whose cube is itself makes the expansion a multiple of the
modulus: the `x³ - x` term vanishes and only `(x²+x)·n` survives. -/
theorem cubic_expansion_multiple
    (x m : Nat) (cube : x * x * x % (m + 1) = x) (bound : x < m + 1) :
    (x + 1) * x * (m + x) % (m + 1) = 0 := by
  have expansion := cubic_expansion x m
  have cubeForm : (m + 1) * (x * x * x / (m + 1)) + x = x * x * x := by
    have split := Nat.div_add_mod (x * x * x) (m + 1)
    rw [cube] at split
    exact split
  have distribute : (m + 1) * (x * x + x + x * x * x / (m + 1))
      = (x * x + x) * (m + 1) + (m + 1) * (x * x * x / (m + 1)) := by
    rw [Nat.mul_add, Nat.mul_comm (m + 1) (x * x + x)]
  have multiple : (x + 1) * x * (m + x)
      = (m + 1) * (x * x + x + x * x * x / (m + 1)) := by
    rw [distribute]
    omega
  rw [multiple, Nat.mul_mod_right]

open Nightstream.SuperNeo.Concrete in
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint in
/-- **The bridge.**  A row-layer residue whose cube is itself is a root of the
frozen cubic.

Composing this with `NormRange.cubicResidual_eq_zero_iff_strictNormTwo` carries
`lowNormRows_sound` to the frozen root classification — the strict `b = 2`
centered window — which is the soundness obligation for this atom. That final
composition inherits `BaseFieldNoZeroDivisors`, the premise the tree already
carries on 297 occurrences. -/
theorem cubicResidual_eq_zero_of_cube
    (x : Nat) (bound : x < goldilocksModulus)
    (cube : x * x % goldilocksModulus * x % goldilocksModulus = x) :
    NormRange.cubicResidual ⟨x, bound⟩ = 0 := by
  have xmod : x % goldilocksModulus = x := Nat.mod_eq_of_lt bound
  have cubeWhole : x * x * x % goldilocksModulus = x := by
    rw [Nat.mul_mod (x * x) x, xmod, cube]
  have positive : goldilocksModulus - 1 + 1 = goldilocksModulus := by decide
  refine Fin.ext ?_
  show ((x + 1) % goldilocksModulus * x % goldilocksModulus)
      * ((goldilocksModulus - 1 + x) % goldilocksModulus)
      % goldilocksModulus = 0
  have strip : ((x + 1) % goldilocksModulus * x % goldilocksModulus)
      * ((goldilocksModulus - 1 + x) % goldilocksModulus) % goldilocksModulus
      = (x + 1) * x * (goldilocksModulus - 1 + x) % goldilocksModulus := by
    simp [Nat.mul_mod]
  rw [strip]
  have multiple := cubic_expansion_multiple x (goldilocksModulus - 1)
    (by rw [positive]; exact cubeWhole) (by rw [positive]; exact bound)
  rw [positive] at multiple
  exact multiple

/-! ## Conservation

The check reads one combination and writes one column. Nothing else is
reachable, and in particular the constant wire is not: unlike an equality row,
neither of these rows carries a literal. -/

/-- **Every column either belongs to the checked combination or is the one
allocated column.** -/
theorem lowNormRows_conservation
    (value : LinComb) (squareColumn : Nat) (row : Row)
    (member : row ∈ lowNormRows value squareColumn) (column : Nat)
    (mentioned : Mentions row.a column ∨ Mentions row.b column
      ∨ Mentions row.c column) :
    Mentions value column ∨ column = squareColumn := by
  have target : Mentions [(squareColumn, 1)] column → column = squareColumn := by
    intro hit
    simpa only [Mentions, List.map_cons, List.map_nil,
      List.mem_singleton] using hit
  simp only [lowNormRows, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl <;> simp only at mentioned
  · rcases mentioned with a | b | c
    · exact Or.inl a
    · exact Or.inl b
    · exact Or.inr (target c)
  · rcases mentioned with a | b | c
    · exact Or.inr (target a)
    · exact Or.inl b
    · exact Or.inl c

/-! ## Cost

Two rows, one auxiliary column. Both components are receipts: the row count is
`lowNormRows_length` and the column count is `lowNormColumns_length`, each read
off the emitted list rather than declared.

`committedColumns` and `publicColumns` are zero because the checked value is a
*read*. The entry being range-checked belongs to whatever recipe allocated it —
here, to Π_DEC's child claim — and counting it again would double-count. -/

/-- **The atom's cost.** -/
def lowNormCost : Lowering.Typed.Cost where
  recurringRows := 2
  committedColumns := 0
  publicColumns := 0
  auxiliaryColumns := 1

theorem lowNormCost_rows (value : LinComb) (squareColumn : Nat) :
    (lowNormRows value squareColumn).length = lowNormCost.recurringRows :=
  lowNormRows_length value squareColumn

theorem lowNormCost_columns (squareColumn : Nat) :
    (lowNormColumns squareColumn).length = lowNormCost.auxiliaryColumns :=
  lowNormColumns_length squareColumn

end Nightstream.Implementation.R1CS.Canonical.KLowNorm
