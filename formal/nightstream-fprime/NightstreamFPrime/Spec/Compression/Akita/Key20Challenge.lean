import NightstreamFPrime.Spec.Compression.Akita.ChallengeFamily
import NightstreamFPrime.Spec.Compression.Akita.Extract.Draw
import NightstreamFPrime.Spec.Compression.Akita.Key20Tower
import NightstreamFPrime.Spec.Compression.Akita.TrigBuild

/-! The challenge families of the schedule file `crates/ns-compress/artifacts/key20.aks` and
their decoders from one slot of the extension field `E` (Akita, ePrint 2026/1983, revision of
2026-10-15, Section 3.2 and Definition 9.2, E1).

The Akita code (revision `d50f700f`, `crates/akita-challenges/src/config.rs`) has the family
`(31, 10)` for the degree 64 and `(31, 0)` for the degree 128. The levels with the Euclidean
route have a filter on the base family `(31, 11)` for the degree 64 and `(31, 0)` for the degree
128.

- `draw_units` and `draw_sparse`: the challenges of `draw (SparseFamily d n₁ n₂) keep` have
  units as differences for `d = 2^{n+1}`, and the norms `ω = n₁ + 2·n₂` and `c_max`.
- `families_large`: the three families have `2^128`, `2^129` and `2^130` members or more, which
  are the slot widths 128, 129 and 130 of the Akita code. `E` has `q² < 2^128` elements, so
  `draw` has an injection of `E` into each family (`slots_le`), and each slot of `E` gives a
  challenge when no filter is present (`draw_isSome`).
- `slotChance_draw`: the slot probability of each decoder `draw` is at most `1/q²`.
  `pow_lt_slotChance`: no decoder from one slot of `E` that gives a challenge has the slot
  probability `2^-128`, because `1/q² > 2^-128`.
- `rootDecode` is the decoder of the three groups of the root, with the norms `rootMass` and
  `rootPeak`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Key20

open NightstreamFPrime.Spec.Compression.Akita.Extract

attribute [local instance low] Classical.propDecidable

section Sparse

variable {Slot : Type} {d ones twos : ℕ} [NeZero d] {keep : Rq d → Prop}

/-- The challenges of `draw` for a sparse family of a degree `2^{n+1}` have units as differences
(Akita Definition 9.2, E1). -/
theorem draw_units (n : ℕ) (power : 2 ^ (n + 1) = d) :
    UnitDifferences {challenge | ∃ slot : Slot,
      draw (SparseFamily d ones twos) keep slot = some challenge} :=
  fun c ⟨_, decoded⟩ c' ⟨_, decoded'⟩ =>
    SparseFamily.unitDifferences n power ones twos c (draw_mem decoded).1 c' (draw_mem decoded').1

/-- The challenges of `draw` for a sparse family have `‖c‖_1 = n₁ + 2·n₂` and `‖c‖_∞ = c_max`. -/
theorem draw_sparse (challenge : Rq d)
    (decoded : ∃ slot : Slot, draw (SparseFamily d ones twos) keep slot = some challenge) :
    Sparse (ones + 2 * twos) (peak ones twos) challenge :=
  SparseFamily.sparse (draw_mem decoded.choose_spec).1

end Sparse

/-- The families `(31, 10)` and `(31, 11)` of the degree 64 and `(31, 0)` of the degree 128 have
`2^128`, `2^130` and `2^129` members or more. -/
theorem families_large : 2 ^ 128 ≤ Nat.card (SparseFamily 64 31 10) ∧
    2 ^ 130 ≤ Nat.card (SparseFamily 64 31 11) ∧ 2 ^ 129 ≤ Nat.card (SparseFamily 128 31 0) := by
  simp only [card_sparseFamily]
  decide

variable {E : Type} [Field E] [Fintype E] [Algebra Zq E] [Degree2 E]

/-- `E` has at most as many elements as a family with `2^128` members or more. -/
theorem slots_le {Ring : Type} {family : Set Ring} (large : 2 ^ 128 ≤ Nat.card family) :
    Fintype.card E ≤ Nat.card family := by
  rw [card_field]
  exact (by decide : fieldModulus ^ 2 ≤ 2 ^ 128).trans large

/-- The slot probability of `draw` from one slot of `E` is at most `1/q²`. -/
theorem slotChance_draw {Ring : Type} (family : Set Ring) (keep : Ring → Prop) :
    slotChance (draw (Slot := E) family keep) ≤ 1 / (fieldModulus : ℝ) ^ 2 := by
  have bound := slotChance_draw_le (Slot := E) family keep
  rwa [card_field, Nat.cast_pow] at bound

/-- A decoder from one slot of `E` that gives a challenge has a slot probability above
`2^-128`. -/
theorem pow_lt_slotChance {Ring : Type} (decode : E → Option Ring) {slot : E}
    (decoded : (decode slot).isSome) : 1 / (2 : ℝ) ^ 128 < slotChance decode := by
  refine lt_of_lt_of_le ?_ (le_slotChance decode decoded)
  rw [card_field]
  exact one_div_lt_one_div_of_lt (Nat.cast_pos.mpr (by decide))
    (by exact_mod_cast (by decide : fieldModulus ^ 2 < 2 ^ 128))

/-- The count of the coefficients `±2` of the family of each group of the root: the key group
has the degree 128, and the groups of `W` and `V` have the degree 64. -/
def rootTwos : Fin 3 → ℕ := ![0, 10, 10]

/-- `ω` of the family of each group of the root: 31, 51 and 51. -/
def rootMass (g : Fin 3) : ℕ := 31 + 2 * rootTwos g

/-- `c_max` of the family of each group of the root: 1, 2 and 2. -/
def rootPeak (g : Fin 3) : ℕ := peak 31 (rootTwos g)

variable (E) in
/-- The decoder of the fold challenges of each group of the root. -/
noncomputable def rootDecode (g : Fin 3) : E → Option (Rq (root.carrierDegree g)) :=
  draw (SparseFamily (root.carrierDegree g) 31 (rootTwos g)) fun _ => True

omit [Field E] [Fintype E] [Algebra Zq E] [Degree2 E] in
theorem rootDecode_units (g : Fin 3) :
    UnitDifferences {challenge | ∃ slot, rootDecode E g slot = some challenge} :=
  draw_units (![6, 5, 5] g) (by revert g; decide)

omit [Field E] [Fintype E] [Algebra Zq E] [Degree2 E] in
theorem rootDecode_sparse (g : Fin 3) (challenge : Rq (root.carrierDegree g))
    (decoded : ∃ slot, rootDecode E g slot = some challenge) :
    Sparse (rootMass g) (rootPeak g) challenge :=
  draw_sparse challenge decoded

/-- Each slot of `E` gives a fold challenge of each group of the root. -/
theorem rootDecode_isSome (g : Fin 3) (slot : E) : (rootDecode E g slot).isSome := by
  refine draw_isSome (slots_le ?_) slot
  fin_cases g
  · exact (by decide : 2 ^ 128 ≤ 2 ^ 129).trans families_large.2.2
  · exact families_large.1
  · exact families_large.1

end NightstreamFPrime.Spec.Compression.Akita.Key20
