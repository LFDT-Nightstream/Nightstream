import NightstreamFPrime.Spec.Compression.Akita.Extract.Staged
import NightstreamFPrime.Spec.Compression.Akita.Extract.Tier

/-! The slot probability of a decoder of the fold challenges from one slot of an answer of the
oracle (Akita, ePrint 2026/1983, revision of 2026-11-23, Section 3.2, (23) and Lemma 10.1).

- `slotChance_le_of_fibres`: a decoder for which each challenge has at most `size` slots has the
  slot probability at most `size/|Slot|`. `slotChance_le_pow` is the bound `2^-k` for
  `size·2^k ≤ |Slot|`, which is the bound of a decoder with the uniform law on `2^k` challenges.
  `slotChance_le_of_injective` is the bound `1/|Slot|` of a decoder that gives each challenge
  for at most one slot, and `le_slotChance` states that no decoder with a challenge has a
  smaller slot probability. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Extract

open scoped BigOperators
open NightstreamFPrime.Spec.RandomOracle (mass)

attribute [local instance low] Classical.propDecidable

section Chance

variable {Slot Ring : Type} [Fintype Slot] [Nonempty Slot] (decode : Slot → Option Ring)

omit [Nonempty Slot] in
/-- The probability of a set of slots is its size over `|Slot|`. -/
theorem mass_eq_card (p : Slot → Prop) :
    mass {slot | p slot} = (Nat.card {slot // p slot} : ℝ) / Fintype.card Slot := by
  rw [Nat.card_eq_fintype_card, Fintype.card_subtype, ← expect_mem]
  unfold mass
  refine Finset.expect_congr rfl fun slot _ => ?_
  simp

/-- A decoder for which each challenge has at most `size` slots has the slot probability at most
`size/|Slot|`. -/
theorem slotChance_le_of_fibres {size : ℕ}
    (fibres : ∀ value, Nat.card {slot // decode slot = some value} ≤ size) :
    slotChance decode ≤ size / Fintype.card Slot := by
  refine Finset.sup'_le _ _ fun slot _ => ?_
  split
  · next decoded =>
    obtain ⟨value, same⟩ := Option.isSome_iff_exists.mp decoded
    rw [same, mass_eq_card]
    exact div_le_div_of_nonneg_right (Nat.cast_le.mpr (fibres value)) (Nat.cast_nonneg _)
  · positivity

/-- A decoder with at most `size` slots for each challenge and `size·2^k ≤ |Slot|` has the slot
probability at most `2^-k`: the bound of the uniform law on `2^k` challenges. -/
theorem slotChance_le_pow {size k : ℕ}
    (fibres : ∀ value, Nat.card {slot // decode slot = some value} ≤ size)
    (enough : size * 2 ^ k ≤ Fintype.card Slot) : slotChance decode ≤ 1 / 2 ^ k := by
  refine (slotChance_le_of_fibres decode fibres).trans ?_
  rw [div_le_div_iff₀ (Nat.cast_pos.mpr Fintype.card_pos) (by positivity), one_mul]
  exact_mod_cast enough

/-- A decoder that gives each challenge for at most one slot has the slot probability at most
`1/|Slot|`. -/
theorem slotChance_le_of_injective
    (injective : ∀ a b value, decode a = some value → decode b = some value → a = b) :
    slotChance decode ≤ 1 / Fintype.card Slot := by
  have bound := slotChance_le_of_fibres decode (size := 1) fun value =>
    Finite.card_le_one_iff_subsingleton.mpr
      ⟨fun a b => Subtype.ext (injective a.1 b.1 value a.2 b.2)⟩
  rwa [Nat.cast_one] at bound

/-- A decoder that gives a challenge for a slot has the slot probability `1/|Slot|` or more. -/
theorem le_slotChance {slot : Slot} (decoded : (decode slot).isSome) :
    1 / (Fintype.card Slot : ℝ) ≤ slotChance decode := by
  refine le_trans ?_ (Finset.le_sup' _ (Finset.mem_univ slot))
  rw [ite_eq_left decoded, mass_eq_card]
  have : Nonempty {other // decode other = decode slot} := ⟨⟨slot, rfl⟩⟩
  exact div_le_div_of_nonneg_right (Nat.one_le_cast.mpr Nat.card_pos) (Nat.cast_nonneg _)

end Chance

end NightstreamFPrime.Spec.Compression.Akita.Extract
