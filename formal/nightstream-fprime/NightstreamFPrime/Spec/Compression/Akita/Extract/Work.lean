import NightstreamFPrime.Spec.Algebra
import NightstreamFPrime.Spec.Compression.Akita.Extract.Staged
import Mathlib.Data.Fintype.Fin
import Mathlib.Data.Nat.Cast.Order.Field

/-! The proof of work of a challenge site of Akita (ePrint 2026/1983, revision of 2026-10-15,
(146), Proposition 10.3 and Remark 10.18) in the model of one squeeze run, and its rule in the
pinned Akita code (`jolt_transcript::grind_bits_for_loss`, `check_grind`).

- An answer of the oracle at a stage input has the slot `none` for the work bits and the slot
  `some c` for each coordinate `c`. `view` is the answer without the work slot: the slots that the
  draws of a challenge read.
- `workZeros Slot g` is a fixed set of `⌊|Slot|/2^g⌋` slot values, and the sets of more bits are
  smaller (`workZeros_anti`). A work slot in this set has `g` zero work bits. `mass_workZeros_le`: a
  uniform slot is in it with probability at most `2^{−g}`.
- `mass_view`: the work slot and the other slots of a uniform answer are independent.
- `workBits L` is the number of work bits of a site with the loss `L` over the field of `q²`
  elements: the least `g ≤ 57` with `L·2^128 ≤ q²·2^g`. `foldWorkBits W K` is the number of
  work bits of the message of a fold with `W` fold coordinates and the smallest challenge set of
  `2^K` elements: the least `g ≤ 57` with `W·2^128 ≤ 2^K·2^g`. `setBits` is `K` of a family. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Extract

open scoped BigOperators
open NightstreamFPrime.Spec.RandomOracle (mass)

attribute [local instance low] Classical.propDecidable

section Work

/-- An answer without its work slot. -/
def view {Coord Slot : Type} (reply : Option Coord → Slot) : Coord → Slot :=
  fun coordinate => reply (some coordinate)

variable (Slot : Type) [Fintype Slot]

/-- The work slots with `g` zero work bits: the first `⌊|Slot|/2^g⌋` values of a fixed order of
the slot values, so that a slot with `g` zero work bits also has `g'` for each `g' ≤ g`. The
definition is irreducible, so that the elaborator does not list the slots of a concrete field. -/
@[irreducible] noncomputable def workZeros (g : ℕ) : Finset Slot :=
  Finset.univ.filter fun slot =>
    ((Fintype.equivFin Slot slot : Fin (Fintype.card Slot)) : ℕ) < Fintype.card Slot / 2 ^ g

theorem workZeros_card (g : ℕ) : (workZeros Slot g).card = Fintype.card Slot / 2 ^ g := by
  rw [workZeros, Finset.card_equiv (Fintype.equivFin Slot)
    (t := Finset.univ.filter fun index : Fin (Fintype.card Slot) =>
      (index : ℕ) < Fintype.card Slot / 2 ^ g) (fun slot => by simp),
    Fin.card_filter_val_lt, min_eq_right (Nat.div_le_self _ _)]

/-- With no work bits, each work slot passes. -/
theorem workZeros_zero : workZeros Slot 0 = Finset.univ :=
  Finset.eq_univ_of_card _ (by rw [workZeros_card, pow_zero, Nat.div_one])

/-- A work slot with `h` zero work bits has `g` zero work bits for each `g ≤ h`. -/
theorem workZeros_anti {g h : ℕ} (le : g ≤ h) : workZeros Slot h ⊆ workZeros Slot g := by
  intro slot member
  simp only [workZeros, Finset.mem_filter, Finset.mem_univ, true_and] at member ⊢
  exact lt_of_lt_of_le member
    (Nat.div_le_div_left (Nat.pow_le_pow_right (by decide) le) (by positivity))

variable [Nonempty Slot]

/-- Proposition 10.3: a uniform work slot has `g` zero work bits with probability at most
`2^{−g}`. -/
theorem mass_workZeros_le (g : ℕ) : mass (workZeros Slot g : Set Slot) ≤ 1 / 2 ^ g := by
  have count : mass (workZeros Slot g : Set Slot) =
      ((workZeros Slot g).card : ℝ) / Fintype.card Slot := by
    rw [← expect_mem (workZeros Slot g)]
    unfold mass
    exact Finset.expect_congr rfl fun slot _ => by
      by_cases inside : slot ∈ workZeros Slot g <;> simp [inside]
  have positive : (0 : ℝ) < Fintype.card Slot := by exact_mod_cast Fintype.card_pos
  rw [count, workZeros_card, div_le_iff₀ positive]
  calc ((Fintype.card Slot / 2 ^ g : ℕ) : ℝ) ≤ (Fintype.card Slot : ℝ) / (2 ^ g : ℕ) :=
        Nat.cast_div_le
    _ = 1 / 2 ^ g * Fintype.card Slot := by push_cast; ring

variable {Slot} {Coord : Type} [Fintype Coord] [DecidableEq Coord]

omit [Nonempty Slot] in
/-- Remark 10.18 in the slot model: the work slot and the other slots of a uniform answer are
independent, so a bad answer with zero work bits has the probability of a bad answer times the
probability of zero work bits. -/
theorem mass_view (zero : Finset Slot) (bad : Set (Coord → Slot)) :
    mass {reply : Option Coord → Slot | reply none ∈ zero ∧ view reply ∈ bad} =
      mass (zero : Set Slot) * mass bad := by
  unfold mass
  rw [Fintype.expect_mul_expect]
  calc
    _ = 𝔼 pair : Slot × (Coord → Slot),
          (if pair.1 ∈ (zero : Set Slot) then (1 : ℝ) else 0) *
            (if pair.2 ∈ bad then 1 else 0) :=
      Fintype.expect_equiv (Equiv.piOptionEquivProd (β := fun _ => Slot)) _ _ fun reply => by
        have first : (Equiv.piOptionEquivProd (β := fun _ => Slot) reply).1 = reply none := rfl
        have second : (Equiv.piOptionEquivProd (β := fun _ => Slot) reply).2 = view reply := rfl
        rw [first, second]
        by_cases worked : reply none ∈ zero <;> by_cases inside : view reply ∈ bad <;>
          simp [worked, inside]
    _ = _ := by
      rw [← Finset.univ_product_univ, Finset.expect_product]
      exact Finset.expect_congr rfl fun slot _ => Finset.expect_congr rfl fun rest _ => by
        by_cases worked : slot ∈ zero <;> by_cases inside : rest ∈ bad <;> simp [worked, inside]

/-- The work slot of a uniform answer has `g` zero work bits with probability at most
`2^{−g}`. -/
theorem mass_work_le (g : ℕ) :
    mass {reply : Option Coord → Slot | reply none ∈ workZeros Slot g} ≤ 1 / 2 ^ g := by
  have same : mass {reply : Option Coord → Slot | reply none ∈ workZeros Slot g} =
      mass (workZeros Slot g : Set Slot) := by
    unfold mass
    calc _ = 𝔼 reply : Option Coord → Slot,
            (if reply none ∈ workZeros Slot g then (1 : ℝ) else 0) :=
          Finset.expect_congr rfl fun reply _ => by
            by_cases inside : reply none ∈ workZeros Slot g <;> simp [inside]
      _ = 𝔼 slot : Slot, (if slot ∈ workZeros Slot g then (1 : ℝ) else 0) :=
          RandomOracle.expect_eval (none : Option Coord) fun slot =>
            if slot ∈ workZeros Slot g then (1 : ℝ) else 0
      _ = _ := Finset.expect_congr rfl fun slot _ => by
            by_cases inside : slot ∈ workZeros Slot g <;> simp [inside]
  rw [same]
  exact mass_workZeros_le Slot g

end Work

/-- `K` of a fold challenge family of the Akita code over the ring of degree `d`
(`specs/transcript-grinding.md`, "Geometry and the sampling denominator"): 128 for a family
with the operator filter of the Euclidean route, and `⌊log_2 |S|⌋` of the family without a
filter of the production ladder for each other family. -/
def setBits (filtered : Bool) (d : ℕ) : ℕ :=
  if filtered then 128
  else if d = 64 then 128 else if d = 128 then 129 else if d = 256 then 131
  else if d = 512 then 132 else 131

/-- The number of work bits of a site with the loss `L` over the field of `q²` elements
(`jolt_transcript::grind_bits_for_loss` with 128 bits, Akita (146)): the least `g ≤ 57` with
`L·2^128 ≤ q²·2^g`, and 57 when there is none. -/
def workBits (loss : ℕ) : ℕ :=
  ((List.range 58).find? fun g => loss * 2 ^ 128 ≤ fieldModulus ^ 2 * 2 ^ g).getD 57

/-- A bounded search over `g < 58` returns at most 57. -/
theorem find_range_le (test : ℕ → Bool) : ((List.range 58).find? test).getD 57 ≤ 57 := by
  rcases found : (List.range 58).find? test with _ | g
  · exact le_rfl
  · exact Nat.le_of_lt_succ (List.mem_range.mp (List.mem_of_find?_eq_some found))

theorem workBits_le (loss : ℕ) : workBits loss ≤ 57 := find_range_le _

theorem map_workBits_le (losses : List ℕ) : ∀ g ∈ losses.map workBits, g ≤ 57 := by
  intro g member
  obtain ⟨loss, _, rfl⟩ := List.mem_map.mp member
  exact workBits_le loss

/-- The number of work bits of the message of a fold (`FoldChallengeWork` of the Akita code):
the least `g ≤ 57` with `W·2^128 ≤ 2^K·2^g`, for `W` fold coordinates and the smallest challenge
set of `2^K` elements, and 57 when there is none. -/
def foldWorkBits (coordinates setBits : ℕ) : ℕ :=
  ((List.range 58).find? fun g => coordinates * 2 ^ 128 ≤ 2 ^ setBits * 2 ^ g).getD 57

theorem foldWorkBits_le (coordinates setBits : ℕ) : foldWorkBits coordinates setBits ≤ 57 :=
  find_range_le _

end NightstreamFPrime.Spec.Compression.Akita.Extract
