import Nightstream.Implementation.R1CS.Canonical.KHornerSupport

/-!
Contract: honest completeness for a Horner evaluation over the canonical
allocator.

Owns: freshness of every allocated frame from a single placement hypothesis,
the inside-out witness for a whole evaluation, and the proof that it satisfies
every emitted row.

Does not own: ownership or conservation for the evaluation program, the
projection identity, or any NIFS structure.

## One hypothesis instead of a bundle

Stated over `KFrames.frameAt base` rather than an arbitrary `frames : Nat →
Frame`. An abstract allocator would need frame distinctness, pairwise frame
disjointness and operand freshness carried as premises; the concrete one turns
all three into theorems, leaving a single hypothesis:

> every operand column is below `base`.

That is `BelowBase`, it is what a caller placing the evaluation above its
inputs already satisfies, and it is checkable by inspection rather than by
proof obligation. Fewer premises also means fewer places for §3's
"obligation moved rather than closed" to hide.

## The witness is built inside out

`hornerRows` recurses from the front, but step `s`'s multiplication consumes
the value carried by steps `s+1 …`. So the witness starts at the deepest step
and each enclosing step extends it. `KHornerSupport.satisfies_extend` keeps the
inner rows satisfied across each extension, and
`KHornerSupport.hornerCarried_mentions` bounds what the suffix can touch so the
enclosing frame is fresh for it.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KHornerHonest

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.KHorner
open Nightstream.Implementation.R1CS.Canonical.KHornerSupport

/-- Every column a combination mentions lies below the allocator's base. -/
def BelowBase (comb : LinComb) (base : Nat) : Prop :=
  ∀ column, Mentions comb column → column < base

/-! ## Freshness from placement -/

theorem frameColumn_ge_base (base step slot : Nat) :
    base ≤ KFrames.frameColumn base step slot := by
  unfold KFrames.frameColumn
  omega

/-- Operands placed below the base are fresh for every frame. -/
theorem fresh_of_belowBase
    (comb : LinComb) (base step : Nat) (below : BelowBase comb base) :
    KMulHonest.Fresh comb (KFrames.frameAt base step) := by
  refine ⟨?_, ?_, ?_⟩ <;> intro mentioned <;>
    exact absurd (below _ mentioned) (by
      have := frameColumn_ge_base base step 0
      have := frameColumn_ge_base base step 1
      have := frameColumn_ge_base base step 2
      simp only [KFrames.frameAt] at *
      omega)

/-- A later frame never collides with an earlier one, so the suffix's frame
columns are fresh for the enclosing step. -/
theorem later_frame_fresh
    (base step later slot otherSlot : Nat) (ordered : step < later)
    (slotLt : slot < 3) (otherSlotLt : otherSlot < 3) :
    KFrames.frameColumn base later slot
      ≠ KFrames.frameColumn base step otherSlot := by
  unfold KFrames.frameColumn KFrames.columnsPerFrame
  omega

/-- **The carried suffix is fresh for its enclosing frame.**  Its columns are
coefficients, which are below the base, or frames strictly later than this
step, which the allocator separates. -/
theorem suffix_fresh
    (beta : Carried) (base : Nat) (coefficients : List Carried) (step : Nat)
    (coeffBelow : ∀ c ∈ coefficients, BelowBase c.low base ∧ BelowBase c.high base) :
    KMulHonest.Fresh
        (hornerCarried beta (KFrames.frameAt base) coefficients (step + 1)).low
        (KFrames.frameAt base step)
      ∧ KMulHonest.Fresh
        (hornerCarried beta (KFrames.frameAt base) coefficients (step + 1)).high
        (KFrames.frameAt base step) := by
  have classify : ∀ column,
      (Mentions (hornerCarried beta (KFrames.frameAt base) coefficients
          (step + 1)).low column
        ∨ Mentions (hornerCarried beta (KFrames.frameAt base) coefficients
          (step + 1)).high column) →
      column ≠ (KFrames.frameAt base step).lowLow
        ∧ column ≠ (KFrames.frameAt base step).highHigh
        ∧ column ≠ (KFrames.frameAt base step).cross := by
    intro column mentioned
    rcases hornerCarried_mentions beta (KFrames.frameAt base) coefficients
      (step + 1) column mentioned with ⟨c, memberC, inC⟩ | ⟨later, ordered, _, inFrame⟩
    · have below : column < base := by
        rcases inC with low | high
        · exact (coeffBelow c memberC).1 column low
        · exact (coeffBelow c memberC).2 column high
      refine ⟨?_, ?_, ?_⟩ <;> intro equal <;>
        exact absurd below (by
          have := frameColumn_ge_base base step 0
          have := frameColumn_ge_base base step 1
          have := frameColumn_ge_base base step 2
          simp only [KFrames.frameAt] at equal
          omega)
    · have stepLt : step < later := by omega
      refine ⟨?_, ?_, ?_⟩ <;> intro equal <;>
        (rcases inFrame with h | h | h <;>
          simp only [KFrames.frameAt] at equal h <;>
          exact absurd (h ▸ equal)
            (later_frame_fresh base step later _ _ stepLt (by decide) (by decide)))
  refine ⟨⟨?_, ?_, ?_⟩, ⟨?_, ?_, ?_⟩⟩ <;> intro mentioned
  · exact (classify _ (Or.inl mentioned)).1 rfl
  · exact (classify _ (Or.inl mentioned)).2.1 rfl
  · exact (classify _ (Or.inl mentioned)).2.2 rfl
  · exact (classify _ (Or.inr mentioned)).1 rfl
  · exact (classify _ (Or.inr mentioned)).2.1 rfl
  · exact (classify _ (Or.inr mentioned)).2.2 rfl

/-! ## The witness, and completeness

Built inside out: the deepest step's frame is written first, and each enclosing
step extends it. -/

/-- **The honest witness for a whole evaluation.** -/
def hornerWitness (z : Nat → Nat) (beta : Carried) (base : Nat) :
    List Carried → Nat → (Nat → Nat)
  | [], _ => z
  | [_], _ => z
  | _ :: next :: rest, step =>
      KMulHonest.witness
        (hornerWitness z beta base (next :: rest) (step + 1))
        beta (hornerCarried beta (KFrames.frameAt base) (next :: rest) (step + 1))
        (KFrames.frameAt base step)

/-- **A block's witness writes only inside its own block.**

Every column the witness touches is a frame column of step `step` or later, so
anything strictly below `base + 3·step` keeps its value. This is the primitive
that lets one block's witness extend another's without disturbing it — the
composition direction, where `hornerWitness_satisfies` gives only the single
block. -/
theorem hornerWitness_off_block (z : Nat → Nat) (beta : Carried) (base : Nat) :
    ∀ (coefficients : List Carried) (step column : Nat),
      column < base + 3 * step →
      hornerWitness z beta base coefficients step column = z column
  | [], _, _, _ => rfl
  | [_], _, _, _ => rfl
  | _ :: next :: rest, step, column, below => by
      show KMulHonest.witness
          (hornerWitness z beta base (next :: rest) (step + 1)) beta
          (hornerCarried beta (KFrames.frameAt base) (next :: rest) (step + 1))
          (KFrames.frameAt base step) column
        = z column
      rw [KMulHonest.witness_off_frame _ _ _ _ column
          (by simp only [KFrames.frameAt, KFrames.frameColumn,
                KFrames.columnsPerFrame]; omega)
          (by simp only [KFrames.frameAt, KFrames.frameColumn,
                KFrames.columnsPerFrame]; omega)
          (by simp only [KFrames.frameAt, KFrames.frameColumn,
                KFrames.columnsPerFrame]; omega),
        hornerWitness_off_block z beta base (next :: rest) (step + 1) column
          (by omega)]

/-- Horner completion preserves canonical representatives at every nested
multiplication frame. -/
theorem hornerWitness_residues
    (z : Nat → Nat) (beta : Carried) (base : Nat) :
    ∀ (coefficients : List Carried) (step : Nat),
      (∀ column, z column < goldilocksP) →
      ∀ column,
        hornerWitness z beta base coefficients step column < goldilocksP
  | [], _, residues => residues
  | [_], _, residues => residues
  | _ :: next :: rest, step, residues => by
      exact KMulHonest.witness_residues
        (hornerWitness z beta base (next :: rest) (step + 1))
        beta
        (hornerCarried beta (KFrames.frameAt base)
          (next :: rest) (step + 1))
        (KFrames.frameAt base step)
        (hornerWitness_residues z beta base
          (next :: rest) (step + 1) residues)

/-- **An honest execution satisfies the whole evaluation.**  The assembly of
`satisfies_extend`, `hornerCarried_mentions`, `hornerRows_mentions` and
`suffix_fresh`, over the canonical allocator. -/
theorem hornerWitness_satisfies
    (z : Nat → Nat) (beta : Carried) (base : Nat)
    (betaLow : BelowBase beta.low base) (betaHigh : BelowBase beta.high base) :
    ∀ (coefficients : List Carried) (step : Nat),
      (∀ c ∈ coefficients, BelowBase c.low base ∧ BelowBase c.high base) →
      Satisfies (hornerRows beta (KFrames.frameAt base) coefficients step)
        (hornerWitness z beta base coefficients step)
  | [], _, _ => by intro row member; simp [hornerRows] at member
  | [_], _, _ => by intro row member; simp [hornerRows] at member
  | c :: next :: rest, step, coeffBelow => by
      have tailBelow : ∀ d ∈ next :: rest,
          BelowBase d.low base ∧ BelowBase d.high base :=
        fun d memberD => coeffBelow d (List.mem_cons_of_mem _ memberD)
      have inner := hornerWitness_satisfies z beta base betaLow betaHigh
        (next :: rest) (step + 1) tailBelow
      have suffixFresh := suffix_fresh beta base (next :: rest) step tailBelow
      intro row member
      simp only [hornerRows, List.mem_append] at member
      rcases member with inMul | inTail
      · exact KMulHonest.witness_satisfies _ beta _ _
          (KMulHonest.canonical_distinct base step)
          (fresh_of_belowBase _ base step betaLow)
          (fresh_of_belowBase _ base step betaHigh)
          suffixFresh.1 suffixFresh.2 row inMul
      · refine satisfies_extend _ _ _ ?_ inner row inTail
        intro other otherMember column mentioned
        refine (KMulHonest.witness_off_frame _ beta _ _ column ?_ ?_ ?_).symm
        all_goals
          rcases hornerRows_mentions beta (KFrames.frameAt base) (next :: rest)
            (step + 1) other otherMember column mentioned with
            inBeta | ⟨d, memberD, inD⟩ | ⟨later, ordered, _, inF⟩
        all_goals first
          | (intro equal
             have below : column < base := by
               rcases inBeta with bl | bh
               · exact betaLow column bl
               · exact betaHigh column bh
             have := frameColumn_ge_base base step 0
             have := frameColumn_ge_base base step 1
             have := frameColumn_ge_base base step 2
             simp only [KFrames.frameAt] at equal
             omega)
          | (intro equal
             have below : column < base := by
               rcases inD with dl | dh
               · exact (tailBelow d memberD).1 column dl
               · exact (tailBelow d memberD).2 column dh
             have := frameColumn_ge_base base step 0
             have := frameColumn_ge_base base step 1
             have := frameColumn_ge_base base step 2
             simp only [KFrames.frameAt] at equal
             omega)
          | (intro equal
             rcases inF with h | h | h <;>
               simp only [KFrames.frameAt] at equal h <;>
               exact absurd (h ▸ equal)
                 (later_frame_fresh base step later _ _ (by omega)
                   (by decide) (by decide)))

end Nightstream.Implementation.R1CS.Canonical.KHornerHonest
