import Nightstream.Implementation.R1CS.Canonical.KHornerHonest
import Nightstream.Implementation.R1CS.Canonical.KMulChain

/-!
Contract: honest completeness for a sequential `K` multiplication chain.

Owns: the forward witness, its placement guarantee, and satisfaction of every
emitted row under the canonical consecutive-frame allocator.

Does not own: semantic selection of the factors. A PiCCS consumer supplies the
paper expressions; this module only proves that their products have an honest
physical realization.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KMulChainHonest

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.KMulChain
open Nightstream.Implementation.R1CS.Canonical.KHornerHonest
open Nightstream.Implementation.R1CS.Canonical.KHornerSupport

/-- Every mentioned column precedes the frame at `step`. -/
def BelowStep (comb : LinComb) (base step : Nat) : Prop :=
  ∀ column, Mentions comb column → column < base + 3 * step

theorem belowBase_to_belowStep
    (comb : LinComb) (base step : Nat) (below : BelowBase comb base) :
    BelowStep comb base step := by
  intro column mentioned
  have := below column mentioned
  omega

/-- A combination placed before the current frame is fresh for it. -/
theorem fresh_of_belowStep
    (comb : LinComb) (base step : Nat) (below : BelowStep comb base step) :
    KMulHonest.Fresh comb (KFrames.frameAt base step) := by
  refine ⟨?_, ?_, ?_⟩ <;> intro mentioned
  · exact absurd (below _ mentioned) (by
      simp only [KFrames.frameAt, KFrames.frameColumn,
        KFrames.columnsPerFrame]
      omega)
  · exact absurd (below _ mentioned) (by
      simp only [KFrames.frameAt, KFrames.frameColumn,
        KFrames.columnsPerFrame]
      omega)
  · exact absurd (below _ mentioned) (by
      simp only [KFrames.frameAt, KFrames.frameColumn,
        KFrames.columnsPerFrame]
      omega)

/-- A frame's carried output lies strictly before the following frame. -/
theorem frameOutput_below_next (base step : Nat) :
    BelowStep (frameOutput (KFrames.frameAt base step)).low base (step + 1)
      ∧ BelowStep (frameOutput (KFrames.frameAt base step)).high base (step + 1) := by
  constructor <;> intro column mentioned
  · simp only [frameOutput, outLow, Mentions, List.map_cons, List.map_nil,
      List.mem_cons, List.not_mem_nil, or_false] at mentioned
    rcases mentioned with rfl | rfl <;>
      simp only [KFrames.frameAt, KFrames.frameColumn,
        KFrames.columnsPerFrame] <;> omega
  · simp only [frameOutput, outHigh, Mentions, List.map_cons, List.map_nil,
      List.mem_cons, List.not_mem_nil, or_false] at mentioned
    rcases mentioned with rfl | rfl | rfl <;>
      simp only [KFrames.frameAt, KFrames.frameColumn,
        KFrames.columnsPerFrame] <;> omega

/-- Extend an assignment in execution order, one multiplication at a time. -/
def witness (assignment : Nat → Nat) (initial : Carried)
    (factors : List Carried) (base step : Nat) : Nat → Nat :=
  match factors with
  | [] => assignment
  | factor :: rest =>
      let extended :=
        KMulHonest.witness assignment initial factor (KFrames.frameAt base step)
      witness extended (frameOutput (KFrames.frameAt base step))
        rest base (step + 1)

/-- A chain only writes its own frame block and later blocks. -/
theorem witness_off_before
    (assignment : Nat → Nat) (initial : Carried) (base : Nat) :
    ∀ (factors : List Carried) (step column : Nat),
      column < base + 3 * step →
      witness assignment initial factors base step column = assignment column
  | [], _, _, _ => rfl
  | factor :: rest, step, column, below => by
      rw [witness,
        witness_off_before
          (KMulHonest.witness assignment initial factor
            (KFrames.frameAt base step))
          (frameOutput (KFrames.frameAt base step)) base rest
          (step + 1) column (by omega),
        KMulHonest.witness_off_frame assignment initial factor
          (KFrames.frameAt base step) column]
      all_goals
        simp only [KFrames.frameAt, KFrames.frameColumn,
          KFrames.columnsPerFrame]
        omega

/-- Every column touched by one multiplication lies before the next frame,
provided its two operands do. -/
theorem head_columns_below_next
    (initial factor : Carried) (base step : Nat)
    (initialLow : BelowStep initial.low base step)
    (initialHigh : BelowStep initial.high base step)
    (factorLow : BelowBase factor.low base)
    (factorHigh : BelowBase factor.high base)
    (row : Row)
    (member :
      row ∈ KMul.rows initial factor (KFrames.frameAt base step))
    (column : Nat)
    (mentioned :
      Mentions row.a column ∨ Mentions row.b column ∨ Mentions row.c column) :
    column < base + 3 * (step + 1) := by
  rcases KMulOwnership.rows_conservation initial factor
      (KFrames.frameAt base step) row member column mentioned with
    operand | frameColumn
  · rcases operand with inInitialLow | inInitialHigh | inFactorLow | inFactorHigh
    · have := initialLow column inInitialLow
      omega
    · have := initialHigh column inInitialHigh
      omega
    · have := factorLow column inFactorLow
      omega
    · have := factorHigh column inFactorHigh
      omega
  · rcases frameColumn with rfl | rfl | rfl <;>
      simp only [KFrames.frameAt, KFrames.frameColumn,
        KFrames.columnsPerFrame] <;> omega

/-- The forward witness satisfies the entire chain. -/
theorem witness_satisfies
    (assignment : Nat → Nat) (base : Nat) :
    ∀ (initial : Carried) (factors : List Carried) (step : Nat),
      BelowStep initial.low base step →
      BelowStep initial.high base step →
      (∀ factor ∈ factors,
        BelowBase factor.low base ∧ BelowBase factor.high base) →
      Satisfies
        (KMulChain.rows initial (KFrames.frameAt base) factors step)
        (witness assignment initial factors base step)
  | _, [], _, _, _, _ => by
      intro row member
      simp [KMulChain.rows] at member
  | initial, factor :: rest, step, initialLow, initialHigh, factorsBelow => by
      let extended :=
        KMulHonest.witness assignment initial factor
          (KFrames.frameAt base step)
      have factorBelow := factorsBelow factor (by simp)
      have headSatisfied :
          Satisfies
            (KMul.rows initial factor (KFrames.frameAt base step))
            extended :=
        KMulHonest.witness_satisfies assignment initial factor
          (KFrames.frameAt base step)
          (KMulHonest.canonical_distinct base step)
          (fresh_of_belowStep initial.low base step initialLow)
          (fresh_of_belowStep initial.high base step initialHigh)
          (fresh_of_belowStep factor.low base step
            (belowBase_to_belowStep factor.low base step factorBelow.1))
          (fresh_of_belowStep factor.high base step
            (belowBase_to_belowStep factor.high base step factorBelow.2))
      have outputBelow := frameOutput_below_next base step
      have restBelow : ∀ other ∈ rest,
          BelowBase other.low base ∧ BelowBase other.high base :=
        fun other member =>
          factorsBelow other (List.mem_cons_of_mem factor member)
      have tailSatisfied :=
        witness_satisfies extended base
          (frameOutput (KFrames.frameAt base step)) rest (step + 1)
          outputBelow.1 outputBelow.2 restBelow
      intro row member
      simp only [KMulChain.rows, List.mem_append] at member
      rcases member with inHead | inTail
      · refine satisfies_extend
          (KMul.rows initial factor (KFrames.frameAt base step))
          extended
          (witness extended (frameOutput (KFrames.frameAt base step))
            rest base (step + 1))
          ?_ headSatisfied row inHead
        intro owned ownedMember column mentioned
        symm
        apply witness_off_before
        exact head_columns_below_next initial factor base step
          initialLow initialHigh factorBelow.1 factorBelow.2
          owned ownedMember column mentioned
      · exact tailSatisfied row inTail

/-- The common entry point: all source operands lie below `base`, and the
chain starts at frame zero. -/
theorem witness_satisfies_from_base
    (assignment : Nat → Nat) (initial : Carried) (factors : List Carried)
    (base : Nat)
    (initialLow : BelowBase initial.low base)
    (initialHigh : BelowBase initial.high base)
    (factorsBelow : ∀ factor ∈ factors,
      BelowBase factor.low base ∧ BelowBase factor.high base) :
    Satisfies
      (KMulChain.rows initial (KFrames.frameAt base) factors 0)
      (witness assignment initial factors base 0) :=
  witness_satisfies assignment base initial factors 0
    (belowBase_to_belowStep initial.low base 0 initialLow)
    (belowBase_to_belowStep initial.high base 0 initialHigh)
    factorsBelow

end Nightstream.Implementation.R1CS.Canonical.KMulChainHonest
