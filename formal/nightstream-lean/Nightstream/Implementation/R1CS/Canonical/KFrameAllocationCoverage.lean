import Nightstream.Implementation.R1CS.Canonical.AllocationCoverage
import Nightstream.Implementation.R1CS.Canonical.KFixedPhaseSumCheck
import Nightstream.Implementation.R1CS.Canonical.KMulChain

/-!
Contract: exact converse-to-conservation coverage for the contiguous
three-column `K` multiplication allocator.

Every column declared by a multiplication, Horner evaluation, multiplication
chain, or fixed-phase SumCheck chain occurs in an emitted row operand.  The
proofs use the Lean-owned row constructors and allocator, not row counts.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KFrameAllocationCoverage

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.AllocationCoverage
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.KHorner
open Nightstream.Implementation.R1CS.Canonical.KFixedPhaseSumCheck

private theorem no_columns (rows : List Row) (base : Nat) :
    RowsCover rows (KFrames.frameColumns base 0) := by
  intro column member
  simp [KFrames.frameColumns] at member

/-- All three columns of one canonical multiplication frame occur as row
targets. -/
theorem mul
    (left right : Carried) (base step : Nat) :
    RowsCover
      (KMul.rows left right (KFrames.frameAt base step))
      (KFrames.frameColumns (base + 3 * step) 1) := by
  intro column member
  rw [KFrames.frameColumns_mem_iff] at member
  have classified :
      column = (KFrames.frameAt base step).lowLow ∨
        column = (KFrames.frameAt base step).highHigh ∨
          column = (KFrames.frameAt base step).cross := by
    simp only [KFrames.frameAt, KFrames.frameColumn,
      KFrames.columnsPerFrame]
    omega
  rcases classified with low | high | cross
  · subst column
    refine
      ⟨KMul.productRow left.low right.low
          (KFrames.frameAt base step).lowLow,
        by simp [KMul.rows], Or.inr (Or.inr ?_)⟩
    simp [KMul.productRow, LinCombNormal.Mentions]
  · subst column
    refine
      ⟨KMul.productRow left.high right.high
          (KFrames.frameAt base step).highHigh,
        by simp [KMul.rows], Or.inr (Or.inr ?_)⟩
    simp [KMul.productRow, LinCombNormal.Mentions]
  · subst column
    refine
      ⟨KMul.productRow (KMul.sumComb left) (KMul.sumComb right)
          (KFrames.frameAt base step).cross,
        by simp [KMul.rows], Or.inr (Or.inr ?_)⟩
    simp [KMul.productRow, LinCombNormal.Mentions]

/-- A Horner program covers exactly the frames from its starting step through
its last nontrivial coefficient. -/
theorem horner (beta : Carried) (base : Nat) :
    ∀ (coefficients : List Carried) (step : Nat),
      RowsCover
        (KHorner.hornerRows beta (KFrames.frameAt base) coefficients step)
        (KFrames.frameColumns (base + 3 * step)
          (coefficients.length - 1))
  | [], step => no_columns _ _
  | [_], step => no_columns _ _
  | coefficient :: next :: rest, step => by
      intro column member
      rw [KFrames.frameColumns_mem_iff] at member
      by_cases inHead : column < base + 3 * step + 3
      · have headColumn :
            column ∈ KFrames.frameColumns (base + 3 * step) 1 := by
          rw [KFrames.frameColumns_mem_iff]
          omega
        rcases mul beta
            (KHorner.hornerCarried beta (KFrames.frameAt base)
              (next :: rest) (step + 1))
            base step column headColumn with
          ⟨row, rowMember, mentioned⟩
        exact
          ⟨row, List.mem_append_left _ rowMember, mentioned⟩
      · have tailColumn :
            column ∈
              KFrames.frameColumns (base + 3 * (step + 1))
                ((next :: rest).length - 1) := by
          rw [KFrames.frameColumns_mem_iff]
          simp only [List.length_cons, Nat.succ_sub_one] at member ⊢
          omega
        rcases horner beta base (next :: rest) (step + 1)
            column tailColumn with ⟨row, rowMember, mentioned⟩
        exact
          ⟨row, List.mem_append_right _ rowMember, mentioned⟩

/-- A left-to-right multiplication chain covers exactly one canonical frame
per factor. -/
theorem mulChain (base : Nat) :
    ∀ (initial : Carried) (factors : List Carried) (step : Nat),
      RowsCover
        (KMulChain.rows initial (KFrames.frameAt base) factors step)
        (KFrames.frameColumns (base + 3 * step) factors.length)
  | _, [], step => no_columns _ _
  | initial, factor :: rest, step => by
      intro column member
      rw [KFrames.frameColumns_mem_iff] at member
      by_cases inHead : column < base + 3 * step + 3
      · have headColumn :
            column ∈ KFrames.frameColumns (base + 3 * step) 1 := by
          rw [KFrames.frameColumns_mem_iff]
          omega
        rcases mul initial factor base step column headColumn with
          ⟨row, rowMember, mentioned⟩
        exact
          ⟨row, List.mem_append_left _ rowMember, mentioned⟩
      · have tailColumn :
            column ∈
              KFrames.frameColumns (base + 3 * (step + 1)) rest.length := by
          rw [KFrames.frameColumns_mem_iff]
          simp only [List.length_cons, Nat.succ_mul] at member
          omega
        rcases mulChain base (KMulChain.frameOutput
              (KFrames.frameAt base step))
            rest (step + 1) column tailColumn with
          ⟨row, rowMember, mentioned⟩
        exact
          ⟨row, List.mem_append_right _ rowMember, mentioned⟩

/-- A well-shaped fixed-phase chain covers its exact contiguous Horner
allocation.  Equality rows allocate no columns. -/
theorem fixedPhase
    {degree : Nat}
    (current : Carried)
    (rounds : List (Round degree))
    (challenges : List Carried)
    (terminal : Carried)
    (base : Nat)
    (sameLength : rounds.length = challenges.length) :
    RowsCover
      (chainRows current rounds challenges terminal base)
      (KFrames.frameColumns base (rounds.length * degree)) := by
  induction rounds generalizing current challenges base with
  | nil =>
      cases challenges with
      | nil =>
          simpa [chainRows] using
            (no_columns (KEquality.rows current terminal) base)
      | cons _ _ => simp at sameLength
  | cons round rounds inductionHypothesis =>
      cases challenges with
      | nil => simp at sameLength
      | cons challenge challenges =>
          have tailLength : rounds.length = challenges.length := by
            simpa using sameLength
          intro column member
          rw [KFrames.frameColumns_mem_iff] at member
          by_cases inHead : column < base + 3 * degree
          · have headColumn :
                column ∈ KFrames.frameColumns base degree := by
              rw [KFrames.frameColumns_mem_iff]
              omega
            have hornerCovered :=
              horner challenge base round.coefficients 0
            have coefficientCount :
                round.coefficients.length - 1 = degree := by
              rw [round.coefficients_length]
              omega
            rw [coefficientCount] at hornerCovered
            rcases hornerCovered column headColumn with
              ⟨row, rowMember, mentioned⟩
            exact
              ⟨row,
                List.mem_append_left _
                  (List.mem_append_right _ rowMember),
                mentioned⟩
          · have tailColumn :
                column ∈
                  KFrames.frameColumns (base + 3 * degree)
                    (rounds.length * degree) := by
              rw [KFrames.frameColumns_mem_iff]
              simp only [List.length_cons, Nat.succ_mul] at member
              omega
            rcases inductionHypothesis
                (KHorner.hornerCarried challenge
                  (KFrames.frameAt base) round.coefficients 0)
                challenges (base + 3 * degree) tailLength
                column tailColumn with
              ⟨row, rowMember, mentioned⟩
            exact
              ⟨row,
                List.mem_append_right _ rowMember,
                mentioned⟩

end Nightstream.Implementation.R1CS.Canonical.KFrameAllocationCoverage
