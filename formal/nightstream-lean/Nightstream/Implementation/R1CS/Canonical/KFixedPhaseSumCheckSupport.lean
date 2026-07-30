import Nightstream.Implementation.R1CS.Canonical.KFixedPhaseSumCheckHonest

/-!
Contract: whole-program column conservation for the Lean-owned fixed-phase
SumCheck rows.

Owns the finite allocation boundary: source combinations lie below `base`,
round `i` owns exactly the interval beginning at `base + i*(3*degree)`, and
no emitted row can mention a column at or beyond the end of the selected
round blocks.

Does not own typed `ColumnId`/`RowId` wrapping or the enclosing call receipt.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KFixedPhaseSumCheckSupport

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.KHorner
open Nightstream.Implementation.R1CS.Canonical.KFixedPhaseSumCheck
open Nightstream.Implementation.R1CS.Canonical.KFixedPhaseSumCheckHonest

/-- **Every emitted row stays inside the selected finite column boundary.**

This is stronger than a row-count statement: it follows each recursive
current value into the preceding Horner block and proves the next block starts
after all of those columns. -/
theorem chainRows_columns_below_end
    {degree : Nat}
    (current : Carried)
    (rounds : List (Round degree))
    (challenges : List Carried)
    (terminal : Carried)
    (base : Nat)
    (basePositive : 0 < base)
    (currentBelow : CarriedBelow current base)
    (roundsBelow : ∀ round ∈ rounds, RoundBelow round base)
    (challengesBelow :
      ∀ challenge ∈ challenges, CarriedBelow challenge base)
    (terminalBelow : CarriedBelow terminal base)
    (sameLength : rounds.length = challenges.length)
    (row : Row)
    (member : row ∈ chainRows current rounds challenges terminal base)
    (column : Nat)
    (mentioned : Mentions row.a column ∨ Mentions row.b column
      ∨ Mentions row.c column) :
    column < base + rounds.length * (3 * degree) := by
  induction rounds generalizing current challenges base row column with
  | nil =>
      cases challenges with
      | nil =>
          simp only [chainRows] at member
          simpa using equalityRows_below basePositive currentBelow
            terminalBelow row member column mentioned
      | cons _ _ => simp at sameLength
  | cons round rounds inductionHypothesis =>
      cases challenges with
      | nil => simp at sameLength
      | cons challenge challenges =>
          have tailLength : rounds.length = challenges.length := by
            simpa using sameLength
          have roundBelow : RoundBelow round base :=
            roundsBelow round (by simp)
          have challengeBelow : CarriedBelow challenge base :=
            challengesBelow challenge (by simp)
          simp only [chainRows, List.mem_append] at member
          rcases member with (inEquality | inHorner) | inRest
          · have below := equalityRows_below basePositive currentBelow
              (roundInitial_below round roundBelow)
              row inEquality column mentioned
            simp only [List.length_cons, Nat.succ_mul]
            omega
          · have below := hornerRows_below_next round challenge
              challengeBelow roundBelow row inHorner column mentioned
            simp only [List.length_cons, Nat.succ_mul]
            omega
          · have nextBasePositive : 0 < base + 3 * degree := by omega
            have nextCurrentBelow :=
              hornerCarried_below_next round challenge roundBelow
            have recursive :=
              inductionHypothesis
                (hornerCarried challenge (KFrames.frameAt base)
                  round.coefficients 0)
                challenges (base + 3 * degree)
                nextBasePositive nextCurrentBelow
                (fun next nextMember coefficient coefficientMember =>
                  carriedBelow_mono
                    (roundsBelow next
                      (List.mem_cons_of_mem round nextMember)
                      coefficient coefficientMember)
                    (by omega))
                (fun next nextMember =>
                  carriedBelow_mono
                    (challengesBelow next
                      (List.mem_cons_of_mem challenge nextMember))
                    (by omega))
                (carriedBelow_mono terminalBelow (by omega))
                tailLength row inRest column mentioned
            simp only [List.length_cons, Nat.succ_mul] at recursive ⊢
            omega

/-- The owned auxiliary interval has the exact width advertised by
`chainCost`; the result is definitional arithmetic, not a measured layout. -/
theorem allocationEnd_eq
    (degree roundCount base : Nat) :
    base + roundCount * (3 * degree) =
      base + (chainCost degree roundCount).auxiliaryColumns := by
  rfl

end Nightstream.Implementation.R1CS.Canonical.KFixedPhaseSumCheckSupport
