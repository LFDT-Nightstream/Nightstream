import Nightstream.Implementation.R1CS.Canonical.AllocationCoverage
import Nightstream.Implementation.R1CS.Canonical.KFixedPhaseSumCheck

/-!
Contract: the carried endpoints consumed by a fixed-phase SumCheck chain are
actually mentioned by its emitted equality rows.

These columns are shared reads, not Horner allocations.  The theorem is used
by enclosing programs that deliberately own the endpoint coordinates.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KFixedPhaseEndpointCoverage

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.AllocationCoverage
open Nightstream.Implementation.R1CS.Canonical.KFixedPhaseSumCheck
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal

/-- Ordered support of one carried quadratic-extension value. -/
def columns (value : Carried) : List Nat :=
  value.low.map Prod.fst ++ value.high.map Prod.fst

private theorem equality
    (left right : Carried) :
    RowsCover (KEquality.rows left right)
      (columns left ++ columns right) := by
  intro column member
  rcases List.mem_append.1 member with inLeft | inRight
  · rcases List.mem_append.1 inLeft with inLow | inHigh
    · refine
        ⟨KEquality.equalityRow left.low right.low,
          by simp [KEquality.rows],
          Or.inl ?_⟩
      exact inLow
    · refine
        ⟨KEquality.equalityRow left.high right.high,
          by simp [KEquality.rows],
          Or.inl ?_⟩
      exact inHigh
  · rcases List.mem_append.1 inRight with inLow | inHigh
    · refine
        ⟨KEquality.equalityRow left.low right.low,
          by simp [KEquality.rows],
          Or.inr (Or.inr ?_)⟩
      exact inLow
    · refine
        ⟨KEquality.equalityRow left.high right.high,
          by simp [KEquality.rows],
          Or.inr (Or.inr ?_)⟩
      exact inHigh

/-- The first and final carried values are covered even though the chain may
contain any number of Horner rounds between them. -/
theorem chain
    {degree : Nat}
    (current : Carried)
    (rounds : List (Round degree))
    (challenges : List Carried)
    (terminal : Carried)
    (base : Nat)
    (sameLength : rounds.length = challenges.length) :
    RowsCover
      (chainRows current rounds challenges terminal base)
      (columns current ++ columns terminal) := by
  induction rounds generalizing current challenges base with
  | nil =>
      cases challenges with
      | nil =>
          simpa only [chainRows] using equality current terminal
      | cons _ _ =>
          simp at sameLength
  | cons round rest inductionHypothesis =>
      cases challenges with
      | nil =>
          simp at sameLength
      | cons challenge remaining =>
          have tailSame : rest.length = remaining.length := by
            simpa only [List.length_cons, Nat.succ.injEq] using sameLength
          intro column member
          rcases List.mem_append.1 member with inCurrent | inTerminal
          · have covered := equality current (roundInitial round.coefficients)
            rcases covered column
                (List.mem_append_left _ inCurrent) with
              ⟨row, rowMember, mentioned⟩
            refine ⟨row, ?_, mentioned⟩
            simp [chainRows, rowMember]
          · have covered :=
              inductionHypothesis
                (KHorner.hornerCarried challenge (KFrames.frameAt base)
                  round.coefficients 0)
                remaining (base + 3 * degree) tailSame
            rcases covered column
                (List.mem_append_right _ inTerminal) with
              ⟨row, rowMember, mentioned⟩
            refine ⟨row, ?_, mentioned⟩
            simp [chainRows, rowMember]

end Nightstream.Implementation.R1CS.Canonical.KFixedPhaseEndpointCoverage
