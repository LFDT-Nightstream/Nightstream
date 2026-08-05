import Nightstream.SuperNeo.ProjectionCheck

namespace NightstreamTests.ProjectionCheck

open Nightstream.SuperNeo.ProjectionCheck

def mod97 : Ops Nat where
  zero := 0
  add := fun left right => (left + right) % 97
  mul := fun left right => (left * right) % 97

/-- `X - 7` is nonzero coefficient-wise but vanishes at beta = 7. -/
def fixedBetaForgery : Identity Nat where
  lhs := [90, 1]
  rhs := [0, 0]
  beta := 7
  maxDegree := 1

def exactIdentity : Identity Nat where
  lhs := [3, 4]
  rhs := [3, 4]
  beta := 7
  maxDegree := 1

example : fixedBetaForgery.WellFormed := by decide
example : ¬ fixedBetaForgery.Exact := by decide
example : eval mod97 fixedBetaForgery.lhs fixedBetaForgery.beta = 0 := by decide
example : eval mod97 fixedBetaForgery.rhs fixedBetaForgery.beta = 0 := by decide

theorem fixedBetaForgery_is_badRoot : BadRoot mod97 fixedBetaForgery := by
  exact ⟨by decide, by decide, by decide⟩

example : Accepted mod97 fixedBetaForgery :=
  badRoot_is_accepted mod97 fixedBetaForgery fixedBetaForgery_is_badRoot

example : Accepted mod97 exactIdentity := by decide

example :
    BatchExact [exactIdentity, fixedBetaForgery] ∨
      BatchBadRoot mod97 [exactIdentity, fixedBetaForgery] := by
  apply batchAccepted_implies_exact_or_badRoot
  intro identity member
  simp only [List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl <;> decide

end NightstreamTests.ProjectionCheck
