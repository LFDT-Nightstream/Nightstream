import Nightstream.SuperNeo.SumCheck.FixedPhase.Sequential

/-!
Focused regressions for sequential honest fixed-width SumCheck construction.

| Property under test | Failure caught |
|---|---|
| the round is fixed before its challenge is derived | future-challenge fixed-point assumption |
| replayed challenges match the constructed honest certificate | transcript/honesty drift |
| exact-list reindexing preserves order | certificate permutation |
-/

namespace NightstreamTests.SumCheckFixedPhaseSequential

open Nightstream.SuperNeo.SumCheck.Finite
open Nightstream.SuperNeo.SumCheck.Finite.FixedPhase
open Nightstream.SuperNeo.SumCheck.Finite.FixedPhase.Sequential

private def natOps : Ops Nat where
  zero := 0
  one := 1
  add := Nat.add
  mul := Nat.mul

private def coordinate : List Nat -> Nat
  | value :: _ => value
  | [] => 0

private def paddedCoordinate : FixedPolynomial Nat 4 where
  coefficients := [0, 1, 0, 0, 0]
  coefficients_length := rfl

private def step (state : Nat) (polynomial : FixedPolynomial Nat 4) :
    Nat × Nat :=
  (polynomial.evaluate natOps state, state + 1)

private theorem coordinateRoundRepresentable :
    RoundRepresentable natOps coordinate 4 1 := by
  intro fixed remaining length
  have fixedNil : fixed = [] := by
    cases fixed with
    | nil => rfl
    | cons _ tail =>
        simp only [List.length_cons] at length
        omega
  have remainingZero : remaining = 0 := by
    subst fixed
    simp at length
    omega
  subst fixed
  subst remaining
  refine ⟨paddedCoordinate, ?_⟩
  intro point
  simp [FixedPolynomial.evaluate, FixedPolynomial.toMessage,
    paddedCoordinate, Message.evaluate, Message.evaluateCoefficients,
    natOps, coordinate, HypercubeTruth.sumCompletions]

example :
    ∃ certificate : Certificate Nat 4,
      ∃ challenges : List Nat,
        ∃ finalState : Nat,
          certificate.rounds.length = 1 ∧
          challenges.length = 1 ∧
          run step 3 certificate.rounds = (challenges, finalState) ∧
          Honest natOps coordinate challenges certificate :=
  exists_honest_run natOps coordinate 4 1 step
    coordinateRoundRepresentable 3

example :
    List.ofFn
      (functionOfExactList [4, 5, 6] (by decide : [4, 5, 6].length = 3)) =
        [4, 5, 6] := by
  exact ofFn_functionOfExactList ([4, 5, 6] : List Nat)
    (by decide : [4, 5, 6].length = 3)

end NightstreamTests.SumCheckFixedPhaseSequential
