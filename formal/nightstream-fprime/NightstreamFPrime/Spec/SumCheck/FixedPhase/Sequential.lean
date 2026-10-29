import NightstreamFPrime.Spec.SumCheck.FixedPhase

namespace NightstreamFPrime.Spec.SumCheck.Finite.FixedPhase.Sequential

universe uField uState uElement

def run
    {Field : Type uField}
    {State : Type uState}
    {degree : Nat}
    (step : State -> FixedPolynomial Field degree -> Field × State) :
    State -> List (FixedPolynomial Field degree) -> List Field × State
  | state, [] => ([], state)
  | state, polynomial :: polynomials =>
      let sample := step state polynomial
      let tail := run step sample.2 polynomials
      (sample.1 :: tail.1, tail.2)

def functionOfExactList
    {Element : Type uElement}
    {count : Nat}
    (values : List Element)
    (length : values.length = count) :
    Fin count -> Element :=
  fun index => values.get (Fin.cast length.symm index)

@[simp] theorem ofFn_functionOfExactList
    {Element : Type uElement}
    {count : Nat}
    (values : List Element)
    (length : values.length = count) :
    List.ofFn (functionOfExactList values length) = values := by
  apply List.ext_get
  · simp [length]
  · intro index leftLt rightLt
    simp only [List.get_eq_getElem, List.getElem_ofFn]
    rfl

def RoundRepresentable
    {Field : Type uField}
    (ops : Ops Field)
    (q : List Field -> Field)
    (degree totalRounds : Nat) : Prop :=
  ∀ (fixed : List Field) (remaining : Nat),
    fixed.length + 1 + remaining = totalRounds ->
      ∃ polynomial : FixedPolynomial Field degree,
        Represents ops polynomial fun point =>
          HypercubeTruth.sumCompletions ops q
            (fixed ++ [point]) remaining

private theorem exists_honest_run_from
    {Field : Type uField}
    {State : Type uState}
    (ops : Ops Field)
    (q : List Field -> Field)
    (degree totalRounds : Nat)
    (step : State -> FixedPolynomial Field degree -> Field × State)
    (roundRepresentable :
      RoundRepresentable ops q degree totalRounds)
    (fixed : List Field)
    (state : State)
    (remaining : Nat)
    (length : fixed.length + remaining = totalRounds) :
    ∃ rounds : List (FixedPolynomial Field degree),
      ∃ challenges : List Field,
        ∃ finalState : State,
          rounds.length = remaining ∧
          challenges.length = remaining ∧
          run step state rounds = (challenges, finalState) ∧
          Representations ops rounds
            (HypercubeTruth.expectedPolynomialsFrom
              ops q fixed challenges) := by
  induction remaining generalizing fixed state with
  | zero =>
      exact ⟨[], [], state, rfl, rfl, rfl, by
        simp [HypercubeTruth.expectedPolynomialsFrom, Representations]⟩
  | succ remaining inductionHypothesis =>
      have roundLength :
          fixed.length + 1 + remaining = totalRounds := by
        omega
      rcases roundRepresentable fixed remaining roundLength with
        ⟨polynomial, represents⟩
      let sample := step state polynomial
      have tailLength :
          (fixed ++ [sample.1]).length + remaining = totalRounds := by
        simp only [List.length_append, List.length_singleton]
        omega
      rcases inductionHypothesis
          (fixed := fixed ++ [sample.1])
          (state := sample.2)
          tailLength with
        ⟨rounds, challenges, finalState, roundsLength, challengesLength,
          replay, honestTail⟩
      refine ⟨polynomial :: rounds, sample.1 :: challenges, finalState,
        by simp [roundsLength], by simp [challengesLength], ?_, ?_⟩
      · simp only [run]
        change
          let next := step state polynomial
          let tail := run step next.2 rounds
          (next.1 :: tail.1, tail.2) =
            (sample.1 :: challenges, finalState)
        simp only [sample]
        rw [replay]
      · simp only [HypercubeTruth.expectedPolynomialsFrom,
          Representations]
        constructor
        · intro point
          simpa [challengesLength] using represents point
        · exact honestTail

theorem exists_honest_run
    {Field : Type uField}
    {State : Type uState}
    (ops : Ops Field)
    (q : List Field -> Field)
    (degree totalRounds : Nat)
    (step : State -> FixedPolynomial Field degree -> Field × State)
    (roundRepresentable :
      RoundRepresentable ops q degree totalRounds)
    (initialState : State) :
    ∃ certificate : Certificate Field degree,
      ∃ challenges : List Field,
        ∃ finalState : State,
          certificate.rounds.length = totalRounds ∧
          challenges.length = totalRounds ∧
          run step initialState certificate.rounds =
            (challenges, finalState) ∧
          Honest ops q challenges certificate := by
  rcases exists_honest_run_from ops q degree totalRounds step
      roundRepresentable [] initialState totalRounds (by simp) with
    ⟨rounds, challenges, finalState, roundsLength, challengesLength,
      replay, honest⟩
  refine ⟨{ rounds }, challenges, finalState, roundsLength,
    challengesLength, replay, ?_⟩
  simpa [Honest, expectedRounds, HypercubeTruth.expectedPolynomials] using honest

end NightstreamFPrime.Spec.SumCheck.Finite.FixedPhase.Sequential
