import NightstreamFPrime.Export.Stage1.HyperNovaHistoryProbability

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.HyperNovaHistoryLaw

open HyperNovaHistory

attribute [local instance] Classical.propDecidable

variable (application : Lifecycle.Stage1.Application.Program)
  (fits : PerApplicationFixedPoint.FitsTwoPow28 application)

variable (source : Statement → Payload application → PMF (SourceResult application))

private noncomputable def resultsAt : Nat → Statement → Envelope application → PMF (List (SourceResult application))
  | 0, _, _ => PMF.pure []
  | remaining + 1, statement, proof =>
      match proof with
      | .bottom => PMF.pure []
      | .recursive payload =>
          if statement.iteration = 0 then PMF.pure []
          else if (decodedInput application fits payload).iteration + 1 = statement.iteration then
            if (decodedInput application fits payload).iteration = 0 then PMF.pure []
            else (source statement payload).bind fun result =>
              match result with
              | none => PMF.pure [none]
              | some values =>
                  (resultsAt remaining (predecessorStatement application fits payload)
                    (.recursive (predecessorPayload application fits payload values))).map (some values :: ·)
          else PMF.pure []

noncomputable def results (statement : Statement) (proof : Envelope application) : PMF (List (SourceResult application)) :=
  resultsAt application fits source statement.iteration statement proof

theorem results_eq (statement : Statement) (proof : Envelope application) :
    results application fits source statement proof =
      match proof with
      | .bottom => PMF.pure []
      | .recursive payload =>
          if statement.iteration = 0 then PMF.pure []
          else if (decodedInput application fits payload).iteration + 1 = statement.iteration then
            if (decodedInput application fits payload).iteration = 0 then PMF.pure []
            else (source statement payload).bind fun result =>
              match result with
              | none => PMF.pure [none]
              | some values =>
                  (results application fits source (predecessorStatement application fits payload)
                    (.recursive (predecessorPayload application fits payload values))).map (some values :: ·)
          else PMF.pure [] := by
  cases count : statement.iteration with
  | zero =>
      cases proof <;> simp only [results, count, resultsAt, ite_true]
  | succ remaining =>
      cases proof with
      | bottom => simp only [results, count, resultsAt]
      | recursive payload =>
          simp only [results, count, resultsAt, Nat.succ_ne_zero, ite_false]
          by_cases counter : (decodedInput application fits payload).iteration + 1 = remaining + 1
          · have previousCount : (predecessorStatement application fits payload).iteration = remaining := by
              dsimp only [predecessorStatement]
              omega
            simp only [ite_eq_left counter, previousCount]
          · simp only [ite_eq_right counter]

noncomputable def law (initial : PMF (Statement × Envelope application)) :
    PMF (HyperNovaHistoryProbability.Sample application) :=
  initial.bind fun input =>
    (results application fits source input.1 input.2).map (fun returns => (input.1, input.2, returns))

theorem initial_marginal (initial : PMF (Statement × Envelope application)) :
    (law application fits source initial).map (fun sample => (sample.1, sample.2.1)) = initial := by
  rw [law, PMF.map_bind]
  have forget (input : Statement × Envelope application) :
      ((results application fits source input.1 input.2).map (fun returns => (input.1, input.2, returns))).map
        (fun sample => (sample.1, sample.2.1)) = PMF.pure input := by
    rw [PMF.map_comp]
    exact PMF.map_const (results application fits source input.1 input.2) input
  simp only [forget, PMF.bind_pure]

end NightstreamFPrime.Export.Stage1.HyperNovaHistoryLaw
