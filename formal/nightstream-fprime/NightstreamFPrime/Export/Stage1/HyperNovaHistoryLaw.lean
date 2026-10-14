import NightstreamFPrime.Export.Stage1.HyperNovaHistoryProbability

/-!
Owns the adaptive law of the source returns consumed by the selected reverse
history. Each call uses the exact current statement and decoded payload.
The walk stops at the base or an abort; its natural counter supplies the
recursion bound.
-/

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

/-- Draw source results in the order read by the actual reverse program.
The recursion counter is exactly the advertised iteration, not a separate
caller-supplied depth limit. The base makes no source call; aborts are retained. -/
noncomputable def results (statement : Statement) (proof : Envelope application) : PMF (List (SourceResult application)) :=
  resultsAt application fits source statement.iteration statement proof

/-- The public recursion equation uses the exact predecessor counter. The
private structural bound adds no call and changes no stopped or abort branch. -/
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
      cases proof <;> simp only [results, count, resultsAt, if_true]
  | succ remaining =>
      cases proof with
      | bottom => simp only [results, count, resultsAt]
      | recursive payload =>
          simp only [results, count, resultsAt, Nat.succ_ne_zero, if_false]
          by_cases counter : (decodedInput application fits payload).iteration + 1 = remaining + 1
          · have previousCount : (predecessorStatement application fits payload).iteration = remaining := by
              dsimp only [predecessorStatement]
              omega
            simp only [if_pos counter, previousCount]
          · simp only [if_neg counter]

/-- Retain the initial terminal opening and all source returns from its
adaptive reverse run. -/
noncomputable def law (initial : PMF (Statement × Envelope application)) :
    PMF (HyperNovaHistoryProbability.Sample application) :=
  initial.bind fun input =>
    (results application fits source input.1 input.2).map (fun returns => (input.1, input.2, returns))

/-- Sampling the reverse run preserves the exact initial terminal law. -/
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
